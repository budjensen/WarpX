/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: WarpX contributors
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ICPHeatingModel.H"

#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Array.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Config.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Parser.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <memory>

using namespace amrex;
using warpx::fields::FieldType;

ICPHeatingModel::ICPHeatingModel()
{
    ReadParameters();
}

void ICPHeatingModel::ReadParameters()
{
    ParmParse pp_icp("icp_heating");

    // Check if ICP heating is enabled
    pp_icp.query("do_heating", m_do_icp_heating);
    if (!m_do_icp_heating) {
        return;
    }

    // Check that we are in 1D mode
#ifndef WARPX_DIM_1D_Z
    WARPX_ABORT_WITH_MESSAGE(
        "ICP heating is only supported in 1D (z-direction) simulations. "
        "Please compile WarpX with -DWarpX_DIMS=1 or DIM=1."
    );
#endif

    // Check that we are using electrostatic solver
    if (WarpX::electromagnetic_solver_id != ElectromagneticSolverAlgo::None) {
        WARPX_ABORT_WITH_MESSAGE(
            "ICP heating is only compatible with electrostatic solvers. "
            "Please set warpx.do_electrostatic = labframe (or another ES option)."
        );
    }

    // Read frequency
    utils::parser::queryWithParser(pp_icp, "frequency", m_icp_frequency);
    m_icp_omega = 2.0_rt * MathConst::pi * m_icp_frequency;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_icp_frequency > 0.0_rt,
        "ICP heating frequency must be positive"
    );

    // Read spatial extent
    utils::parser::queryWithParser(pp_icp, "z_min", m_z_min);
    utils::parser::queryWithParser(pp_icp, "z_max", m_z_max);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_z_max > m_z_min,
        "ICP heating region: z_max must be greater than z_min"
    );

    // Read J_0(z,t) profile
    // Can be a constant value or a parser expression
    std::string j0_str;
    amrex::ParticleReal j0_amplitude = 0;
    if (utils::parser::queryWithParser(pp_icp, "j0_amplitude", j0_amplitude)) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (j0_amplitude >= 0),
            "The j0_amplitude must be non-negative.");
        j0_str = std::to_string(j0_amplitude);
        m_j0_parser = std::make_unique<Parser>(
            utils::parser::makeParser(j0_str, {"z", "t"}));
    }
    else {
        utils::parser::Store_parserString(pp_icp, "j0_amplitude(z,t)", j0_str);
        m_j0_parser = std::make_unique<Parser>(
            utils::parser::makeParser(j0_str, {"z", "t"}));
    }

    // Compile parser
    m_j0_parser_exe = m_j0_parser->compile<2>();

    // Read field limiter
    utils::parser::queryWithParser(pp_icp, "ey_max", m_ey_max);

    // Print summary
    if (ParallelDescriptor::IOProcessor()) {
        Print() << "\n";
        Print() << "-------------------------------------------------------------------------------\n";
        Print() << "ICP Heating Model Parameters:\n";
        Print() << "  Frequency:           " << m_icp_frequency << " Hz\n";
        Print() << "  Heating region:      z = [" << m_z_min << ", " << m_z_max << "] m\n";
        Print() << "  J_0(z,t) expression: " << j0_str << "\n";
        Print() << "  E_y field limiter:   " << m_ey_max << " V/m\n";
        Print() << "-------------------------------------------------------------------------------\n";
        Print() << "\n";
    }
}

void ICPHeatingModel::AllocateLevelMFs(
    ablastr::fields::MultiFabRegister& fields,
    int lev,
    const BoxArray& ba,
    const DistributionMapping& dm,
    const IntVect& ng)
{
    if (!m_do_icp_heating) return;

    // Note: We use the existing Efield_fp[1] (E_y) and current_fp[1] (J_y)
    // which are unused in 1D electrostatic simulations.

    // Allocate J_y target (prescribed current density)
    // Fields are cell-centered in 1D electrostatic
    fields.alloc_init("Jy_icp_target", lev, ba, dm, 1, ng, 0.0_rt);
}

void ICPHeatingModel::ComputeTransverseConductionCurrent(
    ablastr::fields::MultiFabRegister& fields,
    int lev,
    Real dt,
    MultiParticleContainer& mpc)
{
    if (!m_do_icp_heating) return;

    WARPX_PROFILE("ICPHeatingModel::ComputeTransverseConductionCurrent");

    using ablastr::fields::Direction;

    // Get all current components
    // In 1D electrostatic, current_fp is unused by the Poisson solver
    MultiFab* Jx_mf = fields.get(FieldType::current_fp, Direction{0}, lev);
    MultiFab* Jy_mf = fields.get(FieldType::current_fp, Direction{1}, lev);
    MultiFab* Jz_mf = fields.get(FieldType::current_fp, Direction{2}, lev);
    AMREX_ALWAYS_ASSERT(Jx_mf != nullptr);
    AMREX_ALWAYS_ASSERT(Jy_mf != nullptr);
    AMREX_ALWAYS_ASSERT(Jz_mf != nullptr);

    // Reset all current components to zero before deposition
    // DepositCurrent writes to all three components
    Jx_mf->setVal(0.0_rt);
    Jy_mf->setVal(0.0_rt);
    Jz_mf->setVal(0.0_rt);

    // Use WarpX's optimized current deposition for all species
    // This deposits J_x, J_y, and J_z from particle velocities
    // Note: We deposit everywhere (not just heating region) for physical correctness
    // The heating region filter is applied only when updating E_y
    ablastr::fields::MultiLevelVectorField J_fp =
        fields.get_mr_levels_alldirs(FieldType::current_fp, lev);

    // Deposit current with dt and relative_time=0 (instantaneous current at current positions)
    mpc.DepositCurrent(J_fp, dt, 0.0_rt);
}

void ICPHeatingModel::UpdateTransverseElectricField(
    ablastr::fields::MultiFabRegister& fields,
    int lev,
    Real time,
    Real dt)
{
    if (!m_do_icp_heating) return;

    WARPX_PROFILE("ICPHeatingModel::UpdateTransverseElectricField");

    // Get MultiFabs - using existing E_y and J_y from field register
    using ablastr::fields::Direction;
    MultiFab* Ey_mf = fields.get(FieldType::Efield_fp, Direction{1}, lev);  // E_y
    MultiFab* Jy_mf = fields.get(FieldType::current_fp, Direction{1}, lev);  // J_y (conduction)
    MultiFab* Jy_target_mf = fields.get("Jy_icp_target", lev);

    AMREX_ALWAYS_ASSERT(Ey_mf != nullptr);
    AMREX_ALWAYS_ASSERT(Jy_mf != nullptr);
    AMREX_ALWAYS_ASSERT(Jy_target_mf != nullptr);

    // Get geometry
    const Geometry& geom = WarpX::GetInstance().Geom(lev);
    const Real* dx = geom.CellSize();
    const Real* plo = geom.ProbLo();
    const Real dz = dx[WARPX_ZINDEX];
    const Real zmin = plo[WARPX_ZINDEX];

    // Constants
    const Real dt_over_eps0 = dt / PhysConst::ep0;
    const Real omega = m_icp_omega;
    const Real phase = omega * time;
    const Real z_min = m_z_min;
    const Real z_max = m_z_max;
    const Real ey_max = m_ey_max;

    // Get J_0(z,t) parser executor for device
    auto j0_exe = m_j0_parser_exe;

    // Update E_y using Ampere's law (without displacement current)
    for (MFIter mfi(*Ey_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto const& Ey_arr = Ey_mf->array(mfi);
        auto const& Jy_cond_arr = Jy_mf->const_array(mfi);
        auto const& Jy_target_arr = Jy_target_mf->array(mfi);

        // Compute J_target and update E_y in a single loop
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            // Get z position at cell center
            // In 1D, the spatial index is 'i', not 'k'
            const Real z = zmin + (i + 0.5_rt) * dz;

            // Apply update only in heating region
            if (z >= z_min && z <= z_max) {
                // Evaluate J_0(z,t) and compute J_target = J_0(z,t) * sin(omega*t)
                const Real J0_zt = j0_exe(z, time);
                const Real J_target = J0_zt * std::sin(phase);
                Jy_target_arr(i, j, k) = J_target;

                // Update E_y using Ampere's law
                const Real dJ = J_target - Jy_cond_arr(i, j, k);
                Real Ey_new = Ey_arr(i, j, k) + dt_over_eps0 * dJ;

                // Apply field limiter
                Ey_new = amrex::max(-ey_max, amrex::min(ey_max, Ey_new));

                Ey_arr(i, j, k) = Ey_new;
            } else {
                // Zero both J_target and E_y outside the heating region
                Jy_target_arr(i, j, k) = 0.0_rt;
                Ey_arr(i, j, k) = 0.0_rt;
            }
        });
    }
}
