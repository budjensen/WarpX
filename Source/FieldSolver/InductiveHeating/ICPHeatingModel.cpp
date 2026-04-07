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

#include "Parallelization/WarpXSumGuardCells.H"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

using namespace amrex;
using warpx::fields::FieldType;

// =============================================================================
// Constructor and parameter parsing
// =============================================================================

ICPHeatingModel::ICPHeatingModel()
{
    ReadParameters();
}

void ICPHeatingModel::ReadParameters()
{
    ParmParse pp_icp("icp_heating");

    pp_icp.query("do_heating", m_do_icp_heating);
    if (!m_do_icp_heating) { return; }

    // Enforce 1D electrostatic mode
#ifndef WARPX_DIM_1D_Z
    WARPX_ABORT_WITH_MESSAGE(
        "ICP heating is only supported in 1D (z-direction) simulations. "
        "Please compile WarpX with -DWarpX_DIMS=1.");
#endif
    if (WarpX::electromagnetic_solver_id != ElectromagneticSolverAlgo::None) {
        WARPX_ABORT_WITH_MESSAGE(
            "ICP heating is only compatible with electrostatic solvers. "
            "Set warpx.do_electrostatic = labframe.");
    }

    utils::parser::queryWithParser(pp_icp, "frequency", m_icp_frequency);
    m_icp_omega = 2.0_rt * MathConst::pi * m_icp_frequency;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_icp_frequency > 0.0_rt, "icp_heating.frequency must be positive");

    utils::parser::queryWithParser(pp_icp, "z_min", m_z_min);
    utils::parser::queryWithParser(pp_icp, "z_max", m_z_max);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_z_max > m_z_min, "icp_heating: z_max must be greater than z_min");

    // J_0(z,t) amplitude — constant scalar or parser expression
    std::string j0_str;
    amrex::ParticleReal j0_amplitude = 0;
    if (utils::parser::queryWithParser(pp_icp, "j0_amplitude", j0_amplitude)) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            j0_amplitude >= 0, "icp_heating.j0_amplitude must be non-negative");
        j0_str = std::to_string(j0_amplitude);
        m_j0_parser = std::make_unique<Parser>(
            utils::parser::makeParser(j0_str, {"z", "t"}));
    } else {
        utils::parser::Store_parserString(pp_icp, "j0_amplitude(z,t)", j0_str);
        m_j0_parser = std::make_unique<Parser>(
            utils::parser::makeParser(j0_str, {"z", "t"}));
    }
    m_j0_parser_exe = m_j0_parser->compile<2>();

    // Field limiter
    utils::parser::queryWithParser(pp_icp, "ey_max", m_ey_max);

    // Integrator selection
    std::string integrator_str = "euler";
    pp_icp.query("integrator", integrator_str);
    if      (integrator_str == "euler") { m_integrator = ICPIntegrator::Euler; }
    else if (integrator_str == "ab2")   { m_integrator = ICPIntegrator::AB2;   }
    else if (integrator_str == "rk2")   { m_integrator = ICPIntegrator::RK2;   }
    else if (integrator_str == "rk4")   { m_integrator = ICPIntegrator::RK4;   }
    else {
        WARPX_ABORT_WITH_MESSAGE(
            "Unknown icp_heating.integrator '" + integrator_str +
            "'. Valid choices: euler, ab2, rk2, rk4.");
    }

    if (ParallelDescriptor::IOProcessor()) {
        Print() << "\n"
                << "----------------------------------------------------------------------\n"
                << "ICP Heating Model Parameters:\n"
                << "  Frequency:           " << m_icp_frequency  << " Hz\n"
                << "  Heating region:      z = [" << m_z_min << ", " << m_z_max << "] m\n"
                << "  J_0(z,t) expression: " << j0_str            << "\n"
                << "  E_y field limiter:   " << m_ey_max          << " V/m\n"
                << "  Integrator:          " << integrator_str     << "\n"
                << "----------------------------------------------------------------------\n\n";
    }
}

// =============================================================================
// MultiFab allocation
// =============================================================================

void ICPHeatingModel::AllocateLevelMFs(
    ablastr::fields::MultiFabRegister& fields,
    int lev,
    const BoxArray& ba,
    const DistributionMapping& dm,
    const IntVect& ng)
{
    if (!m_do_icp_heating) { return; }

    // Prescribed target current density (diagnostic output)
    fields.alloc_init("Jy_icp_target", lev, ba, dm, 1, ng, 0.0_rt);

    // E_y^n snapshot — base for all integrator branches each step
    fields.alloc_init("Ey_icp_base",   lev, ba, dm, 1, ng, 0.0_rt);

    // RK stage storage: F_i = (J_target - J_cond) / eps0
    // k1 is also used by Euler and AB2 for their single F evaluation.
    fields.alloc_init("k1_icp",        lev, ba, dm, 1, ng, 0.0_rt);
    fields.alloc_init("k2_icp",        lev, ba, dm, 1, ng, 0.0_rt);
    fields.alloc_init("k3_icp",        lev, ba, dm, 1, ng, 0.0_rt);
    fields.alloc_init("k4_icp",        lev, ba, dm, 1, ng, 0.0_rt);

    // F_{n-1} for Adams-Bashforth 2-step history
    fields.alloc_init("F_prev_icp",    lev, ba, dm, 1, ng, 0.0_rt);

    // A permanently zero-valued field used as Ex=Ez=Bx=By=Bz=0 in
    // PushTransverseMomenta. Never written to after initialization.
    fields.alloc_init("zero_field_icp", lev, ba, dm, 1, ng, 0.0_rt);
}

// =============================================================================
// Top-level update dispatched to the selected integrator
// =============================================================================

void ICPHeatingModel::UpdateTransverseElectricField(
    ablastr::fields::MultiFabRegister& fields,
    int lev,
    amrex::Real time,
    amrex::Real dt,
    MultiParticleContainer& mpc)
{
    if (!m_do_icp_heating) { return; }
    WARPX_PROFILE("ICPHeatingModel::UpdateTransverseElectricField");

    using ablastr::fields::Direction;

    MultiFab* Ey_fp   = fields.get(FieldType::Efield_fp, Direction{1}, lev);
    MultiFab* Ey_base = fields.get("Ey_icp_base", lev);
    MultiFab* k1      = fields.get("k1_icp",      lev);

    // Snapshot E_y^n — all integrator branches use this as their starting point
    MultiFab::Copy(*Ey_base, *Ey_fp, 0, 0, 1, 0);

    // -------------------------------------------------------------------------
    // Forward Euler (baseline)
    //
    //   E_y^{n+1} = E_y^n + dt * F_n
    //   F_n = (J_target(t_n) - J_cond^n) / eps0
    // -------------------------------------------------------------------------
    if (m_integrator == ICPIntegrator::Euler) {

        DepositTransverseCurrent(fields, lev, dt, mpc);    // current_fp[1] <- J_cond^n
        ComputeEyRHS(fields, lev, time, *k1);              // k1 <- F_n
        SetTransverseElectricField(fields, lev,            // Ey_fp <- Ey^n + dt*k1
                                   *Ey_base, dt, *k1);

        m_is_first_step = false;
    }

    // -------------------------------------------------------------------------
    // Adams-Bashforth 2-step
    //
    //   E_y^{n+1} = E_y^n + dt * (3/2 * F_n - 1/2 * F_{n-1})
    //
    // Bootstrap: on the first step, F_{n-1} is unavailable so fall back to
    // Euler.  F_n is then copied into F_prev to seed the next step.
    // -------------------------------------------------------------------------
    else if (m_integrator == ICPIntegrator::AB2) {

        MultiFab* F_prev = fields.get("F_prev_icp", lev);

        DepositTransverseCurrent(fields, lev, dt, mpc);   // J_cond^n → current_fp[1]
        ComputeEyRHS(fields, lev, time, *k1);             // k1 = F_n

        if (m_is_first_step) {
            // Bootstrap with Euler — F_{n-1} not yet available
            SetTransverseElectricField(fields, lev, *Ey_base, dt, *k1);
        } else {
            ApplyAB2Update(fields, lev, *Ey_base, dt, *k1, *F_prev);
        }

        MultiFab::Copy(*F_prev, *k1, 0, 0, 1, 0);   // F_{n-1} ← F_n for next step
        m_is_first_step = false;
    }

    // -------------------------------------------------------------------------
    // RK2 / Heun's method
    //
    //   k1 = F(E_y^n,        t_n)       — deposit with uy^n
    //   k2 = F(E_y^n + dt*k1, t_n + dt) — push uy^n by dt, deposit
    //   E_y^{n+1} = E_y^n + (dt/2) * (k1 + k2)
    //   Restore uy^n
    // -------------------------------------------------------------------------
    else if (m_integrator == ICPIntegrator::RK2) {

        MultiFab* k2 = fields.get("k2_icp", lev);

        SaveTransverseMomenta(mpc, lev);

        // Stage 1 — k1 = F(E_y^n, t_n)
        DepositTransverseCurrent(fields, lev, dt, mpc);
        ComputeEyRHS(fields, lev, time, *k1);

        // Stage 2 — k2 = F(E_y^n + dt*k1, t_n + dt)
        SetTransverseElectricField(fields, lev, *Ey_base, dt, *k1);   // Ey_fp ← Ey^n + dt*k1
        PushTransverseMomenta(fields, lev, dt, mpc);                   // uy ← uy^n + dt*(q/m)*Ey
        DepositTransverseCurrent(fields, lev, dt, mpc);
        ComputeEyRHS(fields, lev, time + dt, *k2);

        // Final: E_y^{n+1} = E_y^n + (dt/2)*(k1 + k2)
        ApplyRK2Update(fields, lev, *Ey_base, dt, *k1, *k2);
        RestoreTransverseMomenta(mpc, lev);   // uy back to uy^n

        m_is_first_step = false;
    }

    // -------------------------------------------------------------------------
    // Classic 4-stage Runge-Kutta (RK4)
    //
    //   k1 = F(E_y^n,           t_n)
    //   k2 = F(E_y^n + dt/2*k1, t_n + dt/2)   push uy^n by dt/2
    //   k3 = F(E_y^n + dt/2*k2, t_n + dt/2)   restore uy^n, push by dt/2
    //   k4 = F(E_y^n + dt  *k3, t_n + dt  )   restore uy^n, push by dt
    //   E_y^{n+1} = E_y^n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    //   Restore uy^n
    // -------------------------------------------------------------------------
    else if (m_integrator == ICPIntegrator::RK4) {

        MultiFab* k2 = fields.get("k2_icp", lev);
        MultiFab* k3 = fields.get("k3_icp", lev);
        MultiFab* k4 = fields.get("k4_icp", lev);

        SaveTransverseMomenta(mpc, lev);

        // k1 — current particle state (uy^n), field = E_y^n
        DepositTransverseCurrent(fields, lev, dt, mpc);
        ComputeEyRHS(fields, lev, time, *k1);

        // k2 — field = E_y^n + (dt/2)*k1, particles pushed by dt/2
        SetTransverseElectricField(fields, lev, *Ey_base, 0.5_rt*dt, *k1);
        PushTransverseMomenta(fields, lev, 0.5_rt*dt, mpc);
        DepositTransverseCurrent(fields, lev, dt, mpc);
        ComputeEyRHS(fields, lev, time + 0.5_rt*dt, *k2);

        // k3 — field = E_y^n + (dt/2)*k2, particles pushed by dt/2 from uy^n
        RestoreTransverseMomenta(mpc, lev);
        SetTransverseElectricField(fields, lev, *Ey_base, 0.5_rt*dt, *k2);
        PushTransverseMomenta(fields, lev, 0.5_rt*dt, mpc);
        DepositTransverseCurrent(fields, lev, dt, mpc);
        ComputeEyRHS(fields, lev, time + 0.5_rt*dt, *k3);

        // k4 — field = E_y^n + dt*k3, particles pushed by dt from uy^n
        RestoreTransverseMomenta(mpc, lev);
        SetTransverseElectricField(fields, lev, *Ey_base, dt, *k3);
        PushTransverseMomenta(fields, lev, dt, mpc);
        DepositTransverseCurrent(fields, lev, dt, mpc);
        ComputeEyRHS(fields, lev, time + dt, *k4);

        // Final: E_y^{n+1} = E_y^n + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        ApplyRK4Update(fields, lev, *Ey_base, dt, *k1, *k2, *k3, *k4);
        RestoreTransverseMomenta(mpc, lev);   // uy back to uy^n

        m_is_first_step = false;
    }
}

// =============================================================================
// Workhorse: save / restore transverse particle momenta
// =============================================================================

void ICPHeatingModel::SaveTransverseMomenta(MultiParticleContainer& mpc, int lev)
{
    WARPX_PROFILE("ICPHeatingModel::SaveTransverseMomenta");

    const int nspecies = mpc.nSpecies();
    m_saved_uy.resize(nspecies);

    // Iterate tiles without a surrounding omp parallel region so the order is
    // serial and deterministic — required for correct Save/Restore pairing.
    for (int ispec = 0; ispec < nspecies; ++ispec) {
        auto& pc = mpc.GetParticleContainer(ispec);
        auto& saved = m_saved_uy[ispec];
        saved.clear();

        for (WarpXParIter pti(pc, lev); pti.isValid(); ++pti) {
            const auto& uy_vec = pti.GetAttribs()[PIdx::uy];
            saved.insert(saved.end(), uy_vec.begin(), uy_vec.end());
        }
    }
}

void ICPHeatingModel::RestoreTransverseMomenta(MultiParticleContainer& mpc, int lev)
{
    WARPX_PROFILE("ICPHeatingModel::RestoreTransverseMomenta");

    const int nspecies = mpc.nSpecies();

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        static_cast<int>(m_saved_uy.size()) == nspecies,
        "ICPHeatingModel::RestoreTransverseMomenta called before SaveTransverseMomenta, "
        "or species count changed between Save and Restore.");

    for (int ispec = 0; ispec < nspecies; ++ispec) {
        auto& pc = mpc.GetParticleContainer(ispec);
        const auto& saved = m_saved_uy[ispec];
        long offset = 0;

        for (WarpXParIter pti(pc, lev); pti.isValid(); ++pti) {
            auto& uy_vec = pti.GetAttribs()[PIdx::uy];
            const long np = pti.numParticles();

            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                offset + np <= static_cast<long>(saved.size()),
                "ICPHeatingModel::RestoreTransverseMomenta: saved buffer is smaller "
                "than current particle count. Did the particle count change between "
                "Save and Restore (e.g. due to ionization)?");

            std::copy(saved.begin() + offset,
                      saved.begin() + offset + np,
                      uy_vec.begin());
            offset += np;
        }
    }
}

// =============================================================================
// Workhorse: push transverse momenta using current Efield_fp[1]
// =============================================================================

void ICPHeatingModel::PushTransverseMomenta(
    ablastr::fields::MultiFabRegister& fields,
    int lev,
    amrex::Real dt_sub,
    MultiParticleContainer& mpc)
{
    WARPX_PROFILE("ICPHeatingModel::PushTransverseMomenta");

    using ablastr::fields::Direction;

    // Efield_fp[1] holds the intermediate E_y to push with (set by the caller
    // via SetTransverseElectricField before this function is called).
    MultiFab* Ey_mf   = fields.get(FieldType::Efield_fp, Direction{1}, lev);

    // A permanently-zero field used for all other E and B components.
    // With only E_y nonzero and B = 0, the Boris push in PushP reduces to the
    // non-relativistic update:   uy += (q/m) * E_y(z_p) * dt_sub
    // ux and uz are unchanged. All species are pushed with their own q/m.
    MultiFab* zero_mf = fields.get("zero_field_icp", lev);

    mpc.PushP(lev, dt_sub,
              *zero_mf,   // Ex = 0
              *Ey_mf,     // Ey = intermediate ICP field
              *zero_mf,   // Ez = 0
              *zero_mf,   // Bx = 0
              *zero_mf,   // By = 0
              *zero_mf);  // Bz = 0
}

// =============================================================================
// Workhorse: deposit J_y and synchronize across MPI boundaries
// =============================================================================

void ICPHeatingModel::DepositTransverseCurrent(
    ablastr::fields::MultiFabRegister& fields,
    int lev,
    amrex::Real dt,
    MultiParticleContainer& mpc)
{
    WARPX_PROFILE("ICPHeatingModel::DepositTransverseCurrent");

    using ablastr::fields::Direction;

    // Zero all current components. In 1D ES mode only J_y is meaningful for
    // ICP, but DepositCurrent writes all three components so we zero all three.
    fields.get(FieldType::current_fp, Direction{0}, lev)->setVal(0.0_rt);
    fields.get(FieldType::current_fp, Direction{1}, lev)->setVal(0.0_rt);
    fields.get(FieldType::current_fp, Direction{2}, lev)->setVal(0.0_rt);

    ablastr::fields::MultiLevelVectorField J_fp =
        fields.get_mr_levels_alldirs(FieldType::current_fp, lev);

    // Deposit J from all particle species using their current positions and uy.
    // relative_time = 0 means deposit at the current particle positions.
    mpc.DepositCurrent(J_fp, dt, 0.0_rt);

    // Sum J_y ghost-cell contributions across MPI domain boundaries.
    // Only J_y is used by the ICP algorithm; J_x and J_z are discarded.
    const auto& period = WarpX::GetInstance().Geom(lev).periodicity();
    MultiFab* Jy = fields.get(FieldType::current_fp, Direction{1}, lev);
    WarpXSumGuardCells(*Jy, period, Jy->nGrowVect());
}

// =============================================================================
// Workhorse: compute the E_y ODE right-hand side
// =============================================================================

void ICPHeatingModel::ComputeEyRHS(
    ablastr::fields::MultiFabRegister& fields,
    int lev,
    amrex::Real time,
    amrex::MultiFab& F_out)
{
    WARPX_PROFILE("ICPHeatingModel::ComputeEyRHS");

    using ablastr::fields::Direction;

    MultiFab* Jy_cond_mf   = fields.get(FieldType::current_fp, Direction{1}, lev);
    MultiFab* Jy_target_mf = fields.get("Jy_icp_target", lev);

    const Geometry& geom = WarpX::GetInstance().Geom(lev);
    const Real* dx   = geom.CellSize();
    const Real* plo  = geom.ProbLo();
    const Real dz    = dx[WARPX_ZINDEX];
    const Real zmin  = plo[WARPX_ZINDEX];

    const Real inv_eps0 = 1.0_rt / PhysConst::ep0;
    const Real omega    = m_icp_omega;
    const Real phase    = omega * time;
    const Real z_min    = m_z_min;
    const Real z_max    = m_z_max;

    auto j0_exe = m_j0_parser_exe;

    for (MFIter mfi(F_out, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto const& F_arr      = F_out.array(mfi);
        auto const& Jcond_arr  = Jy_cond_mf->const_array(mfi);
        auto const& Jtgt_arr   = Jy_target_mf->array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            const Real z = zmin + (i + 0.5_rt) * dz;

            if (z >= z_min && z <= z_max) {
                const Real J_target = j0_exe(z, time) * std::sin(phase);
                Jtgt_arr(i, j, k) = J_target;                        // diagnostic
                F_arr(i, j, k)    = (J_target - Jcond_arr(i, j, k)) * inv_eps0;
            } else {
                Jtgt_arr(i, j, k) = 0.0_rt;
                F_arr(i, j, k)    = 0.0_rt;
            }
        });
    }
}

// =============================================================================
// Workhorse: set Efield_fp[1] = Ey_base + coeff * F
// =============================================================================

void ICPHeatingModel::SetTransverseElectricField(
    ablastr::fields::MultiFabRegister& fields,
    int lev,
    amrex::MultiFab const& Ey_base,
    amrex::Real coeff,
    amrex::MultiFab const& F)
{
    WARPX_PROFILE("ICPHeatingModel::SetTransverseElectricField");

    using ablastr::fields::Direction;

    MultiFab* Ey_fp = fields.get(FieldType::Efield_fp, Direction{1}, lev);

    const Geometry& geom = WarpX::GetInstance().Geom(lev);
    const Real* dx  = geom.CellSize();
    const Real* plo = geom.ProbLo();
    const Real dz   = dx[WARPX_ZINDEX];
    const Real zmin = plo[WARPX_ZINDEX];

    const Real z_min  = m_z_min;
    const Real z_max  = m_z_max;
    const Real ey_max = m_ey_max;

    for (MFIter mfi(*Ey_fp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto const& Ey_arr   = Ey_fp->array(mfi);
        auto const& base_arr = Ey_base.const_array(mfi);
        auto const& F_arr    = F.const_array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int kk) {
            const Real z = zmin + (i + 0.5_rt) * dz;

            if (z >= z_min && z <= z_max) {
                Real Ey_new = base_arr(i, j, kk) + coeff * F_arr(i, j, kk);
                Ey_new = amrex::max(-ey_max, amrex::min(ey_max, Ey_new));
                Ey_arr(i, j, kk) = Ey_new;
            } else {
                Ey_arr(i, j, kk) = 0.0_rt;
            }
        });
    }
}

// =============================================================================
// Final-update helpers: write weighted RHS sum into Efield_fp[1]
// Each applies the same field limiter and region mask as SetTransverseElectricField.
// =============================================================================

void ICPHeatingModel::ApplyAB2Update(
    ablastr::fields::MultiFabRegister& fields,
    int lev,
    amrex::MultiFab const& Ey_base,
    amrex::Real dt,
    amrex::MultiFab const& F_n,
    amrex::MultiFab const& F_prev)
{
    WARPX_PROFILE("ICPHeatingModel::ApplyAB2Update");

    using ablastr::fields::Direction;

    MultiFab* Ey_fp = fields.get(FieldType::Efield_fp, Direction{1}, lev);

    const Geometry& geom = WarpX::GetInstance().Geom(lev);
    const Real* dx  = geom.CellSize();
    const Real* plo = geom.ProbLo();
    const Real dz   = dx[WARPX_ZINDEX];
    const Real zmin = plo[WARPX_ZINDEX];

    const Real z_min  = m_z_min;
    const Real z_max  = m_z_max;
    const Real ey_max = m_ey_max;

    for (MFIter mfi(*Ey_fp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto const& Ey_arr    = Ey_fp->array(mfi);
        auto const& base_arr  = Ey_base.const_array(mfi);
        auto const& fn_arr    = F_n.const_array(mfi);
        auto const& fprev_arr = F_prev.const_array(mfi);

        // E_y^{n+1} = E_y^n + dt * (3/2 * F_n - 1/2 * F_{n-1})
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int kk) {
            const Real z = zmin + (i + 0.5_rt) * dz;

            if (z >= z_min && z <= z_max) {
                Real Ey_new = base_arr(i, j, kk)
                            + dt * (1.5_rt * fn_arr(i, j, kk)
                                  - 0.5_rt * fprev_arr(i, j, kk));
                Ey_new = amrex::max(-ey_max, amrex::min(ey_max, Ey_new));
                Ey_arr(i, j, kk) = Ey_new;
            } else {
                Ey_arr(i, j, kk) = 0.0_rt;
            }
        });
    }
}

void ICPHeatingModel::ApplyRK2Update(
    ablastr::fields::MultiFabRegister& fields,
    int lev,
    amrex::MultiFab const& Ey_base,
    amrex::Real dt,
    amrex::MultiFab const& k1,
    amrex::MultiFab const& k2)
{
    WARPX_PROFILE("ICPHeatingModel::ApplyRK2Update");

    using ablastr::fields::Direction;

    MultiFab* Ey_fp = fields.get(FieldType::Efield_fp, Direction{1}, lev);

    const Geometry& geom = WarpX::GetInstance().Geom(lev);
    const Real* dx  = geom.CellSize();
    const Real* plo = geom.ProbLo();
    const Real dz   = dx[WARPX_ZINDEX];
    const Real zmin = plo[WARPX_ZINDEX];

    const Real z_min  = m_z_min;
    const Real z_max  = m_z_max;
    const Real ey_max = m_ey_max;
    const Real dt_half = 0.5_rt * dt;

    for (MFIter mfi(*Ey_fp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto const& Ey_arr   = Ey_fp->array(mfi);
        auto const& base_arr = Ey_base.const_array(mfi);
        auto const& k1_arr   = k1.const_array(mfi);
        auto const& k2_arr   = k2.const_array(mfi);

        // E_y^{n+1} = E_y^n + (dt/2) * (k1 + k2)
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int kk) {
            const Real z = zmin + (i + 0.5_rt) * dz;

            if (z >= z_min && z <= z_max) {
                Real Ey_new = base_arr(i, j, kk)
                            + dt_half * (k1_arr(i, j, kk) + k2_arr(i, j, kk));
                Ey_new = amrex::max(-ey_max, amrex::min(ey_max, Ey_new));
                Ey_arr(i, j, kk) = Ey_new;
            } else {
                Ey_arr(i, j, kk) = 0.0_rt;
            }
        });
    }
}

void ICPHeatingModel::ApplyRK4Update(
    ablastr::fields::MultiFabRegister& fields,
    int lev,
    amrex::MultiFab const& Ey_base,
    amrex::Real dt,
    amrex::MultiFab const& k1,
    amrex::MultiFab const& k2,
    amrex::MultiFab const& k3,
    amrex::MultiFab const& k4)
{
    WARPX_PROFILE("ICPHeatingModel::ApplyRK4Update");

    using ablastr::fields::Direction;

    MultiFab* Ey_fp = fields.get(FieldType::Efield_fp, Direction{1}, lev);

    const Geometry& geom = WarpX::GetInstance().Geom(lev);
    const Real* dx  = geom.CellSize();
    const Real* plo = geom.ProbLo();
    const Real dz   = dx[WARPX_ZINDEX];
    const Real zmin = plo[WARPX_ZINDEX];

    const Real z_min    = m_z_min;
    const Real z_max    = m_z_max;
    const Real ey_max   = m_ey_max;
    const Real dt_sixth = dt / 6.0_rt;

    for (MFIter mfi(*Ey_fp, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto const& Ey_arr   = Ey_fp->array(mfi);
        auto const& base_arr = Ey_base.const_array(mfi);
        auto const& k1_arr   = k1.const_array(mfi);
        auto const& k2_arr   = k2.const_array(mfi);
        auto const& k3_arr   = k3.const_array(mfi);
        auto const& k4_arr   = k4.const_array(mfi);

        // E_y^{n+1} = E_y^n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int kk) {
            const Real z = zmin + (i + 0.5_rt) * dz;

            if (z >= z_min && z <= z_max) {
                Real Ey_new = base_arr(i, j, kk)
                            + dt_sixth * (k1_arr(i, j, kk)
                                        + 2.0_rt * k2_arr(i, j, kk)
                                        + 2.0_rt * k3_arr(i, j, kk)
                                        + k4_arr(i, j, kk));
                Ey_new = amrex::max(-ey_max, amrex::min(ey_max, Ey_new));
                Ey_arr(i, j, kk) = Ey_new;
            } else {
                Ey_arr(i, j, kk) = 0.0_rt;
            }
        });
    }
}
