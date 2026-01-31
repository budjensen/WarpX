/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: WarpX contributors
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_ICPHEATINGMODEL_H_
#define WARPX_ICPHEATINGMODEL_H_

#include "ICPHeatingModel_fwd.H"

#include "Fields.H"
#include "Particles/MultiParticleContainer_fwd.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_IntVect.H>
#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <memory>
#include <string>

/**
 * \brief This class implements a quasi-inductive (ICP-like) heating model
 * for electrostatic PIC simulations in 1D.
 *
 * The model applies a prescribed transverse (y-direction) current density
 * J_y_target(z,t) = J_0(z) * sin(omega * t) within a specified spatial region
 * [z_min, z_max]. The transverse electric field E_y is evolved according to:
 *
 * E_y^{n+1}(z) = E_y^n(z) + (dt/epsilon_0) * [J_y_cond(z) - J_y_target(z,t)]
 *
 * where J_y_cond is the actual conduction current computed from particle
 * velocities in each cell.
 *
 * IMPORTANT: This implementation uses the existing Efield_fp[1] (E_y) and
 * current_fp[1] (J_y) MultiFabs directly. In 1D electrostatic simulations,
 * the transverse (x and y) components of E and J are not used by the Poisson
 * solver, so we can safely repurpose them for ICP heating. This has the
 * advantage that particles automatically gather E_y during the standard
 * field gather operation, requiring no modifications to the particle pusher.
 */
class ICPHeatingModel
{
public:
    ICPHeatingModel();

    /** Read user-defined model parameters. Called in constructor. */
    void ReadParameters();

    /**
     * Allocate ICP heating specific multifabs.
     *
     * Note: For the main E_y and J_y fields, we use the existing Efield_fp[1]
     * and current_fp[1] arrays which are unused in 1D electrostatic mode.
     * We only allocate J_y_target here.
     */
    void AllocateLevelMFs(
        ablastr::fields::MultiFabRegister& fields,
        int lev,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        const amrex::IntVect& ng
    );

    /** Check if ICP heating is enabled */
    [[nodiscard]] bool is_enabled() const { return m_do_icp_heating; }

    /** Get the heating frequency for timestep constraint */
    [[nodiscard]] amrex::Real get_frequency() const { return m_icp_frequency; }

    /**
     * \brief Compute the transverse conduction current J_y from particles
     *
     * Uses WarpX's DepositCurrent to compute J_y from particle velocities.
     * Deposits current at all spatial locations (not limited to heating region).
     *
     * IMPORTANT: After calling this function, WarpX::SumBoundaryJ must be called
     * to properly handle MPI domain boundaries before using the current values.
     *
     * \param[in] fields     Field MultiFab register
     * \param[in] lev        AMR level
     * \param[in] dt         Timestep (needed for DepositCurrent)
     * \param[in] mpc        Particle container
     */
    void ComputeTransverseConductionCurrent(
        ablastr::fields::MultiFabRegister& fields,
        int lev,
        amrex::Real dt,
        MultiParticleContainer& mpc
    );

    /**
     * \brief Update the transverse electric field E_y
     *
     * Applies the update equation:
     * E_y^{n+1} = E_y^n + (dt/eps0) * [J_y_cond - J_y_target]
     * within the heating region [z_min, z_max]. E_y is set to zero outside
     * the heating region.
     *
     * IMPORTANT: This function must be called AFTER SumBoundaryJ has been
     * applied to the current arrays to ensure correct MPI synchronization.
     *
     * \param[in] fields     Field MultiFab register
     * \param[in] lev        AMR level
     * \param[in] time       Current simulation time
     * \param[in] dt         Timestep
     */
    void UpdateTransverseElectricField(
        ablastr::fields::MultiFabRegister& fields,
        int lev,
        amrex::Real time,
        amrex::Real dt
    );

private:
    // Enable/disable flag
    bool m_do_icp_heating = false;

    // Frequency parameters
    amrex::Real m_icp_frequency = 0.0;  // Hz
    amrex::Real m_icp_omega = 0.0;      // rad/s

    // Spatial extent of heating region
    amrex::Real m_z_min = 0.0;
    amrex::Real m_z_max = 0.0;

    // J_0(z,t) amplitude profile - parser function of z and t
    std::unique_ptr<amrex::Parser> m_j0_parser;
    amrex::ParserExecutor<2> m_j0_parser_exe;

    // Maximum E_y field limiter for stability
    amrex::Real m_ey_max = 1.0e10;  // V/m (effectively unlimited by default)
};

#endif // WARPX_ICPHEATINGMODEL_H_
