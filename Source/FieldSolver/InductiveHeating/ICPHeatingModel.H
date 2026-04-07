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
 * \brief Selects the time integration scheme used to evolve the ICP E_y ODE.
 *
 * The ODE is: dE_y/dt = F  where F = (J_target - J_cond) / eps0
 *
 * - Euler: forward Euler (1st order, baseline)
 * - AB2:   Adams-Bashforth 2-step (2nd order, multistep)
 * - RK2:   Heun's method (2nd order, 2-stage Runge-Kutta)
 * - RK4:   Classic 4-stage Runge-Kutta (4th order)
 *
 * Set via input parameter: icp_heating.integrator = euler | ab2 | rk2 | rk4
 */
enum class ICPIntegrator { Euler, AB2, RK2, RK4 };

/**
 * \brief Implements a quasi-inductive (ICP-like) heating model for 1D ES PIC.
 *
 * A prescribed transverse current J_y_target(z,t) = J_0(z)*sin(omega*t) is
 * imposed in [z_min, z_max]. E_y is evolved to drive that current:
 *
 *   dE_y/dt = F(z,t) = [J_y_target(z,t) - J_y_cond(z,t)] / eps0
 *
 * where J_y_cond is the conduction current deposited from particle v_y.
 *
 * Implementation notes:
 * - Reuses Efield_fp[1] (E_y) and current_fp[1] (J_y), which are unused by
 *   the 1D electrostatic Poisson solver. Particles therefore gather E_y
 *   automatically in the standard field-gather step.
 * - Higher-order integrators (AB2, RK2, RK4) are built from a set of
 *   workhorse functions: SaveTransverseMomenta, RestoreTransverseMomenta,
 *   PushTransverseMomenta, DepositTransverseCurrent, ComputeEyRHS, and
 *   SetTransverseElectricField.
 */
class ICPHeatingModel
{
public:
    ICPHeatingModel();

    /** Read user-defined model parameters. Called in constructor. */
    void ReadParameters();

    /**
     * \brief Allocate persistent MultiFabs used by the ICP model.
     *
     * Allocates J_y_target (diagnostic), E_y_base (saves E_y^n at the start
     * of each step), k1-k4 (RK stage RHS storage), F_prev (AB2 history),
     * and zero_field (used as a zero-valued E/B field in PushTransverseMomenta).
     */
    void AllocateLevelMFs(
        ablastr::fields::MultiFabRegister& fields,
        int lev,
        const amrex::BoxArray& ba,
        const amrex::DistributionMapping& dm,
        const amrex::IntVect& ng
    );

    /** Returns true if ICP heating is active. */
    [[nodiscard]] bool is_enabled() const { return m_do_icp_heating; }

    /** Returns the drive frequency (Hz), used externally to check timestep constraints. */
    [[nodiscard]] amrex::Real get_frequency() const { return m_icp_frequency; }

    // =========================================================================
    // Top-level field update — called once per timestep from WarpXSolveFieldsES
    // =========================================================================

    /**
     * \brief Advance E_y by one timestep using the selected integrator.
     *
     * This is the single entry point called from WarpXSolveFieldsES.cpp each
     * step. It internally handles current deposition, MPI synchronization, and
     * the chosen integration scheme (Euler / AB2 / RK2 / RK4).
     *
     * \param[in] fields   Field MultiFab register
     * \param[in] lev      AMR level
     * \param[in] time     Current simulation time t_n
     * \param[in] dt       Timestep
     * \param[in] mpc      All particle species (used for deposit and RK pushes)
     */
    void UpdateTransverseElectricField(
        ablastr::fields::MultiFabRegister& fields,
        int lev,
        amrex::Real time,
        amrex::Real dt,
        MultiParticleContainer& mpc
    );

    // =========================================================================
    // Workhorse functions used by the higher-order integrators
    // =========================================================================

    /**
     * \brief Save transverse momenta (uy) for all particles across all species.
     *
     * Must be called once before a sequence of RK or multistep substep
     * evaluations. The saved state is restored by RestoreTransverseMomenta.
     * Tile iteration is serialized to guarantee consistent ordering with Restore.
     *
     * \param[in] mpc  All particle species
     * \param[in] lev  AMR level
     */
    void SaveTransverseMomenta(MultiParticleContainer& mpc, int lev);

    /**
     * \brief Restore transverse momenta (uy) to the state saved by
     *        SaveTransverseMomenta.
     *
     * Call between RK stages (to reset particle velocities to uy^n before each
     * new stage) and once after the final field update.
     *
     * \param[in] mpc  All particle species
     * \param[in] lev  AMR level
     */
    void RestoreTransverseMomenta(MultiParticleContainer& mpc, int lev);

    /**
     * \brief Push transverse momenta uy for all particles by dt_sub using
     *        the current Efield_fp[1] field.
     *
     * Calls mpc.PushP with B=0 and only E_y nonzero. With B=0 the Boris
     * rotation is the identity, so the update reduces to the non-relativistic:
     *   uy += (q/m) * E_y(z_p) * dt_sub
     * ux and uz are unchanged. All species are pushed with their correct q/m.
     *
     * Precondition: Efield_fp[1] must already hold the desired intermediate
     * E_y value (e.g. set by SetTransverseElectricField before calling).
     *
     * \param[in] fields   Field register (provides Efield_fp[1] and zero_field)
     * \param[in] lev      AMR level
     * \param[in] dt_sub   Substep duration (may be dt/2, dt, etc.)
     * \param[in] mpc      All particle species
     */
    void PushTransverseMomenta(
        ablastr::fields::MultiFabRegister& fields,
        int lev,
        amrex::Real dt_sub,
        MultiParticleContainer& mpc
    );

    /**
     * \brief Deposit transverse conduction current J_y from particle uy into
     *        current_fp[1], including MPI boundary synchronization.
     *
     * Zeros all current_fp components, deposits J from all species, then calls
     * SumBoundaryJ. On return, current_fp[1] holds the MPI-consistent J_cond.
     *
     * \param[in] fields  Field register
     * \param[in] lev     AMR level
     * \param[in] dt      Full simulation timestep (required by DepositCurrent)
     * \param[in] mpc     All particle species
     */
    void DepositTransverseCurrent(
        ablastr::fields::MultiFabRegister& fields,
        int lev,
        amrex::Real dt,
        MultiParticleContainer& mpc
    );

    /**
     * \brief Compute the E_y ODE right-hand side and write it to F_out.
     *
     *   F_out(z) = [J_target(z,t) - J_cond(z)] / eps0   inside [z_min, z_max]
     *   F_out(z) = 0                                      outside
     *
     * Also updates the Jy_icp_target diagnostic field as a side effect.
     *
     * Precondition: current_fp[1] must contain the deposited J_cond
     * (i.e. DepositTransverseCurrent must have been called first).
     *
     * \param[in]  fields  Field register
     * \param[in]  lev     AMR level
     * \param[in]  time    Simulation time at which J_target is evaluated
     * \param[out] F_out   MultiFab to receive the computed RHS
     */
    void ComputeEyRHS(
        ablastr::fields::MultiFabRegister& fields,
        int lev,
        amrex::Real time,
        amrex::MultiFab& F_out
    );

    /**
     * \brief Set Efield_fp[1] = Ey_base + coeff * F, applying the field limiter.
     *
     * Used to stage the intermediate E_y value before each RK substep:
     *   - For k2: Efield_fp[1] = E_y^n + dt/2 * k1
     *   - For k3: Efield_fp[1] = E_y^n + dt/2 * k2
     *   - For k4: Efield_fp[1] = E_y^n + dt   * k3
     *   - For the final Euler update: Efield_fp[1] = E_y^n + dt * k1
     *
     * Zeros Efield_fp[1] outside [z_min, z_max] and clamps to [-ey_max, +ey_max].
     *
     * \param[in] fields    Field register (output goes to Efield_fp[1])
     * \param[in] lev       AMR level
     * \param[in] Ey_base   Base field (typically E_y^n saved in Ey_icp_base)
     * \param[in] coeff     Coefficient on F (e.g. dt, dt/2, dt/6)
     * \param[in] F         ODE RHS term (one of k1–k4 from ComputeEyRHS)
     */
    void SetTransverseElectricField(
        ablastr::fields::MultiFabRegister& fields,
        int lev,
        amrex::MultiFab const& Ey_base,
        amrex::Real coeff,
        amrex::MultiFab const& F
    );

private:
    // =========================================================================
    // Final-update helpers — write the weighted RHS sum into Efield_fp[1]
    // =========================================================================

    /**
     * \brief Apply Adams-Bashforth 2-step final update.
     *
     *   E_y^{n+1} = E_y^n + dt * (3/2 * F_n - 1/2 * F_prev)
     *
     * Zeros Efield_fp[1] outside [z_min, z_max] and clamps to [-ey_max, +ey_max].
     */
    void ApplyAB2Update(
        ablastr::fields::MultiFabRegister& fields,
        int lev,
        amrex::MultiFab const& Ey_base,
        amrex::Real dt,
        amrex::MultiFab const& F_n,
        amrex::MultiFab const& F_prev
    );

    /**
     * \brief Apply RK2 (Heun's) final update.
     *
     *   E_y^{n+1} = E_y^n + (dt/2) * (k1 + k2)
     *
     * Zeros Efield_fp[1] outside [z_min, z_max] and clamps to [-ey_max, +ey_max].
     */
    void ApplyRK2Update(
        ablastr::fields::MultiFabRegister& fields,
        int lev,
        amrex::MultiFab const& Ey_base,
        amrex::Real dt,
        amrex::MultiFab const& k1,
        amrex::MultiFab const& k2
    );

    // -------------------------------------------------------------------------
    // Parameters read from input file
    // -------------------------------------------------------------------------

    bool m_do_icp_heating = false;

    amrex::Real m_icp_frequency = 0.0;  ///< drive frequency [Hz]
    amrex::Real m_icp_omega     = 0.0;  ///< angular frequency [rad/s]

    amrex::Real m_z_min = 0.0;  ///< start of heating region [m]
    amrex::Real m_z_max = 0.0;  ///< end of heating region [m]

    std::unique_ptr<amrex::Parser> m_j0_parser;     ///< J_0(z,t) amplitude profile
    amrex::ParserExecutor<2>       m_j0_parser_exe; ///< compiled parser executor

    amrex::Real m_ey_max = 1.0e10;  ///< E_y field limiter [V/m]

    ICPIntegrator m_integrator = ICPIntegrator::Euler;  ///< selected time integrator

    // -------------------------------------------------------------------------
    // State for multi-step integrators
    // -------------------------------------------------------------------------

    /// Set to true until the first step completes; used to bootstrap AB2.
    bool m_is_first_step = true;

    // -------------------------------------------------------------------------
    // Save/restore buffer for RK substep particle momenta
    // -------------------------------------------------------------------------

    /**
     * Per-species saved uy values used by Save/RestoreTransverseMomenta.
     * m_saved_uy[ispec] holds the uy of every particle of species ispec,
     * concatenated in the serial tile-iteration order used by WarpXParIter.
     * Valid only between a matching Save/Restore pair within one timestep.
     */
    amrex::Vector<amrex::Vector<amrex::Real>> m_saved_uy;
};

#endif // WARPX_ICPHEATINGMODEL_H_
