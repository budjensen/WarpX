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

#include <deque>
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
 * \brief Selects what the PID controller regulates by scaling J_0.
 *
 * The modes are exclusive: a run sets exactly one of icp_heating.P_target
 * (Power) or icp_heating.n_target (Density). None means the controller
 * form j0_amplitude(J_0,z,t) is not in use.
 */
enum class ICPControlMode { None, Power, Density };

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
 *
 * PID controller mode:
 * When the amplitude is given as icp_heating.j0_amplitude(J_0,z,t), the
 * scalar J_0 is adjusted every controller period by a velocity-form PID.
 * Exactly one of two exclusive control modes is selected by the inputs:
 * - Power (icp_heating.P_target [W/m^2]): converges the domain-integrated,
 *   period-averaged inductive power absorbed by the plasma (measured from
 *   the per-species power-deposition tracking buffers, y-component).
 * - Density (icp_heating.n_target [m^-3]): converges the period-averaged
 *   number density of all negative-charge species (summed) inside a z-region
 *   [icp_heating.n_region_lo, n_region_hi] (default: central 25% of the
 *   domain), sampled icp_heating.n_samples_per_update times per period.
 * See TickController for the update law and gain conventions.
 */
class ICPHeatingModel
{
public:
    /** One record per completed controller averaging window. */
    struct ControllerHistoryEntry
    {
        int step;                 ///< step at which the window closed
        amrex::Real time;         ///< simulation time at which the window closed [s]
        /** Window-averaged measurement: absorbed power P_bar [W/m^2] in
         *  Power mode, region-averaged density n_bar [m^-3] in Density mode. */
        amrex::Real measurement;
        amrex::Real error;        ///< normalized error e = (target - measurement)/target
        amrex::Real j0;           ///< J_0 amplitude after the update [A/m^2]
    };

    ICPHeatingModel();

    /** Read user-defined model parameters. Called in constructor. */
    void ReadParameters();

    /**
     * \brief Allocate persistent MultiFabs used by the ICP model.
     *
     * Allocates J_y_target (diagnostic), E_y_base (saves E_y^n at the start
     * of each step), k1-k4 (RK stage RHS storage), F_prev (AB2 history),
     * and zero_field (used as a zero-valued E/B field in PushTransverseMomenta).
     *
     * All scratch fields are allocated NODE-centered to match the staggering
     * of Efield_fp[1] and current_fp[1] in 1D-Z; the cell-centered level
     * BoxArray \p ba is converted internally.
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

    /** Returns true if the PID power controller mode is active. */
    [[nodiscard]] bool is_controller_enabled() const { return m_controller_enabled; }

    // =========================================================================
    // PID controller (power or density mode)
    // =========================================================================

    /**
     * \brief Once-per-step controller tick: accumulate this window's
     *        measurement (absorbed power or plasma density, depending on the
     *        control mode) and, every controller period, update J_0.
     *
     * Called from WarpX::Evolve, after the step's momentum push (which fills
     * the power-deposition tracking buffers) and before the beforeEsolve
     * Python callback, so user-side buffer clears can never corrupt a sample.
     * Never called at init time (the tracking buffers do not exist yet).
     * Must run on all MPI ranks every step: both measurements are MPI
     * collectives.
     *
     * Power sampling (every step): for each species, the per-step absorbed
     * power is the delta of the global sum of the tracking buffer's
     * y-component since the previous step (see
     * WarpXParticleContainer::samplePowerDepositionDelta). In 1D-Z the plain
     * sum over cells is the domain-integrated power in W/m^2, because
     * macroparticle weights are per unit area.
     *
     * Density sampling (n_samples_per_update evenly spaced samples per
     * window): each sample sums the weights of all negative-charge species
     * inside [n_region_lo, n_region_hi] and divides by the region length,
     * giving the region-averaged density in m^-3 (see
     * WarpXParticleContainer::sumParticleWeightInZRange).
     *
     * PID update law (velocity/incremental form), applied when a window of
     * n = round(controller_period/dt) steps completes, with M_bar the
     * window-averaged measurement and M_target = P_target or n_target:
     *
     *   e_k   = (M_target - M_bar) / M_target            (dimensionless)
     *   J_0  <- clamp( J_0 * [ 1 + Kp*(e_k - e_{k-1})
     *                            + Ki*e_k
     *                            + Kd*(e_k - 2 e_{k-1} + e_{k-2}) ],
     *                  J_0_min, J_0_max )
     *
     * This is the discrete velocity form of the textbook PID
     * u = Kp'*e + Ki'*I + Kd'*de/dt with I += e*T: differencing consecutive
     * outputs turns the integral term into Ki'*T*e_k, and the controller
     * period T is folded into the gains (Ki = Ki'*T, Kd = Kd'/T). The
     * integral history therefore accumulates in J_0 itself — there is no
     * separate integral state and no windup at the clamps.
     *
     * Gain units: Kp, Ki, Kd are all DIMENSIONLESS per-update gains acting
     * on the normalized error (e.g. Ki = 0.1 moves J_0 by 10% of the relative
     * error each controller period). Because the period is folded in, gains
     * must be retuned if controller_period is changed. Kd = 0 is recommended:
     * the derivative term amplifies the statistical noise of the PIC
     * measurement.
     *
     * The first m_ctrl_delay_periods windows measure and record history but
     * do not update J_0, letting the plasma converge first. The first active
     * update (and the first update after SetJ0) bootstraps
     * e_{k-1} = e_{k-2} = e_k, so only the Ki term acts.
     *
     * \param[in] mpc   All particle species (measurements sampled from here)
     * \param[in] time  Simulation time at the end of the current step
     * \param[in] dt    Timestep (used once to convert the period to steps)
     * \param[in] step  Current step index (for history records)
     */
    void TickController(
        MultiParticleContainer& mpc,
        amrex::Real time,
        amrex::Real dt,
        int step
    );

    /**
     * \brief Cache the indices of the negative-charge species whose summed
     *        density the Density control mode regulates.
     *
     * Called once from the WarpX constructor (after species construction)
     * when the Density mode is active. Aborts if no negative species exists.
     */
    void InitDensityControl(MultiParticleContainer const& mpc);

    /** Returns the active control mode (None if the controller is off). */
    [[nodiscard]] ICPControlMode GetControlMode() const { return m_control_mode; }

    /** Returns the current J_0 amplitude [A/m^2]. */
    [[nodiscard]] amrex::Real GetJ0() const { return m_j0_current; }

    /**
     * \brief Set the J_0 amplitude [A/m^2] (clamped to [J_0_min, J_0_max]).
     *
     * Also restarts the averaging window (so measurements taken under the
     * old amplitude are not mixed in) and re-bootstraps the PID error history
     * (the next update is Ki-only). Must be called on all MPI ranks.
     * Aborts if the controller mode is not enabled.
     */
    void SetJ0(amrex::Real j0);

    /** Returns the mean absorbed power [W/m^2] of the last completed window
     *  (0 before the first window completes). Meaningful only in Power mode;
     *  the Python binding enforces the mode. */
    [[nodiscard]] amrex::Real GetLastMeanPower() const { return m_last_measurement; }

    /** Returns the region-averaged plasma density [m^-3] of the last
     *  completed window (0 before the first window completes). Meaningful
     *  only in Density mode; the Python binding enforces the mode. */
    [[nodiscard]] amrex::Real GetLastMeanDensity() const { return m_last_measurement; }

    /** Returns the recorded history of controller updates (bounded by
     *  icp_heating.controller_history_size, oldest entries dropped first). */
    [[nodiscard]] std::deque<ControllerHistoryEntry> const& GetControllerHistory() const
    { return m_history; }

    /**
     * \brief Write the controller state to <dir>/ICPController_data.txt.
     *
     * Called from FlushFormatCheckpoint when writing a checkpoint. Writes on
     * the IO processor only. The sidecar file is ignored by older code
     * versions, so checkpoint compatibility is preserved.
     */
    void WriteCheckpointData(std::string const& dir) const;

    /**
     * \brief Restore the controller state from <dir>/ICPController_data.txt.
     *
     * Called at the end of ReadParameters when restarting from a checkpoint
     * (unless icp_heating.restore_controller_from_checkpoint = 0). If the
     * file is missing (e.g. checkpoint written by an older code version),
     * prints a notice and keeps J_0_initial. Warns if the checkpointed
     * profile expression differs from the current input.
     */
    void ReadCheckpointData(std::string const& dir);

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

    /**
     * \brief Apply classic RK4 final update.
     *
     *   E_y^{n+1} = E_y^n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
     *
     * Zeros Efield_fp[1] outside [z_min, z_max] and clamps to [-ey_max, +ey_max].
     */
    void ApplyRK4Update(
        ablastr::fields::MultiFabRegister& fields,
        int lev,
        amrex::MultiFab const& Ey_base,
        amrex::Real dt,
        amrex::MultiFab const& k1,
        amrex::MultiFab const& k2,
        amrex::MultiFab const& k3,
        amrex::MultiFab const& k4
    );

    // -------------------------------------------------------------------------
    // Parameters read from input file
    // -------------------------------------------------------------------------

    bool m_do_icp_heating = false;

    amrex::Real m_icp_frequency = 0.0;  ///< drive frequency [Hz]
    amrex::Real m_icp_omega     = 0.0;  ///< angular frequency [rad/s]

    amrex::Real m_z_min = 0.0;  ///< start of heating region [m]
    amrex::Real m_z_max = 0.0;  ///< end of heating region [m]

    std::unique_ptr<amrex::Parser> m_j0_parser;     ///< amplitude profile parser
    /** Compiled executor over (z, t, J_0). In the legacy scalar and (z,t)
     *  forms J_0 is registered but unused, so evaluation is unchanged. */
    amrex::ParserExecutor<3>       m_j0_parser_exe;
    std::string m_j0_expression;    ///< profile expression string (for checkpoint provenance)

    amrex::Real m_ey_max = 1.0e10;  ///< E_y field limiter [V/m]

    ICPIntegrator m_integrator = ICPIntegrator::Euler;  ///< selected time integrator

    // -------------------------------------------------------------------------
    // PID controller parameters and state (power or density mode)
    // -------------------------------------------------------------------------

    bool m_controller_enabled = false;  ///< true iff j0_amplitude(J_0,z,t) form given
    ICPControlMode m_control_mode = ICPControlMode::None;  ///< what the PID regulates

    amrex::Real m_j0_initial = 0.0;  ///< icp_heating.J_0_initial [A/m^2]
    /** Current J_0 value passed to the amplitude parser. Kept at 1.0 in the
     *  legacy (non-controller) forms, where the parser does not use it. */
    amrex::Real m_j0_current = 1.0;
    amrex::Real m_P_target = 0.0;    ///< target absorbed power [W/m^2] (Power mode)
    amrex::Real m_n_target = 0.0;    ///< target plasma density [m^-3] (Density mode)

    amrex::Real m_ctrl_period_seconds = 0.0;  ///< controller period [s] (default 5/frequency)
    int m_ctrl_period_steps = -1;             ///< period in steps; computed on first tick

    // Density-mode measurement region and sampling cadence
    amrex::Real m_n_region_lo = 0.0;  ///< region lower z bound [m] (default: center - L/8)
    amrex::Real m_n_region_hi = 0.0;  ///< region upper z bound [m] (default: center + L/8)
    int m_n_samples_per_update = 100; ///< requested density samples per window
    int m_n_sample_interval = -1;     ///< steps between samples; computed on first tick
    /** Indices (into MultiParticleContainer) of the negative-charge species
     *  whose summed density is regulated. Filled by InitDensityControl. */
    amrex::Vector<int> m_negative_species;

    /** PID gains. All three are DIMENSIONLESS per-update gains on the
     *  normalized error e = (target - measurement)/target: the controller
     *  period T is folded in (Ki = Ki_continuous*T, Kd = Kd_continuous/T),
     *  so they must be retuned if controller_period changes.
     *  Kd = 0 (default) is recommended — the derivative term amplifies
     *  PIC measurement noise. */
    amrex::Real m_pid_kp = 0.2;
    amrex::Real m_pid_ki = 0.1;
    amrex::Real m_pid_kd = 0.0;

    amrex::Real m_j0_min = 0.0;  ///< amplitude clamp [A/m^2] (default 0.01*J_0_initial)
    amrex::Real m_j0_max = 0.0;  ///< amplitude clamp [A/m^2] (default 100*J_0_initial)

    /** Number of controller periods to wait (measuring, not acting) before
     *  the PID becomes active. */
    int m_ctrl_delay_periods = 10;
    int m_windows_completed = 0;  ///< completed averaging windows since start

    /** Running sum of the current window's measurements: per-step powers
     *  [W/m^2] in Power mode, density samples [m^-3] in Density mode. */
    amrex::Real m_meas_accum = 0.0;
    int m_steps_in_window = 0;    ///< steps accumulated in current window
    int m_samples_in_window = 0;  ///< density samples taken in current window
    /** Window-averaged measurement of the last completed window:
     *  P_bar [W/m^2] in Power mode, n_bar [m^-3] in Density mode. */
    amrex::Real m_last_measurement = 0.0;

    amrex::Real m_e_prev  = 0.0;  ///< e_{k-1}
    amrex::Real m_e_prev2 = 0.0;  ///< e_{k-2}
    /** True until the first active PID update (and again after SetJ0):
     *  bootstraps e_{k-1} = e_{k-2} = e_k so the first update is Ki-only. */
    bool m_pid_bootstrap = true;

    bool m_restore_from_checkpoint = true;  ///< icp_heating.restore_controller_from_checkpoint
    int m_history_capacity = 1000;          ///< icp_heating.controller_history_size
    std::deque<ControllerHistoryEntry> m_history;  ///< bounded record of updates

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
