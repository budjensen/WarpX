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
#include <ablastr/warn_manager/WarnManager.H>

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
#include <fstream>
#include <memory>
#include <sstream>
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

    // J_0 amplitude — one of three input forms:
    //   j0_amplitude          : constant scalar [A/m^2]
    //   j0_amplitude(z,t)     : parser expression with fixed amplitude
    //   j0_amplitude(J_0,z,t) : parser expression whose scalar J_0 is adjusted
    //                           by the PID power controller
    const bool has_scalar = pp_icp.contains("j0_amplitude");
    const bool has_zt     = pp_icp.contains("j0_amplitude(z,t)");
    const bool has_ctrl   = pp_icp.contains("j0_amplitude(J_0,z,t)");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (static_cast<int>(has_scalar) + static_cast<int>(has_zt)
         + static_cast<int>(has_ctrl)) == 1,
        "icp_heating: exactly one of j0_amplitude, j0_amplitude(z,t), or "
        "j0_amplitude(J_0,z,t) must be specified");

    if (has_scalar) {
        amrex::ParticleReal j0_amplitude = 0;
        utils::parser::queryWithParser(pp_icp, "j0_amplitude", j0_amplitude);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            j0_amplitude >= 0, "icp_heating.j0_amplitude must be non-negative");
        m_j0_expression = std::to_string(j0_amplitude);
    } else if (has_zt) {
        utils::parser::Store_parserString(pp_icp, "j0_amplitude(z,t)", m_j0_expression);
    } else {
        utils::parser::Store_parserString(pp_icp, "j0_amplitude(J_0,z,t)", m_j0_expression);
        m_controller_enabled = true;
    }
    // All three forms compile onto the same 3-variable executor. In the
    // non-controller forms J_0 is registered but unused, which the amrex
    // parser allows, so the compiled arithmetic is unchanged.
    m_j0_parser = std::make_unique<Parser>(
        utils::parser::makeParser(m_j0_expression, {"z", "t", "J_0"}));
    m_j0_parser_exe = m_j0_parser->compile<3>();

    if (m_controller_enabled) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_j0_parser->symbols().count("J_0") == 1,
            "icp_heating.j0_amplitude(J_0,z,t) must reference the controlled "
            "amplitude symbol J_0, otherwise the controller has no effect");

        utils::parser::getWithParser(pp_icp, "J_0_initial", m_j0_initial);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_j0_initial > 0.0_rt, "icp_heating.J_0_initial must be positive");

        // Control-mode selection: exactly one of the two targets picks what
        // the PID regulates by scaling J_0.
        const bool has_P_target = pp_icp.contains("P_target");
        const bool has_n_target = pp_icp.contains("n_target");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            static_cast<int>(has_P_target) + static_cast<int>(has_n_target) == 1,
            "icp_heating: exactly one of P_target (power control) or "
            "n_target (density control) must be specified");
        if (has_P_target) {
            m_control_mode = ICPControlMode::Power;
            utils::parser::getWithParser(pp_icp, "P_target", m_P_target);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_P_target > 0.0_rt, "icp_heating.P_target must be positive");
            // Density-mode keys are meaningless here — catch stale inputs.
            for (auto const* key : {"n_region_lo", "n_region_hi",
                                    "n_samples_per_update"}) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    !pp_icp.contains(key),
                    "icp_heating." + std::string(key) +
                    " requires density control mode (icp_heating.n_target)");
            }
        } else {
            m_control_mode = ICPControlMode::Density;
            utils::parser::getWithParser(pp_icp, "n_target", m_n_target);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_n_target > 0.0_rt, "icp_heating.n_target must be positive");

            // Density measurement region: default is the central 25% of the
            // domain (midpoint +- L/8). Geom(0) is valid here: geometry is
            // built by the AmrCore base class before the WarpX constructor
            // body (which constructs this model) runs.
            const auto& geom = WarpX::GetInstance().Geom(0);
            const amrex::Real prob_lo = geom.ProbLo(0);  // 1D-Z: index 0 is z
            const amrex::Real prob_hi = geom.ProbHi(0);
            const amrex::Real zc = 0.5_rt * (prob_lo + prob_hi);
            const amrex::Real L  = prob_hi - prob_lo;
            m_n_region_lo = zc - 0.125_rt * L;
            m_n_region_hi = zc + 0.125_rt * L;
            utils::parser::queryWithParser(pp_icp, "n_region_lo", m_n_region_lo);
            utils::parser::queryWithParser(pp_icp, "n_region_hi", m_n_region_hi);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                prob_lo <= m_n_region_lo && m_n_region_lo < m_n_region_hi
                && m_n_region_hi <= prob_hi,
                "icp_heating: must satisfy prob_lo <= n_region_lo < "
                "n_region_hi <= prob_hi");

            utils::parser::queryWithParser(
                pp_icp, "n_samples_per_update", m_n_samples_per_update);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                m_n_samples_per_update >= 1,
                "icp_heating.n_samples_per_update must be at least 1");
        }

        // Controller period: one key for both modes. P_controller_period is
        // kept as a legacy alias from the power-only controller version.
        const bool has_period = pp_icp.contains("controller_period");
        const bool has_legacy_period = pp_icp.contains("P_controller_period");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !(has_period && has_legacy_period),
            "icp_heating: specify only one of controller_period and its "
            "legacy alias P_controller_period");
        m_ctrl_period_seconds = 5.0_rt / m_icp_frequency;
        if (has_legacy_period) {
            utils::parser::getWithParser(
                pp_icp, "P_controller_period", m_ctrl_period_seconds);
            if (ParallelDescriptor::IOProcessor()) {
                Print() << Utils::TextMsg::Info(
                    "icp_heating.P_controller_period is a legacy alias; "
                    "prefer icp_heating.controller_period");
            }
        } else {
            utils::parser::queryWithParser(
                pp_icp, "controller_period", m_ctrl_period_seconds);
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_ctrl_period_seconds > 0.0_rt,
            "icp_heating.controller_period must be positive");

        // Gains are dimensionless per-update gains acting on the normalized
        // error e = (target - measurement)/target; the controller period is
        // folded into them (see TickController), so they must be retuned if
        // controller_period changes. Kd = 0 is recommended: the derivative
        // term amplifies PIC measurement noise.
        utils::parser::queryWithParser(pp_icp, "pid_kp", m_pid_kp);
        utils::parser::queryWithParser(pp_icp, "pid_ki", m_pid_ki);
        utils::parser::queryWithParser(pp_icp, "pid_kd", m_pid_kd);

        m_j0_min = 0.01_rt  * m_j0_initial;
        m_j0_max = 100.0_rt * m_j0_initial;
        utils::parser::queryWithParser(pp_icp, "J_0_min", m_j0_min);
        utils::parser::queryWithParser(pp_icp, "J_0_max", m_j0_max);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            0.0_rt < m_j0_min && m_j0_min <= m_j0_initial && m_j0_initial <= m_j0_max,
            "icp_heating: must satisfy 0 < J_0_min <= J_0_initial <= J_0_max");

        utils::parser::queryWithParser(
            pp_icp, "controller_delay_N_periods", m_ctrl_delay_periods);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_ctrl_delay_periods >= 0,
            "icp_heating.controller_delay_N_periods must be non-negative");
        utils::parser::queryWithParser(
            pp_icp, "controller_history_size", m_history_capacity);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_history_capacity >= 0,
            "icp_heating.controller_history_size must be non-negative");
        pp_icp.query("restore_controller_from_checkpoint", m_restore_from_checkpoint);

        const ParmParse pp_amr("amr");

        // The controller samples the level-0 buffers/particles only.
        int max_level = 0;
        pp_amr.query("max_level", max_level);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            max_level == 0,
            "icp_heating controller requires amr.max_level = 0");

        m_j0_current = m_j0_initial;

        // On restart, restore J_0 and the PID state from the checkpoint
        // sidecar file (missing file falls back to J_0_initial).
        std::string restart_chkfile;
        pp_amr.query("restart", restart_chkfile);
        if (!restart_chkfile.empty() && m_restore_from_checkpoint) {
            ReadCheckpointData(restart_chkfile);
        }
    } else {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            m_j0_parser->symbols().count("J_0") == 0,
            "icp_heating: the amplitude expression references J_0, but the "
            "controller form j0_amplitude(J_0,z,t) was not used; J_0 would "
            "silently evaluate to 1");
        for (auto const* key : {"J_0_initial", "P_target", "n_target",
                                "controller_period", "P_controller_period",
                                "n_region_lo", "n_region_hi", "n_samples_per_update",
                                "pid_kp", "pid_ki", "pid_kd", "J_0_min", "J_0_max",
                                "controller_delay_N_periods", "controller_history_size",
                                "restore_controller_from_checkpoint"}) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                !pp_icp.contains(key),
                "icp_heating." + std::string(key) + " requires the controller "
                "form icp_heating.j0_amplitude(J_0,z,t)");
        }
    }

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
                << "  J_0 expression:      " << m_j0_expression   << "\n"
                << "  E_y field limiter:   " << m_ey_max          << " V/m\n"
                << "  Integrator:          " << integrator_str     << "\n";
        if (m_controller_enabled) {
            const bool power_mode = (m_control_mode == ICPControlMode::Power);
            Print() << "  PID " << (power_mode ? "power" : "density")
                    << " controller:\n"
                    << "    J_0 (initial/restored): " << m_j0_current << " A/m^2\n";
            if (power_mode) {
                Print() << "    P_target:               " << m_P_target << " W/m^2\n";
            } else {
                Print() << "    n_target:               " << m_n_target << " m^-3\n"
                        << "    Measurement region:     z = [" << m_n_region_lo
                        << ", " << m_n_region_hi << "] m\n"
                        << "    Density samples/update: " << m_n_samples_per_update
                        << " (evenly spaced)\n";
            }
            Print() << "    Controller period:      " << m_ctrl_period_seconds << " s\n"
                    << "    Gains (Kp, Ki, Kd):     " << m_pid_kp << ", " << m_pid_ki
                    << ", " << m_pid_kd
                    << "  (dimensionless, per controller period)\n"
                    << "    J_0 clamps:             [" << m_j0_min << ", "
                    << m_j0_max << "] A/m^2\n"
                    << "    Startup delay:          " << m_ctrl_delay_periods
                    << " controller periods\n";
        }
        Print() << "----------------------------------------------------------------------\n\n";
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

    // E_y (Efield_fp[1]) and J_y (current_fp[1]) are node-centered in 1D-Z.
    // Every ICP scratch field must share that staggering: the update kernels
    // iterate the nodal tilebox of Efield_fp[1] and index these fields at
    // every node, including each box's high-edge node. A cell-centered
    // allocation would leave that shared node reading unwritten ghost data,
    // zeroing E_y at every box boundary.
    const BoxArray ba_nodal = amrex::convert(ba, IntVect::TheNodeVector());

    using ablastr::fields::Direction;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        fields.get(FieldType::Efield_fp, Direction{1}, lev)->boxArray() == ba_nodal,
        "ICPHeatingModel: scratch fields must match the staggering of Efield_fp[1]");

    // Prescribed target current density (diagnostic output)
    fields.alloc_init("Jy_icp_target", lev, ba_nodal, dm, 1, ng, 0.0_rt);

    // E_y^n snapshot — base for all integrator branches each step
    fields.alloc_init("Ey_icp_base",   lev, ba_nodal, dm, 1, ng, 0.0_rt);

    // RK stage storage: F_i = (J_target - J_cond) / eps0
    // k1 is also used by Euler and AB2 for their single F evaluation.
    fields.alloc_init("k1_icp",        lev, ba_nodal, dm, 1, ng, 0.0_rt);
    fields.alloc_init("k2_icp",        lev, ba_nodal, dm, 1, ng, 0.0_rt);
    fields.alloc_init("k3_icp",        lev, ba_nodal, dm, 1, ng, 0.0_rt);
    fields.alloc_init("k4_icp",        lev, ba_nodal, dm, 1, ng, 0.0_rt);

    // F_{n-1} for Adams-Bashforth 2-step history
    fields.alloc_init("F_prev_icp",    lev, ba_nodal, dm, 1, ng, 0.0_rt);

    // A permanently zero-valued field used as Ex=Ez=Bx=By=Bz=0 in
    // PushTransverseMomenta. Never written to after initialization.
    fields.alloc_init("zero_field_icp", lev, ba_nodal, dm, 1, ng, 0.0_rt);
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

    // E_y is nodal, so the node shared by adjacent boxes is computed by both.
    // The inputs (post-SumBoundary J_y, Ey_base, the parsed profile) are
    // identical in both copies, so the duplicated values already agree; this
    // sync is insurance against roundoff-order divergence (e.g. three or more
    // SumBoundary contributors on very narrow boxes) accumulating over long
    // runs. Its cost is negligible next to the per-step current depositions.
    const auto& period = WarpX::GetInstance().Geom(lev).periodicity();
    Ey_fp->OverrideSync(period);
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
    const Real J0_now = m_j0_current;

    for (MFIter mfi(F_out, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();

        auto const& F_arr      = F_out.array(mfi);
        auto const& Jcond_arr  = Jy_cond_mf->const_array(mfi);
        auto const& Jtgt_arr   = Jy_target_mf->array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            const Real z = zmin + static_cast<Real>(i) * dz;  // node-centered coordinate

            if (z >= z_min && z <= z_max) {
                const Real J_target = j0_exe(z, time, J0_now) * std::sin(phase);
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
            const Real z = zmin + static_cast<Real>(i) * dz;  // node-centered coordinate

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
            const Real z = zmin + static_cast<Real>(i) * dz;  // node-centered coordinate

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
            const Real z = zmin + static_cast<Real>(i) * dz;  // node-centered coordinate

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
            const Real z = zmin + static_cast<Real>(i) * dz;  // node-centered coordinate

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

// =============================================================================
// PID controller (power or density mode)
// =============================================================================

void ICPHeatingModel::InitDensityControl(MultiParticleContainer const& mpc)
{
    if (m_control_mode != ICPControlMode::Density) { return; }

    m_negative_species.clear();
    std::string species_list;
    const auto names = mpc.GetSpeciesNames();
    for (int i = 0; i < mpc.nSpecies(); ++i) {
        if (mpc.GetParticleContainer(i).getCharge() < 0.0_rt) {
            m_negative_species.push_back(i);
            if (!species_list.empty()) { species_list += ", "; }
            species_list += names[i];
        }
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !m_negative_species.empty(),
        "icp_heating density controller: no species with negative charge "
        "found; the plasma density measurement would always be zero");

    if (ParallelDescriptor::IOProcessor()) {
        Print() << Utils::TextMsg::Info(
            "ICP density controller: regulating the summed density of "
            "negative species [" + species_list + "]");
    }
}

void ICPHeatingModel::TickController(
    MultiParticleContainer& mpc,
    amrex::Real time,
    amrex::Real dt,
    int step)
{
    if (!m_do_icp_heating || !m_controller_enabled) { return; }

    WARPX_PROFILE("ICPHeatingModel::TickController");

    const bool power_mode = (m_control_mode == ICPControlMode::Power);

    // One-time conversion of the controller period to a whole number of
    // steps, and of the requested density samples per window to a sampling
    // stride (at most one sample per step).
    if (m_ctrl_period_steps < 0) {
        m_ctrl_period_steps = std::max(
            1, static_cast<int>(std::round(m_ctrl_period_seconds / dt)));
        if (!power_mode) {
            m_n_sample_interval = std::max(
                1, m_ctrl_period_steps / m_n_samples_per_update);
        }
        if (ParallelDescriptor::IOProcessor()) {
            std::ostringstream ss;
            ss << "ICP " << (power_mode ? "power" : "density")
               << " controller: period of " << m_ctrl_period_seconds
               << " s = " << m_ctrl_period_steps << " steps ("
               << m_ctrl_period_steps * dt << " s effective)";
            if (!power_mode) {
                ss << "; density sampled every " << m_n_sample_interval
                   << " step(s)";
            }
            Print() << Utils::TextMsg::Info(ss.str());
        }
    }

    ++m_steps_in_window;

    if (power_mode) {
        // Absorbed inductive power this step [W/m^2]: delta of the running
        // sum of the tracking buffer's y-component, summed over all species.
        // The sum is an MPI allreduce, so the result (and J_0) is identical
        // on every rank.
        Real step_power = 0.0_rt;
        for (int i = 0; i < mpc.nSpecies(); ++i) {
            step_power += mpc.GetParticleContainer(i).samplePowerDepositionDelta(0, 1);
        }
        m_meas_accum += step_power;
    } else if (m_steps_in_window % m_n_sample_interval == 0) {
        // Region-averaged plasma density sample [m^-3]: summed weight of the
        // negative species inside the region over the region length (weights
        // are per-area in 1D-Z). The weight sum is an MPI allreduce, and the
        // sampling cadence depends only on rank-identical counters, so the
        // collective runs on every rank and J_0 stays rank-identical.
        Real wsum = 0.0_rt;
        for (const int i : m_negative_species) {
            wsum += mpc.GetParticleContainer(i).sumParticleWeightInZRange(
                m_n_region_lo, m_n_region_hi);
        }
        m_meas_accum += wsum / (m_n_region_hi - m_n_region_lo);
        ++m_samples_in_window;
    }

    if (m_steps_in_window < m_ctrl_period_steps) { return; }

    // Averaging window complete. In power mode every step contributed one
    // measurement; in density mode only the sampled steps did.
    const int n_meas = power_mode ? m_steps_in_window : m_samples_in_window;
    const Real mbar = m_meas_accum / static_cast<Real>(n_meas);
    m_last_measurement = mbar;
    ++m_windows_completed;

    const Real target = power_mode ? m_P_target : m_n_target;
    const Real e_k = (target - mbar) / target;

    if (m_windows_completed > m_ctrl_delay_periods) {
        // Velocity-form PID update (see the header for the derivation and
        // gain conventions; gains are dimensionless per-update gains).
        if (m_pid_bootstrap) {
            m_e_prev  = e_k;
            m_e_prev2 = e_k;
            m_pid_bootstrap = false;
        }
        const Real factor = 1.0_rt
            + m_pid_kp * (e_k - m_e_prev)
            + m_pid_ki * e_k
            + m_pid_kd * (e_k - 2.0_rt * m_e_prev + m_e_prev2);
        m_j0_current = std::clamp(m_j0_current * factor, m_j0_min, m_j0_max);
        m_e_prev2 = m_e_prev;
        m_e_prev  = e_k;
    }
    // During the startup delay J_0 stays at its initial value, and
    // m_pid_bootstrap remains true so the first active update is Ki-only.

    if (m_history_capacity > 0) {
        m_history.push_back({step, time, mbar, e_k, m_j0_current});
        if (static_cast<int>(m_history.size()) > m_history_capacity) {
            m_history.pop_front();
        }
    }

    m_meas_accum = 0.0_rt;
    m_steps_in_window = 0;
    m_samples_in_window = 0;
}

void ICPHeatingModel::SetJ0(amrex::Real j0)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        m_controller_enabled,
        "ICPHeatingModel::SetJ0 requires the PID controller mode "
        "(icp_heating.j0_amplitude(J_0,z,t))");

    m_j0_current = std::clamp(j0, m_j0_min, m_j0_max);
    // Restart the averaging window (do not mix measurements taken under two
    // different amplitudes) and re-bootstrap the PID error history.
    m_meas_accum = 0.0_rt;
    m_steps_in_window = 0;
    m_samples_in_window = 0;
    m_pid_bootstrap = true;
}

// =============================================================================
// Checkpoint sidecar file (ICPController_data.txt)
// =============================================================================

namespace
{
    // Version history:
    //   1: power-only controller; no mode line
    //   2: adds the control-mode token ("power"/"density") as line 2
    constexpr int icp_controller_checkpoint_version = 2;
    const std::string icp_controller_checkpoint_name = "ICPController_data.txt";
}

void ICPHeatingModel::WriteCheckpointData(std::string const& dir) const
{
    if (!m_controller_enabled) { return; }
    if (!ParallelDescriptor::IOProcessor()) { return; }

    const std::string filename = dir + "/" + icp_controller_checkpoint_name;
    std::ofstream chkfile{filename, std::ofstream::out};
    if (!chkfile.good()) {
        WARPX_ABORT_WITH_MESSAGE(
            "ICP controller: could not open checkpoint file " + filename);
    }
    chkfile.precision(17);
    chkfile << icp_controller_checkpoint_version << "\n";
    chkfile << (m_control_mode == ICPControlMode::Power ? "power" : "density") << "\n";
    chkfile << m_j0_current << "\n";
    chkfile << m_e_prev << "\n";
    chkfile << m_e_prev2 << "\n";
    chkfile << m_windows_completed << "\n";
    chkfile << static_cast<int>(m_pid_bootstrap) << "\n";
    // Profile expression last (may contain spaces); used only for a
    // consistency warning on restart.
    chkfile << m_j0_expression << "\n";
}

void ICPHeatingModel::ReadCheckpointData(std::string const& dir)
{
    const std::string filename = dir + "/" + icp_controller_checkpoint_name;
    std::ifstream chkfile{filename};
    if (!chkfile.good()) {
        // Checkpoint written before this feature existed (or by a run
        // without the controller) — keep J_0_initial.
        ablastr::warn_manager::WMRecordWarning(
            "ICP heating",
            "ICP controller: no " + icp_controller_checkpoint_name
            + " found in restart checkpoint " + dir
            + "; starting the controller from J_0_initial.",
            ablastr::warn_manager::WarnPriority::low);
        return;
    }

    int version = 0;
    int bootstrap_int = 1;
    std::string chk_expression;
    chkfile >> version;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        version >= 1 && version <= icp_controller_checkpoint_version,
        "ICP controller: unsupported " + icp_controller_checkpoint_name
        + " format version in " + dir);

    // Version 1 predates the density mode and implies power mode.
    ICPControlMode chk_mode = ICPControlMode::Power;
    if (version >= 2) {
        std::string mode_str;
        chkfile >> mode_str;
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            mode_str == "power" || mode_str == "density",
            "ICP controller: unknown control mode '" + mode_str + "' in " + filename);
        chk_mode = (mode_str == "power") ? ICPControlMode::Power
                                         : ICPControlMode::Density;
    }

    Real chk_j0 = 0.0_rt;
    Real chk_e_prev = 0.0_rt;
    Real chk_e_prev2 = 0.0_rt;
    int chk_windows = 0;
    chkfile >> chk_j0;
    chkfile >> chk_e_prev;
    chkfile >> chk_e_prev2;
    chkfile >> chk_windows;
    chkfile >> bootstrap_int;
    chkfile >> std::ws;
    std::getline(chkfile, chk_expression);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !chkfile.fail(),
        "ICP controller: failed to parse " + filename);

    // The restored J_0 must respect the (possibly re-specified) clamps.
    m_j0_current = std::clamp(chk_j0, m_j0_min, m_j0_max);
    m_windows_completed = chk_windows;

    if (chk_mode == m_control_mode) {
        // Same mode: full restore — the PID continues seamlessly.
        m_e_prev = chk_e_prev;
        m_e_prev2 = chk_e_prev2;
        m_pid_bootstrap = (bootstrap_int != 0);
    } else {
        // Mode switch across the restart (e.g. converge under power control,
        // then hold density): the checkpointed error history is in the other
        // mode's normalization and must not seed this mode's PID. Keep J_0
        // and windows_completed (the plasma is already converged, so the
        // startup delay stays served) but re-bootstrap the error history.
        m_e_prev = 0.0_rt;
        m_e_prev2 = 0.0_rt;
        m_pid_bootstrap = true;
        const auto mode_name = [](ICPControlMode m) {
            return (m == ICPControlMode::Power) ? std::string("power")
                                                : std::string("density");
        };
        ablastr::warn_manager::WMRecordWarning(
            "ICP heating",
            "ICP controller: the checkpoint was written in " + mode_name(chk_mode)
            + " control mode but this run uses " + mode_name(m_control_mode)
            + " control mode; restored J_0 and windows_completed only and "
            "re-bootstrapped the PID error history.",
            ablastr::warn_manager::WarnPriority::medium);
    }

    if (chk_expression != m_j0_expression) {
        ablastr::warn_manager::WMRecordWarning(
            "ICP heating",
            "ICP controller: the amplitude expression in the restart "
            "input differs from the one the checkpoint was written with.\n"
            "  checkpoint: " + chk_expression + "\n"
            "  input:      " + m_j0_expression,
            ablastr::warn_manager::WarnPriority::medium);
    }

    if (ParallelDescriptor::IOProcessor()) {
        std::ostringstream ss;
        ss << "ICP controller: restored J_0 = " << m_j0_current
           << " A/m^2 and PID state from " << filename;
        Print() << Utils::TextMsg::Info(ss.str());
    }
}
