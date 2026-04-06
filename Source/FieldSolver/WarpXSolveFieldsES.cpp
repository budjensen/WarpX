/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Remi Lehe, Roelof Groenewald, Arianna Formenti, Revathi Jambunathan
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.H"
#include "FieldSolver/InductiveHeating/ICPHeatingModel.H"

#include "Fields.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"


void WarpX::ComputeSpaceChargeField (bool const reset_fields)
{
    WARPX_PROFILE("WarpX::ComputeSpaceChargeField");
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    if (reset_fields) {
        // Reset all E and B fields to 0, before calculating space-charge fields
        // Exception: Skip E_y if ICP heating is enabled, since it accumulates over time
        WARPX_PROFILE("WarpX::ComputeSpaceChargeField::reset_fields");
        bool const skip_Ey = (m_icp_heating_model && m_icp_heating_model->is_enabled());

        for (int lev = 0; lev <= max_level; lev++) {
            for (int comp=0; comp<3; comp++) {
                m_fields.get(FieldType::Bfield_fp, Direction{comp}, lev)->setVal(0);

                // Skip clearing E_y (comp=1) if ICP heating is active
                if (comp == 1 && skip_Ey) continue;
                m_fields.get(FieldType::Efield_fp, Direction{comp}, lev)->setVal(0);
            }
        }
    }

    m_electrostatic_solver->ComputeSpaceChargeField(
        m_fields, *mypc, myfl.get(), max_level );

    // Apply ICP heating if enabled (for electrostatic simulations).
    // UpdateTransverseElectricField handles current deposition, MPI sync,
    // and the full field update for whichever integrator is selected.
    if (m_icp_heating_model && m_icp_heating_model->is_enabled()) {
        for (int lev = 0; lev <= max_level; ++lev) {
            m_icp_heating_model->UpdateTransverseElectricField(
                m_fields, lev, t_new[lev], dt[lev], *mypc
            );
        }
    }
}
