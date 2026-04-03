/* Copyright 2026
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_BACKGROUNDMCC_XIVIEW_H_
#define WARPX_PARTICLES_COLLISION_BACKGROUNDMCC_XIVIEW_H_

#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <ablastr/constant.H>

/**
 * \brief Compact, device-copyable view used to evaluate xi(E) in kernels.
 *
 * When m_data is nullptr, the screened-Coulomb analytic model is used.
 * Otherwise xi(E) is linearly interpolated from the shared collision-level
 * xi_data table.
 */
struct MCCXiView
{
    amrex::ParticleReal const* m_data = nullptr;
    amrex::ParticleReal m_energy_lo = 0;
    amrex::ParticleReal m_energy_hi = 0;
    amrex::ParticleReal m_dE = 0;
    amrex::ParticleReal m_lo = 0;
    amrex::ParticleReal m_hi = 0;

    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::ParticleReal getXi(amrex::ParticleReal E_coll) const
    {
        using namespace amrex::literals;

        if (m_data != nullptr) {
            if (E_coll < m_energy_lo) {
                return m_lo;
            } else if (E_coll > m_energy_hi) {
                return m_hi;
            } else {
                using amrex::Math::ceil;
                using amrex::Math::floor;
                amrex::ParticleReal temp = (E_coll - m_energy_lo) / m_dE;
                const int idx_1 = static_cast<int>(floor(temp));
                const int idx_2 = static_cast<int>(ceil(temp));
                temp -= idx_1;
                return m_data[idx_1] + (m_data[idx_2] - m_data[idx_1]) * temp;
            }
        }

        // Screened-Coulomb analytic model with energy normalized by Hartree energy.
        constexpr amrex::ParticleReal E_h_eV =
            static_cast<amrex::ParticleReal>(ablastr::constant::SI::E_h_eV);
        return 4.0_prt * E_coll / (E_h_eV + 4.0_prt * E_coll);
    }
};

#endif // WARPX_PARTICLES_COLLISION_BACKGROUNDMCC_XIVIEW_H_
