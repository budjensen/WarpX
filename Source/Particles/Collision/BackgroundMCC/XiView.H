/* Copyright 2026
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_BACKGROUNDMCC_XIVIEW_H_
#define WARPX_PARTICLES_COLLISION_BACKGROUNDMCC_XIVIEW_H_

#include <AMReX_Algorithm.H>
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
    amrex::ParticleReal const* m_energies = nullptr;
    int m_grid_size = 0;
    amrex::ParticleReal m_energy_lo = 0;
    amrex::ParticleReal m_energy_hi = 0;
    amrex::ParticleReal m_dE = 0;
    amrex::ParticleReal m_lo = 0;
    amrex::ParticleReal m_hi = 0;
    /** whether the energy grid is evenly spaced, enabling the fast index lookup */
    bool m_uniform = false;

    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::ParticleReal getXi(amrex::ParticleReal E_coll) const
    {
        using namespace amrex::literals;

        if (m_data != nullptr) {
            if (E_coll < m_energy_lo) {
                return m_lo;
            } else if (E_coll >= m_energy_hi) {
                // Note the `>=`: `amrex::bisect` below requires E_coll to be strictly
                // below the last grid point, otherwise it reads one past the end of the
                // energy array. The interpolation would return m_hi at that energy anyway.
                return m_hi;
            } else if (m_uniform) {
                using amrex::Math::ceil;
                using amrex::Math::floor;
                // Evenly spaced grid: the bracketing indices follow directly from the
                // energy step, which avoids a search in this particle-loop hot path.
                amrex::ParticleReal temp = (E_coll - m_energy_lo) / m_dE;
                const int idx_1 = static_cast<int>(floor(temp));
                const int idx_2 = static_cast<int>(ceil(temp));
                temp -= idx_1;
                return m_data[idx_1] + (m_data[idx_2] - m_data[idx_1]) * temp;
            } else {
                // Locate the pair of energy grid points bracketing E_coll using a
                // bisection search. This does not assume an evenly spaced energy grid.
                const int idx_1 = amrex::bisect(m_energies, 0, m_grid_size - 1, E_coll);
                const int idx_2 = idx_1 + 1;
                const amrex::ParticleReal temp =
                    (E_coll - m_energies[idx_1]) / (m_energies[idx_2] - m_energies[idx_1]);
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
