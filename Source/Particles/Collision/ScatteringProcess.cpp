/* Copyright 2021-2023 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Modern Electron, Roelof Groenewald (TAE Technologies)
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ScatteringProcess.H"

#include "Utils/TextMsg.H"

#include <algorithm>

ScatteringProcess::ScatteringProcess (
                        const std::string& scattering_process,
                        const std::string& cross_section_file,
                        const amrex::ParticleReal energy )
{
    // read the cross-section data file into memory
    readCrossSectionFile(cross_section_file, m_energies, m_sigmas_h);

    init(scattering_process, energy);
}

template <typename InputVector>
ScatteringProcess::ScatteringProcess (
                        const std::string& scattering_process,
                        const InputVector&& energies,
                        const InputVector&& sigmas,
                        const amrex::ParticleReal energy )
{
    m_energies.insert(m_energies.begin(), std::begin(energies), std::end(energies));
    m_sigmas_h.insert(m_sigmas_h.begin(), std::begin(sigmas),   std::end(sigmas));

    init(scattering_process, energy);
}

void
ScatteringProcess::init (const std::string& scattering_process, const amrex::ParticleReal energy)
{
    using namespace amrex::literals;
    m_name = scattering_process;
    m_exe_h.m_energies_data = m_energies.data();
    m_exe_h.m_sigmas_data = m_sigmas_h.data();

    // save energy grid parameters for easy use
    const int grid_size = static_cast<int>(m_energies.size());
    m_exe_h.m_grid_size = grid_size;
    m_exe_h.m_energy_lo = m_energies[0];
    m_exe_h.m_energy_hi = m_energies[grid_size-1];
    m_exe_h.m_sigma_lo = m_sigmas_h[0];
    m_exe_h.m_sigma_hi = m_sigmas_h[grid_size-1];
    // The energy grid does not need to be evenly spaced. Track both the smallest and
    // the largest spacing: comparing the two tells us whether the fast, search-free
    // index lookup in `Executor::getCrossSection` can be used, and the smallest spacing
    // is what a non-uniform grid reports as its representative energy step (e.g. to set
    // the scan resolution when computing the maximum collision frequency), so that
    // finely resolved regions of the grid are not skipped over.
    amrex::ParticleReal dE_min = 0._prt;
    amrex::ParticleReal dE_max = 0._prt;
    if (grid_size > 1) {
        dE_min = m_energies[grid_size-1] - m_energies[0];
        for (int i = 1; i < grid_size; i++) {
            const amrex::ParticleReal dE_i = m_energies[i] - m_energies[i-1];
            dE_min = std::min(dE_min, dE_i);
            dE_max = std::max(dE_max, dE_i);
        }
    }
    // Same tolerance that the evenly-spaced grid check used historically.
    m_exe_h.m_uniform = (grid_size > 1) && (dE_max - dE_min < dE_min / 100._prt);
    // For an evenly spaced grid, keep the exact step that the fast lookup expects; this
    // also reproduces the pre-existing cross-section values bit-for-bit.
    m_exe_h.m_dE = m_exe_h.m_uniform
                 ? (m_exe_h.m_energy_hi - m_exe_h.m_energy_lo) / (grid_size - 1._prt)
                 : dE_min;
    m_exe_h.m_energy_penalty = energy;
    m_exe_h.m_type = parseProcessType(scattering_process);

    // sanity check cross-section energy grid
    sanityCheckEnergyGrid(m_energies);

    // check that the cross-section is 0 at the energy cost if the energy
    // cost is > 0 - this is to prevent the possibility of negative left
    // over energy after a collision event
    if (m_exe_h.m_energy_penalty > 0 and m_exe_h.m_type != ScatteringProcessType::RECOMBINATION) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (getCrossSection(m_exe_h.m_energy_penalty) == 0),
            "Cross-section > 0 at energy cost for collision."
        );
    }

#ifdef AMREX_USE_GPU
    m_exe_d = m_exe_h;
    m_energies_d.resize(m_energies.size());
    m_sigmas_d.resize(m_sigmas_h.size());
    m_exe_d.m_energies_data = m_energies_d.data();
    m_exe_d.m_sigmas_data = m_sigmas_d.data();
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_energies.begin(), m_energies.end(),
                          m_energies_d.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, m_sigmas_h.begin(), m_sigmas_h.end(),
                          m_sigmas_d.begin());
    amrex::Gpu::streamSynchronize();
#endif
}

ScatteringProcessType
ScatteringProcess::parseProcessType(const std::string& scattering_process)
{
    if (scattering_process.find("elastic") != std::string::npos) {
        return ScatteringProcessType::ELASTIC;
    } else if (scattering_process.find("back") != std::string::npos) {
        return ScatteringProcessType::BACK;
    } else if (scattering_process == "charge_exchange") {
        return ScatteringProcessType::TWOPRODUCT_REACTION;
    } else if (scattering_process == "two_product_reaction") {
        return ScatteringProcessType::TWOPRODUCT_REACTION;
    } else if (scattering_process == "ionization") {
        return ScatteringProcessType::IONIZATION;
    } else if (scattering_process.find("excitation") != std::string::npos) {
        return ScatteringProcessType::EXCITATION;
    } else if (scattering_process.find("forward") != std::string::npos) {
        return ScatteringProcessType::FORWARD;
    } else if (scattering_process == "recombination") {
        return ScatteringProcessType::RECOMBINATION;
    } else {
        return ScatteringProcessType::INVALID;
    }
}

void
ScatteringProcess::readCrossSectionFile (
                                  const std::string& cross_section_file,
                                  amrex::Vector<amrex::ParticleReal>& energies,
                                  amrex::Gpu::HostVector<amrex::ParticleReal>& sigmas )
{
    std::ifstream infile(cross_section_file);
    if(!infile.is_open()) { WARPX_ABORT_WITH_MESSAGE("Failed to open cross-section data file"); }

    amrex::ParticleReal energy, sigma;
    while (infile >> energy >> sigma) {
        energies.push_back(energy);
        sigmas.push_back(sigma);
    }
    if (infile.bad()) { WARPX_ABORT_WITH_MESSAGE("Failed to read cross-section data from file."); }
    infile.close();
}

void
ScatteringProcess::sanityCheckEnergyGrid (
                                   const amrex::Vector<amrex::ParticleReal>& energies
                                   )
{
    // The energy grid does not need to be evenly spaced, but it must be sorted in
    // strictly increasing order for the bisection search and linear interpolation
    // used in `Executor::getCrossSection` to work correctly.
    for (unsigned i = 1; i < energies.size(); i++) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                                         (energies[i] > energies[i-1]),
                                         "Cross-section energy grid must be sorted in "
                                         "strictly increasing order."
                                         );
    }
}
