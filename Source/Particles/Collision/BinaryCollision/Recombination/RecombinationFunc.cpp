#include "RecombinationFunc.H"
#include "Utils/TextMsg.H"

/**
 * \brief Constructor of the RecombinationFunc class
 *
 * @param[in] collision_name the name of the collision
 * @param[in] mypc pointer to the MultiParticleContainer
 */
RecombinationFunc::RecombinationFunc (
    const std::string& collision_name,
    [[maybe_unused]] MultiParticleContainer const * const mypc,
    bool /*isSameSpecies*/ )
{
    using namespace amrex::literals;

    const amrex::ParmParse pp_collision_name(collision_name);

    // Read the cross section file for recombination
    const std::string kw_cross_section = "cross_section";
    std::string cross_section_file;
    bool has_cross_section = pp_collision_name.query(kw_cross_section.c_str(), cross_section_file);

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(has_cross_section,
        "Recombination collision '" + collision_name +
        "' requires a 'cross_section' parameter specifying the cross section data file.");

    // Recombination doesn't have an energy penalty (particles are removed, not de-excited)
    amrex::ParticleReal energy = 0._prt;

    // Create the scattering process for recombination
    // We use a generic "recombination" process type which will be treated as absorption
    m_recombination_process.emplace_back("recombination", cross_section_file, energy);

    // Store executor (similar to BackgroundMCC pattern)
#ifdef AMREX_USE_GPU
    amrex::Gpu::HostVector<ScatteringProcess::Executor> h_recombination_process_exe;
    h_recombination_process_exe.push_back(m_recombination_process[0].executor());
    m_recombination_process_exe.resize(1);
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_recombination_process_exe.begin(),
                          h_recombination_process_exe.end(), m_recombination_process_exe.begin());
    amrex::Gpu::streamSynchronize();
    m_exe.m_recombination_process = m_recombination_process_exe[0];
#else
    m_exe.m_recombination_process = m_recombination_process[0].executor();
#endif
}
