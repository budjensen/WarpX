#include "RecombinationFunc.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_ParallelDescriptor.H>

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

    // Recombination replaces the energy cost with the maximum energy
    amrex::ParticleReal energy = 0._prt;
    pp_collision_name.query("max_energy", energy);

    // Check if collision tracking is enabled
    pp_collision_name.query("enable_collision_tracking", m_do_tracking);

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

    m_exe.m_do_tracking = m_do_tracking;
}

void RecombinationFunc::initializeTracking(
    int lev,
    amrex::BoxArray const& ba,
    amrex::DistributionMapping const& dm)
{
    if (!m_do_tracking) {
        return;
    }

    // Resize tracking data vector if needed
    if (static_cast<std::size_t>(m_tracking_data.size()) <= static_cast<std::size_t>(lev)) {
        m_tracking_data.resize(lev + 1);
    }

    // Create MultiFab with 2 components (count, energy) for recombination
    const int ncomps = getNumTrackingComponents();
    const int ngrow = 0;
    m_tracking_data[lev] = std::make_unique<amrex::MultiFab>(ba, dm, ncomps, ngrow);
    m_tracking_data[lev]->setVal(0.0);

    m_tracking_initialized = true;
}

amrex::MultiFab* RecombinationFunc::getCollisionTracking(int lev)
{
    if (!m_tracking_initialized || lev >= static_cast<int>(m_tracking_data.size())) {
        return nullptr;
    }
    return m_tracking_data[lev].get();
}

void RecombinationFunc::resetCollisionTracking(int lev)
{
    if (!m_tracking_initialized || lev >= static_cast<int>(m_tracking_data.size())) {
        return;
    }
    m_tracking_data[lev]->setVal(0.0);
}

amrex::Vector<std::string> RecombinationFunc::getProcessNames() const
{
    amrex::Vector<std::string> names;
    names.push_back("recombination");
    return names;
}

void RecombinationFunc::gatherCollisionTracking(
    int lev,
    amrex::Vector<amrex::Real>& data,
    amrex::Box& box,
    int ngrow)
{
    data.clear();
    box = amrex::Box();

    // Early return if not initialized - safe for all ranks
    if (!m_tracking_initialized || lev >= static_cast<int>(m_tracking_data.size())) {
        return;
    }

    auto* mf = m_tracking_data[lev].get();
    if (mf == nullptr) {
        return;
    }

    const int ncomp = mf->nComp();
    const int ioproc = amrex::ParallelDescriptor::IOProcessorNumber();

    // Get the domain box for this level
    const auto& geom = WarpX::GetInstance().Geom(lev);
    amrex::Box domain = geom.Domain();
    if (ngrow > 0) {
        domain.grow(ngrow);
    }

    // Create a single-box BoxArray and DistributionMapping on IO processor
    amrex::BoxArray ba_single(domain);
    amrex::DistributionMapping dm_single;
    amrex::Vector<int> pmap(1, ioproc);
    dm_single.define(std::move(pmap));

    // Create a MultiFab on the IO processor to gather data into
    amrex::MultiFab mf_gathered(ba_single, dm_single, ncomp, 0);
    mf_gathered.setVal(0.0);

    // ALL RANKS participate in ParallelCopy (collective MPI operation)
    mf_gathered.ParallelCopy(*mf, 0, 0, ncomp);

    // Only IO processor extracts the data
    if (amrex::ParallelDescriptor::IOProcessor()) {
        box = domain;
        const amrex::Long ncells = domain.numPts();
        data.resize(ncells * ncomp);

        const amrex::Array4<const amrex::Real> arr = mf_gathered.array(0);
        amrex::Long idx = 0;

        // Iterate through the box in standard order and pack data
        amrex::LoopOnCpu(domain, [&](int i, int j, int k) {
            for (int comp = 0; comp < ncomp; ++comp) {
                data[idx * ncomp + comp] = arr(i, j, k, comp);
            }
            idx++;
        });

        // Validate that we filled all the expected data
        if (idx != ncells) {
            amrex::Print() << "Warning: Expected " << ncells << " cells but filled "
                           << idx << " in RecombinationFunc::gatherCollisionTracking" << std::endl;
        }
    }
}
