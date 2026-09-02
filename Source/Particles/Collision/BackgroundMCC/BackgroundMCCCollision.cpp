/* Copyright 2021 Modern Electron
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "BackgroundMCCCollision.H"

#include "ImpactIonization.H"
#include "Particles/Algorithms/KineticEnergy.H"
#include "Particles/ParticleCreation/FilterCopyTransform.H"
#include "Particles/ParticleCreation/SmartCopy.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>

namespace {

void ReadEnergyValueFile(
    std::string const& input_file,
    amrex::Vector<amrex::ParticleReal>& energies,
    amrex::Gpu::HostVector<amrex::ParticleReal>& values)
{
    energies.clear();
    values.clear();

    std::ifstream infile(input_file);
    if (!infile.is_open()) {
        WARPX_ABORT_WITH_MESSAGE("Failed to open xi_data file");
    }

    amrex::ParticleReal energy, value;
    while (infile >> energy >> value) {
        energies.push_back(energy);
        values.push_back(value);
    }
    if (infile.bad()) {
        WARPX_ABORT_WITH_MESSAGE("Failed to read xi_data from file.");
    }
    infile.close();
}

void SanityCheckEnergyGrid(
    amrex::Vector<amrex::ParticleReal> const& energies)
{
    // The energy grid does not need to be evenly spaced, but it must be sorted in
    // strictly increasing order for the bisection search and linear interpolation
    // used in `MCCXiView::getXi` to work correctly.
    for (unsigned int i = 1; i < energies.size(); ++i) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (energies[i] > energies[i-1]),
            "xi_data energy grid must be sorted in strictly increasing order.");
    }
}

} // namespace

BackgroundMCCCollision::BackgroundMCCCollision (std::string const& collision_name)
    : CollisionBase(collision_name)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(m_species_names.size() == 1,
                                     "Background MCC must have exactly one species.");

    const amrex::ParmParse pp_collision_name(collision_name);

    amrex::ParticleReal background_density = 0;
    if (utils::parser::queryWithParser(pp_collision_name, "background_density", background_density)) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (background_density > 0),
            "The background density must be greater than 0.");
        m_background_density_parser =
            utils::parser::makeParser(
                std::to_string(background_density), {"x", "y", "z", "t"});
    }
    else {
        std::string background_density_str;
        utils::parser::Store_parserString(pp_collision_name, "background_density(x,y,z,t)", background_density_str);
        m_background_density_parser =
            utils::parser::makeParser(background_density_str, {"x", "y", "z", "t"});
    }

    amrex::ParticleReal background_temperature;
    if (utils::parser::queryWithParser(pp_collision_name, "background_temperature", background_temperature)) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            (background_temperature >= 0), "The background temperature must be positive."
        );
        m_background_temperature_parser =
            utils::parser::makeParser(std::to_string(background_temperature), {"x", "y", "z", "t"});
    }
    else {
        std::string background_temperature_str;
        utils::parser::Store_parserString(pp_collision_name, "background_temperature(x,y,z,t)", background_temperature_str);
        m_background_temperature_parser =
            utils::parser::makeParser(background_temperature_str, {"x", "y", "z", "t"});
    }

    // compile parsers for background density and temperature
    m_background_density_func = m_background_density_parser.compile<4>();
    m_background_temperature_func = m_background_temperature_parser.compile<4>();

    utils::parser::queryWithParser(
        pp_collision_name, "max_background_density", m_max_background_density);
    // if the background density is constant we can use that number to calculate
    // the maximum collision probability, if `max_background_density` was not
    // specified
    if (m_max_background_density == 0 && background_density != 0) {
        m_max_background_density = background_density;
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        (m_max_background_density > 0),
        "The maximum background density must be greater than 0."
    );

    // if the neutral mass is specified use it, but if ionization is
    // included the mass of the secondary species of that interaction
    // will be used. If no neutral mass is specified and ionization is not
    // included the mass of the colliding species will be used
    m_background_mass = -1;
    utils::parser::queryWithParser(
        pp_collision_name, "background_mass", m_background_mass);

    // Check if collision tracking is enabled
    pp_collision_name.query("enable_collision_tracking", m_enable_collision_tracking);

    // Check if anisotropic scattering is enabled for scattering processes
    pp_collision_name.query("anisotropic_scatter", m_anisotropic_scatter);

    // Configure a single xi source for this collision object. This avoids
    // duplicating identical xi(E) data for each individual scattering process.
    if (m_anisotropic_scatter) {
        std::string xi_file;
        const bool has_xi_file = pp_collision_name.query("xi_data", xi_file);
        pp_collision_name.query("screened_coulomb", m_use_screened_coulomb);

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            has_xi_file || m_use_screened_coulomb,
            "When anisotropic_scatter = true, specify either xi_data = <path> "
            "(shared table for this collision type) or screened_coulomb = 1."
        );
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !(has_xi_file && m_use_screened_coulomb),
            "xi_data and screened_coulomb are mutually exclusive; choose one."
        );

        if (has_xi_file) {
            ReadEnergyValueFile(xi_file, m_xi_energies, m_xi_values_h);

            const int xi_grid_size = static_cast<int>(m_xi_energies.size());
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                xi_grid_size >= 2,
                "xi_data file must contain at least two (energy, xi) points."
            );

            m_xi_energy_lo = m_xi_energies[0];
            m_xi_energy_hi = m_xi_energies[xi_grid_size-1];
            m_xi_lo = m_xi_values_h[0];
            m_xi_hi = m_xi_values_h[xi_grid_size-1];

            SanityCheckEnergyGrid(m_xi_energies);

            // The xi energy grid does not need to be evenly spaced. Compare the
            // smallest and largest spacing to decide whether `MCCXiView::getXi` can use
            // the fast, search-free index lookup instead of a bisection search.
            amrex::ParticleReal xi_dE_min = m_xi_energy_hi - m_xi_energy_lo;
            amrex::ParticleReal xi_dE_max = 0;
            for (int i = 1; i < xi_grid_size; ++i) {
                const amrex::ParticleReal dE_i = m_xi_energies[i] - m_xi_energies[i-1];
                xi_dE_min = std::min(xi_dE_min, dE_i);
                xi_dE_max = std::max(xi_dE_max, dE_i);
            }
            // Same tolerance that the evenly-spaced grid check used historically.
            m_xi_uniform = (xi_dE_max - xi_dE_min < xi_dE_min / 100.0);
            // For an evenly spaced grid, keep the exact step that the fast lookup
            // expects; this also reproduces the pre-existing xi values bit-for-bit.
            m_xi_dE = m_xi_uniform
                    ? (m_xi_energy_hi - m_xi_energy_lo)
                      / static_cast<amrex::ParticleReal>(xi_grid_size - 1)
                    : xi_dE_min;

#ifdef AMREX_USE_GPU
            m_xi_energies_d.resize(m_xi_energies.size());
            m_xi_values_d.resize(m_xi_values_h.size());
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                                  m_xi_energies.begin(), m_xi_energies.end(),
                                  m_xi_energies_d.begin());
            amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                                  m_xi_values_h.begin(), m_xi_values_h.end(),
                                  m_xi_values_d.begin());
            amrex::Gpu::streamSynchronize();
#endif
        }
    }

    // query for a list of collision processes
    // these could be elastic, excitation, charge_exchange, back, etc.
    amrex::Vector<std::string> scattering_process_names;
    pp_collision_name.queryarr("scattering_processes", scattering_process_names);

    // create a vector of ScatteringProcess objects from each scattering
    // process name
    for (const auto& scattering_process : scattering_process_names) {
        const std::string kw_cross_section = scattering_process + "_cross_section";
        std::string cross_section_file;
        pp_collision_name.query(kw_cross_section.c_str(), cross_section_file);

        amrex::ParticleReal energy = 0.0;
        // if the scattering process is excitation or ionization get the
        // energy associated with that process
        if (scattering_process.find("excitation") != std::string::npos ||
            scattering_process.find("ionization") != std::string::npos) {
            const std::string kw_energy = scattering_process + "_energy";
            utils::parser::getWithParser(
                pp_collision_name, kw_energy.c_str(), energy);
        }
        // if the scattering process is forward scattering get the energy
        // associated with the process if it is given (this allows forward
        // scattering to be used both with and without a fixed energy loss)
        else if (scattering_process.find("forward") != std::string::npos ||
                 scattering_process.find("back") != std::string::npos) {
            const std::string kw_energy = scattering_process + "_energy";
            utils::parser::queryWithParser(
                pp_collision_name, kw_energy.c_str(), energy);
        }

        ScatteringProcess process(scattering_process, cross_section_file, energy);

        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(process.type() != ScatteringProcessType::INVALID,
                                         "Cannot add an unknown scattering process type");

        // if the scattering process is ionization get the secondary species
        // only one ionization process is supported, the vector
        // m_ionization_processes is only used to make it simple to calculate
        // the maximum collision frequency with the same function used for
        // particle conserving processes
        if (process.type() == ScatteringProcessType::IONIZATION) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!ionization_flag,
                                             "Background MCC only supports a single ionization process");
            ionization_flag = true;

            std::string secondary_species;
            pp_collision_name.get("ionization_species", secondary_species);
            m_species_names.push_back(secondary_species);

            // Optional ionization energy-partition parameter (eV).
            // If B_ioniz <= 0, equal split is used in the transform functor.
            utils::parser::queryWithParser(pp_collision_name, "B_ioniz", m_B_ioniz);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                (m_B_ioniz >= 0.0_prt),
                "B_ioniz must be >= 0 (eV). Use 0 for equal energy split."
            );

            m_ionization_processes.push_back(std::move(process));
        } else {
            m_scattering_processes.push_back(std::move(process));
        }
    }


#ifdef AMREX_USE_GPU
    amrex::Gpu::HostVector<ScatteringProcess::Executor> h_scattering_processes_exe;
    amrex::Gpu::HostVector<ScatteringProcess::Executor> h_ionization_processes_exe;
    for (auto const& p : m_scattering_processes) {
        h_scattering_processes_exe.push_back(p.executor());
    }
    for (auto const& p : m_ionization_processes) {
        h_ionization_processes_exe.push_back(p.executor());
    }
    m_scattering_processes_exe.resize(h_scattering_processes_exe.size());
    m_ionization_processes_exe.resize(h_ionization_processes_exe.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_scattering_processes_exe.begin(),
                          h_scattering_processes_exe.end(), m_scattering_processes_exe.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice, h_ionization_processes_exe.begin(),
                          h_ionization_processes_exe.end(), m_ionization_processes_exe.begin());
    amrex::Gpu::streamSynchronize();
#else
    for (auto const& p : m_scattering_processes) {
        m_scattering_processes_exe.push_back(p.executor());
    }
    for (auto const& p : m_ionization_processes) {
        m_ionization_processes_exe.push_back(p.executor());
    }
#endif
}

/** Calculate the maximum collision frequency using a fixed energy grid that
 *  ranges from 1e-4 to 5000 eV in 0.2 eV increments
 */
amrex::ParticleReal
BackgroundMCCCollision::get_nu_max(amrex::Vector<ScatteringProcess> const& mcc_processes) const
{
    using namespace amrex::literals;
    amrex::ParticleReal nu, nu_max = 0.0;
    amrex::ParticleReal E_start = 1e-4_prt;
    amrex::ParticleReal E_end = 5000._prt;
    amrex::ParticleReal E_step = 0.2_prt;

    // set the energy limits and step size for calculating nu_max based
    // on the given cross-section inputs
    for (const auto &process : mcc_processes) {
        auto energy_lo = process.getMinEnergyInput();
        E_start = (energy_lo < E_start) ? energy_lo : E_start;
        auto energy_hi = process.getMaxEnergyInput();
        E_end = (energy_hi > E_end) ? energy_hi : E_end;
        auto energy_step = process.getEnergyInputStep();
        E_step = (energy_step < E_step) ? energy_step : E_step;
    }

    amrex::ParticleReal E = E_start;
    while(E < E_end){
        amrex::ParticleReal sigma_E = 0.0;

        // loop through all collision pathways
        for (const auto &scattering_process : mcc_processes) {
            // get collision cross-section
            sigma_E += scattering_process.getCrossSection(E);
        }

        // calculate collision frequency
        nu = (
              m_max_background_density
              * std::sqrt(2.0_prt / m_mass1 * PhysConst::q_e)
              * sigma_E * std::sqrt(E)
              );
        if (nu > nu_max) {
            nu_max = nu;
        }

        E+=E_step;
    }
    return nu_max;
}

void
BackgroundMCCCollision::doCollisions (amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc)
{
    WARPX_PROFILE("BackgroundMCCCollision::doCollisions()");
    using namespace amrex::literals;

    auto& species1 = mypc->GetParticleContainerFromName(m_species_names[0]);
    // this is a very ugly hack to have species2 be a reference and be
    // defined in the scope of doCollisions
    auto& species2 = (
                      (m_species_names.size() == 2) ?
                      mypc->GetParticleContainerFromName(m_species_names[1]) :
                      mypc->GetParticleContainerFromName(m_species_names[0])
                      );

    if (!init_flag) {
        m_mass1 = species1.getMass();

        // calculate maximum collision frequency without ionization
        m_nu_max = get_nu_max(m_scattering_processes);

        // calculate total collision probability
        auto coll_n = m_nu_max * dt;
        m_total_collision_prob = 1.0_prt - std::exp(-coll_n);

        // dt has to be small enough that a linear expansion of the collision
        // probability is sufficiently accurately, otherwise the MCC results
        // will be very heavily affected by small changes in the timestep
        if (coll_n > 0.1_prt) {
            ablastr::warn_manager::WMRecordWarning("BackgroundMCC Collisions",
                     "dt is too large to ensure accurate MCC results , coll_n: " +
                      std::to_string(coll_n) + " is > 0.1 and collision probability is = " +
                      std::to_string(m_total_collision_prob) + "\n");
        }

        if (ionization_flag) {
            // calculate maximum collision frequency for ionization
            m_nu_max_ioniz = get_nu_max(m_ionization_processes);

            // calculate total ionization probability
            auto coll_n_ioniz = m_nu_max_ioniz * dt;
            m_total_collision_prob_ioniz = 1.0_prt - std::exp(-coll_n_ioniz);

            if (coll_n_ioniz > 0.1_prt) {
                ablastr::warn_manager::WMRecordWarning("BackgroundMCC Collisions",
                         "dt is too large to ensure accurate MCC ionization , coll_n_ionization: " +
                          std::to_string(coll_n_ioniz) + " is > 0.1 and ionization probability is = " +
                          std::to_string(m_total_collision_prob_ioniz) + "\n");
            }

            // if an ionization process is included the secondary species mass
            // is taken as the background mass
            m_background_mass = species2.getMass();
        }
        // if no neutral species mass was specified and ionization is not
        // included assume that the collisions will be with neutrals of the
        // same mass as the colliding species (as in ion-neutral collisions)
        else if (m_background_mass == -1) {
            m_background_mass = species1.getMass();
        }

        amrex::Print() << Utils::TextMsg::Info(
            "Setting up Monte-Carlo collisions for " + m_species_names[0] + " with:\n"
            + "     total non-ionization collision probability: "
            + std::to_string(m_total_collision_prob)
            + "\n     total ionization collision probability: "
            + std::to_string(m_total_collision_prob_ioniz)
        );

        init_flag = true;
    }

    // Initialize collision tracking if enabled
    if (m_enable_collision_tracking && !m_tracking_initialized) {
        auto const flvl = species1.finestLevel();
        amrex::Vector<amrex::BoxArray> ba(flvl + 1);
        amrex::Vector<amrex::DistributionMapping> dm(flvl + 1);
        for (int lev = 0; lev <= flvl; ++lev) {
            ba[lev] = species1.ParticleBoxArray(lev);
            dm[lev] = species1.ParticleDistributionMap(lev);
        }
        InitializeCollisionTracking(flvl + 1, ba, dm);
    }

    // Loop over refinement levels
    auto const flvl = species1.finestLevel();
    for (int lev = 0; lev <= flvl; ++lev) {

        auto *cost = WarpX::getCosts(lev);

        // firstly loop over particles box by box and do all particle conserving
        // scattering
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (WarpXParIter pti(species1, lev); pti.isValid(); ++pti) {
            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
            }
            auto wt = static_cast<amrex::Real>(amrex::second());

            doBackgroundCollisionsWithinTile(pti, cur_time);

            if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
            {
                amrex::Gpu::synchronize();
                wt = static_cast<amrex::Real>(amrex::second()) - wt;
                amrex::HostDevice::Atomic::Add( &(*cost)[pti.index()], wt);
            }
        }

        // secondly perform ionization through the SmartCopyFactory if needed
        if (ionization_flag) {
            doBackgroundIonization(lev, cost, species1, species2, cur_time);
        }
    }
}

MCCXiView
BackgroundMCCCollision::makeXiView () const
{
    MCCXiView xi_view;
    xi_view.m_energy_lo = m_xi_energy_lo;
    xi_view.m_energy_hi = m_xi_energy_hi;
    xi_view.m_dE = m_xi_dE;
    xi_view.m_lo = m_xi_lo;
    xi_view.m_hi = m_xi_hi;
    xi_view.m_uniform = m_xi_uniform;
    xi_view.m_grid_size = static_cast<int>(m_xi_energies.size());

    if (m_anisotropic_scatter && !m_use_screened_coulomb) {
#ifdef AMREX_USE_GPU
        xi_view.m_data = m_xi_values_d.data();
        xi_view.m_energies = m_xi_energies_d.data();
#else
        xi_view.m_data = m_xi_values_h.data();
        xi_view.m_energies = m_xi_energies.data();
#endif
    }

    return xi_view;
}


void BackgroundMCCCollision::doBackgroundCollisionsWithinTile
( WarpXParIter& pti, amrex::Real t )
{
    using namespace amrex::literals;

    // So that CUDA code gets its intrinsic, not the host-only C++ library version
    using std::sqrt;

    // get particle count
    const long np = pti.numParticles();

    // get parsers for the background density and temperature
    auto n_a_func = m_background_density_func;
    auto T_a_func = m_background_temperature_func;

    // get collision parameters
    auto *scattering_processes = m_scattering_processes_exe.data();
    auto const process_count  = static_cast<int>(m_scattering_processes_exe.size());

    auto const total_collision_prob = m_total_collision_prob;
    auto const nu_max = m_nu_max;

    // store projectile and target masses
    auto const m = m_mass1;
    auto const M = m_background_mass;

    // we need particle positions in order to calculate the local density
    // and temperature
    auto GetPosition = GetParticlePosition<PIdx>(pti);

    // get Struct-Of-Array particle data, also called attribs
    auto& attribs = pti.GetAttribs();
    amrex::ParticleReal* const AMREX_RESTRICT w = attribs[PIdx::w].dataPtr();
    amrex::ParticleReal* const AMREX_RESTRICT ux = attribs[PIdx::ux].dataPtr();
    amrex::ParticleReal* const AMREX_RESTRICT uy = attribs[PIdx::uy].dataPtr();
    amrex::ParticleReal* const AMREX_RESTRICT uz = attribs[PIdx::uz].dataPtr();

    // Get collision tracking array for cell-by-cell tracking
    const int lev = pti.GetLevel();
    amrex::Array4<amrex::Real> tracking_arr;
    bool do_tracking = false;
    if (m_tracking_initialized && lev < static_cast<int>(m_collision_tracking_mf.size())) {
        auto& tracking_mf = m_collision_tracking_mf[lev];
        tracking_arr = tracking_mf->array(pti);
        do_tracking = true;
    }

    // Get geometry and box info for cell index calculation
    const auto& geom = WarpX::GetInstance().Geom(lev);
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();

    // Capture anisotropic_scatter flag for use inside the GPU lambda
    auto const anisotropic_scatter = m_anisotropic_scatter;
    auto const xi_view = makeXiView();

    amrex::ParallelForRNG(np,
                          [=] AMREX_GPU_HOST_DEVICE (long ip, amrex::RandomEngine const& engine)
                          {
                              // determine if this particle should collide
                              if (amrex::Random(engine) > total_collision_prob) { return; }

                              amrex::ParticleReal x, y, z;
                              GetPosition.AsStored(ip, x, y, z);

                              // Calculate cell indices for this particle
                              int i = 0, j = 0, k = 0;
                              if (do_tracking) {
                                  getCellIndices(x, y, z, plo, dxi, i, j, k);
                              }

                              const amrex::ParticleReal n_a = n_a_func(x, y, z, t);
                              const amrex::ParticleReal T_a = T_a_func(x, y, z, t);

                              amrex::ParticleReal v_coll, v_coll2, sigma_E, nu_i = 0;
                              double E_coll;
                              amrex::ParticleReal ua_x, ua_y, ua_z, vx, vy, vz;
                              amrex::ParticleReal uCOM_x, uCOM_y, uCOM_z;
                              const amrex::ParticleReal col_select = amrex::Random(engine);
                              amrex::ParticleReal xi;

                              // get velocities of gas particles from a Maxwellian distribution
                              auto const vel_std = sqrt(PhysConst::kb * T_a / M);
                              ua_x = vel_std * amrex::RandomNormal(0_prt, 1.0_prt, engine);
                              ua_y = vel_std * amrex::RandomNormal(0_prt, 1.0_prt, engine);
                              ua_z = vel_std * amrex::RandomNormal(0_prt, 1.0_prt, engine);

                              // we assume the target particle is not relativistic (in
                              // the lab frame) and therefore we can transform the projectile
                              // velocity to a frame in which the target is stationary with
                              // a simple Galilean boost
                              // not doing the full Lorentz boost here saves us computation
                              // since most particles will not actually collide
                              vx = ux[ip] - ua_x;
                              vy = uy[ip] - ua_y;
                              vz = uz[ip] - ua_z;
                              v_coll2 = (vx*vx + vy*vy + vz*vz);
                              v_coll = std::sqrt(v_coll2);

                              // calculate the collision energy in eV
                              ParticleUtils::getNonrelativisticCollisionEnergy(v_coll2, m, M, E_coll);

                              // Store initial kinetic energy for tracking
                              const double E_initial = (do_tracking) ?
                                  Algorithms::NonrelativisticKineticEnergy<double>(ux[ip], uy[ip], uz[ip], m) : 0.0;

                              // loop through all collision pathways
                              for (int iproc = 0; iproc < process_count; iproc++) {
                                  auto const& scattering_process = *(scattering_processes + iproc);

                                  // get collision cross-section
                                  sigma_E = scattering_process.getCrossSection(static_cast<amrex::ParticleReal>(E_coll));

                                  // calculate normalized collision frequency
                                  nu_i += n_a * sigma_E * v_coll / nu_max;

                                  // check if this collision should be performed
                                  if (col_select > nu_i) { continue; }

                                  // charge exchange is implemented as a simple swap of the projectile
                                  // and target velocities which doesn't require any of the Lorentz
                                  // transformations below; note that if the projectile and target
                                  // have the same mass this is identical to back scattering
                                  if (scattering_process.m_type == ScatteringProcessType::TWOPRODUCT_REACTION) {
                                      ux[ip] = ua_x;
                                      uy[ip] = ua_y;
                                      uz[ip] = ua_z;

                                      // Track collision (interleaved: count at 2*iproc, energy at 2*iproc+1)
                                      if (do_tracking) {
                                          const double E_final = Algorithms::NonrelativisticKineticEnergy<double>(ua_x, ua_y, ua_z, m);
                                          const double E_transfer = E_initial - E_final;
                                          const auto weight = static_cast<amrex::Real>(w[ip]);
                                          amrex::Gpu::Atomic::AddNoRet(&tracking_arr(i, j, k, 2*iproc), weight);
                                          amrex::Gpu::Atomic::AddNoRet(&tracking_arr(i, j, k, 2*iproc + 1),
                                              weight * static_cast<amrex::Real>(E_transfer));
                                      }
                                      break;
                                  }

                                  // At this point the given particle has been chosen for a collision
                                  // and so we perform the needed calculations to transform to the
                                  // COM frame.
                                  uCOM_x = static_cast<amrex::ParticleReal>(m * vx / (m + M));
                                  uCOM_y = static_cast<amrex::ParticleReal>(m * vy / (m + M));
                                  uCOM_z = static_cast<amrex::ParticleReal>(m * vz / (m + M));

                                  // subtract any energy penalty of the collision from the
                                  // projectile energy
                                  if (scattering_process.m_energy_penalty > 0.0_prt) {
                                      const double E_coll_before = E_coll;
                                      E_coll -= scattering_process.m_energy_penalty;
                                      const auto scale_fac = static_cast<amrex::ParticleReal>(
                                        std::sqrt(E_coll / E_coll_before));
                                      vx *= scale_fac;
                                      vy *= scale_fac;
                                      vz *= scale_fac;
                                  }

                                  // Early exit when projectile does not scatter
                                  if (scattering_process.m_type == ScatteringProcessType::FORWARD) {
                                      ux[ip] = vx;
                                      uy[ip] = vy;
                                      uz[ip] = vz;

                                      // Track collision (interleaved: count at 2*iproc, energy at 2*iproc+1)
                                      if (do_tracking) {
                                          const double E_final = Algorithms::NonrelativisticKineticEnergy<double>(vx, vy, vz, m);
                                          const double E_transfer = E_initial - E_final;
                                          const auto weight = static_cast<amrex::Real>(w[ip]);
                                          amrex::Gpu::Atomic::AddNoRet(&tracking_arr(i, j, k, 2*iproc), weight);
                                          amrex::Gpu::Atomic::AddNoRet(&tracking_arr(i, j, k, 2*iproc + 1),
                                              weight * static_cast<amrex::Real>(E_transfer));
                                      }
                                      break;
                                  }

                                  // transform to COM frame
                                  ParticleUtils::doGalileanTransform(vx, vy, vz, uCOM_x, uCOM_y, uCOM_z);

                                  if ((scattering_process.m_type == ScatteringProcessType::ELASTIC)
                                      || (scattering_process.m_type == ScatteringProcessType::EXCITATION)) {
                                      if (anisotropic_scatter) {
                                          // xi is evaluated with the post-threshold energy
                                          xi = xi_view.getXi(static_cast<amrex::ParticleReal>(E_coll));
                                          const amrex::ParticleReal v_mag = sqrt(vx*vx + vy*vy + vz*vz);
                                          const amrex::ParticleReal vT = sqrt(vx*vx + vy*vy);
                                          ParticleUtils::AnisotropicScatterAndScaleVelocity(
                                              vx, vy, vz, v_mag, vT, xi, m, M, engine
                                          );
                                      } else {
                                          ParticleUtils::RandomizeVelocity(
                                              vx, vy, vz, sqrt(vx*vx + vy*vy + vz*vz), engine
                                          );
                                      }
                                  }
                                  else if (scattering_process.m_type == ScatteringProcessType::BACK) {
                                      // elastic scattering with cos(chi) = -1 (i.e. 180 degrees)
                                      vx *= -1.0_prt;
                                      vy *= -1.0_prt;
                                      vz *= -1.0_prt;
                                  }

                                  // transform back to scattering frame
                                  ParticleUtils::doGalileanTransform(vx, vy, vz, -uCOM_x, -uCOM_y, -uCOM_z);

                                  // update particle velocity with new components in labframe
                                  ux[ip] = vx + ua_x;
                                  uy[ip] = vy + ua_y;
                                  uz[ip] = vz + ua_z;

                                  // Track collision (interleaved: count at 2*iproc, energy at 2*iproc+1)
                                  if (do_tracking) {
                                      const double E_final = Algorithms::NonrelativisticKineticEnergy<double>(
                                          vx + ua_x, vy + ua_y, vz + ua_z, m);
                                      const double E_transfer = E_initial - E_final;
                                      const auto weight = static_cast<amrex::Real>(w[ip]);
                                      amrex::Gpu::Atomic::AddNoRet(&tracking_arr(i, j, k, 2*iproc), weight);
                                      amrex::Gpu::Atomic::AddNoRet(&tracking_arr(i, j, k, 2*iproc + 1),
                                          weight * static_cast<amrex::Real>(E_transfer));
                                  }
                                  break;
                              }
                          }
                          );
}


void BackgroundMCCCollision::doBackgroundIonization
( int lev, amrex::LayoutData<amrex::Real>* cost,
  WarpXParticleContainer& species1, WarpXParticleContainer& species2, amrex::Real t)
{
    WARPX_PROFILE("BackgroundMCCCollision::doBackgroundIonization()");

    const SmartCopyFactory copy_factory_elec(species1, species1);
    const SmartCopyFactory copy_factory_ion(species1, species2);
    const auto CopyElec = copy_factory_elec.getSmartCopy();
    const auto CopyIon = copy_factory_ion.getSmartCopy();

    const auto Filter = ImpactIonizationFilterFunc(
                                                   m_ionization_processes[0],
                                                   m_mass1, m_total_collision_prob_ioniz,
                                                   m_nu_max_ioniz, m_background_density_func, t
                                                   );

    const amrex::ParticleReal sqrt_kb_m = std::sqrt(PhysConst::kb / m_background_mass);

    // Get tracking array for ionization if enabled
    amrex::Array4<amrex::Real> tracking_arr;
    bool do_tracking = false;
    if (m_tracking_initialized && lev < static_cast<int>(m_collision_tracking_mf.size())) {
        do_tracking = true;
    }

    // Get geometry for cell index calculation
    const auto& geom = WarpX::GetInstance().Geom(lev);
    const auto plo = geom.ProbLoArray();
    const auto dxi = geom.InvCellSizeArray();

    // Get ionization process index (it's after all scattering processes)
    const int ionization_comp_idx = static_cast<int>(m_scattering_processes.size());

    auto const anisotropic_scatter = m_anisotropic_scatter;
    auto const xi_view = makeXiView();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (WarpXParIter pti(species1, lev); pti.isValid(); ++pti) {

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
        }
        auto wt = static_cast<amrex::Real>(amrex::second());

        // Get tracking array for this tile
        if (do_tracking) {
            auto& tracking_mf = m_collision_tracking_mf[lev];
            tracking_arr = tracking_mf->array(pti);
        }

        auto& elec_tile = species1.ParticlesAt(lev, pti);
        auto& ion_tile = species2.ParticlesAt(lev, pti);

        const auto np_elec = elec_tile.numParticles();
        const auto np_ion = ion_tile.numParticles();

        auto Transform = ImpactIonizationTransformFunc(
                                                       m_ionization_processes[0].getEnergyPenalty(),
                                                       m_mass1, m_background_mass, sqrt_kb_m,
                                                       m_background_temperature_func, t,
                                                       anisotropic_scatter, m_B_ioniz, xi_view,
                                                       do_tracking, tracking_arr, ionization_comp_idx,
                                                       plo, dxi
                                                       );

        const auto num_added = filterCopyTransformParticles<1>(species1, species2,
                                                               elec_tile, ion_tile, elec_tile, np_elec, np_ion,
                                                               Filter, CopyElec, CopyIon, Transform
                                                               );

        setNewParticleIDs(elec_tile, np_elec, num_added);
        setNewParticleIDs(ion_tile, np_ion, num_added);

        if (cost && WarpX::load_balance_costs_update_algo == LoadBalanceCostsUpdateAlgo::Timers)
        {
            amrex::Gpu::synchronize();
            wt = static_cast<amrex::Real>(amrex::second()) - wt;
            amrex::HostDevice::Atomic::Add( &(*cost)[pti.index()], wt);
        }
    }
}

void BackgroundMCCCollision::InitializeCollisionTracking(
    int nlevs,
    amrex::Vector<amrex::BoxArray> const& ba,
    amrex::Vector<amrex::DistributionMapping> const& dm)
{
    if (m_tracking_initialized) {
        return;  // Already initialized
    }

    const int ncomps = getNumTrackingComponents();
    if (ncomps == 0) {
        return;  // No scattering processes to track
    }

    m_collision_tracking_mf.resize(nlevs);
    for (int lev = 0; lev < nlevs; ++lev) {
        m_collision_tracking_mf[lev] = std::make_unique<amrex::MultiFab>(
            ba[lev], dm[lev], ncomps, 0);
        m_collision_tracking_mf[lev]->setVal(0.0);
    }

    m_tracking_initialized = true;
}

amrex::MultiFab* BackgroundMCCCollision::getCollisionTracking(int lev)
{
    if (!m_tracking_initialized || lev >= static_cast<int>(m_collision_tracking_mf.size())) {
        return nullptr;
    }
    return m_collision_tracking_mf[lev].get();
}

void BackgroundMCCCollision::resetCollisionTracking(int lev)
{
    if (m_tracking_initialized && lev < static_cast<int>(m_collision_tracking_mf.size())) {
        m_collision_tracking_mf[lev]->setVal(0.0);
    }
}

amrex::Vector<std::string> BackgroundMCCCollision::getProcessNames() const
{
    amrex::Vector<std::string> names;
    for (const auto& process : m_scattering_processes) {
        names.push_back(process.name());
    }
    for (const auto& process : m_ionization_processes) {
        names.push_back(process.name());
    }
    return names;
}

void BackgroundMCCCollision::gatherCollisionTracking(int lev, amrex::Vector<amrex::Real>& data,
                                                     amrex::Box& box, int ngrow)
{
    data.clear();
    box = amrex::Box();

    if (!m_tracking_initialized || lev >= static_cast<int>(m_collision_tracking_mf.size())) {
        return;
    }

    auto* mf = m_collision_tracking_mf[lev].get();
    if (mf == nullptr) {
        return;
    }

    const int ncomp = getNumTrackingComponents();
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

    // Copy from the distributed MultiFab to the gathered one
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
    }
}
