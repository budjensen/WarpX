/* Copyright 2021 Modern Electron
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_BACKGROUNDMCCCOLLISION_H_
#define WARPX_PARTICLES_COLLISION_BACKGROUNDMCCCOLLISION_H_

#include "Particles/MultiParticleContainer.H"
#include "Particles/Collision/CollisionBase.H"
#include "Particles/Collision/ScatteringProcess.H"
#include "XiView.H"

#include <AMReX_Parser.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_GpuContainers.H>

#include <memory>
#include <string>

class BackgroundMCCCollision final
    : public CollisionBase
{
public:
    BackgroundMCCCollision (std::string const& collision_name);

    ~BackgroundMCCCollision () override = default;

    BackgroundMCCCollision ( BackgroundMCCCollision const &)             = delete;
    BackgroundMCCCollision& operator= ( BackgroundMCCCollision const & ) = delete;
    BackgroundMCCCollision ( BackgroundMCCCollision&& )                  = delete;
    BackgroundMCCCollision& operator= ( BackgroundMCCCollision&& )       = delete;

    [[nodiscard]] amrex::ParticleReal get_nu_max (amrex::Vector<ScatteringProcess> const& mcc_processes) const;

    /** Perform the collisions
     *
     * @param cur_time Current time
     * @param dt Time step size
     * @param mypc Container of species involved
     *
     */
    void doCollisions (amrex::Real cur_time, amrex::Real dt, MultiParticleContainer* mypc) override;

    /** Perform particle conserving MCC collisions within a tile
     *
     * @param pti particle iterator
     * @param t current time
     *
     */
    void doBackgroundCollisionsWithinTile ( WarpXParIter& pti, amrex::Real t);

    /** Perform MCC ionization interactions
     *
     * @param[in] lev the mesh-refinement level
     * @param[in,out] cost pointer to (load balancing) cost corresponding to box where present particles are ionized.
     * @param[in,out] species1,species2 reference to species container used to inject
     * new particles
     * @param t current time
     *
     */
    void doBackgroundIonization (
                                 int lev,
                                 amrex::LayoutData<amrex::Real>* cost,
                                 WarpXParticleContainer& species1,
                                 WarpXParticleContainer& species2,
                                 amrex::Real t
                                 );

    /** Initialize collision tracking multifabs for all refinement levels
     *
     * @param[in] nlevs number of refinement levels
     * @param[in] ba box array for each level
     * @param[in] dm distribution mapping for each level
     */
    void InitializeCollisionTracking(
        int nlevs,
        amrex::Vector<amrex::BoxArray> const& ba,
        amrex::Vector<amrex::DistributionMapping> const& dm);

    /** Get collision tracking multifab for a given level
     *
     * @param[in] lev refinement level
     * @return pointer to the collision tracking multifab
     */
    amrex::MultiFab* getCollisionTracking(int lev);

    /** Reset collision tracking data (zero all cells) for a given level
     *
     * @param[in] lev refinement level
     */
    void resetCollisionTracking(int lev);

    /** Get the number of tracking components (2 * total number of processes)
     *
     * @return number of components in tracking multifab
     */
    [[nodiscard]] int getNumTrackingComponents() const {
        return 2 * static_cast<int>(m_scattering_processes.size() + m_ionization_processes.size());
    }

    /** Get the names of scattering processes for tracking
     *
     * @return vector of process names
     */
    [[nodiscard]] amrex::Vector<std::string> getProcessNames() const;

    /** Gather collision tracking data from all boxes into a single spatial array
     *
     * Collects data from all boxes and arranges them spatially into a single
     * contiguous array. Only the IO processor (rank 0) gets the full data;
     * other ranks get an empty vector.
     *
     * @param[in] lev refinement level
     * @param[out] data output vector containing the gathered spatial data
     * @param[out] box output Box describing the index space of the gathered data
     * @param[in] ngrow number of grow cells to include (default=0)
     */
    void gatherCollisionTracking(int lev, amrex::Vector<amrex::Real>& data,
                                 amrex::Box& box, int ngrow=0);

private:

    /**
     * \brief Calculate cell indices from particle position
     *
     * \param[in] x, y, z particle coordinates
     * \param[in] plo lower corner of problem domain
     * \param[in] dxi inverse cell size array
     * \param[out] i, j, k cell indices
     */
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static void getCellIndices(amrex::ParticleReal x, amrex::ParticleReal y, amrex::ParticleReal z,
                               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
                               amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi,
                               int& i, int& j, int& k)
    {
#if defined(WARPX_DIM_1D_Z)
        amrex::ignore_unused(x, y);
        i = static_cast<int>((z - plo[0]) * dxi[0]);
        j = 0;
        k = 0;
#elif defined(WARPX_DIM_XZ) || defined(WARPX_DIM_RZ)
        amrex::ignore_unused(y);
        i = static_cast<int>((x - plo[0]) * dxi[0]);
        j = 0;
        k = static_cast<int>((z - plo[1]) * dxi[1]);
#else
        i = static_cast<int>((x - plo[0]) * dxi[0]);
        j = static_cast<int>((y - plo[1]) * dxi[1]);
        k = static_cast<int>((z - plo[2]) * dxi[2]);
#endif
    }

    [[nodiscard]]
    MCCXiView makeXiView () const;

    amrex::Vector<ScatteringProcess> m_scattering_processes;
    amrex::Vector<ScatteringProcess> m_ionization_processes;
    amrex::Gpu::DeviceVector<ScatteringProcess::Executor> m_scattering_processes_exe;
    amrex::Gpu::DeviceVector<ScatteringProcess::Executor> m_ionization_processes_exe;

    bool init_flag = false;
    bool ionization_flag = false;

    // When true, ELASTIC and EXCITATION processes use the anisotropic
    // Okhrimovskyy/Vahedi scattering model instead of isotropic scattering
    bool m_anisotropic_scatter = false;
    bool m_use_screened_coulomb = false;
    amrex::ParticleReal m_B_ioniz = 0;

    // Collision-level xi(E) data shared across all pathways.
    amrex::Vector<amrex::ParticleReal> m_xi_energies;
    amrex::Gpu::HostVector<amrex::ParticleReal> m_xi_values_h;
#ifdef AMREX_USE_GPU
    amrex::Gpu::DeviceVector<amrex::ParticleReal> m_xi_values_d;
#endif
    amrex::ParticleReal m_xi_energy_lo = 0;
    amrex::ParticleReal m_xi_energy_hi = 0;
    amrex::ParticleReal m_xi_dE = 0;
    amrex::ParticleReal m_xi_lo = 0;
    amrex::ParticleReal m_xi_hi = 0;

    amrex::ParticleReal m_mass1;

    amrex::ParticleReal m_max_background_density = 0;
    amrex::ParticleReal m_background_mass;
    amrex::ParticleReal m_total_collision_prob;
    amrex::ParticleReal m_total_collision_prob_ioniz = 0;
    amrex::ParticleReal m_nu_max;
    amrex::ParticleReal m_nu_max_ioniz;

    amrex::Parser m_background_density_parser;
    amrex::Parser m_background_temperature_parser;

    amrex::ParserExecutor<4> m_background_density_func;
    amrex::ParserExecutor<4> m_background_temperature_func;

    // Collision tracking multifabs (one per refinement level)
    // Component layout: [0, 2, 4, ...]: collision counts, [1, 3, 5, ...]: energy transfer (in Joules)
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> m_collision_tracking_mf;
    bool m_tracking_initialized = false;
    bool m_enable_collision_tracking = false;
};

#endif // WARPX_PARTICLES_COLLISION_BACKGROUNDMCCCOLLISION_H_
