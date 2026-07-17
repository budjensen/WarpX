/* Copyright 2021 Modern Electron
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_PARTICLES_COLLISION_IMPACT_IONIZATION_H_
#define WARPX_PARTICLES_COLLISION_IMPACT_IONIZATION_H_

#include "Particles/Collision/ScatteringProcess.H"
#include "XiView.H"

#include "Particles/Algorithms/KineticEnergy.H"
#include "Utils/ParticleUtils.H"
#include "Utils/WarpXConst.H"

#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

/** @file
 *
 * This file contains filter and transform functors for impact ionization
 */

/**
 * \brief Filter functor for impact ionization
 */
class ImpactIonizationFilterFunc
{
public:

    /**
    * \brief Constructor of the ImpactIonizationFilterFunc functor.
    *
    * This function sample a random number and compares it to the total
    * collision probability to see if the given particle should be considered
    * for an ionization event. If the particle passes this stage the collision
    * cross-section is calculated given its energy and another random number
    * is used to determine whether it actually collides.
    * Note that the mass and energy quantities have to be stored as doubles to
    * ensure accurate energy calculations (otherwise errors occur with single
    * or mixed precision builds of WarpX).
    *
    * @param[in] mcc_process an ScatteringProcess object associated with the ionization
    * @param[in] mass colliding particle's mass (could also assume electron)
    * @param[in] total_collision_prob total probability for a collision to occur
    * @param[in] nu_max maximum collision frequency
    * @param[in] n_a_func ParserExecutor<4> function to get the background
                 density in m^-3 as a function of space and time
    * @param[in] t the current simulation time
    */
    ImpactIonizationFilterFunc(
        ScatteringProcess const& mcc_process,
        double const mass,
        amrex::ParticleReal const total_collision_prob,
        amrex::ParticleReal const nu_max,
        amrex::ParserExecutor<4> const& n_a_func,
        amrex::Real t
    ) : m_mcc_process(mcc_process.executor()), m_mass(mass),
        m_total_collision_prob(total_collision_prob),
        m_nu_max(nu_max), m_n_a_func(n_a_func), m_t(t) { }

    /**
    * \brief Functor call. This method determines if a given (electron) particle
    * should undergo an ionization collision.
    *
    * @param[in] ptd particle tile data
    * @param[in] i particle index
    * @param[in] engine the random number state and factory
    * @return true if a collision occurs, false otherwise
    */
    template <typename PData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool operator() (
        const PData& ptd, int const i, amrex::RandomEngine const& engine
    ) const noexcept
    {
        using namespace amrex;
        using std::sqrt;

        // determine if this particle should collide
        if (Random(engine) > m_total_collision_prob) { return false; }

        // get references to the particle to get its position
        const auto& p = ptd.getSuperParticle(i);
        ParticleReal x, y, z;
        double E_coll;
        get_particle_position(p, x, y, z);

        // calculate neutral density at particle location
        const ParticleReal n_a = m_n_a_func(x, y, z, m_t);

        // get the particle velocity
        const ParticleReal ux = ptd.m_rdata[PIdx::ux][i];
        const ParticleReal uy = ptd.m_rdata[PIdx::uy][i];
        const ParticleReal uz = ptd.m_rdata[PIdx::uz][i];

        // calculate kinetic energy
        constexpr auto eV = PhysConst::q_e;
        E_coll = Algorithms::NonrelativisticKineticEnergy<double>(ux, uy, uz, m_mass)/eV;

        // get collision cross-section
        const ParticleReal sigma_E = m_mcc_process.getCrossSection(static_cast<amrex::ParticleReal>(E_coll));

        // calculate normalized collision frequency
        const ParticleReal u_coll2 = ux*ux + uy*uy + uz*uz;
        const ParticleReal nu_i = n_a * sigma_E * sqrt(u_coll2) / m_nu_max;

        // check if this collision should be performed
        return (Random(engine) <= nu_i);
    }

private:
    ScatteringProcess::Executor m_mcc_process;
    double m_mass;
    amrex::ParticleReal m_total_collision_prob = 0;
    amrex::ParticleReal m_nu_max;
    amrex::ParserExecutor<4> m_n_a_func;
    amrex::Real m_t;
};


/**
 * \brief Transform functor for impact ionization
 */
class ImpactIonizationTransformFunc
{
public:

    /**
    * \brief Constructor of the ImpactIonizationTransformFunc functor.
    *
    * The transform is responsible for appropriately decreasing the kinetic
    * energy of the colliding particle and assigning appropriate velocities
    * to the two newly created particles. To this end the energy cost of
    * ionization is passed to the constructor as well as the mass of the
    * colliding species and the standard deviation of the ion velocity
    * (normalized temperature).
    * Note that the mass and energy quantities have to be stored as doubles to
    * ensure accurate energy calculations (otherwise errors occur with single
    * or mixed precision builds of WarpX).
    *
    * @param[in] energy_cost energy cost of ionization
    * @param[in] mass1 mass of the colliding species
    * @param[in] mass2 mass of the background/ionized species (used for the
                 emitted ion's kinetic energy when collision tracking is enabled)
    * @param[in] sqrt_kb_m value of sqrt(kB/m), where kB is Boltzmann's constant
                 and m is the background neutral mass
    * @param[in] T_a_func ParserExecutor<4> function to get the background
                 temperature in Kelvin as a function of space and time
    * @param[in] t the current simulation time
    * @param[in] anisotropic_scatter toggle anisotropic ionization electron scattering
    * @param[in] B_ioniz Vahedi energy partition parameter (eV).
    *            If B_ioniz <= 0, equal post-threshold energy split is used.
    * @param[in] xi_view collision-level xi(E) source for anisotropic scattering
    * @param[in] do_tracking whether to accumulate collision tracking data
    * @param[in] tracking_arr Array4 into which tracking data is accumulated
                 (only dereferenced when do_tracking is true)
    * @param[in] tracking_comp_idx the tracking component index for this
                 ionization process (count is written to 2*tracking_comp_idx,
                 energy to 2*tracking_comp_idx + 1)
    * @param[in] plo lower corner of the problem domain (for cell index lookup)
    * @param[in] dxi inverse cell size array (for cell index lookup)
    */
    ImpactIonizationTransformFunc(
        amrex::ParticleReal energy_cost, double mass1, double mass2, amrex::ParticleReal sqrt_kb_m,
        amrex::ParserExecutor<4> const& T_a_func, amrex::Real t,
        bool anisotropic_scatter, amrex::ParticleReal B_ioniz, MCCXiView xi_view,
        bool do_tracking, amrex::Array4<amrex::Real> const& tracking_arr, int tracking_comp_idx,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& plo,
        amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dxi
    ) :  m_energy_cost(energy_cost), m_mass1(mass1), m_mass2(mass2),
         m_sqrt_kb_m(sqrt_kb_m), m_T_a_func(T_a_func), m_t(t),
         m_anisotropic_scatter(anisotropic_scatter), m_B_ioniz(B_ioniz),
         m_xi_view(xi_view), m_do_tracking(do_tracking), m_tracking_arr(tracking_arr),
         m_tracking_comp_idx(tracking_comp_idx), m_plo(plo), m_dxi(dxi) { }

    /**
    * \brief Functor call. It determines the properties of the generated pair
    * and decreases the kinetic energy of the colliding particle. Inputs are
    * standard from the FilterCopyTransfrom::filterCopyTransformParticles
    * function.
    *
    * @param[in,out] dst1 target species 1 (electrons)
    * @param[in,out] dst2 target species 2 (ions)
    * @param[in] src source species (electrons)
    * @param[in] i_src particle index of the source species
    * @param[in] i_dst1 particle index of target species 1
    * @param[in] i_dst2 particle index of target species 2
    * @param[in] engine random number generator engine
    */
    template <typename DstData, typename SrcData>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    void operator() (DstData& dst1, DstData& dst2, SrcData& src,
        int const i_src, int const i_dst1, int const i_dst2,
        amrex::RandomEngine const& engine) const noexcept
    {
        using namespace amrex;
        using std::sqrt;
        using std::atan;
        using std::tan;

        // get references to the particle to get its position
        const auto& p = src.getSuperParticle(i_src);
        ParticleReal x, y, z;
        double E_coll;
        get_particle_position(p, x, y, z);

        // calculate standard deviation in neutral velocity distribution using
        // the local temperature
        const ParticleReal ion_vel_std = m_sqrt_kb_m * std::sqrt(m_T_a_func(x, y, z, m_t));

        // get references to the original particle's velocity
        auto& ux = src.m_rdata[PIdx::ux][i_src];
        auto& uy = src.m_rdata[PIdx::uy][i_src];
        auto& uz = src.m_rdata[PIdx::uz][i_src];

        // get references to the new particles' velocities
        auto& e_ux = dst1.m_rdata[PIdx::ux][i_dst1];
        auto& e_uy = dst1.m_rdata[PIdx::uy][i_dst1];
        auto& e_uz = dst1.m_rdata[PIdx::uz][i_dst1];
        auto& i_ux = dst2.m_rdata[PIdx::ux][i_dst2];
        auto& i_uy = dst2.m_rdata[PIdx::uy][i_dst2];
        auto& i_uz = dst2.m_rdata[PIdx::uz][i_dst2];

        // calculate kinetic energy
        constexpr auto eV = PhysConst::q_e;
        E_coll = Algorithms::NonrelativisticKineticEnergy<double>(ux, uy, uz, m_mass1)/eV;

        amrex::ParticleReal E_inc = static_cast<amrex::ParticleReal>(E_coll);
        amrex::ParticleReal E_avail = E_inc - m_energy_cost;
        if (E_avail < 0.0_prt) {
            E_avail = 0.0_prt;
        }

        amrex::ParticleReal E_ej = 0.0_prt;
        amrex::ParticleReal E_prim = 0.0_prt;

        // If B_ioniz <= 0, use equal split. Otherwise, sample Vahedi partition.
        if (m_B_ioniz > 0.0_prt) {
            if (E_avail > 0.0_prt) {
                const amrex::ParticleReal theta_max = atan(E_avail / (2.0_prt * m_B_ioniz));
                E_ej = m_B_ioniz * tan(Random(engine) * theta_max);
                if (E_ej < 0.0_prt) {
                    E_ej = 0.0_prt;
                } else if (E_ej > E_avail) {
                    E_ej = E_avail;
                }
            }
            E_prim = E_avail - E_ej;
            if (E_prim < 0.0_prt) {
                E_prim = 0.0_prt;
            }
        } else {
            // Equal post-threshold energy split.
            E_prim = 0.5_prt * E_avail;
            E_ej = E_prim;
        }

        const amrex::ParticleReal up_prim = sqrt(2.0_prt * E_prim * eV / m_mass1);
        const amrex::ParticleReal up_ej = sqrt(2.0_prt * E_ej * eV / m_mass1);

        if (m_anisotropic_scatter) {
            // Scale incident velocities to the appropriate post-collision energies
            const amrex::ParticleReal v_inc = sqrt(ux*ux + uy*uy + uz*uz);

            const amrex::ParticleReal scale_ej = up_ej / v_inc;
            e_ux = ux * scale_ej;
            e_uy = uy * scale_ej;
            e_uz = uz * scale_ej;

            const amrex::ParticleReal scale_prim = up_prim / v_inc;
            ux *= scale_prim;
            uy *= scale_prim;
            uz *= scale_prim;

            const amrex::ParticleReal xi_prim = m_xi_view.getXi(E_prim);
            const amrex::ParticleReal xi_ej = m_xi_view.getXi(E_ej);

            // Compute transverse velocity components for the anisotropic scattering
            const amrex::ParticleReal vT_prim = sqrt(ux*ux + uy*uy);
            const amrex::ParticleReal vT_ej = sqrt(e_ux*e_ux + e_uy*e_uy);

            ParticleUtils::AnisotropicScatterVelocity(ux, uy, uz, up_prim, vT_prim, xi_prim, engine);
            ParticleUtils::AnisotropicScatterVelocity(e_ux, e_uy, e_uz, up_ej, vT_ej, xi_ej, engine);
        } else {
            ParticleUtils::RandomizeVelocity(ux, uy, uz, up_prim, engine);
            ParticleUtils::RandomizeVelocity(e_ux, e_uy, e_uz, up_ej, engine);
        }

        // get velocities for the ion from a Maxwellian distribution
        i_ux = ion_vel_std * RandomNormal(0_prt, 1.0_prt, engine);
        i_uy = ion_vel_std * RandomNormal(0_prt, 1.0_prt, engine);
        i_uz = ion_vel_std * RandomNormal(0_prt, 1.0_prt, engine);

        // Record the exact energy balance of the plasma species involved in
        // this collision:
        //   \Delta E = E_inc_elec_i - (E_inc_elec_f + E_emit_elec + E_emit_ion)
        // This is not simply the ionization energy cost,
        // since the emitted ion's energy is drawn from the
        // background profile. As such, there is a small amount of
        // "heating" as the ion is created.
        if (m_do_tracking) {
            const double E_inc0 = E_coll * eV;
            const double E_inc1 = Algorithms::NonrelativisticKineticEnergy<double>(ux, uy, uz, m_mass1);
            const double E_emit_elec = Algorithms::NonrelativisticKineticEnergy<double>(e_ux, e_uy, e_uz, m_mass1);
            const double E_emit_ion = Algorithms::NonrelativisticKineticEnergy<double>(i_ux, i_uy, i_uz, m_mass2);
            const double E_transfer = E_inc0 - E_inc1 - E_emit_elec - E_emit_ion;

            int ii = 0, jj = 0, kk = 0;
            getCellIndices(x, y, z, m_plo, m_dxi, ii, jj, kk);

            const auto weight = static_cast<amrex::Real>(src.m_rdata[PIdx::w][i_src]);
            amrex::Gpu::Atomic::AddNoRet(&m_tracking_arr(ii, jj, kk, 2*m_tracking_comp_idx), weight);
            amrex::Gpu::Atomic::AddNoRet(&m_tracking_arr(ii, jj, kk, 2*m_tracking_comp_idx + 1),
                weight * static_cast<amrex::Real>(E_transfer));
        }
    }

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

    amrex::ParticleReal m_energy_cost;
    double m_mass1;
    double m_mass2;
    amrex::ParticleReal m_sqrt_kb_m;
    amrex::ParserExecutor<4> m_T_a_func;
    amrex::Real m_t;
    bool m_anisotropic_scatter = false;
    amrex::ParticleReal m_B_ioniz = 0;
    MCCXiView m_xi_view;

    bool m_do_tracking = false;
    amrex::Array4<amrex::Real> m_tracking_arr;
    int m_tracking_comp_idx = 0;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_plo{};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> m_dxi{};
};
#endif // WARPX_PARTICLES_COLLISION_IMPACT_IONIZATION_H_
