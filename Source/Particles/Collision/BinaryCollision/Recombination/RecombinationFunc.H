#ifndef WARPX_RECOMBINATION_FUNC_H_
#define WARPX_RECOMBINATION_FUNC_H_

#include "Particles/Collision/BinaryCollision/BinaryCollisionUtils.H"
#include "Particles/Collision/CollisionBase.H"
#include "Particles/Collision/ScatteringProcess.H"
#include "Particles/MultiParticleContainer.H"
#include "Particles/Pusher/GetAndSetPosition.H"
#include "Particles/WarpXParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/ParticleUtils.H"
#include "Utils/TextMsg.H"

#include <AMReX_DenseBins.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Random.H>
#include <AMReX_REAL.H>

/**
 * \brief This class performs recombination collisions within a cell.
 * Two kinetic particles are selected, their relative energy is computed,
 * and a probability based on a user-provided cross section determines
 * if recombination occurs. If successful, both particles are removed.
 *
 * This is similar to MCC collisions but with both projectile and target
 * being kinetic particles, following the DSMC pairing algorithm.
 */
class RecombinationFunc
{
    // Define shortcuts for frequently-used type names
    using ParticleType = WarpXParticleContainer::ParticleType;
    using ParticleTileType = WarpXParticleContainer::ParticleTileType;
    using ParticleTileDataType = ParticleTileType::ParticleTileDataType;
    using ParticleBins = amrex::DenseBins<ParticleTileDataType>;
    using index_type = ParticleBins::index_type;
    using SoaData_type = WarpXParticleContainer::ParticleTileType::ParticleTileDataType;

public:

    /**
     * \brief Default constructor of the RecombinationFunc class.
     */
    RecombinationFunc () = default;

    /**
     * \brief Constructor of the RecombinationFunc class
     *
     * @param[in] collision_name the name of the collision
     * @param[in] mypc pointer to the MultiParticleContainer
     */
    RecombinationFunc ( const std::string& collision_name,
                        MultiParticleContainer const * mypc,
                        bool isSameSpecies );

    struct Executor {
        /**
         * \brief Executor of the RecombinationFunc class. Performs recombination
         * collisions at the cell level following the DSMC algorithm.
         *
         * For each pair of particles:
         * 1. Calculate relative kinetic energy in center-of-momentum frame
         * 2. Get cross section from user-provided data
         * 3. Calculate collision probability
         * 4. Randomly determine if recombination occurs
         * 5. If yes, remove weight from both particles (mark for deletion)
         *
         * @param[in] I1s,I2s is the start index for I1,I2 (inclusive).
         * @param[in] I1e,I2e is the stop index for I1,I2 (exclusive).
         * @param[in] I1,I2 index arrays. They determine all elements that will be used.
         * @param[in] soa_1,soa_2 contain the struct of array data of the two species
         * @param[in] dt is the time step length between two collision calls.
         * @param[in] dV is the volume of the corresponding cell.
         * @param[in] coll_idx is the collision index offset.
         * @param[in] engine the random engine.
         */
        AMREX_GPU_HOST_DEVICE AMREX_INLINE
        void operator() (
            index_type const I1s, index_type const I1e,
            index_type const I2s, index_type const I2e,
            index_type const* AMREX_RESTRICT I1,
            index_type const* AMREX_RESTRICT I2,
            const SoaData_type& soa_1, const SoaData_type& soa_2,
            GetParticlePosition<PIdx> /*get_position_1*/, GetParticlePosition<PIdx> /*get_position_2*/,
            amrex::ParticleReal const /*n1*/, amrex::ParticleReal const /*n2*/,
            amrex::ParticleReal const /*T1*/, amrex::ParticleReal const /*T2*/,
            amrex::ParticleReal const  /*q1*/, amrex::ParticleReal const  /*q2*/,
            amrex::ParticleReal const  m1, amrex::ParticleReal const  m2,
            amrex::Real const  dt, amrex::Real const dV, index_type coll_idx,
            index_type const /*cell_start_pair*/, index_type* AMREX_RESTRICT /*p_mask*/,
            index_type* AMREX_RESTRICT /*p_pair_indices_1*/, index_type* AMREX_RESTRICT /*p_pair_indices_2*/,
            amrex::ParticleReal* AMREX_RESTRICT /*p_pair_reaction_weight*/,
            amrex::RandomEngine const& engine) const
        {
            using namespace amrex::literals;

            amrex::ParticleReal * const AMREX_RESTRICT w1 = soa_1.m_rdata[PIdx::w];
            amrex::ParticleReal * const AMREX_RESTRICT u1x = soa_1.m_rdata[PIdx::ux];
            amrex::ParticleReal * const AMREX_RESTRICT u1y = soa_1.m_rdata[PIdx::uy];
            amrex::ParticleReal * const AMREX_RESTRICT u1z = soa_1.m_rdata[PIdx::uz];
            uint64_t* AMREX_RESTRICT idcpu1 = soa_1.m_idcpu;

            amrex::ParticleReal * const AMREX_RESTRICT w2 = soa_2.m_rdata[PIdx::w];
            amrex::ParticleReal * const AMREX_RESTRICT u2x = soa_2.m_rdata[PIdx::ux];
            amrex::ParticleReal * const AMREX_RESTRICT u2y = soa_2.m_rdata[PIdx::uy];
            amrex::ParticleReal * const AMREX_RESTRICT u2z = soa_2.m_rdata[PIdx::uz];
            uint64_t* AMREX_RESTRICT idcpu2 = soa_2.m_idcpu;

            // Number of macroparticles of each species
            const index_type NI1 = I1e - I1s;
            const index_type NI2 = I2e - I2s;
            const index_type max_N = amrex::max(NI1,NI2);
            const index_type min_N = amrex::min(NI1,NI2);

            // multiplier ratio to take into account unsampled pairs
            // For recombination (always different species), use min_N
            const auto multiplier_ratio = static_cast<int>(min_N);

#if (defined WARPX_DIM_RZ)
            amrex::ParticleReal * const AMREX_RESTRICT theta1 = soa_1.m_rdata[PIdx::theta];
            amrex::ParticleReal * const AMREX_RESTRICT theta2 = soa_2.m_rdata[PIdx::theta];
#endif

            index_type i1 = I1s + coll_idx;
            index_type i2 = I2s + coll_idx;

            // Loop over particle pairs
            for (index_type k = coll_idx; k < max_N; k += min_N)
            {
                // Skip if a particle's weight was reduced to zero from a previous collision
                if (idcpu1[ I1[i1] ] == amrex::ParticleIdCpus::Invalid ||
                    idcpu2[ I2[i2] ] == amrex::ParticleIdCpus::Invalid ) {
                    if (max_N == NI1) { i1 += min_N; }
                    if (max_N == NI2) { i2 += min_N; }
                    continue;
                }

#if (defined WARPX_DIM_RZ)
                /* In RZ geometry, temporarily rotate the momentum of one macroparticle
                 * to account for cylindrical symmetry */
                amrex::ParticleReal const theta = theta2[I2[i2]]-theta1[I1[i1]];
                amrex::ParticleReal const u1xbuf = u1x[I1[i1]];
                u1x[I1[i1]] = u1xbuf*std::cos(theta) - u1y[I1[i1]]*std::sin(theta);
                u1y[I1[i1]] = u1xbuf*std::sin(theta) + u1y[I1[i1]]*std::cos(theta);
#endif

                // Get collision parameters (relative energy and velocity in COM frame)
                amrex::ParticleReal E_coll_eV, v_coll, lorentz_factor;
                BinaryCollisionUtils::get_collision_parameters(
                    u1x[I1[i1]], u1y[I1[i1]], u1z[I1[i1]],
                    u2x[I2[i2]], u2y[I2[i2]], u2z[I2[i2]],
                    m1, m2, E_coll_eV, v_coll, lorentz_factor
                );

                // Convert collision energy from Joules to eV for cross section lookup
                E_coll_eV /= PhysConst::q_e;

                // If energy is above the maximum energy, skip this pair (use energy penalty for recombination maximum)
                if (E_coll_eV > m_recombination_process.m_energy_penalty) {
#if (defined WARPX_DIM_RZ)
                    /* Undo the earlier velocity rotation */
                    amrex::ParticleReal const u1xbuf_new = u1x[I1[i1]];
                    u1x[I1[i1]] = u1xbuf_new*std::cos(-theta) - u1y[I1[i1]]*std::sin(-theta);
                    u1y[I1[i1]] = u1xbuf_new*std::sin(-theta) + u1y[I1[i1]]*std::cos(-theta);
#endif
                    if (max_N == NI1) { i1 += min_N; }
                    if (max_N == NI2) { i2 += min_N; }
                    continue;
                }

                // Get cross section at this collision energy
                const amrex::ParticleReal sigma = m_recombination_process.getCrossSection(E_coll_eV);

                // Calculate the collision probability using DSMC/MCC formula
                // P = n * sigma * v_rel * dt * multiplier_ratio / V
                // where n is the number density of the species with fewer particles
                const amrex::ParticleReal n_density = static_cast<amrex::ParticleReal>(min_N) / dV;
                const amrex::ParticleReal exponent = n_density * sigma * v_coll * dt
                                                     * static_cast<amrex::ParticleReal>(multiplier_ratio)
                                                     * lorentz_factor;

                const amrex::ParticleReal probability = -std::expm1(-exponent);

                // Determine if recombination occurs
                const amrex::ParticleReal random_number = amrex::Random(engine);
                if (random_number < probability) {
                    // Recombination occurs - remove both particles
                    // Use the minimum weight to ensure both particles can be removed
                    const amrex::ParticleReal reaction_weight = amrex::min(w1[I1[i1]], w2[I2[i2]]);

                    BinaryCollisionUtils::remove_weight_from_colliding_particle(
                        w1[I1[i1]], idcpu1[I1[i1]], reaction_weight);
                    BinaryCollisionUtils::remove_weight_from_colliding_particle(
                        w2[I2[i2]], idcpu2[I2[i2]], reaction_weight);
                }

#if (defined WARPX_DIM_RZ)
                /* Undo the earlier velocity rotation */
                amrex::ParticleReal const u1xbuf_new = u1x[I1[i1]];
                u1x[I1[i1]] = u1xbuf_new*std::cos(-theta) - u1y[I1[i1]]*std::sin(-theta);
                u1y[I1[i1]] = u1xbuf_new*std::sin(-theta) + u1y[I1[i1]]*std::cos(-theta);
#endif

                // Move to next pair
                if (max_N == NI1) { i1 += min_N; }
                if (max_N == NI2) { i2 += min_N; }
            }
        }

        // These flags must be present for compatibility with BinaryCollision.H
        static constexpr bool m_computeSpeciesDensities = false;
        static constexpr bool m_computeSpeciesTemperatures = false;

        ScatteringProcess::Executor m_recombination_process;
    };

    [[nodiscard]] Executor const& executor () const { return m_exe; }

private:
    amrex::Vector<ScatteringProcess> m_recombination_process;
    amrex::Gpu::DeviceVector<ScatteringProcess::Executor> m_recombination_process_exe;

    Executor m_exe;
};

#endif // WARPX_RECOMBINATION_FUNC_H_
