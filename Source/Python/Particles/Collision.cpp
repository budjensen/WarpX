/* Copyright 2026 The WarpX Community
 *
 * Authors: Contributors
 * License: BSD-3-Clause-LBNL
 */

#include "Python/pyWarpX.H"

#include <Particles/Collision/CollisionBase.H>
#include <Particles/Collision/BackgroundMCC/BackgroundMCCCollision.H>
#include <Particles/Collision/BinaryCollision/BinaryCollision.H>
#include <Particles/Collision/BinaryCollision/Recombination/RecombinationFunc.H>
#include <Particles/Collision/BinaryCollision/ParticleCreationFunc.H>

#include <AMReX_MultiFab.H>

#include <pybind11/stl.h>


void init_Collision (py::module& m)
{
    // Expose BackgroundMCCCollision methods for collision tracking
    py::class_<CollisionBase>(m, "CollisionBase");

    py::class_<BackgroundMCCCollision, CollisionBase>(m, "BackgroundMCCCollision")
        .def("get_collision_tracking",
             &BackgroundMCCCollision::getCollisionTracking,
             py::arg("lev"),
             py::return_value_policy::reference_internal,
             R"doc(Get collision tracking MultiFab for a given refinement level.

             The MultiFab has cell-by-cell resolution with 2*N components where N is
             the total number of collision processes (scattering + ionization).
             Data is stored in an interleaved layout for better cache performance:
             - Even components (0, 2, 4, ...): collision counts for each process (per cell)
             - Odd components (1, 3, 5, ...): energy transfer in Joules for each process (per cell)

             For example, with 3 processes:
               [count₀, energy₀, count₁, energy₁, count₂, energy₂]

             This layout ensures count and energy for the same process are adjacent in
             memory, improving cache efficiency during collision tracking.

             Each cell independently tracks the number of collisions and total energy
             transferred for particles within that cell.

             Parameters
             ----------
             lev : int
                 Refinement level (0 for base level)

             Returns
             -------
             amrex.MultiFab or None
                 The collision tracking MultiFab, or None if not initialized
             )doc"
        )
        .def("reset_collision_tracking",
             &BackgroundMCCCollision::resetCollisionTracking,
             py::arg("lev"),
             R"doc(Reset collision tracking data (zero all cells) for a given level.

             Parameters
             ----------
             lev : int
                 Refinement level (0 for base level)
             )doc"
        )
        .def("get_process_names",
             [](BackgroundMCCCollision& collision) {
                 auto amrex_vec = collision.getProcessNames();
                 return std::vector<std::string>(amrex_vec.begin(), amrex_vec.end());
             },
             R"doc(Get the names of collision processes for tracking.

             Returns
             -------
             list of str
                 List of process names (e.g., ['elastic1', 'excitation1', 'ionization'])
             )doc"
        )
        .def("gather_collision_tracking",
             [](BackgroundMCCCollision& collision, int lev, int ngrow) {
                 amrex::Vector<amrex::Real> data;
                 amrex::Box box;
                 collision.gatherCollisionTracking(lev, data, box, ngrow);

                 // Create return tuple with data and box dimensions
                 std::vector<amrex::Real> data_vec(data.begin(), data.end());
                 std::vector<int> box_lo(AMREX_SPACEDIM);
                 std::vector<int> box_hi(AMREX_SPACEDIM);

                 for (int i = 0; i < AMREX_SPACEDIM; ++i) {
                     box_lo[i] = box.smallEnd(i);
                     box_hi[i] = box.bigEnd(i);
                 }

                 return py::make_tuple(data_vec, box_lo, box_hi);
             },
             py::arg("lev"), py::arg("ngrow") = 0,
             R"doc(Gather collision tracking data from all boxes into a single spatial array.

             Collects data from all boxes and arranges them spatially into a single
             contiguous array. Only the IO processor (rank 0) gets the full data;
             other ranks get empty arrays.

             The returned data is flattened in row-major (C) order with interleaved
             components: for each cell (i,j,k), data contains [count₀, energy₀, count₁, energy₁, ...].

             Parameters
             ----------
             lev : int
                 Refinement level (0 for base level)
             ngrow : int, optional
                 Number of grow cells to include (default=0)

             Returns
             -------
             tuple of (data, box_lo, box_hi)
                 data : list of float
                     Flattened array with shape (ncells * ncomponents,)
                 box_lo : list of int
                     Lower corner indices of the box [ilo, jlo, klo]
                 box_hi : list of int
                     Upper corner indices (inclusive) [ihi, jhi, khi]
             )doc"
        );

    // Expose RecombinationCollision methods for collision tracking
    // RecombinationCollision is a BinaryCollision<RecombinationFunc>
    using RecombinationCollision = BinaryCollision<RecombinationFunc, NoParticleCreationFunc>;

    py::class_<RecombinationCollision, CollisionBase>(m, "RecombinationCollision")
        .def("get_collision_tracking",
             [](RecombinationCollision& collision, int lev) -> amrex::MultiFab* {
                 return collision.getCollisionFunctor().getCollisionTracking(lev);
             },
             py::arg("lev"),
             py::return_value_policy::reference_internal,
             R"doc(Get collision tracking MultiFab for a given refinement level.

             The MultiFab has cell-by-cell resolution with 2 components:
             - Component 0: recombination collision counts (per cell)
             - Component 1: energy transfer in Joules (per cell)

             Parameters
             ----------
             lev : int
                 Refinement level (0 for base level)

             Returns
             -------
             amrex.MultiFab or None
                 The collision tracking MultiFab, or None if not initialized
             )doc"
        )
        .def("reset_collision_tracking",
             [](RecombinationCollision& collision, int lev) {
                 collision.getCollisionFunctor().resetCollisionTracking(lev);
             },
             py::arg("lev"),
             R"doc(Reset collision tracking data (zero all cells) for a given level.

             Parameters
             ----------
             lev : int
                 Refinement level (0 for base level)
             )doc"
        )
        .def("get_process_names",
             [](RecombinationCollision& collision) {
                 auto amrex_vec = collision.getCollisionFunctor().getProcessNames();
                 return std::vector<std::string>(amrex_vec.begin(), amrex_vec.end());
             },
             R"doc(Get the names of collision processes for tracking.

             Returns
             -------
             list of str
                 List of process names (typically ['recombination'])
             )doc"
        )
        .def("gather_collision_tracking",
             [](RecombinationCollision& collision, int lev, int ngrow) {
                 amrex::Vector<amrex::Real> data;
                 amrex::Box box;
                 collision.getCollisionFunctor().gatherCollisionTracking(lev, data, box, ngrow);

                 // Create return tuple with data and box dimensions
                 std::vector<amrex::Real> data_vec(data.begin(), data.end());
                 std::vector<int> box_lo(AMREX_SPACEDIM);
                 std::vector<int> box_hi(AMREX_SPACEDIM);

                 for (int i = 0; i < AMREX_SPACEDIM; ++i) {
                     box_lo[i] = box.smallEnd(i);
                     box_hi[i] = box.bigEnd(i);
                 }

                 return py::make_tuple(data_vec, box_lo, box_hi);
             },
             py::arg("lev"), py::arg("ngrow") = 0,
             R"doc(Gather collision tracking data from all boxes into a single spatial array.

             Parameters
             ----------
             lev : int
                 Refinement level (0 for base level)
             ngrow : int, optional
                 Number of grow cells to include (default=0)

             Returns
             -------
             tuple of (data, box_lo, box_hi)
                 data : list of float
                     Flattened array with shape (ncells * 2,) for [count, energy]
                 box_lo : list of int
                     Lower corner indices of the box
                 box_hi : list of int
                     Upper corner indices (inclusive)
             )doc"
        );
}
