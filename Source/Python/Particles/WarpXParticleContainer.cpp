/* Copyright 2021-2022 The WarpX Community
 *
 * Authors: Axel Huebl, Remi Lehe
 * License: BSD-3-Clause-LBNL
 */

#include "Python/pyWarpX.H"

#include <Particles/WarpXParticleContainer.H>

#include <AMReX_MultiFab.H>

#include <pybind11/stl.h>


void init_WarpXParIter (py::module& m)
{
    py::class_<
        WarpXParIter, amrex::ParIterSoA<PIdx::nattribs, 0>
    >(m, "WarpXParIter")
        .def(py::init<amrex::ParIterSoA<PIdx::nattribs, 0>::ContainerType&, int>(),
            py::arg("particle_container"), py::arg("level"))
        .def(py::init<amrex::ParIterSoA<PIdx::nattribs, 0>::ContainerType&, int, amrex::MFItInfo&>(),
            py::arg("particle_container"), py::arg("level"),
            py::arg("info"))
    ;
}

void init_WarpXParticleContainer (py::module& m)
{
    py::class_<
        WarpXParticleContainer,
        amrex::ParticleContainerPureSoA<PIdx::nattribs, 0>
    > wpc (m, "WarpXParticleContainer");
    wpc
        .def("add_real_comp",
            [](WarpXParticleContainer& pc, const std::string& name, bool comm) { pc.AddRealComp(name, comm); },
            py::arg("name"), py::arg("comm")
        )
        .def("add_n_particles",
            [](WarpXParticleContainer& pc, int lev,
                int n, py::array_t<double> &x,
                py::array_t<double> &y,
                py::array_t<double> &z,
                py::array_t<double> &ux,
                py::array_t<double> &uy,
                py::array_t<double> &uz,
                const int nattr_real, py::array_t<double> &attr_real,
                const int nattr_int, py::array_t<int> &attr_int,
                int uniqueparticles, int id
            ) {
                amrex::Vector<amrex::ParticleReal> xp(x.data(), x.data() + n);
                amrex::Vector<amrex::ParticleReal> yp(y.data(), y.data() + n);
                amrex::Vector<amrex::ParticleReal> zp(z.data(), z.data() + n);
                amrex::Vector<amrex::ParticleReal> uxp(ux.data(), ux.data() + n);
                amrex::Vector<amrex::ParticleReal> uyp(uy.data(), uy.data() + n);
                amrex::Vector<amrex::ParticleReal> uzp(uz.data(), uz.data() + n);

                // create 2d arrays of real and in attributes
                amrex::Vector<amrex::Vector<amrex::ParticleReal>> attr;
                const double *attr_data = attr_real.data();
                for (int ii=0; ii<nattr_real; ii++) {
                    amrex::Vector<amrex::ParticleReal> attr_ii(n);
                    for (int jj=0; jj<n; jj++) {
                        attr_ii[jj] = attr_data[ii + jj*nattr_real];
                    }
                    attr.push_back(attr_ii);
                }

                amrex::Vector<amrex::Vector<int>> iattr;
                const int *iattr_data = attr_int.data();
                for (int ii=0; ii<nattr_int; ii++) {
                    amrex::Vector<int> attr_ii(n);
                    for (int jj=0; jj<n; jj++) {
                        attr_ii[jj] = iattr_data[ii + jj*nattr_int];
                    }
                    iattr.push_back(attr_ii);
                }

                pc.AddNParticles(
                    lev, n, xp, yp, zp, uxp, uyp, uzp, nattr_real, attr,
                    nattr_int, iattr, uniqueparticles, id
                );
            },
            py::arg("lev"), py::arg("n"),
            py::arg("x"), py::arg("y"), py::arg("z"),
            py::arg("ux"), py::arg("uy"), py::arg("uz"),
            py::arg("nattr_real"), py::arg("attr_real"),
            py::arg("nattr_int"), py::arg("attr_int"),
            py::arg("uniqueparticles"), py::arg("id")=-1
        )
        .def("get_comp_index",  // deprecated: use pyAMReX get_real_comp_index
            [](WarpXParticleContainer& pc, std::string comp_name)
            {
                py::print("get_comp_index is deprecated. Use get_real_comp_index instead.");
                return pc.GetRealCompIndex(comp_name);
            },
            py::arg("comp_name")
        )
        .def("get_icomp_index",  // deprecated: use pyAMReX get_int_comp_index
            [](WarpXParticleContainer& pc, std::string comp_name)
            {
                py::print("get_icomp_index is deprecated. Use get_int_comp_index instead.");
                return pc.GetIntCompIndex(comp_name);
            },
            py::arg("comp_name")
        )
        .def("num_local_tiles_at_level",
            &WarpXParticleContainer::numLocalTilesAtLevel,
            py::arg("level")
        )
        .def("total_number_of_particles",
            &WarpXParticleContainer::TotalNumberOfParticles,
            py::arg("valid_particles_only"), py::arg("local")
        )
        .def("sum_particle_weight",
            &WarpXParticleContainer::sumParticleWeight,
            py::arg("local")
        )
        .def("sum_particle_charge",
            &WarpXParticleContainer::sumParticleCharge,
            py::arg("local")
        )
        .def("sum_particle_energy",
            &WarpXParticleContainer::sumParticleEnergy,
            py::arg("local")
        )
        .def("deposit_charge",
            [](WarpXParticleContainer& pc,
            amrex::MultiFab* rho, const int lev)
            {
                for (WarpXParIter pti(pc, lev); pti.isValid(); ++pti)
                {
                    const long np = pti.numParticles();
                    auto& wp = pti.GetAttribs(PIdx::w);
                    pc.DepositCharge(pti, wp, nullptr, rho, 0, 0, np, 0, lev, lev);
                }
            },
            py::arg("rho"), py::arg("lev")
        )
        .def("get_charge_density",
            [](WarpXParticleContainer& pc, int lev, bool local)
            {
                return pc.GetChargeDensity(lev, local);
            },
            py::arg("lev"), py::arg("local")
        )
        .def("set_do_not_push",
            [](WarpXParticleContainer& pc, bool flag) { pc.setDoNotPush(flag); },
            py::arg("flag")
        )
        .def("set_do_not_gather",
            [](WarpXParticleContainer& pc, int flag) { pc.setDoNotGather(flag); },
            py::arg("flag")
        )
        .def("set_do_not_deposit",
            [](WarpXParticleContainer& pc, int flag) { pc.setDoNotDeposit(flag); },
            py::arg("flag")
        )
        .def("get_power_deposition_tracking",
            &WarpXParticleContainer::getPowerDepositionTracking,
            py::arg("lev"),
            py::return_value_policy::reference_internal,
            R"doc(Get power deposition tracking MultiFab for a given refinement level.

            The MultiFab has cell-by-cell resolution with 3 components, the
            J.E power deposited into this species per cell, per direction:
            [Px, Py, Pz]. Values are sampled at the exact velocity and field
            values used in the particle's momentum push.

            Parameters
            ----------
            lev : int
                Refinement level (0 for base level)

            Returns
            -------
            amrex.MultiFab or None
                The power deposition tracking MultiFab, or None if tracking
                is not enabled for this species (see the species'
                ``enable_power_deposition_tracking`` flag)
            )doc"
        )
        .def("reset_power_deposition_tracking",
            &WarpXParticleContainer::resetPowerDepositionTracking,
            py::arg("lev"),
            R"doc(Reset power deposition tracking data (zero all cells) for a given level.

            Parameters
            ----------
            lev : int
                Refinement level (0 for base level)
            )doc"
        )
        .def("gather_power_deposition_tracking",
            [](WarpXParticleContainer& pc, int lev, int ngrow) {
                amrex::Vector<amrex::Real> data;
                amrex::Box box;
                pc.gatherPowerDepositionTracking(lev, data, box, ngrow);

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
            R"doc(Gather power deposition tracking data from all boxes into a single spatial array.

            Collects data from all boxes and arranges them spatially into a single
            contiguous array. Only the IO processor (rank 0) gets the full data;
            other ranks get empty arrays.

            The returned data is flattened in row-major (C) order with interleaved
            components: for each cell (i,j,k), data contains [Px, Py, Pz].

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
                    Flattened array with shape (ncells * 3,)
                box_lo : list of int
                    Lower corner indices of the box [ilo, jlo, klo]
                box_hi : list of int
                    Upper corner indices (inclusive) [ihi, jhi, khi]
            )doc"
        )
    ;
}
