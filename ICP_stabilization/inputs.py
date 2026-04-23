#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
from mpi4py import MPI as mpi

from pywarpx import callbacks, fields, particle_containers, picmi

comm = mpi.COMM_WORLD
num_proc = comm.Get_size()

constants = picmi.constants

milli = 1e-3
micro = 1e-6
nano = 1e-9
pico = 1e-12

eV_in_K = 11605.41586


class CapacitiveDischargeExample(object):

    # ---------------------------------------------------------------
    # Stability related parameters
    # ---------------------------------------------------------------
    dz = 2.e-5                      # Cell size, will be checked against Debye length
    dt = 1e-11                      # Time step, will be calculated based on plasma frequency and grid size
    plasma_density = 5e15           # [m^-3]
    seed_nppc = 16                  # Number of particles per cell (only loosely related to stability)

    # ICP heating parameters
    freq = 10e6                     # Hz, ICP frequency
    ICP_mag = 500.0                 # A/m^2, ICP current density amplitude
    zmin_icp = 5 * milli            # m, ICP region minimum z
    zmax_icp = 15 * milli           # m, ICP region maximum z
    integrator = "euler"              # euler | ab2 | rk2 | rk4

    # ---------------------------------------------------------------
    # Diagnostic collection parameters
    # ---------------------------------------------------------------

    # Run time
    convergence_time = 0 / freq    # Convergence time
    diag_time = 20 / freq           # Time of diagnostic evaluation
    collect_every_n_steps = 400      # Collect diagnostics every n steps

    # Total simulation time in seconds
    total_time = convergence_time + diag_time

    # ---------------------------------------------------------------
    # Simulation parameters
    # ---------------------------------------------------------------
    zmin = 0                        # m
    zmax = 20 * milli               # m

    elec_temp = 2.5 * eV_in_K       # [eV]
    ion_temp = 300                  # [K]

    gas_temp = 300                  # [K]
    gas_pressure = 5 * milli        # [Torr]
    gas_density = 133.32 * gas_pressure / gas_temp / constants.kb
    m_ion = 6.634e-26               # [kg]
    m_gas = m_ion                   # [kg]

    # ---------------------------------------------------------------
    # Grid and time step check
    # ---------------------------------------------------------------
    target_density = 1e17           # [m^-3], target density for baseline PIC stability
    target_energy = 5 * eV_in_K     # [eV], target energy for baseline PIC stability

    lambda_De = np.sqrt(
        constants.ep0 * constants.kb * target_energy / (target_density * constants.q_e**2)
    )
    omega_p = np.sqrt(target_density * constants.q_e**2 / (constants.ep0 * constants.m_e))

    def __init__(self, verbose=False, diag_outfolder="./diags"):
        """Setup the simulation."""
        # Control verbose (-v) flag output
        self.verbose = verbose

        # Output folder for diagnostics (-d flag)
        self.diag_outfolder = os.path.abspath(diag_outfolder)

        if self.dz > self.lambda_De:
            raise ValueError(
                f"Cell size dz={self.dz:.2e} m is too large for the Debye length lambda_De={self.lambda_De:.2e} m. Please reduce dz."
            )
        else:
            self.nz = int(self.zmax / self.dz)
            self.dz = self.zmax / self.nz
        if self.dt > 1.0 / (5 * self.omega_p):
            raise ValueError(
                f"Time step dt={self.dt:.2e} s is too large for the target plasma frequency omega_p={self.omega_p:.2e} Hz. Please reduce dt, or modify the plasma frequency."
            )

        self.convergence_steps = int(self.convergence_time / self.dt)
        # When convergence_time is 0 we need at least 1 pre-diagnostic step so
        # the displacement-current buffer can be seeded from a real field state.
        if self.convergence_steps == 0:
            self.convergence_steps = 1
        self.max_steps = int(self.total_time / self.dt)
        self.collection_steps = (
            self.max_steps - self.convergence_steps
        ) // self.collect_every_n_steps

        self.setup_run()

    def setup_run(self):
        """Setup simulation components."""

        #######################################################################
        # Set geometry and boundary conditions                                #
        #######################################################################

        self.grid = picmi.Cartesian1DGrid(
            number_of_cells=[self.nz],
            warpx_blocking_factor=8,
            warpx_max_grid_size=128,
            lower_bound=[self.zmin],
            upper_bound=[self.zmax],
            lower_boundary_conditions=["dirichlet"],
            upper_boundary_conditions=["dirichlet"],
            lower_boundary_conditions_particles=["absorbing"],
            upper_boundary_conditions_particles=["absorbing"],
            warpx_potential_lo_z=0.0,
            warpx_potential_hi_z=0.0,
        )

        #######################################################################
        # Field setup                                                         #
        #######################################################################

        # This will use the tridiagonal solver
        self.solver = picmi.ElectrostaticSolver(grid=self.grid)

        #######################################################################
        # Particle types setup                                                #
        #######################################################################

        elec_distribution = picmi.UniformDistribution(
            density=self.plasma_density,
            rms_velocity=[np.sqrt(constants.kb * self.elec_temp / constants.m_e)] * 3,
        )
        ion_distribution = picmi.UniformDistribution(
            density=self.plasma_density,
            rms_velocity=[np.sqrt(constants.kb * self.ion_temp / self.m_ion)] * 3,
        )

        self.electrons = picmi.Species(
            particle_type="electron",
            name="electrons",
            initial_distribution=elec_distribution,
        )
        ion_name = "ar_ions"
        self.ions = picmi.Species(
            particle_type="Ar",
            name=ion_name,
            charge="q_e",
            mass=self.m_ion,
            initial_distribution=ion_distribution,
        )

        #######################################################################
        # Collision initialization                                            #
        #######################################################################

        cross_sec_direc = (
            "/scratch/network/bj8080/warpx-data/MCC_cross_sections/Ar/"
        )
        electron_colls = picmi.MCCCollisions(
            name="coll_elec",
            species=self.electrons,
            background_density=self.gas_density,
            background_temperature=self.gas_temp,
            background_mass=self.ions.mass,
            ndt=None,
            scattering_processes={
                "elastic": {
                    "cross_section": cross_sec_direc + "electron_scattering.dat"
                },
                "excitation": {
                    "cross_section": cross_sec_direc + "excitation_1.dat",
                    "energy": 11.5,
                },
                "ionization": {
                    "cross_section": cross_sec_direc + "ionization.dat",
                    "energy": 15.7596112,
                    "species": self.ions,
                },
            },
        )

        ion_scattering_processes = {
            "elastic": {"cross_section": cross_sec_direc + "ion_scattering.dat"},
            "back": {"cross_section": cross_sec_direc + "ion_back_scatter.dat"},
        }
        ion_colls = picmi.MCCCollisions(
            name="coll_ion",
            species=self.ions,
            background_density=self.gas_density,
            background_temperature=self.gas_temp,
            ndt=None,
            scattering_processes=ion_scattering_processes,
        )

        #######################################################################
        # Initialize simulation                                               #
        #######################################################################

        self.sim = picmi.Simulation(
            solver=self.solver,
            time_step_size=self.dt,
            max_steps=self.max_steps + 1,
            warpx_collisions=[electron_colls, ion_colls],
            verbose=self.verbose,
            warpx_break_signals="USR1",
            warpx_numprocs=[num_proc],
            warpx_field_gathering_algo="energy-conserving",
            warpx_used_inputs_file=os.path.join(
                self.diag_outfolder, "warpx_used_inputs"
            ),
        )
        self.solver.sim = self.sim

        self.sim.add_species(
            self.electrons,
            layout=picmi.GriddedLayout(
                n_macroparticle_per_cell=[self.seed_nppc], grid=self.grid
            ),
        )
        self.sim.add_species(
            self.ions,
            layout=picmi.GriddedLayout(
                n_macroparticle_per_cell=[self.seed_nppc], grid=self.grid
            ),
        )
        self.solver.sim_ext = self.sim.extension

        #######################################################################
        # ICP heating setup                                                   #
        #######################################################################
        self.icp_heating = picmi.ICPHeating(
            frequency=self.freq,  # Hz
            z_min=self.zmin_icp,  # m
            z_max=self.zmax_icp,  # m
            j0_amplitude=self.ICP_mag,  # A/m^2
            ey_max=1e12,  # V/m (field limiter)
            integrator=self.integrator,  # euler | ab2 | rk2 | rk4
        )
        self.sim.add_icp_heating(self.icp_heating)

        #######################################################################
        # Initialize                                                          #
        #######################################################################
        # Initialize everything
        self.sim.initialize_inputs()
        self.sim.initialize_warpx()

        # Save cell size and time step for diagnostic
        with open(os.path.join(self.diag_outfolder, "run_parameters.txt"), "w") as f:
            f.write(f"dz = {self.dz:.2e} m\n")
            f.write(f"dt = {self.dt:.2e} s\n")
            f.write(f"icp_zmin = {self.zmin_icp:.2e} m\n")
            f.write(f"icp_zmax = {self.zmax_icp:.2e} m\n")

        # Save the nodes for diagnostics
        np.save(os.path.join(self.diag_outfolder, "z_nodes.npy"), np.linspace(self.zmin, self.zmax, self.nz + 1))

        # Precompute and save the diagnostic collection steps
        # time_bw_diagnostic = self.collect_every_n_steps * self.dt
        self.collection_times = np.linspace(
            self.convergence_time, self.total_time,
            self.collection_steps,
            endpoint=False
        )
        np.save(os.path.join(self.diag_outfolder, "collection_times.npy"), self.collection_times)

        #######################################################################
        # Add diagnostics                                                     #
        #######################################################################
        # To add another diagnostic, simply add another array
        #     self.<diagnostic_name> = np.zeros((self.collection_steps, self.nz + 1))
        # and then implement a save_<diagnostic_name> function that fills in the array
        #     def save_<diagnostic_name>(self, step):
        # then add the call to save the diagnostic in the do_diagnostics function.

        # Add custom diagnostic arrays
        self.N_e = np.zeros((self.collection_steps, self.nz + 1))
        self.E_y = np.zeros((self.collection_steps, self.nz + 1))
        self.J_disp_y = np.zeros((self.collection_steps, self.nz + 1))
        self.J_cond_y = np.zeros((self.collection_steps, self.nz + 1))

    def save_Ne(self, step_idx):
        """Save electron density diagnostic."""
        he_electrons_wrapper = particle_containers.ParticleContainerWrapper("electrons")
        he_electrons_wrapper.deposit_charge_density(level=0)

        rho_data = self.rho_wrapper[...]
        self.N_e[step_idx] = rho_data / (-constants.q_e)

        # Check if all of N_e is zero, and prepare to exit early if so
        if np.all(np.abs(self.N_e[step_idx]) < 1e-20):
            self.early_exit = True

    def save_Ey(self, step_idx):
        """Save electric field diagnostic."""
        self.E_y[step_idx] = self.Ey_wrapper[...]

    def save_Jcond_y(self, step_index):
        """Save total particle conduction current density in the y direction in the ICP region."""
        species_charges = {
            "electrons": -constants.q_e,
            "ar_ions": constants.q_e,
        }
        J_total = np.zeros(self.nz + 1)
        for species_name, charge in species_charges.items():
            species_wrapper = particle_containers.ParticleContainerWrapper(species_name)
            try:
                uy = np.concatenate(species_wrapper.get_particle_uy())
                w = np.concatenate(species_wrapper.get_particle_weight())
                z = np.concatenate(species_wrapper.get_particle_z())
            except ValueError:
                uy = np.array([])
                w = np.array([])
                z = np.array([])

            cell_idx = np.floor(z / self.dz).astype(int)
            frac_pos = (z / self.dz) - cell_idx
            frac_l = 1 - frac_pos
            frac_r = frac_pos

            temp_J = np.zeros(self.nz + 1)
            np.add.at(temp_J, cell_idx, uy * w * frac_l)
            valid_idxs = cell_idx != self.nz
            np.add.at(
                temp_J, cell_idx[valid_idxs] + 1,
                uy[valid_idxs] * w[valid_idxs] * frac_r[valid_idxs],
            )

            J_species = np.zeros_like(temp_J)
            comm.Allreduce(temp_J, J_species, op=mpi.SUM)
            J_total += charge / self.dz * J_species

        self.J_cond_y[step_index] = J_total

        if self.early_exit:
            num_particles = {}
            for species_name in species_charges.keys():
                species_wrapper = particle_containers.ParticleContainerWrapper(species_name)
                try:
                    z = np.concatenate(species_wrapper.get_particle_z())
                except ValueError:
                    z = np.array([])
                num_particles[species_name] = len(z)

            if comm.rank == 0:
                print(f"Early exit triggered at step {step_index}")
            for species_name, count in num_particles.items():
                print(f"Rank {comm.rank} {species_name} particle count at early exit: {count}")

    def do_diagnostics(self):
        """Callback function to save diagnostics during the simulation."""
        step = self.sim.extension.warpx.getistep(lev=0) - 1
        current_Ey = np.copy(self.Ey_wrapper[...])

        # Compute displacement current for the previous step using a centered
        # finite difference: J_disp(n) = eps0 * (E_y(n+1) - E_y(n-1)) / (2*dt).
        # When the callback fires at step s, E_y(s) is available as current_Ey and
        # E_y(s-2) is in self.Ey_two_ago so we fill the diagnostic for step s-1 = n.
        prev_step = step - 1
        if (prev_step >= self.convergence_steps and
                (prev_step - self.convergence_steps) % self.collect_every_n_steps == 0):
            diag_idx = (prev_step - self.convergence_steps) // self.collect_every_n_steps
            if diag_idx < self.collection_steps:
                self.J_disp_y[diag_idx] = (
                    constants.ep0 * (current_Ey - self.Ey_two_ago) / (2.0 * self.dt)
                )

        # Shift the rolling E_y buffer for the next step
        self.Ey_two_ago = self.Ey_one_ago
        self.Ey_one_ago = current_Ey

        if step < self.convergence_steps:
            return
        elif (step - self.convergence_steps) % self.collect_every_n_steps != 0:
            return
        step_index = (step - self.convergence_steps) // self.collect_every_n_steps
        if step_index >= self.collection_steps:
            return

        self.sim.extension.warpx.synchronize_velocity_with_position()

        self.save_Ne(step_index)
        self.save_Ey(step_index)
        self.save_Jcond_y(step_index)

        if self.early_exit:
            # Save all diagnostics up to the current step
            if comm.rank == 0:
                self.write_diagnostics(early_exit=True, step_index=step_index)
            sys.exit(0)

    def write_diagnostics(self, early_exit=False, step_index=None):
        """Save diagnostics to file."""
        if comm.rank == 0:
            save_slice = slice(step_index + 1) if early_exit else slice(None)
            np.save(os.path.join(self.diag_outfolder, "N_e.npy"), self.N_e[save_slice])
            np.save(os.path.join(self.diag_outfolder, "E_y.npy"), self.E_y[save_slice])
            np.save(os.path.join(self.diag_outfolder, "J_disp_y.npy"), self.J_disp_y[save_slice])
            np.save(os.path.join(self.diag_outfolder, "J_cond_y.npy"), self.J_cond_y[save_slice])

            # Write total current density (displacement + conduction) in the ICP region to file for easier analysis
            top_icp_idx = len(self.J_disp_y[0]) - 1 - np.argmax(np.abs(self.J_disp_y[0][::-1]) > 0.)
            bottom_icp_idx = np.argmax(np.abs(self.J_disp_y[0]) > 0.)
            J_total = self.J_disp_y[save_slice] + self.J_cond_y[save_slice]

            # zero current density outside of ICP region
            J_total[:, :bottom_icp_idx] = 0.0
            J_total[:, top_icp_idx + 1 :] = 0.0
            np.save(os.path.join(self.diag_outfolder, "J_total_y.npy"), J_total)

            if early_exit:
                np.save(os.path.join(self.diag_outfolder, "collection_times.npy"), self.collection_times[save_slice])
                print(f"Early exit at diagnostic index {step_index} due to zero conduction current density. All diagnostics up to this point have been saved.")

    #######################################################################
    # Run Simulation                                                      #
    #######################################################################
    def run_sim(self):
        # Set up a flag for early exit if all particles are removed
        self.early_exit = False

        self.rho_wrapper = fields.RhoFPWrapper(0)
        self.Ey_wrapper = fields.EyFPWrapper(0)

        # Seed the E_y rolling buffer with the field state one step before the
        # first diagnostic.
        if self.convergence_steps > 1:
            self.sim.step(self.convergence_steps - 1)
        self.Ey_one_ago = np.copy(self.Ey_wrapper[...])
        self.Ey_two_ago = None  # populated on the first callback iteration
        self.sim.step(1)

        callbacks.installafterstep(self.do_diagnostics)

        # Run the diagnostic phase plus one extra step for the centered-difference
        # displacement current diagnostic.
        self.sim.step(self.max_steps - self.convergence_steps + 1)

        callbacks.uninstallcallback("afterstep", self.do_diagnostics)

        # Write diagnostics to file
        self.write_diagnostics()

##########################
### Execute Simulation ###
##########################
parser = argparse.ArgumentParser()
parser.add_argument(
    "-v", "--verbose", help="Verbose run, default = False", action="store_true"
)
parser.add_argument(
    "-d",
    "--diag_outfolder",
    type=str,
    default="diags",
    help="Output folder for diagnostics, default = diags",
)
args, left = parser.parse_known_args()
sys.argv = sys.argv[:1] + left  # keep other libs able to parse remaining args

# normalize and ensure directory exists
diag_out = os.path.abspath(args.diag_outfolder)
os.makedirs(diag_out, exist_ok=True)

run = CapacitiveDischargeExample(verbose=args.verbose, diag_outfolder=diag_out)
run.run_sim()
