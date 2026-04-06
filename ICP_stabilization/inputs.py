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
    # Simulation parameters
    zmin = 0  # m
    zmax = 40 * milli  # m

    # ICP heating parameters
    freq = 10e6  # Hz, ICP frequency
    ICP_mag = 250.0  # A/m^2, ICP current density amplitude
    zmin_icp = 5 * milli  # m, ICP region minimum z
    zmax_icp = 15 * milli  # m, ICP region maximum z
    integrator = "euler"  # euler | ab2 | rk2 | rk4, ICP heating integrator

    # T and n profile parameters
    gas_temp = 300  # [K]
    gas_pressure = 1 * milli  # [Torr]
    gas_density = 133.32 * gas_pressure / gas_temp / constants.kb  # [m^-3]
    m_ion = 6.634e-26  # [kg]
    m_gas = m_ion  # [kg]

    # Particle setup
    plasma_density = 5e15  # [m^-3]
    elec_temp = 3.5 * eV_in_K  # [eV] to [K]
    ion_temp = 300  # [K]
    seed_nppc = 4  # Number of particles per cell

    # Grid and time step setup
    lambda_De = np.sqrt(
        constants.ep0 * constants.kb * elec_temp / (1.5e17 * constants.q_e**2)
    )
    omega_p = np.sqrt(1.5e17 * constants.q_e**2 / (constants.ep0 * constants.m_e))
    dz = lambda_De / 2  # Approximate cell size
    nz = int(zmax / dz)  # Number of cells
    dz = zmax / nz  # True cell size [m]
    dt = 1.0 / (5 * omega_p)  # [s]
    max_electron_velocity = np.sqrt(2 * 20 * constants.q_e / constants.m_e)  # [m/s]
    # If the time step is too large, the electron will go through the grid
    # Fix this by changing the time step to be the grid size divided by the maximum electron velocity
    if dz < max_electron_velocity * dt:
        dt = dz / max_electron_velocity

    # Run time
    convergence_time = 30 / freq  # Convergence time
    diag_time = 15 / freq  # Time of diagnostic evaluation

    collect_every_n_steps = 10  # Collect diagnostics every n steps

    # Total simulation time in seconds
    total_time = convergence_time + diag_time

    def __init__(self, verbose=False, diag_outfolder="./diags"):
        """Get input parameters for the specific case (n) desired."""
        # Control verbose (-v) flag output
        self.verbose = verbose

        # Output folder for diagnostics (-d flag)
        self.diag_outfolder = os.path.abspath(diag_outfolder)

        self.convergence_steps = int(self.convergence_time / self.dt)
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
            "/scratch/gpfs/DGRAVES/bj8080/warpx-data/MCC_cross_sections/Ar/"
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
            max_steps=self.max_steps,
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
        # Add diagnostics                                                     #
        #######################################################################

        # Initialize everything
        self.sim.initialize_inputs()
        self.sim.initialize_warpx()

        # Set up diagnostics
        self.convergence_steps = int(self.convergence_time / self.dt)

        # Add custom diagnostic arrays
        self.N_e = np.zeros((self.collection_steps, self.nz + 1))
        self.E_y = np.zeros((self.collection_steps, self.nz + 1))

    def save_Ne(self, step):
        """Save electron density diagnostic."""
        he_electrons_wrapper = particle_containers.ParticleContainerWrapper("electrons")
        he_electrons_wrapper.deposit_charge_density(level=0)

        rho_data = self.rho_wrapper[...]
        self.N_e[step] = rho_data / (-constants.q_e)

    def save_Ey(self, step):
        """Save electric field diagnostic."""
        self.E_y[step] = self.Ey_wrapper[...]

    def do_diagnostics(self):
        """Callback function to save diagnostics during the simulation."""
        step = self.sim.extension.warpx.getistep(lev=0)

        if step < self.convergence_steps:
            return
        elif (step - self.convergence_steps) % self.collect_every_n_steps != 0:
            return

        self.save_Ne(step)
        self.save_Ey(step)

    #######################################################################
    # Run Simulation                                                      #
    #######################################################################
    def run_sim(self):
        self.rho_wrapper = fields.RhoFPWrapper(0)
        self.Ey_wrapper = fields.EyFPWrapper(0)
        elapsed_steps = 0

        # Run until convergence
        self.sim.step(self.convergence_steps - elapsed_steps - 1)
        elapsed_steps = self.convergence_steps - 1

        callbacks.installafterstep(self.do_diagnostics)

        # Run the simulation until the end
        self.sim.step(self.max_steps - elapsed_steps)

        callbacks.uninstallcallback("afterstep", self.do_diagnostics)

        # Write diagnostics to file
        np.save(os.path.join(self.diag_outfolder, "N_e.npy"), self.N_e)
        np.save(os.path.join(self.diag_outfolder, "E_y.npy"), self.E_y)


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
