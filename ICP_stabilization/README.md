## Implementation in the code
The ICP heating algorithms are in the folder `WarpX/Source/FieldSolver/InductiveHeating`.
Main algorithms are in `WarpX/Source/FieldSolver/InductiveHeating/ICPHeatingModel.cpp`.
Our simulations

## Running a simulation
To run a simulation, write a script in this directory (`WarpX/ICP_stabilization`) like
```bash
#!/bin/bash
#SBATCH -J sample_ICP_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=3G
#SBATCH --output warpx.%j.out

# setup environment
module purge
module load /home/bj8080/Modules/modulefiles/warpx_icp_stabilization/1.0

# Name the input file and output directory
RUNFILE="inputs.py"
OUTFILE="diags_output"

# copy the input file and submission script to the output directory
mkdir -p "$OUTFILE"
cp "$RUNFILE" "$OUTFILE"
cp "$0" "$OUTFILE"

# Set parameters and run simulation
ARGS=( # Note: Bash arrays support inline comments, unlike backslash-continued commands we had before
    --dz 5.e-5                      # Cell size
    --dt 5e-11                      # Time step
    --ICP_mag 500.                  # ICP current source strength [A/m^2]
    --ICP_freq 10e6                 # ICP source frequency
    --Nppc 20                       # Number of particles per cell
    --initial_density 5e15          # Initial plasma density [m^-3], should be near the expected final density
    --convergence_periods 0.        # Number of ICP periods in the convergence window
    --diagnostic_periods 20.        # Number of ICP periods in the diagnostic window
    --collections_per_period 25     # Number of collection steps per ICP period
    --solver euler                  # euler, rk2, rk4, ab2
)
mpirun -n 24 python "$RUNFILE" -d "$OUTFILE" "${ARGS[@]}"
```

and then submit the job (for a script named batch_adroit) with the command

```bash
$ sbatch batch_adroit
```

make sure that you have more than enough time allocated for the job.

To view simulation outputs, run the `make_videos.ipynb` and `plot_data.ipynb` notebooks. Note:
This example script demonstrates an unstable simulation.