## Implementation in the code
The ICP heating algorithms are in the folder `WarpX/Source/FieldSolver/InductiveHeating`.
Main algorithms are in `WarpX/Source/FieldSolver/InductiveHeating/ICPHeatingModel.cpp`.

## Running a simulation
To run a simulation, write a script like
```bash
#!/bin/bash
#SBATCH -J sample_ICP_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
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

# run program with settings set on the command line
mpirun -n 24 python "$RUNFILE" -d "$OUTFILE" \
    --dz 2.e-5 \                        # Cell size
    --dt 1e-11 \                        # Time step
    --ICP_mag 500. \                    # ICP current source strength [A/m^2]
    --ICP_freq 10e6 \                   # ICP source frequency
    --Nppc 30 \                         # Number of particles per cell, should be high enough to reduce noise but low enough to be computationally feasible
    --initial_density 1e16 \            # Initial plasma density [m^-3], should be near the expected final density
    --convergence_periods 100. \        # Number of RF periods to run before saving diagnostics
    --diagnostic_periods 20. \          # Number of RF periods to collect diagnostics over
    --steps_bw_diagnostics 400 \        # Number of steps between each diagnostics collection
    --solver euler                      # euler, rk2, rk4, ab2
```

and then submit the job (for a script named job_submission) with the command

```bash
$ sbatch job_submission
```

make sure that you have more than enough time allocated for the job.