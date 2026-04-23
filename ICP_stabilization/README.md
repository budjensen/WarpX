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

# copy the input file to the output directory
mkdir -p "$OUTFILE"
cp "$RUNFILE" "$OUTFILE"

# run program
mpirun -n 24 python "$RUNFILE" -d "$OUTFILE"
```

and then submit with

```bash
$ sbatch batch_adroit
```

make sure that you have more than enough time allocated for the job.