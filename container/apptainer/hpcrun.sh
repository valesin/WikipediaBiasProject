#!/bin/bash

#SBATCH --job-name=wiki_graph        # Job name
#SBATCH --output=job.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=1        # Schedule one core
#SBATCH --time=01:00:00          # Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=scavenge    # Run on scavenge queue (all nodes, low priority)

export PROJECT=project_folder export
export APPTAINER_CACHEDIR=$HOME/APPTAINER_CACHEDIR
export APPTAINER_TMPDIR=$HOME/APPTAINER_TMPDIR
export TMPDIR=$HOME/tmp

apptainer run python_project.sif --input data/notable_ids_attributes.csv --limit 2 --output data/outduck.db
