#!/bin/bash
# The name of the job, can be whatever makes sense to you
#$ -N preprocessing
# The job should be placed into the queue 'media'.
#$ -q larga
# Redirect output stream to this file.
#$ -o preprocessing_output.dat
# Redirect error stream to this file.
#$ -e preprocessing_error.dat
# The batchsystem should use the current directory as working directory.
# Both files (output.dat and error.dat) will be placed in the current
# directory. The batchsystem assumes to find the executable in this directory.
#$ -cwd
# If a different working directory is preferred, then #$ -wd <dir>
# must be used instead
python preprocessing.py > preprocessing.out
