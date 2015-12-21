#!/bin/bash
# The name of the job, can be whatever makes sense to you
#$ -N session_length
# The job should be placed into the queue 'media'.
#$ -q larga
# Redirect output stream to this file.
#$ -o session_length_output.dat
# Redirect error stream to this file.
#$ -e session_length_error.dat
# The batchsystem should use the current directory as working directory.
# Both files (output.dat and error.dat) will be placed in the current
# directory. The batchsystem assumes to find the executable in this directory.
#$ -cwd
# If a different working directory is preferred, then #$ -wd <dir>
# must be used instead
python session_length.py > session_length.out
