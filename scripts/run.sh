#!/bin/bash

if [ $# -eq 0 ]; then
    echo "No arguments provided"
    exit 1
fi

#$ -N $1
#$ -o output/$JOB_NAME_$JOB_ID.out
#$ -e output/$JOB_NAME_$JOB_ID.error
#$ -q larga
#$ -cwd

python $1
