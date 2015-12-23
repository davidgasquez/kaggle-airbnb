#!/bin/bash

#$ -N gb
#$ -o output/$JOB_NAME_$JOB_ID.out
#$ -e output/$JOB_NAME_$JOB_ID.error
#$ -q larga
#$ -cwd

python -u gb.py
