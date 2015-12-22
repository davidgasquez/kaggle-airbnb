#!/bin/bash

#$ -N age_modeling
#$ -o output/$JOB_NAME_$JOB_ID.out
#$ -e output/$JOB_NAME_$JOB_ID.error
#$ -q larga
#$ -cwd

python age_modeling.py
