#!/bin/bash

#$ -N session_length
#$ -o output/$JOB_NAME_$JOB_ID.out
#$ -e output/$JOB_NAME_$JOB_ID.error
#$ -q larga
#$ -cwd

python session_length.py
