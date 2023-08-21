#!/bin/bash
FOLDER=$1

for fname in $FOLDER/swectrl_swequad_ft;
do
  echo "Working on $fname"
  for checkpoint in $fname/checkpoint-*/;
  do
    if [[ -d "$checkpoint" && ! -L "$checkpoint" ]]; then
      sh scripts/auto_eval/run_docker_swequad_dev_generate.sh dmytroka_swectrl_sp5 swectrl_based $checkpoint;
    fi
  done
done
