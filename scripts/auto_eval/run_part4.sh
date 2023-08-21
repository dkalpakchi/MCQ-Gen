#!/bin/bash
FOLDER=$1

for fname in $FOLDER/kbbert_swequad_*_aot;
do
  echo "Working on $fname"
  for checkpoint in $fname/checkpoint-*/;
  do
    if [[ -d "$checkpoint" && ! -L "$checkpoint" ]]; then
      sh scripts/auto_eval/run_docker_swequad_dev_generate.sh dmytroka_swectrl_p4 kbbert_based $checkpoint;
    fi
  done
done
