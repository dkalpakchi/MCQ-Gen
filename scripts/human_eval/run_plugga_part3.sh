#!/bin/bash
FOLDER=$1

declare -A to_process=([swectrl_swequad_ft]=605)

for fname in $FOLDER/swectrl_swequad_ft;
do
  echo "Working on $fname"
  mf=`basename $fname`      # remove the trailing "/"
  mf=${mf%/}
  idp=${to_process[$mf]}
  
  for checkpoint in $fname/checkpoint-*/;
  do
    cid=`echo "$checkpoint" | cut -d "-" -f 2`
    cid=${cid%/}
    if [ $cid -ne $idp ]; then
      continue
    fi

    if [[ -d "$checkpoint" && ! -L "$checkpoint" ]]; then
      echo $checkpoint
      sh scripts/human_eval/run_docker_plugga_generate.sh dmytroka_swectrl_p3 swectrl_based $checkpoint;
    fi
  done
done
