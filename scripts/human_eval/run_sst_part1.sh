#!/bin/bash
FOLDER=$1

declare -A to_process=([kbbert_quasi_ft_arb]=19820 [kbbert_quasi_ft_ltr]=6288 [kbbert_quasi_ft_arb_aot]=2382)

for fname in $FOLDER/kbbert_quasi_*;
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
      sh scripts/human_eval/run_docker_sst_generate.sh dmytroka_swectrl_p1 kbbert_based $checkpoint;
    fi
  done
done
