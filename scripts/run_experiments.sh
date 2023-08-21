#!/bin/sh
for fname in scripts/run_docker_*_finetune*.sh;
do
  echo $fname
  sh $fname
done
