#!/bin/sh
DOCKER_CONTAINER_NAME=$1
MODEL=$2
CHECKPOINT_DIR=$3
END_ARGS=$4

MODEL_DIR="$(dirname "$CHECKPOINT_DIR")"

docker run --rm -t --shm-size=10g -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -v=$(pwd):/workspace --name $DOCKER_CONTAINER_NAME --gpus all --cpus=10 -m=60g qg_exp bash -c "cd /workspace; python -m models.$MODEL.generate -f $CHECKPOINT_DIR -fa $MODEL_DIR/ft_args.bin -l swequad-mc -d data/swequad-mc/dev.json -m 100 -o sqd $END_ARGS"

