#!/bin/sh
python3.8 -m models.baseline.ud_based -l sv -tf models/baseline/templates/sv/swequad/1667045512413156 -tr data/swequad-mc/training.json -n 5 -d data/sst.yaml -o sst
