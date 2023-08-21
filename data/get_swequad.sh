#!/bin/sh
URL_BASE="https://raw.githubusercontent.com/dkalpakchi/SweQUAD-MC/main/data"

mkdir swequad
cd swequad
wget $URL_BASE/urls.json
wget $URL_BASE/training.json
wget $URL_BASE/dev.json
wget $URL_BASE/test.json
wget $URL_BASE/data_collection_instructions.txt
wget $URL_BASE/LICENSE
