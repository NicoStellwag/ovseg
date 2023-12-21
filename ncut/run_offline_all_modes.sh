#!/bin/bash

python ./ncut/run_offline.py ncut.data.dataset.mode=train
python ./ncut/run_offline.py ncut.data.dataset.mode=val
python ./ncut/run_offline.py ncut.data.dataset.mode=test