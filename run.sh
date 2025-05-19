#!/bin/bash

timestamp=$(date +"%Y%m%d_%H%M%S")

# 
mkdir -p nohup

nohup python pissm_train.py > nohup/output_$timestamp.log 2>&1 &