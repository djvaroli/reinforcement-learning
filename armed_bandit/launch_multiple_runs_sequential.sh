#!/bin/bash

# define the experiment launch commands

declare -a launch_commands=(
    "python3 run_k_armed_bandit.py -k 10 -n 1000 -r 2000 --av_bias 0"
    "python3 run_k_armed_bandit.py -k 10 -n 1000 -r 2000 --av_bias 1"
    "python3 run_k_armed_bandit.py -k 10 -n 1000 -r 2000 --av_bias 2"
    "python3 run_k_armed_bandit.py -k 10 -n 1000 -r 2000 --av_bias 5"
    "python3 run_k_armed_bandit.py -k 10 -n 1000 -r 2000 --av_bias 10"
)

# colors
RED="\033[0;31m"
NC="\033[0m"
BLUE="\033[0;34m"

for command in "${launch_commands[@]}"
do
    echo -e "${BLUE}Running command:${NC} $command"
    $command
done


