#!/bin/bash

# run_mvp.sh

#echo "Activating Conda environment..."
#conda activate memorization_mvp

if [ $? -ne 0 ]; then
    echo "Failed to activate Conda environment. Make sure 'memorization_mvp' exists."
    echo "You might need to run: conda create -n memorization_mvp python=3.9 && conda activate memorization_mvp && conda install pytorch cpuonly numpy tqdm -c pytorch"
    exit 1
fi

echo "Running MVP training script..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
python src/train.py

echo "MVP script finished."
