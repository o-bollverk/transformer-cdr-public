#!/bin/bash

python -c 'print ("Starting run for mapmatching")'
srun python run_single_transformer.py \
--main_dir=YOUR_PROJECT_DIRECTORY \
--dfname=YOUR_FILENAME \
--seq=1 \
--device=cuda \
--epochs=7 \
--dropout=0.1 \
--batchsize=4 \
--with_dataloader=1 \
--no_masking=False \
--run_on_multiple_gpus=1 \
--cross_validation=1 \
--cross_validation_folds_perc=0.05 \
--learning_rate=0.0005 \
--reduction=sum \
--max_folds=5 \
--timestamp_file="recorded_timestamp.json"

python -c 'print ("Starting run for cellmatching")'
srun python run_single_transformer.py \
--main_dir=YOUR_PROJECT_DIRECTORY \
--dfname=YOUR_FILENAME \
--seq=1 \
--device=cuda \
--epochs=7 \
--dropout=0.1 \
--batchsize=4 \
--with_dataloader=1 \
--no_masking=False \
--run_on_multiple_gpus=1 \
--cross_validation=1 \
--learning_rate=0.0005 \
--reduction=sum \
--load_cv_indices=1 \
--max_folds=5 \
--cell_transformer=1 \
--src_col=cell_id \
--timestamp_file="recorded_timestamp.json"

