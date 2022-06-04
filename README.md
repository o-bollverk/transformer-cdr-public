# transformer-cdr-public

This is the initial public repository for Trajectory Reconstruction Based on Mobile Phone Network Data Using Transformers.

<img width="1440" alt="initial_visual_abstract" src="https://user-images.githubusercontent.com/65232333/172000568-76922895-78a8-4ac5-8429-2d9055a5bb42.png">

Computations were performed on University of Tartu High Computing Center: https://ut.ee/en/high-performance-computing-center

### Overview of the data processing steps

1) Trajectories are extracted from the cellular network data and enriched with CDR information
2) Each gps point is attached a road id in a road network, constructed using OSM linestrings for Tartu area
3) Trajectories containing more than 200 points are split into 3, for memory concuption purposes and for avoiding data mismatch problems
4) Each trajectory is padded on the road-id dimension. As a result, every road contains an equal no of GPS points (padding is done to the road, which contains the maximum no of points in the whole dataset).
5) Road-ids are scaled to 2:max(len(unique(road_id)))
6) Cell-ids are scaled to  2:max(len(unique(cell_id)))

Corresponding scrips are as follows:
1) extract_trajectories.py
2) create_edge_id.R (not public)
3) cut_to_max_gps_count.R
4) custom_padding_for_edge.py
5) id_column_proccesor, with --id_column=road_id
6) id_column_processor, with --id_column=cell_id

### Example of a model run:

Example run is provided in example_run.sh

```bash
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
