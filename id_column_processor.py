# Script to transform road id s/cell ids so that the number of unique road id s/cell ids is equal to the maximum road id/cell id in the training data
# And to provide a mapping with the original id-s

import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--main_dir', help='Main directory, has subdirectories: data, real_data, trained_model, preds', type = str)
parser.add_argument('--dfname', help='csv file name', type = str)
parser.add_argument('--id_column', help='column to be scaled', type = str, default = "road_id")

args = vars(parser.parse_args())

main_dir = args["main_dir"]
dfname = args["dfname"]
id_column = args["id_column"]

print("Main directory: " + main_dir)
print(dfname)

data_dir = main_dir + "real_data/"
fname = data_dir + dfname

input_df = pd.read_csv(fname)
if 'Unnamed: 0' in input_df.columns:
    input_df = input_df.drop('Unnamed: 0',axis = 1)

max_road_id = input_df[id_column].max()
unique_road_ids = input_df[id_column].unique() # unique road id-s
unique_road_ids = unique_road_ids[unique_road_ids != 0]

mapping_df = pd.DataFrame({id_column:unique_road_ids,
                           id_column + '_scaled': np.arange(2, len(unique_road_ids) + 2)})

mapping_df = pd.concat([mapping_df, pd.DataFrame({id_column: [0], id_column + "_scaled":[0]})], axis = 0, ignore_index=True)

input_df_w_scaled_road_id = input_df.merge(mapping_df, on = id_column, how = "left").drop(id_column, axis = 1)\
    .rename(columns = {id_column + '_scaled':id_column})

input_df_w_scaled_road_id.to_csv(fname[:-4] + "_w_mapped_" + id_column + ".csv", index = None)
mapping_df.to_csv(fname[:-4] + "_" + id_column + "_mapping.csv", index=None)

print("Took in: " + str(input_df.shape[0]) + " rows")
print("Wrote out: " +  str(input_df_w_scaled_road_id.shape[0]) + " rows")


