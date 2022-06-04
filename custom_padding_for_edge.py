# Pad in a way, that all edges have equal amount of points
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--main_dir', help='Main directory, has subdirectories: data, real_data, trained_model, preds', type = str)
parser.add_argument('--dfname', help='csv file name', type = str)

args = vars(parser.parse_args())

dfname = args["dfname"]
data_dir = args["main_dir"]

# load data
df = pd.read_csv(data_dir + dfname)

if "epoch"  in df.columns:
    df = df.drop("epoch", axis = 1)
# Find the maximum no of points for an edge ( padding to max)
# pad the start of each edge

max_no_of_edges = df.groupby("batch_id").nunique("road_id")["road_id"].max()
selected_edge_length = df.groupby(["batch_id", "road_id"]).count()["X"].max()

# filtering out ids that have more than a certain amount of points on any of its edges, to test not to get memory errors

df_copy = df.merge(df\
         .groupby(["batch_id", "road_id"])\
         .count()["X"]\
         .reset_index()\
         .rename(columns = {"X":"point_count"}),
                   on = ["batch_id", "road_id"],
                   how = "left")

# removing all batches that have lower than selected edge length
df = df.loc[df.batch_id.isin(df_copy.loc[df_copy.point_count > selected_edge_length,"batch_id"].unique()) == False,:] # we dont want batches, where the point count is larger

print("Dataframe shape:")
print(df.shape)
print("selected edge length:")
print(selected_edge_length)

if (df.columns == ["X", "Y", "road_id", "batch_id", "cell_id"]).all() == False:
    print("Check columns!")
    breakpoint()

padded_by_road_id_list = []

for batch_id in df.batch_id.unique():
    df_sub = df.loc[df.batch_id == batch_id,]
    for road_id in df_sub.road_id.unique():
        df_sub_road = df_sub.loc[df_sub.road_id == road_id,]
        if df_sub_road.shape[0] < selected_edge_length:
            padding = pd.DataFrame({"X":[0], "Y":[0], "road_id": [road_id], "batch_id": [batch_id],"cell_id": [0] }) # source pad index stays the same for cell transformer
            padding = pd.concat([padding]*(selected_edge_length - df_sub_road.shape[0]))
            #df_sub_road_padded = pd.concat([padding, df_sub_road], axis = 0, ignore_index=True) # pad from start
            df_sub_road_padded = pd.concat([df_sub_road, padding], axis = 0, ignore_index=True) # pad from end, post-padding as it should be in a transformer

            padded_by_road_id_list.append(df_sub_road_padded)
        else:
            padded_by_road_id_list.append(df_sub_road)


padded_by_road_id = pd.concat(padded_by_road_id_list, axis = 0, ignore_index=True)

padded_by_road_id_w_count = \
    padded_by_road_id \
        .merge(padded_by_road_id \
               .groupby("batch_id").count().reset_index().rename(columns = {"road_id": "road_id_count"})[["batch_id", "road_id_count"]],
               how = "left",
               on = "batch_id")

padded_by_road_id_w_count_filtered = padded_by_road_id_w_count

# Export only equal length batches

print("Input df shape:")
print(df.shape)

padded_by_road_id_w_count_filtered2 = padded_by_road_id_w_count_filtered.loc[padded_by_road_id_w_count_filtered.batch_id.isin(
    padded_by_road_id_w_count_filtered.batch_id.unique()
),].drop("road_id_count", axis = 1)

print("Output df shape")
print(padded_by_road_id_w_count_filtered2.shape)

padded_by_road_id_w_count_filtered2.to_csv(f"{data_dir}{dfname[:-4]}_postpadded_by_road_id.csv", index = False)
