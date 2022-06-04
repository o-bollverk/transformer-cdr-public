import pandas as pd
import numpy as np
from shapely import wkt
from shapely.geometry import Point
import geopandas as gpd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='Directory that holds all data files', type = str)
parser.add_argument('--trace1', help='csv file name for trace of user 1', type = str)
parser.add_argument('--trace2', help='csv file name for trace of user 2', type = str)
parser.add_argument('--celldata1', help='csv file name for celldata for trace 1', type = str)
parser.add_argument('--celldata2', help='csv file name for celldata for trace 2', type = str)
parser.add_argument('--wkt_data', help='csv file name for wkt data', type = str)
parser.add_argument('--output_fname', help='csv output file name', type = str)

args = vars(parser.parse_args())

realdata_dir = "~/Desktop/masters_thesis/real_data/"
trace_1 = pd.read_csv(f'{realdata_dir}{args["trace1"]}', sep = "|")
trace_2 = pd.read_csv(f'{realdata_dir}{args["trace2"]}', sep = "|")
celldata_1 = pd.read_csv(f'{realdata_dir}{args["celldata1"]}', sep = ";")
celldata_2 = pd.read_csv(f'{realdata_dir}{args["celldata2"]}', sep = ";")
wkt_data = pd.read_csv(f'{realdata_dir}{args["wkt_data"]}', sep = "\t")

# For each trace, get the polygon it belongs to
# In case of duplication, keep smaller polygon
# In case of no match, keep activation to previous cell?

# Get ci in wkt data
wkt_data["ci"] = wkt_data["CGI"].apply(lambda x: x.split("-")[3])
wkt_data["geometry"] = wkt_data["WKT"].apply(lambda x: wkt.loads(x))

# Create gpd
wkt_data_gpd = gpd.GeoDataFrame(wkt_data,  crs="EPSG:4326" )
wkt_data_gpd["ci"] = wkt_data_gpd["ci"].astype(object)

trace_1["point"] = trace_1.apply(lambda x: Point(x.LONGITUDE, x.LATITUDE), axis = 1)
trace_1["geometry"] = trace_1["point"]
trace_1 = gpd.GeoDataFrame(trace_1, crs="EPSG:4326")

trace_2["point"] = trace_2.apply(lambda x: Point(x.LONGITUDE, x.LATITUDE), axis = 1)
trace_2["geometry"] = trace_2["point"]
trace_2 = gpd.GeoDataFrame(trace_2, crs="EPSG:4326")

trace_1["TIMESTAMP"] = trace_1["TIMESTAMP"].str[:21]
trace_2["TIMESTAMP"] = trace_2["TIMESTAMP"].str[:21]



def match_smallest_cell(point, cells_gpd):
    containing_cells = cells_gpd.loc[cells_gpd.contains(point),:]
    if containing_cells.shape[0] == 0:
        return 0

    containing_cells  = containing_cells.to_crs("EPSG:3301") # for area calculation
    containing_cells["area"] = containing_cells["geometry"].area
    smallest_cell_ci = containing_cells.loc[containing_cells.area == containing_cells.area.min(), "ci"].values[0]
    return smallest_cell_ci

# For simplicity, use geographical match at last and first connection, when cell value is -1 (always)

def create_batched_df(trace_df, celldata_df,
                      bbox = [58.3459925 , 58.3786785, 26.6825294, 26.7408943]):

    min_y = bbox[0]#58.3459925
    max_y = bbox[1]#58.3786785
    min_x = bbox[2]#26.6825294
    max_x = bbox[3]#26.7408943

    trace_df = trace_df.loc[((trace_df.LATITUDE < max_y) &
                             (trace_df.LATITUDE > min_y) &
                             (trace_df.LONGITUDE < max_x)&
                             (trace_df.LONGITUDE > min_x)),:]

    trace_df = trace_df.reset_index().drop("index", axis = 1)

    trace_df["time_diff"] = pd.to_datetime(trace_df.TIMESTAMP) \
        .diff().astype("timedelta64[m]")

    trace_df["cut_point"] = trace_df["time_diff"] >= 1

    traces_list = []
    cut_points = np.where(trace_df["cut_point"].values)[0]

    new_batch_id = 0
    for i in range(1, len(cut_points)): # start of data is ignored, as no cell is known
        start_point = cut_points[i - 1]
        end_point = cut_points[i] - 1 # i is where the time difference is already too large
        trace_df_sub = trace_df.loc[start_point:end_point,]
        if trace_df_sub.shape[0] > 50:
            trace_df_sub = trace_df_sub.merge(
                celldata_df.rename(columns = {"t_ascii":"TIMESTAMP"})[["TIMESTAMP", "ci"]].drop_duplicates(),
                how = "left",
                on = "TIMESTAMP")
            trace_df_sub["within_timediff"] = pd.to_datetime(trace_df_sub.TIMESTAMP) \
                .diff().astype("timedelta64[m]")

            if not all(trace_df_sub.ci.unique() == -1):
                trace_df_sub["matches_in_celldata"] = 1
            else:
                trace_df_sub["matches_in_celldata"] = 0

            trace_df_sub = trace_df_sub.reset_index().drop("index", axis = 1)

            if trace_df_sub.ci.iloc[-1] == -1:
                last_point_ci = match_smallest_cell(trace_df_sub.point.iloc[-1], wkt_data_gpd)
                trace_df_sub.at[trace_df_sub.index[-1], "ci"] = last_point_ci
            if trace_df_sub.ci.iloc[0] == -1:
                first_point_ci = match_smallest_cell(trace_df_sub.point.iloc[0], wkt_data_gpd)
                trace_df_sub.at[0, "ci"] = first_point_ci

            new_batch_id += 1
            trace_df_sub["new_batch_id"] = new_batch_id
            # minimum trajectory length:
            traces_list.append(trace_df_sub)

    trace_df_corrected = pd.concat(traces_list, axis  = 0)
    return trace_df_corrected

trace_1_batched = create_batched_df(trace_1, celldata_1)
trace_2_batched = create_batched_df(trace_2, celldata_2)

trace_2_batched = trace_2_batched.loc[trace_2_batched.ci.isna() == False,:].copy()

trace_2_batched["new_batch_id"] = trace_2_batched["new_batch_id"] + trace_1_batched.new_batch_id.max() + 1
traces_combined = pd.concat([trace_1_batched, trace_2_batched], axis = 0)

traces_combined.ci.isna().sum()

traces_combined.drop(["accuracy",
                      "stop_probability",
                      "mask",
                      "point",
                      "geometry"], axis = 1).to_csv(realdata_dir + args["output_fname"], index = None)

print("Wrote file:")
print(realdata_dir + args["output_fname"])