import numpy as np

def normalize_to_onezero(x, src_pad_idx):
    min_over_gps = np.min(x[x != src_pad_idx])
    max_over_gps = np.max(x[x != src_pad_idx])
    res_list = []
    for value in x:
        if value != src_pad_idx:
            res = (value - min_over_gps)/(max_over_gps - min_over_gps)
            res_list.append(res)
        else:
            res_list.append(src_pad_idx)
    return res_list

def get_gps_data_by_batch_id(gps_df, batch_id):
    gps_df_coords = gps_df.loc[gps_df.batch_id == batch_id,["X", "Y"]].to_numpy()
    res_gtruth = gps_df.loc[gps_df.batch_id == batch_id,"road_id"].values
    return gps_df_coords, res_gtruth

def get_celldata_by_batch_id(cell_df, batch_id):
    road_ids = cell_df.loc[cell_df.batch_id == batch_id,"road_id"].values
    cell_ids = cell_df.loc[cell_df.batch_id == batch_id,"cell_id"].values
    return cell_ids, road_ids
