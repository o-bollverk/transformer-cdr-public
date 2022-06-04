import pandas as pd
import numpy as np
import shapely
import shapely.wkt

from shapely.geometry import polygon

data_dir = "~/Desktop/masters_thesis/data/"
synth_mobility_df = pd.read_csv(f'{data_dir}'"synth_mobility_wgs84.csv")
validation_df = pd.read_csv(f'{data_dir}'"synth_validation_dataset.csv")

# Join cell id to gps traces
# Check for duplicates - only keep the smallest cell when activation was at the same time
# Extend the currently activated cell until the next activation

synth_mobility_df["uid"] = synth_mobility_df["id"]
synth_w_val_df = pd.merge(synth_mobility_df, validation_df.drop(["mod", "speed"], axis = 1), on = ["epoch", "uid"], how = "left")
synth_w_val_df = synth_w_val_df.sort_values(["id", "epoch"])

print(synth_w_val_df.shape)

added_cols = validation_df.columns
added_cols = added_cols[added_cols.isin(["mod", "speed", "epoch", "uid"]) == False]

synth_w_val_df_filled_list = []

def get_wkt_area(wkt_str):
    if pd.isna(wkt_str):
        return np.nan
    else:
        return shapely.wkt.loads(wkt_str).area

for id in synth_w_val_df.id.unique():
    print("current_id: " + str(id))

    synth_w_val_df_sub = synth_w_val_df.loc[synth_w_val_df.id == id,:]
    synth_w_val_list = []

    for epoch in synth_w_val_df_sub.epoch.unique():
        synth_w_val_df_sub_epoch = synth_w_val_df_sub.loc[synth_w_val_df_sub.epoch == epoch,:]

        if synth_w_val_df_sub_epoch.shape[0] > 1:
            synth_w_val_df_sub_epoch.at[:,"wkt_area"] = synth_w_val_df_sub_epoch.cell_wkt.apply(lambda x: get_wkt_area(x))
            synth_w_val_df_sub_epoch = synth_w_val_df_sub_epoch.loc[synth_w_val_df_sub_epoch.wkt_area == synth_w_val_df_sub_epoch.wkt_area.max(),:]\
                .drop(["wkt_area"], axis = 1)

            synth_w_val_list.append(synth_w_val_df_sub_epoch)
        else:
            synth_w_val_list.append(synth_w_val_df_sub_epoch)

    synth_w_val_df_dedupl = pd.concat(synth_w_val_list, axis = 0)

    if synth_w_val_df_dedupl.shape[0] != synth_w_val_df_dedupl.shape[0]:
        print("Deduplicated shape:")
        print(synth_w_val_df_dedupl.shape)
        print("Original shape: ")
        print(synth_w_val_df_sub.shape)

    del synth_w_val_list

    # Extend current cell info to next cell id
    # If we get nan values in the cell id, we take the latest cell id to fill that na

    synth_w_val_df_dedupl_copy = synth_w_val_df_dedupl.copy()

    for i in range(1, synth_w_val_df_dedupl.shape[0]):
        if synth_w_val_df_dedupl.iloc[i,][["ci"]].isna().values:
            synth_w_val_df_dedupl_bef = synth_w_val_df_dedupl.iloc[:(i-1),]
            if all(synth_w_val_df_dedupl_bef["ci"].isna()):
                continue
            synth_w_val_df_dedupl_bef = synth_w_val_df_dedupl_bef.loc[synth_w_val_df_dedupl_bef["ci"].isna() == False,].iloc[-1, ]
            #print("Latest wkt df:")
            #print(synth_w_val_df_dedupl_bef)

            for added_col in added_cols:
                added_col_index = np.where(synth_w_val_df_dedupl_copy.columns == added_col)[0][0]
                synth_w_val_df_dedupl_copy.iat[i, added_col_index] = synth_w_val_df_dedupl_bef[[added_col]].values
                #print(synth_w_val_df_dedupl_copy.iloc[i, added_col])
    synth_w_val_df_filled_list.append(synth_w_val_df_dedupl_copy)

    print(f"Added {synth_w_val_df_dedupl_copy.shape[0]} rows to the list")
    del synth_w_val_df_dedupl_copy

synth_w_val_df_filled = pd.concat(synth_w_val_df_filled_list, axis = 0)
del synth_w_val_df_filled_list

synth_w_val_df_filled.to_csv(f'{data_dir}synth_w_val_df_filled.csv', index = None)


print(synth_w_val_df_filled)
#print(synth_mobility_df.columns)
#print(synth_w_val_df)
#print(synth_w_val_df.columns)




