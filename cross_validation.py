import pandas as pd
from sklearn.model_selection import KFold
import re
import os as os
from training_and_validation import *
from timeit import default_timer as timer
import codecs, json
import numpy as np
from torch.utils.data import TensorDataset

def acc_calculator(preds, target, src_pad_idx, tgt_pad_idx):
    matches = (preds == target)
    acc = np.mean(matches[np.where(((target != src_pad_idx) &
                                    (target != tgt_pad_idx) &
                                    (preds != src_pad_idx) &
                                    (preds != tgt_pad_idx)))
                  ])
    return acc

def no_of_splits_from_perc(TensorDataset, perc):
    no_of_folds = int(len(TensorDataset)/int(len(TensorDataset)*perc))
    return no_of_folds

def load_splits(splits_dfname):
    f = open(splits_dfname, "r")
    rows = []
    for row in f:
        rows.append(row)
    rows_as_string = "".join(rows)
    indices_list = eval(rows_as_string)

    common_list = []

    for i in range(len(indices_list[0])):
        common_list.append((indices_list[0][i], indices_list[1][i]))

    return common_list

def load_acc(acc_printout_dfnames, main_dir):
    starting_list = []
    for acc_printout_dfname in acc_printout_dfnames:

        f = open(main_dir + "/preds/printout/" + acc_printout_dfname, "r")
        rows = []
        for row in f:
            rows.append(row)
        rows_as_string = "".join(rows)
        values_list = eval(rows_as_string)

        test_acc = values_list[2]
        train_acc = values_list[3]
        epoch = values_list[0]
        fold  = values_list[len(values_list) - 1]
        starting_list.append([fold, epoch, test_acc, train_acc])
    df = pd.DataFrame(starting_list)
    df.columns = ["fold", "epoch", "test_acc", "train_acc"]
    return df

def load_preds(preds_dfname, fold, which_type = "train"):
    preds_dfname = preds_dfname + "_" + which_type + "__fold_" + str(fold) + ".csv"
    return pd.read_csv(preds_dfname)

def load_previous_timestamp(main_dir, timestamp_file):
    if timestamp_file == "":
        print("NO PREVIOUS TIMESTAMP FILE PROVIDED")
        exit()

    with open(main_dir + "preds/printout/" + timestamp_file, "r") as f:
        res = json.load(f)
    prev_w_timestamp = res['dfname_w_timestamp'][0]
    timestamp = prev_w_timestamp[(len(prev_w_timestamp)- 19):]
    return prev_w_timestamp, timestamp

def test_match(pattern, text):
    re_res = re.search(string = text,
                       pattern = pattern)
    if re_res != None:
        return True
    else:
        return False

def train_and_validate_with_cv(model,
                               device,
                               loss_fn,
                               NUM_EPOCHS,
                               no_of_folds_perc,
                               dataset,
                               batchsize,
                               max_folds,
                               main_dir,
                               timestamp_file,
                               cell_transformer,
                               src_pad_idx,
                               tgt_pad_idx,
                               dfname,
                               tgt_trajectory_len,
                               optimizer,
                               json_path,
                               load_cv_indices = False):
    # Performs cross validation; can load previous predictions from map-matching step as data and previous folds from json

    if load_cv_indices:
        fold_train_indices = []
        fold_test_indices = []

        # load previous timestamp

        prev_timestamp_w_filename, prev_timestamp = load_previous_timestamp(main_dir, timestamp_file)
        splits_dfname = main_dir + "preds/" + prev_timestamp_w_filename + "_preds_fold_indices.json"

        train_test_split = load_splits(splits_dfname)
        preds_dfnames_string = main_dir + "preds/" + prev_timestamp_w_filename  + "_preds"
        pattern_matches = [test_match(prev_timestamp, x) for x in os.listdir(main_dir + "preds/printout/")]
        acc_printout_dfnames = np.array(os.listdir(main_dir + "preds/printout/"))[pattern_matches]

        # for each fold, find epoch with highest test accuracy
        acc_df = load_acc(acc_printout_dfnames)
        highest_test_acc = acc_df.groupby("fold").max().reset_index()[["fold", "test_acc"]] \
            .rename(columns = {"test_acc":"max_test_acc"} )

        acc_df = acc_df.merge(highest_test_acc,
                              how = "left",
                              on = "fold")

        epoch_for_fold_df = acc_df.loc[acc_df.test_acc ==  acc_df.max_test_acc,["epoch", "fold"]]
    else:
        kfold = KFold(n_splits= no_of_splits_from_perc(dataset, no_of_folds_perc), shuffle=True)
        fold_train_indices = []
        fold_test_indices = []
        train_test_split = kfold.split(dataset)

    for fold,(train_idx,test_idx) in enumerate(train_test_split):
        if fold >= max_folds:
            break
        if load_cv_indices:
            fold_train_indices.append(train_idx)
            fold_test_indices.append(test_idx)
        else:
            fold_train_indices.append(train_idx.tolist())
            fold_test_indices.append(test_idx.tolist())

        train_dataset = TensorDataset(dataset[train_idx][0],dataset[train_idx][1])
        test_dataset = TensorDataset(dataset[test_idx][0],dataset[test_idx][1])

        train_dataloader = DataLoader(train_dataset, batch_size=batchsize)
        test_dataloader = DataLoader(test_dataset, batch_size=batchsize)

        target_list = []
        for _, target in test_dataloader:
            target_list.append(target.numpy())
        gtruth_target = np.row_stack(target_list)[:, :-1]

        train_target_list = []
        for _, train_target in train_dataloader:
            train_target_list.append(train_target.numpy())
        gtruth_train_target = np.row_stack(train_target_list)[:, :-1]

        if load_cv_indices: # target will be loaded from predictions
            selected_epoch_for_fold = epoch_for_fold_df.loc[epoch_for_fold_df.fold == fold,"epoch"].values[0]

            train_target_df  = load_preds(preds_dfnames_string, fold, which_type = "train") \
                                   .iloc[:,selected_epoch_for_fold] # first column is batch-id

            train_target = train_target_df.to_numpy() \
                .reshape(int(train_target_df.shape[0]/tgt_trajectory_len), tgt_trajectory_len)

            target_df = load_preds(preds_dfnames_string, fold, which_type = "test") \
                            .iloc[:,selected_epoch_for_fold]

            target = target_df.to_numpy() \
                .reshape(int(target_df.shape[0]/tgt_trajectory_len), tgt_trajectory_len)
        else:
            target = gtruth_target
            train_target = gtruth_train_target

        estimated_routes_list_test = []
        estimated_routes_list_train = []

        # zero gradients at each cross validation step
        optimizer.zero_grad()

        for epoch in range(1, NUM_EPOCHS+1):
            start_time = timer()

            train_loss = train_epoch_dataloader(model = model,  cdr_dataset = train_dataset, device = device, batchsize=batchsize, loss_fn = loss_fn,  optimizer=optimizer)
            preds_train = np.array(predict(model, train_dataset).data.tolist())

            preds_initial, valid_loss = predict(model, test_dataset, device = device, batchsize = batchsize, loss_fn = loss_fn, return_losses=True)
            preds = np.array(preds_initial.data.tolist())

            end_time = timer()

            estimated_routes_list_test.append(preds.tolist())
            estimated_routes_list_train.append(preds_train.tolist())

            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f} "f"Epoch time = {(end_time - start_time):.3f}s"))

            # acc calc is never over the padding indices
            train_acc = acc_calculator(preds_train, train_target, src_pad_idx, tgt_pad_idx)
            acc = acc_calculator(preds, target, src_pad_idx, tgt_pad_idx)

            # calculate final accuracy with regard to ground truth of the original labels
            # For both train and test
            if load_cv_indices:
                gtruth_test_acc = acc_calculator(preds, gtruth_target)
                gtruth_train_acc = acc_calculator(preds_train, gtruth_train_target)
            else:
                gtruth_test_acc = "Not calculated"
                gtruth_train_acc = "Not calculated"


            if cell_transformer:
                json_printout_path = main_dir + "cell_preds/printout/" +  dfname + "_" + ct + "_epoch_" + str(epoch) +  "_acc_printout" + "_" +  str(fold) +  ".json"
            else:
                json_printout_path = main_dir + "preds/printout/" +  dfname + "_" + ct + "_epoch_" + str(epoch) +  "_acc_printout" + "_" +  str(fold) +  ".json"

            json.dump(
                [epoch, train_loss,valid_loss,gtruth_train_acc,gtruth_test_acc,  train_acc, acc, end_time - start_time, fold],
                codecs.open(json_printout_path, 'w', encoding='utf-8'),
                separators=(',', ':'),
                sort_keys=True,
                indent=4
            ) # dump accuracy printout

            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Valid loss: {valid_loss:.3f} "
                   f" Test accuracy: {acc}, Train accuracy: {train_acc}, "
                   f" Ground truth train accuracy: {gtruth_train_acc}, Ground truth test accuracy: {gtruth_test_acc},"
                   f" Epoch train and validation time: {(end_time - start_time):.3f}s, Fold: {fold}"))

        csv_path_for_fold_test = json_path[:-5] + "_test_" + "_fold_" + str(fold) + ".csv"
        csv_path_for_fold_train = json_path[:-5] + "_train_" + "_fold_" + str(fold) + ".csv"

        # csv instead of json
        train_output_df = pd.DataFrame(estimated_routes_list_train).T
        train_output_df = pd.concat([train_output_df[x].explode() for x in train_output_df.columns], axis = 1) \
            .reset_index() \
            .rename({"index":"batch_id_scaled"})

        test_output_df = pd.DataFrame(estimated_routes_list_test).T
        test_output_df = pd.concat([test_output_df[x].explode() for x in test_output_df.columns], axis = 1) \
            .reset_index() \
            .rename({"index":"batch_id_scaled"})

        train_output_df.to_csv(csv_path_for_fold_train, sep = ",", index = None)
        test_output_df.to_csv(csv_path_for_fold_test, sep = ",", index = None)

    json_path_for_fold_indices = json_path[:-5] + "_fold_indices"  + ".json"
    json.dump([fold_train_indices, fold_test_indices], codecs.open(json_path_for_fold_indices, 'w', encoding='utf-8'),
              separators=(',', ':'),
              sort_keys=True,
              indent=4)