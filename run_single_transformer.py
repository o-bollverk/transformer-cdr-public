import datetime
import argparse
import torch.nn.functional as F
from data_processing import *
from cdr_transformer import *
from training_and_validation import *
from cross_validation import *

# Setting main directory arguments using argparser

parser = argparse.ArgumentParser()
parser.add_argument('--main_dir', help='Main directory, has subdirectories: data, real_data, trained_model, preds, cell_preds', type = str)
parser.add_argument('--dfname', help='csv file name', type = str)
parser.add_argument("--device", help = "device: cuda or cpu", type = str, default = "cpu")
parser.add_argument("--epochs", help = "Number of epochs", type = int, default = 5)
parser.add_argument("--batchsize", help = "Batch size. Defaults to whole data being one batch.", type = int, default = 0)
parser.add_argument("--dropout", help = "Dropout", type = float, default = 0)
parser.add_argument("--src_pad_idx", help = "Source pad index, for cases when the data is pre-padded on the road level dimension", type = int, default = 0)
parser.add_argument("--tgt_pad_idx", help = "Target pad index, for cases when the data is pre-padded on the road level dimension", type = int, default = 1)
parser.add_argument("--run_on_multiple_gpus", help = "1: run on multiple GPUS, 0: run on single GPU", type = int, default = 1)
parser.add_argument("--evaluate_all_epochs", help = "1: evaluate all epochs, 0: evaluate only last epoch", type = int, default = 1)
parser.add_argument("--cross_validation", help = "1: do cross validation for as many folds as max_folds, where the number of folds is calculated from the data based on --cross_validation_folds_perc, 0: no cross validation, only training accuracy reported", type = int, default = 0)
parser.add_argument("--cross_validation_folds_perc", help = "Size of the test/dev fold as a ratio of train fold", type = float, default = 0.05)
parser.add_argument("--cell_transformer", help = "if 1, perform road id to cell id mapmatching; if 0, perform gps trajectory mapmatching", type = int, default = 0)
parser.add_argument("--src_col", help = "source column for cell transformer", type = str, default = "cell_id")
parser.add_argument("--target_col", help = "target column for cell transformer", type = str, default = "road_id")
parser.add_argument("--learning_rate", help = "Learning rate, default: 0.0007 from Transformer based map-matching paper", type = float, default = 0.0007) # from mapmatching paper
parser.add_argument("--reduction", help = "Reduction argument value for torch.nn.CrossEntropyLoss. Default: mean ", type = str, default = 'mean') # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
parser.add_argument("--max_folds", help = "Maximum number of folds to validate over. Folds are taken in order from the train test split, which has shuffle = T. Default: 5.", type = int, default = 5)
parser.add_argument("--load_cv_indices", help = "Whether to load cross validation indices from the last recorded ones. Can be used only with cell transformer.", type = int, default = 0)
parser.add_argument("--timestamp_file", help = "File to write the current filename and starting timestamp. Cell transformer reads from here, gps transformer writes to here.", type = str, default = "")
parser.add_argument("--forward_expansion", help = "Forward expansion parameter. In the original transformer paper = 4, default = 1", type = int, default = 1)

args = vars(parser.parse_args())

print("Main directory: " + args["main_dir"])
print("File name: " +  args["dfname"])
print("Device count: " + str(torch.cuda.device_count()))
print("The task matching road id and cell id: " + str(args["cell_transformer"] == 1))
cell_transformer = args["cell_transformer"] == 1
load_cv_indices = args["load_cv_indices"]

if load_cv_indices and not cell_transformer:
    print("Previous cross validation indices can only be loaded in the cell transformer phase.")
    exit

#os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
#PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128#
#torch.backends.cuda.cufft_plan_cache.max_size = 2000
#torch.backends.cuda.cufft_plan_cache[0].max_size = 10

if __name__ == "__main__":
    # 1) Current datetime to use in json dumping of predictions
    ct = datetime.datetime.now().__str__()[:-7]
    ct1 = ct[:10]
    ct2 = ct[11:len(ct)]
    ct = ct1 + "_" +  ct2
    ct = re.sub(":", "-", ct)
    complete_ct = ct

    print("Starting time:" + complete_ct)

    # 2) Parsing device, seq and batchsize arguments
    device = args["device"]
    batchsize = args["batchsize"]

    if batchsize == 0:
        batchsize = None

    torch.backends.cudnn.enabled = False #https://stackoverflow.com/questions/48445942/pytorch-training-with-gpu-gives-worse-error-than-training-the-same-thing-with-c
    torch.manual_seed(33) #https://stackoverflow.com/questions/67511658/training-pytorch-models-on-different-machines-leads-to-different-results
    #torch.backends.cudnn.deterministic = True
    print("Used device: " + device)

    # 3) Defining transformer parameters

    embed_size =  512
    heads = 8
    num_layers = 6

    forward_expansion = args['forward_expansion']
    dropout = args['dropout']
    src_pad_idx = args["src_pad_idx"] #0
    tgt_pad_idx = args["tgt_pad_idx"] #0
    if cell_transformer:
        target_col = args["target_col"] #0
        src_col = args["src_col"] #0
    else:
        target_col = "road_id"

# no of link, src and tgt trajectory len read in from the data

    # 4) Parsing directory, data file and epoch arguments
    # dfname chema: X, Y, road_id, batch_id

    main_dir = args["main_dir"]
    dfname = args["dfname"]
    data_dir = main_dir + "data/"
    fname = data_dir + dfname
    NUM_EPOCHS = args["epochs"]

    # 5) Directory to dump json predictions

    preds_dfname = dfname[:-4] + "_" + ct + "_preds.json"

    if cell_transformer:
        json_path = main_dir  + "cell_preds/" + preds_dfname
    else:
        json_path = main_dir + "preds/" + preds_dfname ## your path variable


    # 5.2) Dumping the current filename and timestamp for next step of cell transformer

    if not cell_transformer and args["timestamp_file"] != "": # timestamp file argument is provided explicitly
        json.dump(
            {'dfname_w_timestamp': [dfname[:-4] + "_" + ct]},
            codecs.open(main_dir + "preds/printout/"+ args["timestamp_file"], 'w', encoding='utf-8'),
            separators=(',', ':'),
            sort_keys=True,
            indent=4
        )

    # 6) Directory to save trained model

    trained_model_dir = main_dir + "trained_model/"

    # Guessed delimiter
    import csv
    s = csv.Sniffer()

    for line in open(fname, "r"):
        first_line = line
        break

    guessed_delimiter = s.sniff(first_line).delimiter

    src_trajectory_len = pd.read_csv(fname, sep=guessed_delimiter).groupby("batch_id")\
        .count().reset_index()[target_col].max()

    tgt_trajectory_len = src_trajectory_len - 1

    print("Source trajectory length:" +  str(src_trajectory_len))
    print("Target trajectory length:" +  str(tgt_trajectory_len))

    # 7) determine no of links from the data:

    no_of_link =  pd.read_csv(fname, sep=guessed_delimiter)[target_col].max()
    print("No of links in the data:"  + str(no_of_link))
    print("Size of test fold as a ratio of the entire dataset: " + str(args["cross_validation_folds_perc"]))
    print("Dropout: " + str(args["dropout"]))
    print("Learning rate: " + str(args["learning_rate"]))
    print("Loading cross validation indices from previous step: " + str(load_cv_indices == 1))

    # https://discuss.pytorch.org/t/embedding-error-index-out-of-range-in-self/81550/10

    # 8) Define loss functions outside of training functions
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tgt_pad_idx, reduction = args["reduction"], label_smoothing= 0.1)

    # 9) Load data

    batch_ids = pd.read_csv(fname, sep = guessed_delimiter).batch_id.unique().tolist()

    # 10) Applying gps trajectory normalization for the whole data, not by batch

    def coords_changer(x):
        if x == -1:
            return 0
        else:
            return x

    loaded_df = pd.read_csv(fname, sep = guessed_delimiter)

    source_array_list = [] # list of numpy arrays
    target_array_list = []
    loaded_df_sub_list = []

    if cell_transformer:
        if args["cross_validation"] == 1:
            load_cv_indices = True
        for i in batch_ids:
            cell_id_array, road_id_array  = get_celldata_by_batch_id(loaded_df,
                                                                    batch_id=i)

            if src_col == "road_id" and target_col == "cell_id":
                source_array_list.append(road_id_array[:src_trajectory_len])
                target_array_list.append(cell_id_array)
            elif src_col == "cell_id" and target_col == "road_id":
                source_array_list.append(cell_id_array[:src_trajectory_len])
                target_array_list.append(road_id_array)

    else:
        loaded_df["X"] = loaded_df["X"].apply(lambda x: coords_changer(x))
        loaded_df["Y"] = loaded_df["Y"].apply(lambda x: coords_changer(x))

        # apply normalization on batch basis for gps trajectories
        loaded_df["X"] = normalize_to_onezero(loaded_df["X"], src_pad_idx=src_pad_idx)
        loaded_df["Y"] = normalize_to_onezero(loaded_df["Y"],  src_pad_idx=src_pad_idx)

        for i in batch_ids:
            gps_array, gtruth_array = get_gps_data_by_batch_id(loaded_df,
                                                     batch_id=i) # sep = " "
            gps_array = gps_array[:src_trajectory_len,:]
            source_array_list = source_array_list + [gps_array]
            gtruth_array = gtruth_array[:src_trajectory_len]
            target_array_list.append(gtruth_array)

    # 11) Dataloader
    max_batch_size = loaded_df.groupby("batch_id").count()["road_id"].max()
    source_tensors_list = []
    target_tensors_list = []

    for i in range(len(source_array_list)):
        source = torch.Tensor(np.array(source_array_list[i]))
        target = torch.Tensor(np.array(target_array_list[i])).long()
        pad_len = max_batch_size - source.shape[0]
        if cell_transformer:
            source = source.long()
            source_padded = F.pad(input=source, pad=(0, pad_len), mode='constant', value= src_pad_idx)
            target_padded = F.pad(input=target, pad=(0, pad_len), mode='constant', value= tgt_pad_idx)
        else:
            source = source.float()
            source_padded = F.pad(input=source, pad=(0, 0, 0, pad_len), mode='constant', value= src_pad_idx)
            target_padded = F.pad(input=target, pad=(0, pad_len), mode='constant', value= tgt_pad_idx)
        source_tensors_list.append(source_padded)
        target_tensors_list.append(target_padded)

    source_tensor = torch.stack(source_tensors_list)
    target_tensor = torch.stack(target_tensors_list)
    cdr_dataset = TensorDataset(source_tensor, target_tensor)

    # 12) Define transformer along with optimizer

    no_masking = False # this would cancel both source and target mask, only used when testing functionality without masking

    if cell_transformer:
        transformer = Transformer(
            src_trajectory_len,
            tgt_trajectory_len,
            src_pad_idx = src_pad_idx,
            trg_pad_idx = tgt_pad_idx,
            embed_size=embed_size,
            num_layers= num_layers,
            forward_expansion=forward_expansion,
            heads=heads,
            dropout=dropout,
            device=device,
            no_of_link=no_of_link,
            coord_dim = 2,
            batch_size=batchsize,
            no_masking=no_masking,
            max_batch_size = max_batch_size,
            cell_transformer=True
        )
    else:
        transformer = Transformer(
            src_trajectory_len,
            tgt_trajectory_len,
            src_pad_idx = src_pad_idx,
            trg_pad_idx = tgt_pad_idx,
            embed_size=embed_size,
            num_layers= num_layers,
            forward_expansion=forward_expansion,
            heads=heads,
            dropout=dropout,
            device=device,
            no_of_link=no_of_link,
            coord_dim = 2,
            batch_size=batchsize,
            no_masking=no_masking,
            max_batch_size = max_batch_size,
            cell_transformer=False
        )

    optimizer = torch.optim.Adam(transformer.parameters(), lr=args["learning_rate"], eps=1e-9)  # betas=(0.9, 0.98)? # 0.0007 from paper, p16
    CUDA_LAUNCH_BLOCKING=1

    # 13) Multiple GPU definition
    if args["run_on_multiple_gpus"] == 1:
        transformer.to(device)
    else:
        transformer.to(device)

    # 14) Start run
    if args["cross_validation"] == 1:
        train_and_validate_with_cv(model = transformer,
                                   device = device,
                                   loss_fn = loss_fn,
                                   NUM_EPOCHS = NUM_EPOCHS,
                                   no_of_folds_perc =  args["cross_validation_folds_perc"],
                                   dataset = cdr_dataset,
                                   batchsize = batchsize,
                                   max_folds = args["max_folds"],
                                   main_dir = main_dir,
                                   timestamp_file = args["timestamp_file"],
                                   cell_transformer = cell_transformer,
                                   src_pad_idx=src_pad_idx,
                                   tgt_pad_idx=tgt_pad_idx,
                                   dfname=dfname,
                                   tgt_trajectory_len=tgt_trajectory_len,
                                   optimizer=optimizer,
                                   json_path=json_path,
                                   load_cv_indices = load_cv_indices)
    else:
        estimated_routes_list = []
        train_dataloader = DataLoader(cdr_dataset, batch_size=batchsize)     # get the ground truth from the dataloader
        target_list = []
        for _, target in train_dataloader:
            target_list.append(target.numpy())

        for epoch in range(1, NUM_EPOCHS+1):
            start_time = timer()
            train_loss = train_epoch_dataloader(transformer, cdr_dataset, device, batchsize, loss_fn, optimizer)
            end_time = timer()

            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f} "f"Epoch time = {(end_time - start_time):.3f}s"))

            target = np.row_stack(target_list)[:, :-1]

            if args["evaluate_all_epochs"] == 1:
                preds = np.array(predict(transformer, cdr_dataset, device, batchsize, loss_fn).data.tolist())
            else:
                if epoch == NUM_EPOCHS:
                    preds = np.array(predict(transformer, cdr_dataset, device, batchsize, loss_fn).data.tolist())#

                else:
                    preds = np.zeros(target.shape)

            estimated_routes_list.append(preds.tolist())
            acc = acc_calculator(preds, target, src_pad_idx, tgt_pad_idx)

            if cell_transformer:
                json_printout_path = main_dir + "cell_preds/printout/" +  dfname + "_" + ct + "_epoch_" + str(epoch) +  "_acc_printout.json"
            else:
                json_printout_path = main_dir + "preds/printout/" +  dfname + "_" + ct + "_epoch_" + str(epoch) +  "_acc_printout.json"

            if not cell_transformer:
                json.dump(
                    {'dfname_w_timestamp': [dfname + "_" + ct]},
                    codecs.open(main_dir + "preds/printout/"+ args["timestamp_file"], 'w', encoding='utf-8'),
                    separators=(',', ':'),
                    sort_keys=True,
                    indent=4
                )
            json.dump(
                [epoch, train_loss, acc, end_time - start_time],
                codecs.open(json_printout_path, 'w', encoding='utf-8'),
                separators=(',', ':'),
                sort_keys=True,
                indent=4
            ) # dump accuracy printout

            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Accuracy: {acc} "f"Epoch time = {(end_time - start_time):.3f}s"))

        json.dump(estimated_routes_list, codecs.open(json_path, 'w', encoding='utf-8'),
                  separators=(',', ':'),
                  sort_keys=True,
                  indent=4)

    # Save the model
    # torch save results in a pickle error and is not the recommended way of saving
    # saving using parameters

    torch.save(transformer.state_dict(), trained_model_dir + "trained_" + complete_ct + ".pt")

