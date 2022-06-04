# Cut butches to ensure that no batch contains no more than 200 gps points ------
library(dplyr)
library(purrr)
library(dplyr)
library(data.table)

# Data variables -----------
args <-  commandArgs(trailingOnly=TRUE)
data_dir <- args[1] #"~/Desktop/masters_thesis/real_data/"
dfname <- args[2] #"revised_realdata_w_batch_id_w_road_id.csv"
output_df_name <- args[3] #"revised_realdata_w_batch_id_w_road_id_batchcut.csv"

# Print --------
print(paste0("Directory: ", data_dir))
print(paste0("Input filename: ", dfname))
print(paste0("Output filename: ", output_df_name))

# load data -----
final_df <- fread(paste0(data_dir, dfname))
final_df <- final_df %>% 
  mutate(cut_id = batch_id)

# extend function --------
extend <- function(x, desired_length){
  no_to_extend <- desired_length - length(x)
  if(no_to_extend > 0){
    y <- c(x, rep(x[length(x)], no_to_extend))
  } else {
    y = x
  }
  return(y)
}
floor2 <- function(x){
  y <- floor(x)
  if(y == 0){
    return(round(x))
  }
  else return(y)
}

# Cut ---------
no_of_splits <- 3

final_df <- final_df %>% 
  group_by(cut_id) %>% 
  mutate(count_n = n()) %>% 
  mutate(cut_index = ifelse(
    count_n > 200,
    extend(rep(1:no_of_splits,
               each = floor2((count_n %>% unique())/no_of_splits)), count_n %>% unique()),
    1)) %>% 
  ungroup()

final_df <- final_df %>% 
  left_join(
    final_df %>% 
      filter(cut_index != 1) %>% 
      group_by(cut_id, cut_index) %>% 
      summarise(test = T) %>% 
      ungroup() %>% 
      mutate(new_index = row_number() + final_df$cut_id %>% max()) %>% 
      select(-test)
  ) %>% 
  mutate(new_index = ifelse(is.na(new_index), cut_id, new_index)) 

# export ---------
export_df <- final_df %>% 
  select(-batch_id) %>% 
  rename(batch_id = new_index) %>% 
  select(-cut_id, -cut_index, -count_n) %>% 
  select(c("X", "Y", "road_id", "batch_id", "cell_id"))

write.table(export_df, paste0(data_dir, output_df_name), sep = ",", row.names = F)

