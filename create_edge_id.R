args = commandArgs(trailingOnly=TRUE)

# Graph matching based on Tartu area ------------

if(!"remotes" %in% installed.packages()) {
  install.packages("remotes")
}

cran_pkgs = c(
  "dplyr",
  "purrr",
  "rgdal",
  "sf",
  "tidygraph",
  "igraph",
  "osmdata",
  "dplyr",
  "tibble",
  "ggplot2",
  "units",
  "tmap",
  "rgrass7",
  "link2GI",
  "nabor",
  "spNetwork",
  "data.table",
  "rgeos",
  "leaflet",
  "purrr",
  "stringr",
  "spNetwork"
)

for(pkg in cran_pkgs){
  if(require(pkg, character.only = T) == FALSE){
    #next
    remotes::install_cran(cran_pkgs)
  }
}

# prior Ubuntu dependencies ---------------
# FIRST !:
# sudo-apt get dist-upgrade

# DEPENDNCIES FOR RGRASS7:
# https://geocompr.github.io/post/2020/installing-r-spatial-ubuntu/

# sudo aptitude install libproj-dev
#? sudo apt install ruby
# /lib/x86_64-linux-gnu/libproj.so.22
# https://github.com/r-spatial/sf#ubuntu
# sudo apt-get install proj-bin

# for using with rgeos-------------
#devtools::install_github("JeremyGelb/spNetwork", ref = "a3bc982")

# read libraries -----------
library(sf)
library(tidygraph)
library(igraph)
library(dplyr)
library(tibble)
library(ggplot2)
library(units)
library(tmap)
library(osmdata)
library(rgrass7)
library(link2GI)
library(nabor)
library(rgdal)
library(stringr)
library(purrr)
library(spNetwork)

# expected warnings upon reading of packages -------
# Loading required package: sp
# Please note that rgdal will be retired by the end of 2023,
# plan transition to sf/stars/terra functions using GDAL and PROJ
# at your earliest convenience.
# 
# rgdal: version: 1.5-28, (SVN revision 1158)
# Geospatial Data Abstraction Library extensions to R successfully loaded
# Loaded GDAL runtime: GDAL 3.4.0, released 2021/11/04
# Path to GDAL shared files: /usr/share/gdal
# GDAL binary built with GEOS: TRUE 
# Loaded PROJ runtime: Rel. 8.2.0, November 1st, 2021, [PJ_VERSION: 820]
# Path to PROJ shared files: /home/revilo/.local/share/proj:/usr/share/proj
# PROJ CDN enabled: FALSE
# Linking to sp version:1.4-6
# To mute warnings of possible GDAL/OSR exportToProj4() degradation,
# use options("rgdal_show_exportToProj4_warnings"="none") before loading sp or rgdal.

# Data variables -------------
data_dir <- args[1] 
dfname <- args[2] 
output_dfname <- args[3] 

id_column <- "new_batch_id"

# Assisting functions ----------
clean_osm_data = function(x, retain_named = F){
  for(i in names(x)){
    if(str_detect(i, "osm") & !is.null(x[[i]])){
      if(nrow(x[[i]]) == 0) { next} 
      if(retain_named){
        x[[i]] = x[[i]] %>% 
          filter(!is.na(name))
      }
      
      # res = x[[i]] %>% keep(~ mean(is.na(na_if(.x, ""))) < 0.2) 
      res = x[[i]] %>% keep(function(.x) if(is.list(.x)){return(T)} else {mean(is.na(na_if(.x, ""))) < 0.2} ) 
      if(("name" %in% colnames(x[[i]])) & (!("name" %in% colnames(res)))){
        name = x[[i]] %>% as.data.frame() %>% select(name) 
        res = res %>% bind_cols(name)
      }
      x[[i]] = res
    }
  }
  
  return(x)
}

# Get road network from OSM data ----------------------
# DEFINING BBOX -------------
bbox_vec <- c(26.6825294, 58.3459925, 26.7408943, 58.3786785)

highways = opq(bbox_vec) %>% 
  add_osm_feature(key = "highway") %>% 
  osmdata_sf() %>% 
  unname_osmdata_sf() %>% 
  clean_osm_data()

non_service <- highways$osm_lines[highways$osm_lines$highway != "service",]
non_service2 <- non_service[!is.na(non_service$name),]

non_service2$geometry[! non_service2$osm_id %in% c("223572243",
                                                   "691519260",
                                                   "223578124")] %>% plot()

non_service2 <- non_service2[! non_service2$osm_id %in% c("223572243",
                                                          "691519260",
                                                          "223578124"),]

tartu_center <- non_service2

# Step 2: Give each edge a unique index -----------
edges <- tartu_center %>%
  mutate(edgeID = c(1:n()))

# Step 3: Create nodes at the start and end point of each edge ------------

nodes <- edges %>%
  st_coordinates() %>%
  as_tibble() %>%
  rename(edgeID = L1) %>%
  group_by(edgeID) %>%
  slice(c(1, n())) %>%
  ungroup() %>%
  mutate(start_end = rep(c('start', 'end'), times = n()/2))


# Step 4: Give each node a unique index -----------
nodes <- nodes %>%
  mutate(xy = paste(.$X, .$Y)) %>% 
  mutate(nodeID = group_indices(., factor(xy, levels = unique(xy)))) %>%
  select(-xy)

# Step 5: Combine the node indices with the edges -----------

source_nodes <- nodes %>%
  filter(start_end == 'start') %>%
  pull(nodeID)

target_nodes <- nodes %>%
  filter(start_end == 'end') %>%
  pull(nodeID)

edges = edges %>%
  mutate(from = source_nodes, to = target_nodes)

# Step 6: Remove duplicate nodes ------------

nodes <- nodes %>%
  distinct(nodeID, .keep_all = TRUE) %>%
  select(-c(edgeID, start_end)) %>%
  st_as_sf(coords = c('X', 'Y')) %>%
  st_set_crs(st_crs(edges))

# Step 7: Convert to tbl_graph --------
graph = tbl_graph(nodes = nodes, edges = as_tibble(edges), directed = FALSE)

# Step 8: Putting it together ---------------

sf_to_tidygraph = function(x, directed = TRUE) {
  
  edges <- x %>%
    mutate(edgeID = c(1:n()))
  
  nodes <- edges %>%
    st_coordinates() %>%
    as_tibble() %>%
    rename(edgeID = L1) %>%
    group_by(edgeID) %>%
    slice(c(1, n())) %>%
    ungroup() %>%
    mutate(start_end = rep(c('start', 'end'), times = n()/2)) %>%
    mutate(xy = paste(.$X, .$Y)) %>% 
    mutate(nodeID = group_indices(., factor(xy, levels = unique(xy)))) %>%
    select(-xy)
  
  source_nodes <- nodes %>%
    filter(start_end == 'start') %>%
    pull(nodeID)
  
  target_nodes <- nodes %>%
    filter(start_end == 'end') %>%
    pull(nodeID)
  
  edges = edges %>%
    mutate(from = source_nodes, to = target_nodes)
  
  nodes <- nodes %>%
    distinct(nodeID, .keep_all = TRUE) %>%
    select(-c(edgeID, start_end)) %>%
    st_as_sf(coords = c('X', 'Y')) %>%
    st_set_crs(st_crs(edges))
  
  tbl_graph(nodes = nodes, edges = as_tibble(edges), directed = directed)
  
}

tartu_center_graph <- sf_to_tidygraph(tartu_center, directed = FALSE)

edge_by_id <- tartu_center_graph %>% 
  activate(edges) %>% 
  select(geometry, edgeID) %>% 
  as.data.frame() 

# attempt to use closest rnn function on built network ---------

edge_by_id_coord <- edge_by_id$geometry %>% 
  st_coordinates()

# REMOVING DOUBLE ROADS ---------------
# OSM IDs are not suitable.
# Here I create a pseudo network, with double roads removed
# and index properly.

p <- st_collection_extract(st_intersection(edge_by_id$geometry), "POINT")
q <- lwgeom::st_split(edge_by_id$geometry, p)
q <- st_collection_extract(q, "LINESTRING")
new_edge_by_id <- st_geometry(q)

# arrange by the center point of linestring across y axis ------------
new_edge_by_id_df <- new_edge_by_id %>% 
  as.data.frame()

line_geom <- new_edge_by_id_df$geometry
edge_by_id_coord <- new_edge_by_id_df$geometry %>% st_coordinates()

# Other ways of indexing -----------
# X and y center

diagonal_order <- st_centroid(line_geom) %>%
  st_coordinates() %>% 
  as.data.frame() %>% 
  mutate(diag = abs(X - mean(X)) + abs(Y - mean(Y))) %>% 
  arrange(diag) %>% 
  mutate(index = row.names(.)) %>% 
  select(index) %>% 
  unlist() %>% 
  as.numeric()

normalize_to_onezero <- function(x){
  res <- (x - min(x))/(max(x) - min(x))
  return(res)
}

# creating linelist ----------
tartu_linelist <- lapply(unique(edge_by_id_coord[,3]),
                         function(x){
                           selected_matrix <- edge_by_id_coord[edge_by_id_coord[,3] == x, ]
                           as_line <- sp::Line(selected_matrix[, c(1,2)])
                           sp::Lines(as_line, ID = unique(selected_matrix[, 3]))
                         })

tartu_spatiallines <- sp::SpatialLines(tartu_linelist,
                                       proj4string = sp::CRS("EPSG:4326"))

edge_by_id_coord_copy <- edge_by_id_coord %>% 
  as.data.frame() %>% 
  distinct(L1)

row.names(edge_by_id_coord_copy) <- edge_by_id_coord_copy$L1

tartu_spatiallines_df <- sp::SpatialLinesDataFrame(tartu_spatiallines,
                                                   data = edge_by_id_coord_copy)

tartu_spatiallines_df <- spTransform(tartu_spatiallines_df, 
                                     CRSobj = sp::CRS("EPSG:3301"))

# load mobility df --------
mobility_df <- data.table::fread(paste0(data_dir, dfname))

mobility_df <- mobility_df %>% 
  group_by(!! as.name(id_column)) %>%
  mutate(epoch = row_number()) %>% 
  ungroup()

# Export line/road_network data -----

new_edge_by_id_copy <- new_edge_by_id
new_edge_by_id_copy$length <- sf::st_length(new_edge_by_id)
road_len_df <- data.frame(
  road_id = 1:length(new_edge_by_id_copy$length ),
  length = new_edge_by_id_copy$length %>% as.numeric()
)
write.table(road_len_df, paste0(data_dir, "road_len_df.csv"), row.names = F,
            sep = ",")


# Write data with network spatial information -------------
# save(new_edge_by_id_df,file = "") 

# Run nearest lines ---------------

nearest_lines <- function(points, lines, snap_dist = 5, # 300
                          max_iter = 10){
  
  # getting the coordinates of the lines
  list_lines <- unlist(sp::coordinates(lines), recursive = FALSE)
  
  # getting the coordinates of the points
  coords <- sp::coordinates(points)
  
  # getting the indexes
  idx <- spNetwork:::find_nearest_object_in_line_rtree(coords, list_lines, snap_dist, max_iter)
  
  # adding 1 to match with c++ indexing
  return(idx+1)
}

coords_df_full <- c()

for (j in unique(mobility_df[[id_column]])){
  coords_df_sub <- mobility_df %>% 
    filter(!! as.name(id_column) == j) %>% 
    select(LONGITUDE, LATITUDE)
  
  sp_points_df <- sp::SpatialPointsDataFrame(coords = 
                                               coords_df_sub,
                                             data = coords_df_sub,
                                             proj4string = sp::CRS("EPSG:4326")) 
  
  sp_points_df <- spTransform(sp_points_df,
                              CRSobj = sp::CRS("EPSG:3301"))
  
  
  
  nearest_lines_vec <- nearest_lines(sp_points_df,
                                     lines = tartu_spatiallines_df)
  
  coords_df_sub <- coords_df_sub %>% 
    mutate(edge = nearest_lines_vec)
  
  coords_df_full <- rbind.data.frame(coords_df_full,
                                     coords_df_sub %>% 
                                       mutate(!! as.name(id_column) := j)) #c(coords_df_list, coords_df_sub)
  
}

mobility_df_export <- mobility_df %>% 
  bind_cols(coords_df_full %>% select(edge))

# check for cases with only one road id per batch and filter out ------

mobility_df_export <- mobility_df_export %>% 
  filter(!
           (!! as.name(id_column) %in%
           (
             mobility_df_export %>% 
               group_by(!! as.name(id_column)) %>%
               filter(n_distinct(edge) == 1) %>%
               pull(!! as.name(id_column))
           ))
  )

# ci proper definition per road -------
# Since we now only use start and end, expand throughout the road
# the ci is the same for the whole road

mobility_df_export <- mobility_df_export %>% 
  mutate(ci = ifelse(ci == 0, max(.$ci) + 1, ci)) %>%  # keep 0 as src_pad_idx to avoid confusion
  group_by(!!as.name(id_column),  edge) %>% 
  mutate(unique_cis = n_distinct(ci)) %>% 
  #mutate(no_ci = all(unique(ci) == 1)) %>% 
  mutate(complete_ci = ifelse(unique_cis > 1,ci[ci != -1], ci)) %>% # starting and ending cis are
  # extended to the starting and ending road ids
  mutate(complete_ci = ifelse(complete_ci == -1, 1, complete_ci)) %>% # inbetween_padding_idx, this will 
  # klater scaled
  ungroup() %>% 
  group_by(!! as.name(id_column)) %>%
  arrange_at(c(id_column, "epoch")) %>%
  ungroup() %>% 
  select(X = LONGITUDE, Y = LATITUDE, road_id = edge, batch_id = !! as.name(id_column),
         cell_id = complete_ci)


# Write out new file with edge id -----------
write.table(mobility_df_export, paste0(data_dir, output_dfname), row.names = F, sep = ",")
print("Wrote file")
print(paste0(data_dir, output_dfname))
