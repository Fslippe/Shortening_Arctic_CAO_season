import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from code.model_training.functions import load_and_predict_encoder, process_label_maps, calculate_area_scores_and_plot, process_model_area_mask, get_area_masks, generate_patches

# from functions import generate_patches, get_area_masks, load_and_predict_encoder, process_model_area_mask, calculate_area_scores_and_plot, plot_img_cluster_mask, process_label_maps
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#tf.config.threading.set_inter_op_parallelism_threads(1)
import itertools
from tensorflow.keras.models import load_model
import joblib
# Visualize the result
import json
from autoencoder import SimpleAutoencoder

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,"  , len(logical_gpus), "Logical GPUs")
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

import socket


data_loc = ""


def import_label_data(label_data_file_path, all_dates=False):
    folder_loc = "/labeling_session/npy_files/"
    dates_block = np.load(f"{data_loc}dates_for_labeling/day_filtered/dates_block.npy")
    times_block = np.load(f"{data_loc}dates_for_labeling/day_filtered/times_block.npy")
    if all_dates:
        dates_rest = np.load(f"{data_loc}dates_for_labeling/day_filtered/dates_rest.npy")
        times_rest = np.load(f"{data_loc}dates_for_labeling/day_filtered/times_rest.npy")
        dates = np.append(dates_block, dates_rest)
        times = np.append(times_block, times_rest)
    else:
        dates = dates_block
        times = times_block

    x_cao = []
    masks_cao = []
    lon_lats_cao = []

    #dates, times = dates_block[10:12], times_block[10:12]
    s=0

    for (d, m) in zip(dates, times):
        s+=1
        arr = np.load(f"{folder_loc}MOD021KM.A%s.%s.combined.npy" %(d, m))
        x_cao.append(arr)
        arr = np.load(f"{folder_loc}masks/masks.A%s.%s.combined.npy" %(d, m))
        masks_cao.append(arr)
        arr = np.load(f"{folder_loc}lon_lats/lon_lats.A%s.%s.combined.npy" %(d, m))
        lon_lats_cao.append(arr)

        #print("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/MOD021KM.A%s.%s_combined" %(d, m))
        #idx = np.where((np.array(dates_cao) == d) & (np.array(mod_min_cao) == m))[0][0]
        #np.save("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/masks/masks.A%s.%s.combined.npy" %(d, m), masks_cao[idx])
        #np.save("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/lon_lats/lon_lats.A%s.%s.combined.npy" %(d, m), lon_lats_cao[idx])
        #np.save("/scratch/fslippe/modis/MOD02/labeling_session/npy_files/new_files/MOD021KM.A%s.%s.combined" %(d, m), arr)
        
    max_vals = np.load(f"{data_loc}models/patch_size128/max_val_dnb_l95_z50_ps128_band29_2018-2023.npy")
    min_vals = np.load(f"{data_loc}models/patch_size128/min_val_dnb_l95_z50_ps128_band29_2018-2023.npy")

    with open(label_data_file_path, "r") as f:
        data = json.load(f)["data"]["image_results"]

    labeled_data = pd.json_normalize(data)
    
    return dates, times, labeled_data, x_cao, masks_cao, lon_lats_cao , max_vals, min_vals 


def get_cluster_results(encoded_patches_flat_cao, patch_size, last_filter, n_K):
    #print("cluster load loc:", f"{data_loc}models/patch_size%s/filter%s/clustering/cluster_dnb_l95_z50_ps128_band29_filter%s_K%s.pkl"  %(patch_size, last_filter, last_filter, n_K))
    cluster = joblib.load(f"{data_loc}models/patch_size%s/filter%s/clustering/cluster_dnb_l95_z50_ps128_band29_filter%s_K%s.pkl" %(patch_size, last_filter, last_filter, n_K))
    labels = cluster.predict(encoded_patches_flat_cao)

    global_min = 0
    global_max = n_K 
    return labels, global_min, global_max

def manually_find_cloud_labels(min_vals, max_vals, autoencoder_predict, patch_size, last_filter, n_K):
    x_test1 = np.load(f"{data_loc}cao_examples/radiance_2021080_1120_combined.npy")
    x_test2 = np.load(f"{data_loc}cao_examples/radiance_2023062_1100_combined.npy")
    x_test3 = np.load(f"{data_loc}cao_examples/radiance_2023065_1125_combined.npy")

    x_test4 = np.load(f"{data_loc}cao_examples/MOD021KM.A2019060.1030.combined.npy")
    x_test5 = np.load(f"{data_loc}cao_examples/MOD021KM.A2022347.1150.combined.npy")
    x_test6 = np.load(f"{data_loc}cao_examples/MOD021KM.A2022120.955.combined.npy")
    x_test = ([x_test1, x_test2, x_test3, x_test4, x_test5, x_test6])


    masks_test1 = np.load(f"{data_loc}cao_examples/mask_2021080_1120_combined.npy")
    masks_test2 = np.load(f"{data_loc}cao_examples/mask_2023062_1100_combined.npy")
    masks_test3 = np.load(f"{data_loc}cao_examples/mask_2023065_1125_combined.npy")

    masks_test4 = np.load(f"{data_loc}cao_examples/masks.A2019060.1030.combined.npy")
    masks_test5 = np.load(f"{data_loc}cao_examples/masks.A2022347.1150.combined.npy")
    masks_test6 = np.load(f"{data_loc}cao_examples/masks.A2022120.955.combined.npy")
    masks_test = ([masks_test1, masks_test2, masks_test3, masks_test4, masks_test5, masks_test6])
    

    lon_lats_test1 = np.load(f"{data_loc}cao_examples/lonlat_2021080_1120_combined.npy")
    lon_lats_test2 = np.load(f"{data_loc}cao_examples/lonlat_2023062_1100_combined.npy")
    lon_lats_test3 = np.load(f"{data_loc}cao_examples/lonlat_2023065_1125_combined.npy")

    lon_lats_test4 = np.load(f"{data_loc}cao_examples/lon_lats.A2019060.1030.combined.npy")
    lon_lats_test5 = np.load(f"{data_loc}cao_examples/lon_lats.A2022347.1150.combined.npy")
    lon_lats_test6 = np.load(f"{data_loc}cao_examples/lon_lats.A2022120.955.combined.npy")
    lon_lats_test = ([lon_lats_test1, lon_lats_test2, lon_lats_test3, lon_lats_test4, lon_lats_test5, lon_lats_test6])

    patches_test, all_lon_patches_test, all_lat_patches_test, starts_test, ends_test, shapes_test, n_patches_tot_test, indices_test = generate_patches([x[:,:,0] for x in x_test],
                                                                                                                                            masks_test,
                                                                                                                                            lon_lats_test,
                                                                                                                                            max_vals,
                                                                                                                                            min_vals,
                                                                                                                                            autoencoder_predict,
                                                                                                                                            strides=[1, patch_size, patch_size,1])
    with tf.device('/CPU:0'):   
        
        encoded_patches_flat_cao = load_and_predict_encoder(patch_size, last_filter, patches_test)
        labels, global_min, global_max = get_cluster_results(encoded_patches_flat_cao, patch_size, last_filter, n_K)

    plot_img_cluster_mask(x_test,
                      labels,#, labels_64],
                      masks_test,
                      starts_test,
                      ends_test,
                      shapes_test,
                      indices_test,
                      ["1", "2", "3", "4", "5", "6"],
                      n_patches_tot_test,
                      patch_size,
                      global_min,
                      global_max,
                      index_list=[0,1,2,3,4,5],
                      chosen_label=3,
                      one_fig=True,
                      save=None)

    plt.ion()
                 
    # plt.show()
    open_label = int(input("Open cell label: "))
    # Prompt for closed cell label
    closed_label = int(input("Closed cell label: "))
    plt.ioff()
    plt.close()
    np.save(f"{data_loc}models/patch_size{patch_size}/filter{last_filter}/clustering/cluster_{model_run_name}_filter{last_filter}_K{n_K}_opencell_label", open_label)
    np.save(f"{data_loc}models/patch_size{patch_size}/filter{last_filter}/clustering/cluster_{model_run_name}_filter{last_filter}_K{n_K}_closedcell_label", closed_label)


import sys


bands=[31]#[22, 31]
band_str = ["_" + str(b) for b in bands]
band_str = "".join(band_str) 
band_str = "31"
band_str = "31"
n_Ks = range(7, 17)
patch_size = 128
last_filter = 128
strides = 64
label_filter = ""
size_threshold = 15
try:
    year_threshold = sys.argv[1]
except:
    print("Either change code or add command line argument for year thresholds")
year_threshold = "_20-24"
# save_addon = f"_full_record_500{year_threshold}"
save_addon = f"_full_record_500{year_threshold}"


with open(f"{data_loc}labeled_data/image_results_filip_500.json", "r") as f:
    data = json.load(f)["data"]["image_results"]
labeled_data = pd.json_normalize(data)

# filtered_data = labeled_data[labeled_data["user_id"] == "Filip_500_profession_metorology-position_phd-1736256271078"]
filtered_data = labeled_data[labeled_data["image_id"].str.contains(year_threshold)]

labeled_data = filtered_data
label_filter = ""

folder = "/labeling_data/full_record_500/"

# files = [f for f in os.listdir(folder) if f.endswith("band31.npz")]
# files = [f for f in os.listdir(folder) if f.endswith(f"band{band_str}.npz")]
files = [f for f in os.listdir(folder) if f.endswith(".npz")]

dates = []
times = [] 
x_cao = [] 
masks_cao = [] 
lon_lats_cao = []

for file in files:
    dat_tmp = np.load(folder + file)
    date = file.split(".")[1][1:]
    time = file.split(".")[2]
    # if (date, time) in zip(dates_old, times_old) or (date, str(int(time)+5)) in zip(dates_old, times_old):
    dates.append(date)
    times.append(time)
    x_cao.append(dat_tmp["data"])
    masks_cao.append(np.where(dat_tmp["mask"]>0.1, 0, 1))#np.where(dat_tmp["mask"]==1, 0, 1))
    lon_lats_cao.append(np.array([dat_tmp["lon"], dat_tmp["lat"]]))


index_list = []
for idx in range(len(x_cao)):

    if f"MOD021KM.A{dates[idx]}.{times[idx]}" in np.array(labeled_data["image_id"].str.split("/").str[1].str.split("_").str[0]):
        index_list.append(idx)

labeled_areas = get_area_masks(x_cao,
                                dates,
                                times,
                                masks_cao,
                                labeled_data,
                                subpixel_resolution=2,
                                reduction=strides,
                                index_list=index_list,
                                plot=False,)
                                # min_outside=26,
                                # max_outside=1024+26)

dates_new = []
times_new = []
x_cao_new = []
masks_cao_new = []
lon_lats_cao_new = []
for i, file in enumerate(files):
    if i in index_list:
        dates_new.append(dates[i])
        times_new.append(times[i])
        x_cao_new.append(x_cao[i])
        masks_cao_new.append(masks_cao[i])#np.where(dat_tmp["mask"]==1, 0, 1))
        lon_lats_cao_new.append(lon_lats_cao[i])
        
dates = dates_new
times = times_new
x_cao = x_cao_new
masks_cao = masks_cao_new
lon_lats_cao = lon_lats_cao_new
index_list = range(len(x_cao))



# model_run_name = f"dnb_ice01_l95_z50_ps128_band_31_6filters"
# patch_load_name = f"dnb_ice01_l95_z50_ps128_band{band_str}"

patch_load_name = f"dnb_ice01_l95_z50_ps128_band{band_str}"
model_run_name = f"dnb_ice01_l95_z50_ps128_band31"

max_vals = np.load(f"{data_loc}models/patch_size128/filter128/max_val_{patch_load_name}_2019-2023.npy")
min_vals = np.load(f"{data_loc}models/patch_size128/filter128/min_val_{patch_load_name}_2019-2023.npy")
encoder = load_model(f"{data_loc}models/patch_size128/filter128/encoder_{model_run_name}.h5")

# model_run_name = f"dnb_l95_z50_ps128_band29"
# max_vals = np.load(f"{data_loc}models/patch_size128/filter128/max_val_{model_run_name}_2018-2023.npy")
# min_vals = np.load(f"{data_loc}models/patch_size128/filter128/min_val_{model_run_name}_2018-2023.npy")
# encoder = load_model(f"{data_loc}models/patch_size128/filter128/encoder_dnb_l95_z50_ps128_f128_1e3_201812-202312.h5")


autoencoder_predict = SimpleAutoencoder(len(bands), patch_size, patch_size)

patches_cao, all_lon_patches_cao, all_lat_patches_cao, starts_cao, ends_cao, shapes_cao, n_patches_tot_cao, indices_cao = generate_patches([x for x in x_cao],
                                                                                                                                            masks_cao,
                                                                                                                                            lon_lats_cao,
                                                                                                                                            max_vals,
                                                                                                                                            min_vals,
                                                                                                                                            autoencoder_predict,
                                                                                                                                            strides=[1, strides, strides,1])


with tf.device('/CPU:0'):
    encoded_patches_cao = encoder.predict(patches_cao)
    encoded_patches_flat_cao = encoded_patches_cao.reshape(encoded_patches_cao.shape[0], -1)


for n_K in n_Ks:

    # if not os.path.exists(f"{data_loc}models/patch_size{patch_size}/filter{last_filter}/clustering/cluster_{model_run_name}_filter{last_filter}_K{n_K}_opencell_label.npy"):
    #     manually_find_cloud_labels(min_vals, max_vals, autoencoder_predict, patch_size, last_filter, n_K)
    # if not os.path.exists(f"{data_loc}models/patch_size{patch_size}/filter{last_filter}/clustering/cluster_{model_run_name}_filter{last_filter}_K{n_K}_closedcell_label.npy"):
    #     manually_find_cloud_labels(min_vals, max_vals, autoencoder_predict, patch_size, last_filter, n_K)

    # open_label = np.load(f"{data_loc}models/patch_size{patch_size}/filter{last_filter}/clustering/cluster_{model_run_name}_filter{last_filter}_K{n_K}_opencell_label.npy")
    # closed_label = np.load(f"{data_loc}models/patch_size{patch_size}/filter{last_filter}/clustering/cluster_{model_run_name}_filter{last_filter}_K{n_K}_closedcell_label.npy")
    cao_labels = np.load(f"{data_loc}models/patch_size{patch_size}/filter{last_filter}/clustering/K{n_K}/cluster_{model_run_name}_filter{last_filter}_K{n_K}_cao_labels.npy")
    likely_label = np.load(f"{data_loc}models/patch_size{patch_size}/filter{last_filter}/clustering/K{n_K}/cluster_{model_run_name}_filter{last_filter}_K{n_K}_cao_likely_labels.npy")
    unlikely_label = np.load(f"{data_loc}models/patch_size{patch_size}/filter{last_filter}/clustering/K{n_K}/cluster_{model_run_name}_filter{last_filter}_K{n_K}_cao_unlikely_labels.npy")
    all_labels = np.append(np.append(cao_labels, likely_label), unlikely_label)

    
    combos = []
    for i in range(1, len(all_labels[:len(cao_labels)]) + 1):
        for combo in itertools.combinations(all_labels[:len(cao_labels)], i):
            combos.append(list(combo))

    for i in range(1, len(all_labels[len(cao_labels):]) + 1):
        for combo in itertools.combinations(all_labels[len(cao_labels):], i):
            combos.append(np.append(cao_labels, list(combo)))

    # Using the combinations in your loop
    for combo in combos:
        label_str = "".join([f"_{int(lab)}" for lab in combo])


        # labels, global_min, global_max = get_cluster_results(encoded_patches_flat_cao, patch_size, last_filter, n_K)
        cluster = joblib.load(f"{data_loc}models/patch_size128/filter128/clustering/K{n_K}/cluster_{model_run_name}_filter128_K{n_K}.pkl" )
        labels = cluster.predict(encoded_patches_flat_cao)
        global_min = 0
        global_max = n_K 

        label_map, lon_map, lat_map = process_label_maps(labels,
                                                        all_lon_patches_cao,
                                                        all_lat_patches_cao,
                                                        starts_cao,
                                                        ends_cao,
                                                        shapes_cao,
                                                        indices_cao,
                                                        global_max,
                                                        n_patches_tot_cao,
                                                        patch_size,
                                                        strides,
                                                        combo, 
                                                        [], 
                                                        size_thr_1=size_threshold, 
                                                        size_thr_2=size_threshold)

        # label_map, lon_map, lat_map = process_label_maps(labels

        model_areas = process_model_area_mask(index_list, lon_map, lat_map, indices_cao, label_map, combo, [], plot=False)
        
        ######################### BUg IN SCORE CALCULATION WHERE MISSING LABELED AREA IS STILL APPENDED 
        area_scores = calculate_area_scores_and_plot(model_areas, labeled_areas, dates, times)

        folder = f"{data_loc}models/patch_size{patch_size}/filter{last_filter}/clustering/K{n_K}/scores/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        tot_points = np.sum(area_scores["tot_points"]) 
        print(f"--- K {n_K} --- ")
        print("True:", np.sum(area_scores["area_true_prediction_scores"]) / tot_points)
        # print("TP:", np.sum(area_scores["area_true_positive_scores"]) / tot_points)
        # print("FP:", np.sum(area_scores["area_false_positive_scores"]) / tot_points)
        # print("TN:", np.sum(area_scores["area_true_negative_scores"]) / tot_points)
        # print("FN:", np.sum(area_scores["area_false_negative_scores"]) / tot_points)
        TP = np.sum(area_scores["area_true_positive_scores"])
        FP = np.sum(area_scores["area_false_positive_scores"])
        TN = np.sum(area_scores["area_true_negative_scores"])
        FN = np.sum(area_scores["area_false_negative_scores"])
        precission = TP / (TP + FP)
        TPR = TP / (TP + FN)
        print("TPR:", TPR)
        print("FPR:", FP / (FP + TN))
        print("TNR:", TN / (TN + FP))
        print("FNR:", FN / (FN + TP))
        print("F1:",  2* (TPR * precission) / (precission + TPR))

        print(f"area_scores_cluster_{model_run_name}_filter{last_filter}_K{n_K}_res{strides}_thr{size_threshold}_labels{label_str}")
        np.save(folder + f"area_scores_cluster_{model_run_name}_filter{last_filter}_K{n_K}_res{strides}_thr{size_threshold}_labels{label_str}{label_filter}{save_addon}", area_scores)
        


