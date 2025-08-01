import os
import sys

import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import time
from extract_modis_data import extract_1km_data
from code.model_training.functions import process_pred_info_all_labels, process_label_maps, generate_xy_grid, generate_patches
from autoencoder import SimpleAutoencoder
import sys
from tensorflow.keras.models import load_model
import joblib
from pyproj import Proj, transform
import datetime
from concurrent.futures import ProcessPoolExecutor
if len(sys.argv) > 1:
    year = sys.argv[1]
else:
    year = input("Enter the year: ")

if __name__ == "__main__":
    # Visualize the result
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,"  , len(logical_gpus), "Logical GPUs")
            print("DICT LIST")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    data_loc = "loc_of_data_example/"
    merra_folder = "merra_path_used_for_lon_lat_mesh"



    bands = [31]
    band_str = ["_" + str(b) for b in bands]
    band_str = "".join(band_str) 
    model_run_name = f"dnb_ice01_l95_z50_ps128_band31"
    #encoder = load_model(f"{data_loc}models/winter_2020_21_band(6,20,29)_encoder")
    encoder = load_model(f"{data_loc}models/patch_size128/filter128/encoder_{model_run_name}.h5")
    max_vals = np.load(f"{data_loc}models/patch_size128/filter128/max_val_{model_run_name}_2019-2023.npy")
    min_vals = np.load(f"{data_loc}models/patch_size128/filter128/min_val_{model_run_name}_2019-2023.npy")

    # open_label = np.load(f"{data_loc}models/patch_size{patch_size}/filter{last_filter}/clustering/cluster_{model_run_name}_filter{last_filter}_K{n_K}_opencell_label.npy")
    # closed_label = np.load(f"{data_loc}models/patch_size{patch_size}/filter{last_filter}/clustering/cluster_{model_run_name}_filter{last_filter}_K{n_K}_closedcell_label.npy")
    closed_label = None 
    open_label = None 
    get_no_pred = False       ############### Get none predicted areas 
    get_border_index = False 
    non_cao_swaths = False 
    patch_size = 128
    last_filter = 128
    lon_lat_min_max=[-55, 65, 55, 82]
    
    n_K = 7
    threshold = 0
    multi_labels = [3, 6]
    size_thresholds = [15]#, 0, 25]#, 25, 50, 100, 300]#, 25, 50, 100, 300]   #[0, 25, 50, 100, 300] #[300, 0, 100]#, 100, 300]
    # years = [2002, 2003, 2004,2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]#, 2020, 2021, 2022, 2023]#, 2021, 2022, 2023]
    years = [year]
    stride = 64
    split_up_size = 62#152 ################################################3 MAX 6 FOR GPU RUNS
    raw_data_folder = ""
    base_modis_folder = "base_folder_of_modis_data/"#"
    for i, yr in enumerate(years):
        if i > 0:
            raw_data_folder += " "
        raw_data_folder += base_modis_folder + f"{yr}/"
    time_periods = []
    # time_periods.append(f"20180901-20181231")
    for yr in years:
        time_periods.append(f"{yr}0101-{yr}0531")
        time_periods.append(f"{yr}0901-{yr}1231")


    times_folder = f"{data_loc}models/patch_size{patch_size}/filter{last_filter}/clustering/K{n_K}/cao_date_time_lists/"
    
    import numpy as np

def predict_in_batches(encoder, patches, batch_size=32):
    """
    Predicts using the encoder model in smaller batches to avoid GPU memory issues.
    
    Arguments:
    encoder -- the encoder model
    patches -- the input dataset that needs prediction
    batch_size -- the size of each batch
    
    Returns:
    encoded_patches_flat -- the flattened, encoded patches
    """
    
    num_patches = patches.shape[0]
    encoded_patches_flat = []

    for start in range(0, num_patches, batch_size):
        end = min(start + batch_size, num_patches)
        batch = patches[start:end]
        # with tf.device('/CPU:0'):   
        encoded_batch = encoder.predict(batch, verbose=1)
        encoded_batch_flat = encoded_batch.reshape(encoded_batch.shape[0], -1)
        encoded_patches_flat.append(encoded_batch_flat)
    import time
    time_start = time.time()
    encoded_patches_flat = np.concatenate(encoded_patches_flat, axis=0)
    print("time used to concatenate:", time.time()- time_start)
    return encoded_patches_flat



def extract_for_nKs(dates,
                    time_period,
                    n_K,
                    size_thresholds,
                    threshold,
                    open_label,
                    closed_label,
                    stride=16,
                    multi_labels=None,
                    split_up_index=None,
                    split_up_size=None,
                    get_border_index=True,
                    get_no_pred=False,
                    non_cao_swaths=False):
    
    dates_cao = dates
    # for yr in years:                       

    start_time = time.time()
    dates_extract = np.unique(dates_cao)
    if split_up_index != None:
        split_to = (split_up_index + 1) * split_up_size if (split_up_index + 1) * split_up_size < len(dates_extract) else len(dates_extract)
        dates_extract = dates_extract[split_up_index*split_up_size: split_to]             
    day_time_shift = "0100"

    ds_water_mask=xr.open_dataset(f"{data_loc}land_sea_ice_mask/nimbus/with_lonlat/NSIDC0051_SEAICE_PS_N25km_20200302_v2.0.nc")
    print("EXTRATING SWATHS")
    print(raw_data_folder)
    print(dates_extract)
    x, dates, masks, lon_lats, mod_min = extract_1km_data(raw_data_folder,
                                                            bands=bands,
                                                            ds_water_mask=ds_water_mask,
                                                            date_list=dates_extract,
                                                            return_lon_lat=True,
                                                            data_loc=data_loc,
                                                            data_type="npz",
                                                            combine_pics=False,
                                                            lon_min=-45,
                                                            lon_max=65,
                                                            lat_min=55,
                                                            lat_max=82,
                                                            day_time_shift=day_time_shift)
    print("FINISHED EXTRATING SWATHS")

    x, dates, masks, lon_lats, mod_min = zip(*[(xi, date, mask, lon_lat, mod_min) for xi, date, mask, lon_lat, mod_min in zip(x, dates, masks, lon_lats, mod_min) if (xi.shape[0] > patch_size) and (xi.shape[1] > patch_size)])
    x = list(x)
    print(len(x))
    dates = list(dates)

    print("unique dates", len(np.unique(dates_cao)))
    print("Total images", len((dates_cao)))

    x_cao = x
    dates_cao = dates
    masks_cao = [np.where(mask> 0.1, 0, 1) for mask in masks]
    lon_lats_cao = lon_lats
    mod_min_cao = mod_min
    del x
    del dates
    del masks
    del lon_lats
    del mod_min 
    gc.collect()
    print("TIME used for swath extraction:", time.time()-start_time)
    start_time = time.time()
    autoencoder_predict = SimpleAutoencoder(len(bands), patch_size, patch_size)
    print("GENERATING PATCHES")

    patches, all_lon_patches, all_lat_patches, starts, ends, shapes, n_patches_tot, indices = generate_patches([x for x in x_cao],
                                                                                                        masks_cao,
                                                                                                        lon_lats_cao,
                                                                                                        max_vals,
                                                                                                        min_vals,
                                                                                                        autoencoder_predict,
                                                                                                        lon_lat_min_max=lon_lat_min_max, 
                                                                                                        strides=[1, stride, stride, 1],
                                                                                                        mask_lon_lats=False)     
                                                             

    
    del masks_cao 
    del lon_lats_cao 
    gc.collect()
    print("TIME used for patch extraction:", time.time()-start_time)
    
    start_time = time.time()

    encoded_patches_flat_cao = predict_in_batches(encoder, patches, batch_size=len(patches) // 20 + len(patches) % 20 )

    del patches 
    gc.collect()


    lon = xr.open_dataset(f"{merra_folder}MERRA2_101.const_2d_asm_Nx.00000000.nc4.nc4").lon.values
    lat = xr.open_dataset(f"{merra_folder}MERRA2_101.const_2d_asm_Nx.00000000.nc4.nc4").lat.values
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
    north_polar_stereo = Proj(proj='stere', lat_ts=70, lat_0=90, lon_0=0, k=1, x_0=0, y_0=0, datum='WGS84')
    geographic = Proj(proj='latlong', datum='WGS84')
    lon_lat_res = 100
    lon_mesh, lat_mesh = generate_xy_grid(x_extent=[-3e6, 2.2e6], y_extent=[-4e6, -0.5e6], grid_resolution=lon_lat_res*1e3)
    lon_mesh, lat_mesh = transform(north_polar_stereo, geographic, lon_mesh, lat_mesh)

    global_min = 0 
    global_max = n_K
    cluster = joblib.load(f"{data_loc}models/patch_size128/filter128/clustering/K{n_K}/cluster_{model_run_name}_filter128_K{n_K}.pkl" )
    labels = cluster.predict(encoded_patches_flat_cao)

    del encoded_patches_flat_cao 
    gc.collect()
    print("TIME used for encoding and clustering:", time.time()-start_time)

    start_time = time.time()
    
    all_labels = range(n_K)
    if get_no_pred:
        multi_labels = [item for item in all_labels if item not in multi_labels]
    if multi_labels != None:
        closed_label = multi_labels
        open_label = []
    for size_threshold in size_thresholds:
        label_map, lon_map, lat_map = process_label_maps(labels,
                                                all_lon_patches,
                                                all_lat_patches,
                                                starts,                 
                                                ends,
                                                shapes,         
                                                indices,
                                                n_K,             
                                                n_patches_tot,              
                                                patch_size,
                                                stride,
                                                closed_label, 
                                                open_label, 
                                                size_thr_1=size_threshold if size_threshold != 0 else None, 
                                                size_thr_2=size_threshold if size_threshold != 0 else None)

        dict_list = []

        # Setup arguments for parallel processing
        args = []
        for i in range(len(x_cao)):
            args.append((dates_cao[i], mod_min_cao[i], lon_map[i], lat_map[i], label_map[i], all_labels, lon_mesh, lat_mesh))
        # Perform parallel processing
        with ProcessPoolExecutor(max_workers=len(x_cao) if len(x_cao) < 128 else 128) as executor:
            dict_list = list(executor.map(process_pred_info_all_labels, args))

        del label_map
        del lon_map
        del lat_map
        del args
        # Save results to file
        # if len(years) > 1:
        #     np.save(f"{data_loc}model_pred_info/filter{last_filter}/dict_{model_run_name}_filter{last_filter}_nK{n_K}_caothr{threshold}_sizethr{size_threshold}_stride{stride}_{years[0]}-{years[-1]}", dict_list)
        # else:
        labels_str = ""
        for l in multi_labels:
            labels_str += f"_{l}"
        folder = f"{data_loc}model_pred_info/filter{last_filter}/{model_run_name}_filter{last_filter}_nK{n_K}/caothr{threshold}/sizethr{size_threshold}/labels{labels_str}"
        os.makedirs(folder, exist_ok=True)
        file_path = f"{folder}/dict_all_labels_{model_run_name}_filter{last_filter}_nK{n_K}_caothr{threshold}_sizethr{size_threshold}_stride{stride}_res{lon_lat_res}_{time_period}"
        count = 1
        if split_up_index > 0:
            new_file_path = file_path + f"_{split_up_index}"
        else:
            new_file_path = file_path


        # Save the data to the new file name
        print("SAVING AT", new_file_path)
        np.save(new_file_path, dict_list)

    print("TIME used for label_map and dict_list extraction:", time.time()-start_time)
    

    del x_cao 
    del dates_cao 
    del mod_min_cao 
    gc.collect()

if __name__ == "__main__":

    print(time_periods)
    for time_period in time_periods:
        start, end = time_period.split('-')
        if start == "20000101": 
            start = "20000224" 
        start_date = datetime.datetime.strptime(start, "%Y%m%d")
        end_date = datetime.datetime.strptime(end, "%Y%m%d")
        
        day_count = (end_date - start_date).days + 1
        dates = [(start_date + datetime.timedelta(days=day)).strftime("%Y%j") for day in range(day_count)]

        for split_up_index in range(151 // split_up_size + 1): 
            # try:
            extract_for_nKs(dates,
                            time_period,
                            n_K,
                            size_thresholds,
                            threshold,
                            closed_label,
                            open_label,
                            stride,
                            multi_labels,
                            split_up_index,
                            split_up_size,
                            get_border_index,
                            get_no_pred=get_no_pred,
                            non_cao_swaths=non_cao_swaths)
            # except:
            #     print("failed for split up index:", split_up_index )
            #     continue

print("--- FINISHED ---")
