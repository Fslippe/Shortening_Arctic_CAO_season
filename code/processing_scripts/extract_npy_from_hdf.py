import os
import sys
import numpy as np
from pyhdf.SD import SD, SDC
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm

def combine_images_based_on_time_2(ds_all, dates, lon_lats, mod_min):
    combined_ds = []
    combined_dates = []
    combined_lon_lats = []
    # Sort everything based on dates and mod_min
    sorted_indices = sorted(range(len(dates)), key=lambda x: (dates[x], mod_min[x]))
    ds_all = [ds_all[i] for i in sorted_indices]
    dates = [dates[i] for i in sorted_indices]
    lon_lats = [lon_lats[i] for i in sorted_indices]
    mod_min = [mod_min[i] for i in sorted_indices]
    combined_mod_min = []
   
    i = 0

    while i < len(dates) - 1:

        imgs_to_combine = [ds_all[i]]
        lon_lats_to_combine = [lon_lats[i]]
        date = dates[i]
        min_time = mod_min[i]
        mod_min_start = min_time

        while i < len(dates) - 1 and ((mod_min[i+1] - min_time) == 5 or (mod_min[i+1] % 100 == 0 and ((min_time+45) % 100 == 0))):#or (min_time == 2355 and mod_min[i+1] == 0)):
            
            imgs_to_combine.append(ds_all[i+1])
            lon_lats_to_combine.append(lon_lats[i+1])
            i += 1
            min_time = mod_min[i]

        # Find overlapping columns
        num_columns = imgs_to_combine[0].shape[1]

        # Initialize the mask as all True
        full_final_valid_cols_mask = np.ones(num_columns, dtype=bool)


        # Use the mask to extract the overlapping columns
        combined_mod_min.append(mod_min_start)
        try:
            combined_ds.append(np.vstack([img[:, full_final_valid_cols_mask] for img in imgs_to_combine]))
            combined_dates.append(date)  # Taking the first date
            combined_lon_lats.append(np.concatenate([ll[:,:, full_final_valid_cols_mask] for ll in lon_lats_to_combine], axis=1))
        except:
            print(f"failed on image date {date} min {min_time}")
            print("TRYING ANYWAY")
            combined_ds.append(np.vstack([img[:, full_final_valid_cols_mask] for img in imgs_to_combine]))
            combined_dates.append(date)  # Taking the first date
            combined_lon_lats.append(np.concatenate([ll[:,:, full_final_valid_cols_mask] for ll in lon_lats_to_combine], axis=1))
        i += 1

    print(len(combined_mod_min))
    print(len(combined_dates))

    # For the last image if it's standalone
    if len(imgs_to_combine) == 1:
        combined_mod_min.append(mod_min[-1])
        combined_ds.append(ds_all[-1]) 
        combined_lon_lats.append(lon_lats[-1]) 
        combined_dates.append(dates[-1])
        # combined_masks.append(masks[-1][valid_cols_to_combine[0]])
        # combined_lon_lats.append(lon_lats[-1][valid_cols_to_combine[0]])

    return combined_ds, combined_dates, combined_lon_lats, combined_mod_min

from datetime import datetime, timedelta
import os

def parse_timestamp_from_filename(filename, pattern="%Y%j%H%M"):
    """
    Extracts and parses the timestamp from the given filename using the provided pattern.
    """
    try:
        # Split the filename on dots and take the necessary parts
        # The file name is in the form MOD021KM.AYYYYDOY.HHMM.OTHER.INFO.hdf
        parts = filename.split(".")
        date_part = parts[1][1:]  # Taking 'YYYYDOY' part after 'A'
        time_part = parts[2]  # Taking 'HHMM' part
        timestamp_str = date_part + time_part  # Concatenate date and time parts
        
        # Parse the timestamp string into a datetime object
        timestamp = datetime.strptime(timestamp_str, pattern)
        return timestamp
    except ValueError as e:
        raise ValueError(f"Error parsing timestamp from filename: {filename}, error: {e}")


def group_files_by_time(files, time_delta=timedelta(minutes=5)):
    """
    Groups files based on the time difference between their timestamps.

    :param files: List of file paths
    :param time_delta: The maximum time delta between file timestamps for them to be grouped
    :return: List of file groups
    """
    # First, sort the files by their timestamps
    files.sort(key=lambda x: parse_timestamp_from_filename(os.path.basename(x)))
    # Then, group the files
    file_groups = []
    current_group = [files[0]]

    for file in files[1:]:
        current_file_timestamp = parse_timestamp_from_filename(os.path.basename(file))
        last_file_in_group = current_group[-1]
        last_file_timestamp = parse_timestamp_from_filename(os.path.basename(last_file_in_group))

        if current_file_timestamp - last_file_timestamp <= time_delta:
            # If the current file is within the time_delta, add it to the current group
            current_group.append(file)
        else:
            # If it's not, start a new group and add the current file to it
            file_groups.append(current_group)
            current_group = [file]

    # Make sure to add the last group if it's not already added
    if current_group not in file_groups:
        file_groups.append(current_group)
    print(file_groups)
    
    return file_groups


def get_all_files_in_folders(folders):
    all_files = []
    for folder in folders:
        all_files.extend([os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.hdf')])
    return all_files

def process_hdf_file(file, key, idx, band, attrs, base_save_folder="/scratch/fslippe/modis/MOD02/"):
    filename = file.split("/")[-1][:-4]
    year = filename.split(".")[1][1:5]
    file_loc = f"{base_save_folder}{year}/{filename}_ll_band_{band}.npz"

    if not os.path.exists(file_loc):
        try:
            hdf = SD(file, SDC.READ)
            data = hdf.select(key)[:][idx]
            data = np.where(data == attrs["_FillValue"], np.nan, data)
            out_of_range = np.where(data > attrs["valid_range"][1])
            data = np.float32((data - attrs["radiance_offsets"][idx]) * attrs["radiance_scales"][idx])
            data[out_of_range] = np.nan
            lat = hdf.select("Latitude")[:]
            lon = hdf.select("Longitude")[:]

            data_dict = {
                'lon': lon,
                'lat': lat,
                'data': data
            }
            np.savez_compressed(file_loc, lon=lon, lat=lat, data=data)

            #np.save(file_loc, data_dict)
        except:
            print("FAILED ON HDF FILE", file)



def process_hdf_files_combos(files, keys, idxs, bands, attrs_list, base_save_folder="/scratch/fslippe/modis/MOD02/"):
    combined_data = []  # To hold the stacked arrays of data
    combined_lat = []
    combined_lon = []
    filenames = []
    for file in files:
        filename = file.split("/")[-1][:-4]
        year = filename.split(".")[1][1:5]
        try:
            hdf = SD(file, SDC.READ)
        except Exception as e:
            print(f"Failed to open file: {file}. Error: {e}")
            sys.exit(1)  # Exit the script with an error code

        datas = []
        for idx, key, attrs in zip(idxs, keys, attrs_list):

            data = hdf.select(key)[:][idx]
            data = np.where(data == attrs["_FillValue"], np.nan, data)
            out_of_range = np.where(data > attrs["valid_range"][1])
            offsets = attrs["radiance_offsets"][idx]
            scales = attrs["radiance_scales"][idx]

            data = np.float32((data - offsets) * scales)
            data[out_of_range] = np.nan
            datas.append(data)
        data = np.stack(datas, axis=0)
        lat = hdf.select("Latitude")[:]
        lon = hdf.select("Longitude")[:]
        data = np.transpose(data, (1, 2, 0))
        combined_data.append(data)
        combined_lat.append(lat)
        combined_lon.append(lon)
        filenames.append(filename)

    ndims = data.ndim

    if combined_data:
        data = np.vstack(combined_data)
        lat = np.vstack(combined_lat)
        lon = np.vstack(combined_lon)
        band_str = ["_" + str(band) for band in bands]
        band_str = "".join(band_str)
        filename = filenames[0]
        year = filename.split(".")[1][1:5]
        file_loc = f"{base_save_folder}{year}/band{band_str}/{filename}_ll_band{band_str}.npz"
        np.savez_compressed(file_loc, lon=lon, lat=lat, data=data)
        print(f"Saved {len(files)} swath combined data to {file_loc}")
    

from functools import partial

def process_func(file, key, idx, band, attrs, base_save_folder):
    return process_hdf_file(file, key, idx, band, attrs, base_save_folder)



def process_files_parallel(files, key, idx, band, attrs, base_save_folder, workers=4):      
    process_partial = partial(process_func, key=key, idx=idx, band=band, attrs=attrs, base_save_folder=base_save_folder)
    for _ in executor.map(process_partial, files):
        pbar.update(1)
    # with ProcessPoolExecutor(max_workers=workers) as executor, tqdm(total=len(files), desc="Processing Files") as pbar:
    #     for _ in executor.map(lambda file: process_hdf_file(file, key, idx, band, attrs, base_save_folder), files):
    #         pbar.update(1)

# def process_files_serial(files, key, idx, band, attrs, base_save_folder):
#     with tqdm(total=len(files), desc="Processing Files") as pbar:
#         for file in files:
#             process_hdf_file(file, key, idx, band, attrs, base_save_folder)
#             pbar.update(1)


def process_files_serial(files, key, idx, band, attrs, base_save_folder):
    # with tqdm(total=len(files), desc="Processing Files") as pbar:
    file_groups = group_files_by_time(files)#[2016 +153:]
    
    with tqdm(total=len(file_groups), desc="Processing Files") as pbar:
        for file_group in file_groups:
            process_hdf_files_combos(file_group, key, idx, band, attrs, base_save_folder)
            pbar.update(1)

def run_parallelized_over_folders(folders):

    base_save_folder = "/nird/projects/NS9600K/data/modis/cao/MOD02_npz/"
    all_files = get_all_files_in_folders(folders)[:]#[1000:]
    
    # date_list = ["2021080", "2023062", "2023065", "2019060", "2022347", "2022120"]
    date_list = [f.split(".")[1][1:] for f in os.listdir("/nird/projects/NS9600K/fslippe/mimi_backup/labeling_session/npy_files/") if f.endswith("npy")]

    final_files = []
    for f in all_files:
        date = f.split("/")[-1].split(".")[1][1:]
        if date in date_list:
            final_files.append(f)
    all_files = final_files
    length = (len(all_files))
    print(length)
    #all_files_2 = all_files[length // 2:]
    #all_files = all_files[:length // 2]
    #print(len(all_files_2), len(all_files)) 
    key = "EV_1KM_Emissive"
    hdf_attrs = SD(all_files[0], SDC.READ) 
    attrs = hdf_attrs.select(key).attributes()
    idx = [(np.where(np.array(attrs["band_names"].split(",")) == "%s" %band)[0][0]) for band in bands]

    # Assuming attrs are the same for all files, using the first file to get attributes
    
    process_files_serial(all_files, key, idx, bands, attrs, base_save_folder)


def main():
    year = 2024
    bands = [31]
    band_str = ["_" + str(band) for band in bands]
    band_str = "".join(band_str)
    folders = [f"/loc_of_modis_hdf-files/{year}/"]
    
    base_save_folder = "loc_to_save_modis_npz_files/"
    all_files = get_all_files_in_folders(folders)[:]#[1000:]
    all_files = [f for f in all_files if f.split("/")[-1].split(".")[1][1:5] == str(year)]
    hdf = SD(all_files[0], SDC.READ)

    length = (len(all_files))
    print(length)
    list1 = [int(num_str) for num_str in hdf.select("EV_250_Aggr1km_RefSB").attributes()["band_names"].split(",")]
    list2 = [int(num_str) for num_str in hdf.select("EV_500_Aggr1km_RefSB").attributes()["band_names"].split(",")]
    list3 = [int(num_str) for num_str in hdf.select("EV_1KM_RefSB").attributes()["band_names"].split(",") if num_str.isdigit()]
    list4 = [int(num_str) for num_str in hdf.select("EV_1KM_Emissive").attributes()["band_names"].split(",")]

    file_layers = np.empty(36, dtype=object)
    for i, (band) in enumerate(list1):
        file_layers[band-1] = {"EV_250_Aggr1km_RefSB": i}
    for i, (band) in enumerate(list2):
        file_layers[band-1] = {"EV_500_Aggr1km_RefSB": i}    
    for i, (band) in enumerate(list3):
        file_layers[band-1] = {"EV_1KM_RefSB": i}
    for i, (band) in enumerate(list4):
        file_layers[band-1] = {"EV_1KM_Emissive": i}
    hdf.end()
    keys = []
    idxs = []
    for j, (band) in enumerate(bands):
        key = list(file_layers[band-1].keys())[0]
        idx = list(file_layers[band-1].values())[0]
        keys.append(key)
        idxs.append(idx)

    print(keys)
    # key = "EV_1KM_Emissive"
    hdf_attrs = SD(all_files[0], SDC.READ) 
    attrs_list = []
    for key in keys:
        attrs = hdf_attrs.select(key).attributes()
        attrs_list.append(attrs)
    # idx = [(np.where(np.array(attrs["band_names"].split(",")) == "%s" %band)[0][0]) for band in bands]

    # Assuming attrs are the same for all files, using the first file to get attributes
    os.makedirs(f"{base_save_folder}{year}/band{band_str}", exist_ok=True)
    process_files_serial(all_files, keys, idxs, bands, attrs_list, base_save_folder)


if __name__ == "__main__":
    main()

