from pyhdf.SD import SD, SDC
import numpy as np 
import os
import sys
import socket
import xarray as xr
#from extract_modis_data import extract_1km_data, process_key, combine_images_based_on_time, extract_250m_data, replace_out_of_bounds_with_nearest, process_file, process_hdf_file, process_npy_file, append_data, normalize_data, create_nc_from_extracted_data, parse_date


from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from collections import defaultdict
import gc
from functions import convert_to_date
import multiprocessing
from scipy.ndimage import zoom
# from create_water_mask import * 
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
import socket
import calendar


total_cores = multiprocessing.cpu_count()
print("total cores:", total_cores)

def extract_1km_data(folder="folder_of_modis_files",
                     bands = [6,7,20,28,28,31],
                     ds_water_mask=None,
                     save=None,
                     start_date=None,
                     end_date=None,
                     date_list=None,
                     return_lon_lat=False,
                     workers=None,
                     max_zenith=47.4,
                     combine_pics=True,
                     data_loc="data_loc_including_nimbus_sea_ice_mask_if_used/",
                     data_type="hdf",
                     specified_date_min_zip=None,
                     lon_min=-45,
                     lon_max=55,
                     lat_min=55,
                     lat_max=82,
                     day_time_shift=None):
    
    date_list = np.unique(np.array(date_list))
    if ds_water_mask is None:
        ds_water_mask = xr.open_dataset("%sland_sea_ice_mask/sea_land_mask.nc" %data_loc)
        water_mask_ravel = ds_water_mask["sea_ice_region_surface_mask"].values.ravel()
        coords_lowres = np.vstack((ds_water_mask.latitude.values.ravel(), ds_water_mask.longitude.values.ravel())).T
        tree = cKDTree(coords_lowres)
    else:
        water_mask_ravel = None
        coords_lowres = np.vstack((ds_water_mask.lat.values.ravel(), ds_water_mask.lon.values.ravel())).T
        tree = cKDTree(coords_lowres)
    all_files = []

    
    folders = folder.split(" ")

    for f in folders:
        all_files.extend([os.path.join(f, file) for file in os.listdir(f) if file.endswith(data_type if data_type != "mod06" else "hdf")])

    if data_type == "hdf":
        hdf = SD(all_files[0], SDC.READ)

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
        
    elif data_type == "mod06":
        hdf = SD(all_files[0], SDC.READ)
        file_layers = "Cloud_Water_Path"
        bands = None
        hdf.end()


    elif data_type == "npy":
        file_layers = None
    elif data_type == "npz":
        file_layers = None
    file_groups = defaultdict(list)
    mod_mins = defaultdict(list)


    for file in all_files:
        minute = int(file.split(".")[2])
        date = str(file.split('.')[1][1:])  # This will give e.g., '2021120' for 'MOD02QKM.A2021120'
        if specified_date_min_zip:
            if (date, minute) in specified_date_min_zip:
                file_groups[date].append(file)
                mod_mins[date].append(minute)
        else:
            if day_time_shift is not None:
                day_time_shift = int(day_time_shift)
                if minute >= day_time_shift:
                    file_groups[date].append(file)
                    mod_mins[date].append(minute)
                else:
                    day = date[-3:]
                    year = int(date[:4])
                    if day == "001":
                        # Calculating the last day of the previous year
                        day_before = "366" if calendar.isleap(year - 1) else "365"
                        year -= 1  # Go to the previous year
                    else:
                        day_before = str(int(day) - 1).zfill(3)

                    date_before = f"{year}{day_before}"  # Proper string formatting
                    file_groups[date_before].append(file)
                    mod_mins[date_before].append(minute)

            else:
                file_groups[date].append(file)
                mod_mins[date].append(minute)
    print(mod_mins)
    sorted_keys = sorted(file_groups.keys(), key=int)  # Convert keys to integers for sorting

    # Extract the keys between start and end dates
    
    if start_date == None and end_date == None:
        # print(sorted_keys)
        selected_keys = [key for key in date_list if key in sorted_keys]
        # print(selected_keys)
    else:
        selected_keys = [key for key in sorted_keys if int(start_date) <= int(key) <= int(end_date)]
    #all_files = os.listdir(folder)[16:18]

    #X = np.empty((len(all_files), 2030, 1354, len(bands)))
    # if day_time_shift is not None:
    #     extra_keys = []
    #     for selected_key in selected_keys:
    #         day = selected_key[-3:]
    #         if day != "365" or day != "366":
    #             next_day_key = str(int(selected_key) + 1)
    #             if next_day_key not in selected_keys:
    #                 extra_keys.append(next_day_key)
    #         else: 

    #             next_day_key = str(int(selected_key[:4]) + 1) + "001"
    #             if next_day_key not in selected_keys:
    #                 extra_keys.append(next_day_key)

    #     is_extra_key = [-int(day_time_shift)]*len(selected_keys) + [int(day_time_shift)]*len(extra_keys)
    #     selected_keys = selected_keys + extra_keys

    # else:
    #     is_extra_key = [0]*len(selected_keys)


    print(mod_mins)
    ds_all = []
    ##### FOLLOWING CODE IS USED IF PARALLELIZING OVER EACH DATE
    # if save == None:
    #     for key in list(selected_keys):
    #         file_group = file_groups[key]
    #         print("Date:", convert_to_standard_date(key))
    #         with ProcessPoolExecutor(max_workers=len(file_group)) as executor:
    #             X = list(tqdm(executor.map(append_data, [folder]*len(file_group), file_group, [file_layers]*len(file_group), [bands]*len(file_group), [min_mean]*len(file_group), [normalize]*len(file_group)), total=len(file_group)))
    #         ds_all.extend([xi for xi in X if xi.ndim>1])
    #     return ds_all
    

    ##### FOLLOWING CODE IS USED IF PARALLELIZING all files
    if workers == None:
        if len(selected_keys) < 10:
            workers = len(selected_keys)
        else:
            workers = 128
    if workers == 1:        
        if save is None:
            results = []
            for key in tqdm(selected_keys, total=len(selected_keys)):
                
                result = process_key(
                    key,
                    file_groups,
                    file_layers,
                    bands,
                    water_mask_ravel,
                    tree,
                    mod_mins,
                    return_lon_lat,
                    max_zenith,
                    lon_min,
                    lon_max,
                    lat_min,
                    lat_max,
                )
                results.append(result)
            
            if return_lon_lat:
                ds_all, dates, masks, lon_lats, mod_min, valid_cols_lon = zip(*results)
                ds_all = [item for sublist in ds_all for item in sublist]  # Flatten the list
                dates = [item for sublist in dates for item in sublist]
                masks = [item for sublist in masks for item in sublist]
                lon_lats = [item for sublist in lon_lats for item in sublist]
                mod_min = [item for sublist in mod_min for item in sublist]
                valid_cols_lon = [item for sublist in valid_cols_lon for item in sublist]

                if combine_pics:
                    ds_all, dates, masks, lon_lats, mod_min = combine_images_based_on_time(ds_all, dates, masks, lon_lats, mod_min, valid_cols_lon)

                return ds_all, dates, masks, lon_lats, mod_min

            else:
                ds_all, dates, masks, valid_cols_lon = zip(*results)
                ds_all = [item for sublist in ds_all for item in sublist]  # Flatten the list
                dates = [item for sublist in dates for item in sublist]
                masks = [item for sublist in masks for item in sublist]
                return ds_all, dates, masks
    else:
        if save == None:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(
                    tqdm(
                        executor.map(
                            process_key, 
                            selected_keys, 
                            [file_groups] * len(selected_keys),
                            [file_layers] * len(selected_keys),
                            [bands] * len(selected_keys),
                            [water_mask_ravel] * len(selected_keys),
                            [tree] * len(selected_keys),
                            [mod_mins] * len(selected_keys),
                            [return_lon_lat] * len(selected_keys),
                            [max_zenith] * len(selected_keys),
                            [lon_min]*len(selected_keys),
                            [lon_max]*len(selected_keys),
                            [lat_min]*len(selected_keys),
                            [lat_max]*len(selected_keys),
                        ), 
                        total=len(selected_keys)
                    )
                )
            if return_lon_lat:
                ds_all, dates, masks, lon_lats, mod_min, valid_cols_lon = zip(*results)
                ds_all = [item for sublist in ds_all for item in sublist]  # Flatten the list
                dates = [item for sublist in dates for item in sublist]
                masks = [item for sublist in masks for item in sublist]
                lon_lats = [item for sublist in lon_lats for item in sublist]
                mod_min = [item for sublist in mod_min for item in sublist]
                valid_cols_lon = [item for sublist in valid_cols_lon for item in sublist]


                if combine_pics:
                    ds_all, dates, masks, lon_lats, mod_min = combine_images_based_on_time(ds_all, dates, masks, lon_lats, mod_min, valid_cols_lon)
                
                return ds_all, dates, masks, lon_lats, mod_min
            
            else:
                ds_all, dates, masks, valid_cols_lon = zip(*results)
                ds_all = [item for sublist in ds_all for item in sublist]  # Flatten the list
                dates = [item for sublist in dates for item in sublist]
                masks = [item for sublist in masks for item in sublist]

                return ds_all, dates, masks

def process_key(key, file_groups, file_layers, bands, full_water_mask, tree, mod_mins, return_lon_lat=False, max_zenith=47.4, lon_min=-35, lon_max=45, lat_min=60, lat_max=82):
    file_group = file_groups[key]
    selected_mod_mins  = mod_mins[key]

    X = []
    dates = []
    masks = []
    lon_lats = []
    mod_min_list = []
    valid_cols_lon_list = []
    
    if return_lon_lat:
        for (file, mod_min) in zip(file_group, selected_mod_mins):
            result, mask, lon_lat, valid_cols_lon = process_file(file, file_layers, bands, full_water_mask, tree, return_lon_lat, max_zenith, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)

            if result.shape[0] > 1 and result.shape[1] > 1:
                X.append(result) 
                dates.append(key)
                masks.append(mask)
                lon_lats.append(lon_lat)
                mod_min_list.append(mod_min)
                valid_cols_lon_list.append(valid_cols_lon)
        
        return X, dates, masks, lon_lats, mod_min_list, valid_cols_lon_list

    else:
        for file in file_group:
            result, mask,valid_cols_lon = process_file(file, file_layers, bands, full_water_mask, tree, return_lon_lat, max_zenith, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)

            if  result.shape[0] > 1 and result.shape[1] > 1:
                X.append(result) 
                dates.append(key)
                masks.append(mask)
                valid_cols_lon_list.append(valid_cols_lon)

        return X, dates, masks, valid_cols_lon_list

    
def combine_images_based_on_time(ds_all, dates, masks, lon_lats, mod_min, valid_cols_lon):
    combined_ds = []
    combined_dates = []
    combined_masks = []
    combined_lon_lats = []
    # Sort everything based on dates and mod_min
    sorted_indices = sorted(range(len(dates)), key=lambda x: (dates[x], mod_min[x]))
    ds_all = [ds_all[i] for i in sorted_indices]
    dates = [dates[i] for i in sorted_indices]
    masks = [masks[i] for i in sorted_indices]
    lon_lats = [lon_lats[i] for i in sorted_indices]
    mod_min = [mod_min[i] for i in sorted_indices]
    valid_cols_lon = [valid_cols_lon[i] for i in sorted_indices]
    combined_mod_min = []
   
    i = 0

    while i < len(dates) - 1:

        imgs_to_combine = [ds_all[i]]
        masks_to_combine = [masks[i]]
        lon_lats_to_combine = [lon_lats[i]]
        valid_cols_to_combine = [valid_cols_lon[i]]
        date = dates[i]
        min_time = mod_min[i]
        mod_min_start = min_time

        while i < len(dates) - 1 and ((mod_min[i+1] - min_time) == 5 or (mod_min[i+1] % 100 == 0 and ((min_time+45) % 100 == 0))):#or (min_time == 2355 and mod_min[i+1] == 0)):
            
            imgs_to_combine.append(ds_all[i+1])
            masks_to_combine.append(masks[i+1])
            lon_lats_to_combine.append(lon_lats[i+1])
            valid_cols_to_combine.append(valid_cols_lon[i+1])
            i += 1
            min_time = mod_min[i]

        # Find overlapping columns
        num_columns = imgs_to_combine[0].shape[1]

        # Initialize the mask as all True
        full_final_valid_cols_mask = np.ones(num_columns, dtype=bool)

        # Update the mask for each valid_cols
        for valid_cols in valid_cols_to_combine:
            full_final_valid_cols_mask &= valid_cols
        
        full_final_valid_cols_mask = np.any(valid_cols_to_combine, axis=0)

        # Use the mask to extract the overlapping columns
        combined_mod_min.append(mod_min_start)
        try:
            combined_ds.append(np.vstack([img[:, full_final_valid_cols_mask] for img in imgs_to_combine]))
            combined_dates.append(date)  # Taking the first date
            combined_masks.append(np.vstack([mask[:, full_final_valid_cols_mask] for mask in masks_to_combine]))
            combined_lon_lats.append(np.concatenate([ll[:,:, full_final_valid_cols_mask] for ll in lon_lats_to_combine], axis=1))
        except:
            print(f"failed on image date {date} min {min_time}")
            print("TRYING ANYWAY")
            combined_ds.append(np.vstack([img[:, full_final_valid_cols_mask] for img in imgs_to_combine]))
            combined_dates.append(date)  # Taking the first date
            combined_masks.append(np.vstack([mask[:, full_final_valid_cols_mask] for mask in masks_to_combine]))
            combined_lon_lats.append(np.concatenate([ll[:,:, full_final_valid_cols_mask] for ll in lon_lats_to_combine], axis=1))
        i += 1

    print(len(combined_mod_min))
    print(len(combined_dates))

    # For the last image if it's standalone
    if len(imgs_to_combine) == 1:
        combined_mod_min.append(mod_min[-1])
        combined_ds.append(ds_all[-1][:, valid_cols_to_combine[0]]) 
        combined_masks.append(masks[-1][:, valid_cols_to_combine[0]]) 
        combined_lon_lats.append(lon_lats[-1][:,:, valid_cols_to_combine[0]]) 
        combined_dates.append(dates[-1])
        # combined_masks.append(masks[-1][valid_cols_to_combine[0]])
        # combined_lon_lats.append(lon_lats[-1][valid_cols_to_combine[0]])

    return combined_ds, combined_dates, combined_masks, combined_lon_lats, combined_mod_min


def replace_out_of_bounds_with_nearest(data, low_bound, high_bound):
    out_of_bounds = ~((data >= low_bound) & (data <= high_bound))

    if np.any(out_of_bounds):

        # Compute the distances to the nearest valid value
        distances, (i, j) = distance_transform_edt(out_of_bounds, return_indices=True)
        
        # Use the indices to get the values from the nearest valid value
        data[out_of_bounds] = data[i[out_of_bounds], j[out_of_bounds]]

    return data


def process_file(file, file_layers, bands, full_water_mask=None, tree=None, return_lon_lat=False, max_zenith=47.4, lon_min=-35, lon_max=45, lat_min=60, lat_max=82, data_loc="data_loc_including_nimbus_sea_ice_mask_if_used/", npz_compressed=False):
    if file.endswith(".hdf"):
        return process_hdf_file(file, file_layers, bands, max_zenith, data_loc, full_water_mask, tree, return_lon_lat, lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)
    elif file.endswith(".npy"):
        return process_npy_file(file, file_layers, bands, max_zenith, data_loc, full_water_mask, tree, return_lon_lat,  lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max)
    elif file.endswith(".npz"):
        return process_npy_file(file, file_layers, bands, max_zenith, data_loc, full_water_mask, tree, return_lon_lat,  lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max, npz_compressed=True)
    else:
        raise ValueError(f"Unsupported file format: {file}")

def process_hdf_file(file, file_layers, bands, max_zenith, data_loc, full_water_mask, tree, return_lon_lat, lon_min=-35, lon_max=45, lat_min=60, lat_max=82):
    zenith = np.load("%sland_sea_ice_mask/sensor_zenith_bilinear_1km.npy" %data_loc)
    zenith_mask = zenith < max_zenith
    current_data_list = []

    hdf = SD(file, SDC.READ)

    #### IF FILE LAYER IS A STRING DO NOT INDEX ....

    try:
        dty = file_layers.get("dtype")
    except:
        dty = "mod06"
    if dty != "mod06":
        key = list(file_layers[bands[0]-1].keys())[0]
        idx = list(file_layers[bands[0]-1].values())[0]
        data = hdf.select(key)[:][idx]
        mod06 = False
    else:
        key = list(file_layers[bands[0]-1].keys())[0]
        idx = list(file_layers[bands[0]-1].values())[0]
        #key = file_layers 
        data = hdf.select(key)[:][idx]
        mod06 = True
        mod06 = False

    
    lat = hdf.select("Latitude")[:]
    lon = hdf.select("Longitude")[:]

    lat = replace_out_of_bounds_with_nearest(lat, -90, 90)
    lon = replace_out_of_bounds_with_nearest(lon, -180, 180)

    
    
    mask_lowres = (lat >= lat_min) & (lat <= lat_max) & (lon > lon_min) & (lon < lon_max)
    zoom_factor_y = data.shape[0] / lat.shape[0]
    zoom_factor_x = data.shape[1] / lat.shape[1]
    zoom_factors = (zoom_factor_y, zoom_factor_x)  # Now considering only 2 dimensions
    
        
    mask_highres = zoom(mask_lowres, zoom_factors, order=0)  # nearest neighbor interpolation
    lat_highres = zoom(lat, zoom_factors, order=1)  # bilinear interpolation
    lon_highres = zoom(lon, zoom_factors, order=1) 
    valid_rows_ll = np.any(mask_highres , axis=1)
    valid_cols_ll = zenith_mask
    valid_cols_lon = np.any(mask_highres[valid_rows_ll][:,valid_cols_ll] , axis=0)  ### remove last for matching x-axis
    data = data[valid_rows_ll][:, valid_cols_ll]
    if mod06:
        SolarZenith = hdf.select("Solar_Zenith")[:]
        SolarZenith_attrs = hdf.select("Solar_Zenith").attributes()
        SolarZenith = np.where((SolarZenith > SolarZenith_attrs["valid_range"][1]) | (SolarZenith < SolarZenith_attrs["valid_range"][0]), 99999, SolarZenith)
        SolarZenith = zoom(SolarZenith, zoom_factors, order=0)  
        SolarZenith = SolarZenith*SolarZenith_attrs["scale_factor"]
        SolarZenith = SolarZenith[valid_rows_ll][:, valid_cols_ll]

    lat_highres = lat_highres[valid_rows_ll][:, valid_cols_ll] 
    lon_highres = lon_highres[valid_rows_ll][:, valid_cols_ll] 

    #coords_lowres = np.column_stack((full_water_mask.latitude.values.ravel(), full_water_mask.longitude.values.ravel()))

    coords_highres = np.column_stack((lat_highres.ravel(), lon_highres.ravel()))
    distances, indices = tree.query(coords_highres, k=1,  eps=0.5)

    if full_water_mask is not None:
        mask = full_water_mask[indices].reshape(data.shape)
    else:
        date = convert_to_date(file.split(".")[1][1:])
        ds = xr.open_dataset(f"{data_loc}land_sea_ice_mask/nimbus/NSIDC0051_SEAICE_PS_N25km_{date}_v2.0.nc")
        full_water_mask = ds.F17_ICECON.values.ravel()
        mask = full_water_mask[indices].reshape(data.shape)
        ds.close()

    attrs = hdf.select(key).attributes()
    is_nan = data == attrs["_FillValue"]
    
    if not mod06:
        valid_rows = ~np.all(is_nan, axis=1)
        valid_cols = ~np.all(is_nan, axis=0)
        data = data[valid_rows][:, valid_cols]
        mask = mask[valid_rows][:, valid_cols]
        lon_highres = lon_highres[valid_rows][:, valid_cols]
        lat_highres = lat_highres[valid_rows][:, valid_cols]
      


    data_shape_bool = data.shape[0] !=0 and data.shape[1] != 0
    if data_shape_bool:
        data = np.where(data > attrs["valid_range"][1], np.mean(data), data)
        if not mod06:
            data = np.float32((data - attrs["radiance_offsets"][idx])*attrs["radiance_scales"][idx])
        else:
            data = np.where(SolarZenith > 90, np.nan, data)

        current_data_list.append(data)
    else:
        current_data_list.append(np.empty((0,0)))

    if not mod06:
        for j, (band) in enumerate(bands[1:]):
            key = list(file_layers[band-1].keys())[0]
            idx = list(file_layers[band-1].values())[0]

            attrs = hdf.select(key).attributes()
            data = hdf.select(key)[:][idx]
            data = data[valid_rows_ll][:, valid_cols_ll]

            data = data[valid_rows][:, valid_cols]
            data = np.where(data > attrs["valid_range"][1], np.mean(data), data)

            data = np.float32((data - attrs["radiance_offsets"][idx])*attrs["radiance_scales"][idx])
                        
            current_data_list.append(data)

    hdf.end()
        
    ##### ALGORITHM LOOKS ONLY AT NEAREST LAT LON AND NOT THE EXACT DISTANCE IN DETERMINATION   
    x_bands = np.stack(current_data_list, axis=-1)
   
    if return_lon_lat:
        return x_bands, mask, np.array([lon_highres, lat_highres]), valid_cols_lon 
    else:
        return x_bands, mask, valid_cols_lon 

def process_npy_file(file, file_layers, bands, max_zenith, data_loc, full_water_mask, tree, return_lon_lat, npz_compressed=False, lon_min=-35, lon_max=45, lat_min=60, lat_max=82):
    zenith = np.load("%sland_sea_ice_mask/sensor_zenith_bilinear_1km.npy" %data_loc)
    zenith_mask = zenith < max_zenith
    current_data_list = []
    if npz_compressed:
        ds = np.load(file)
    else:
        ds = np.load(file, allow_pickle=True).item()

    data = ds["data"]
    lon = ds["lon"]
    lat = ds["lat"]
    date_shape = data.shape

    lat = replace_out_of_bounds_with_nearest(lat, -90, 90)
    lon = replace_out_of_bounds_with_nearest(lon, -180, 180)


    
    mask_lowres = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)
    zoom_factor_y = date_shape[0] / lat.shape[0]
    zoom_factor_x = date_shape[1] / lat.shape[1]
    zoom_factors = (zoom_factor_y, zoom_factor_x)  # Now considering only 2 dimensions
    

    mask_highres = zoom(mask_lowres, zoom_factors, order=0)  # nearest neighbor interpolation
    lat_highres = zoom(lat, zoom_factors, order=1)  # bilinear interpolation
    lon_highres = zoom(lon, zoom_factors, order=1) 
    valid_rows_ll = np.any(mask_highres , axis=1)

    valid_cols_ll = zenith_mask
    valid_cols_lon = np.any(mask_highres[valid_rows_ll][:,valid_cols_ll] , axis=0)  ### remove last for matching x-axis
    data = data[valid_rows_ll][:, valid_cols_ll]
    lat_highres = lat_highres[valid_rows_ll][:, valid_cols_ll] 
    lon_highres = lon_highres[valid_rows_ll][:, valid_cols_ll] 


    coords_highres = np.column_stack((lat_highres.ravel(), lon_highres.ravel()))
    distances, indices = tree.query(coords_highres, k=1,  eps=0.5)

    if full_water_mask is not None:
        mask = full_water_mask[indices].reshape(data.shape[:2])
    else:
        date = convert_to_date(file.split(".")[1][1:])
        try:
            ds = xr.open_dataset(f"{data_loc}land_sea_ice_mask/nimbus/NSIDC0051_SEAICE_PS_N25km_{date}_v2.0.nc")
        except:
            # Use last available nimbus sea ice mask from 2024 - doesnt change results for following analysis
            print("Using 20241231 sea ice mask")
            ds = xr.open_dataset(f"{data_loc}land_sea_ice_mask/nimbus/NSIDC0051_SEAICE_PS_N25km_20241231_v2.0.nc")

        if "F13_ICECON" in ds:
            full_water_mask = ds.F13_ICECON.values.ravel()
        else:
            full_water_mask = ds.F17_ICECON.values.ravel()
        mask = full_water_mask[indices].reshape(data.shape[:2])
        ds.close()

    is_nan = np.isnan(data)
    valid_cols = ~np.all(is_nan, axis=0)
    valid_rows = ~np.all(is_nan, axis=1)

    ### NOT 100% THAT THIS IS RIGHT. BUT WORKS FOR NOW...
    data_dim = data.ndim 
    if data_dim == 3:
        valid_cols = np.any(valid_cols, axis=1)
        valid_rows = np.any(valid_rows, axis=1)
    
    # Use only valid data
    data = data[valid_rows][:, valid_cols]
    mask = mask[valid_rows][:, valid_cols]
    lon_highres = lon_highres[valid_rows][:, valid_cols]
    lat_highres = lat_highres[valid_rows][:, valid_cols]

    # nan check of single values inside dataset
    is_nan = np.isnan(data)
    data_shape_bool = data.shape[0] !=0 and data.shape[1] != 0

    if data_shape_bool:
        data = np.where(is_nan, np.nanmean(data), data)
    else:
        data = np.empty((0, 0))

    
    ##### ALGORITHM LOOKS ONLY AT NEAREST LAT LON AND NOT THE EXACT DISTANCE IN DETERMINATION   
    if data_dim == 2:
        current_data_list.append(data)
        data = np.stack(current_data_list, axis=-1)

    if return_lon_lat:
        return data, mask, np.array([lon_highres, lat_highres]), valid_cols_lon 
    else:
        return data, mask, valid_cols_lon


def append_data(file, file_layers, bands, min_mean=0, full_water_mask=None, tree=None, return_lon_lat=False, normalize=False, max_zenith=47.4, lon_min=-35, lon_max=45, lat_min=60, lat_max=82):
    zenith = np.load("%sland_sea_ice_mask/sensor_zenith_bilinear_1km.npy" %data_loc)
    zenith_mask = zenith < max_zenith
    current_data_list = []

    if file.endswith(".hdf"):
        hdf = SD(file, SDC.READ)
        key = list(file_layers[bands[0]-1].keys())[0]
        idx = list(file_layers[bands[0]-1].values())[0]
        data = hdf.select(key)[:][idx]
        lat = hdf.select("Latitude")[:]
        lon = hdf.select("Longitude")[:]
    elif file.endswith(".npy"):
        ds = np.load(file, allow_pickle=True).item()
        data = ds["data"]
        lon = ds["lon"]
        lat = ds["lat"]

    lat = replace_out_of_bounds_with_nearest(lat, -90, 90)
    lon = replace_out_of_bounds_with_nearest(lon, -180, 180)

    mask_lowres = (lat >= lat_min) & (lat <= lat_max) & (lon > lon_min) & (lon < lon_max)
    zoom_factor_y = data.shape[0] / lat.shape[0]
    zoom_factor_x = data.shape[1] / lat.shape[1]
    zoom_factors = (zoom_factor_y, zoom_factor_x)  # Now considering only 2 dimensions
    

    mask_highres = zoom(mask_lowres, zoom_factors, order=0)  # nearest neighbor interpolation
    lat_highres = zoom(lat, zoom_factors, order=1)  # bilinear interpolation
    lon_highres = zoom(lon, zoom_factors, order=1) 
    valid_rows_ll = np.any(mask_highres , axis=1)

    valid_cols_ll = zenith_mask
    valid_cols_lon = np.any(mask_highres[:][:,valid_cols_ll] , axis=0)  ### remove last for matching x-axis
    data = data[valid_rows_ll][:, valid_cols_ll]
    lat_highres = lat_highres[valid_rows_ll][:, valid_cols_ll] 
    lon_highres = lon_highres[valid_rows_ll][:, valid_cols_ll] 

    #coords_lowres = np.column_stack((full_water_mask.latitude.values.ravel(), full_water_mask.longitude.values.ravel()))

    coords_highres = np.column_stack((lat_highres.ravel(), lon_highres.ravel()))
    distances, indices = tree.query(coords_highres, k=1,  eps=0.5)

    mask = full_water_mask[indices].reshape(data.shape)
    if file.endswith(".hdf"):
        attrs = hdf.select(key).attributes()
        is_nan = data == attrs["_FillValue"]
    elif file.endswith(".npy"):
        is_nan = np.isnan(data)

    valid_rows = ~np.all(is_nan, axis=1)
    valid_cols = ~np.all(is_nan, axis=0)
    data = data[valid_rows][:, valid_cols]
    mask = mask[valid_rows][:, valid_cols]
    lon_highres = lon_highres[valid_rows][:, valid_cols]
    lat_highres = lat_highres[valid_rows][:, valid_cols]
    
    data_shape_bool = data.shape[0] !=0 and data.shape[1] != 0
    if data_shape_bool:
        if file.endswith(".hdf"):
            data = np.where(data > attrs["valid_range"][1], 0, data)
            data = np.float32((data - attrs["radiance_offsets"][idx])*attrs["radiance_scales"][idx])
        elif file.endswith(".npy"):
            data = np.where(is_nan, np.nanmean(data), data)
        current_data_list.append(data)
    else:
        current_data_list.append(np.empty((0,0)))

    if file.endswith(".hdf"):
        for j, (band) in enumerate(bands[1:]):
            key = list(file_layers[band-1].keys())[0]
            idx = list(file_layers[band-1].values())[0]

            attrs = hdf.select(key).attributes()
            data = hdf.select(key)[:][idx]
            data = data[valid_rows_ll][:, valid_cols_ll]

            data = data[valid_rows][:, valid_cols]
            data = np.where(data > attrs["valid_range"][1], 0, data)

            data = np.float32((data - attrs["radiance_offsets"][idx])*attrs["radiance_scales"][idx])
                        
            current_data_list.append(data)
    
    hdf.end()
    ##### ALGORITHM LOOKS ONLY AT NEAREST LAT LON AND NOT THE EXACT DISTANCE IN DETERMINATION   
    x_bands = np.stack(current_data_list, axis=-1)
   
    if return_lon_lat:
        return x_bands, mask, np.array([lon_highres, lat_highres]), valid_cols_lon 
    else:
        return x_bands, mask, valid_cols_lon 
    

def normalize_data(data):
    normalized_data = []    
    normalized_data.append((data - np.nanmin(data, axis=(0,1), keepdims=True)) / (np.nanmax(data, axis=(0,1), keepdims=True) - np.nanmin(data, axis=(0,1), keepdims=True)))
    #normalized_data = (data - np.nanmin(data, axis=(1,2), keepdims=True)) / (np.nanmax(data, axis=(1,2), keepdims=True) - np.nanmin(data, axis=(1,2), keepdims=True))
    return normalized_data
    


def create_nc_from_extracted_data(x, dates, masks, lon_lats, times, save_folder):
    from datetime import datetime, timedelta

    def parse_date(date_str, time_str):
        # Parse the date string 'YYYYddd'
        year = int(date_str[:4])
        day_of_year = int(date_str[4:])
        
        # Convert the day of the year to month and day
        date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
        
        # Parse the time string 'HHMM' and create a time delta
        hours = int(time_str[:-2])
        minutes = int(time_str[-2:])
        
        # Add the time delta to the date
        date_time = date + timedelta(hours=hours, minutes=minutes)
        
        return date_time

    # Create an empty list to hold your datasets
    datasets = []
    lon = [lon_lats[0][0, :, :], lon_lats[1][0, :, :]]
    lat = [lon_lats[0][1, :, :], lon_lats[1][1, :, :]]
    for i, (x_array, date, time) in enumerate(zip(x, dates, times)):
        # Parse the date and time using the `parse_date` function
        date_time = parse_date(str(date), str(time).zfill(4))  # Assuming date & time are integers

        # Create a DataArray for 'x' and add 'time' dimension and coordinate
        x_da = xr.DataArray(
            data=x_array[None, ...], 
            dims=('time', 'x', 'y', 'band'),  
            coords={'time': [date_time]}      
        )

        # Create the Dataset for the current time step
        ds = xr.Dataset(
            data_vars={'radiance': x_da},
            coords={'time': [date_time]}  
        )
        
        # Add 'masks', 'lon', and 'lat' to the Dataset for the current time step
        ds['masks'] = xr.DataArray(masks[i][None, ...], dims=('time', 'x', 'y'), coords={'time': [date_time]})
        ds['lon'] = xr.DataArray(lon[i][None, ...], dims=('time', 'x', 'y'), coords={'time': [date_time]})
        ds['lat'] = xr.DataArray(lat[i][None, ...], dims=('time', 'x', 'y'), coords={'time': [date_time]})
        
        # And do the same for any other variables, such as 'mod_min'
        
        # Append the dataset to the list of datasets
        ds.to_netcdf(save_folder + f"MOD021KM.A{date}.{str(time).zfill(4)}.nc")

