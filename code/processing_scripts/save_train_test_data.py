import tensorflow as tf
import numpy as np

from extract_modis_data import extract_1km_data
from code.model_training.functions import generate_date_list
from autoencoder import SimpleAutoencoder
import numpy as np
from sklearn.model_selection import train_test_split
import xarray as xr
import os
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


# get year as command line argument:
import sys
year = sys.argv[1] 

bands=[31]
print(len(bands))

data_loc = "/location_of_data_including_nimbus_sea_ice_mask_if_used"
folder = "folder_containing_MODIS_npz_files multiple_folders_is_seperated_by_space_in_this_string"
folder_save = "/where_to_save_train_test_npz_files/"
os.makedirs(folder_save, exist_ok=True)

start = "%s0101" %(year)
end = "%s0430" %(year)
dates = generate_date_list(start, end)
start = "%s1001" %(year)
end = "%s1231" %(year)
dates.extend(generate_date_list(start, end))

ds_water_mask=xr.open_dataset(f"{data_loc}/land_sea_ice_mask/nimbus/with_lonlat/NSIDC0051_SEAICE_PS_N25km_20200302_v2.0.nc")

x_cao, dates_cao, masks_cao, lon_lats_cao, mod_min_cao = extract_1km_data(folder,
                                                        #  start_date=start_converted,
                                                        bands=bands,
                                                        #  end_date=end_converted,
                                                        ds_water_mask=ds_water_mask,
                                                        date_list=dates,
                                                         return_lon_lat=True,
                                                         data_loc=data_loc,
                                                         data_type="npz",
                                                         combine_pics=False,
                                                         lon_min=-55,
                                                         lon_max=65,
                                                         lat_min=55,
                                                         lat_max=82)






def save_patches(patch_size, lon_lat_min_max = [-45, 55, 55, 82]):
    x, dates, masks, lon_lats, mod_min = zip(*[(xi, date, mask, lon_lat, mod_min) for xi, date, mask, lon_lat, mod_min in zip(x_cao, dates_cao, masks_cao, lon_lats_cao, mod_min_cao) if (xi.shape[0] > patch_size) and (xi.shape[1] > patch_size)])
   

   
    all_patches = []
    i=0
    tot = len(x)
    strides = [1, patch_size, patch_size, 1]
    
    autoencoder = SimpleAutoencoder(len(bands), patch_size, patch_size)

    random_start_max = 1024 % patch_size



    for (image, mask, lon_lat) in zip(x, masks, lon_lats):
        print(f"{i+1}/{tot}", end="\r")
        if random_start_max > 0:
            rand_start = np.random.randint(0, random_start_max)
        else:
            rand_start = 0
        patches, idx, n_patches, lon, lat = autoencoder.extract_patches(image[:, rand_start:],
                                                                        mask[:, rand_start:],
                                                                        mask_threshold=0.95,
                                                                        lon_lat=lon_lat[:, :, rand_start:],
                                                                        extract_lon_lat=True,
                                                                        strides=strides,
                                                                        lon_lat_min_max=lon_lat_min_max) 
        
        all_patches.append(patches)
        i+=1

    patches = np.concatenate(all_patches, axis=0)

    print(len(patches))

    # TRAIN TEST SPLIT
    patches, val_data = train_test_split(patches, test_size=0.15, random_state=42, shuffle=True)

    print(patches.shape)

    model_run_name = "dnb_l95_z50_ps%s_band%s_%s" %(patch_size, bands[0], str(dates[0])[:4])

    np.save(folder_save + "train_" + model_run_name, patches)
    np.save(folder_save + "test_" + model_run_name, val_data)



# save_patches(256)
# save_patches(384)

## SAVING patches of size 128
save_patches(128)

