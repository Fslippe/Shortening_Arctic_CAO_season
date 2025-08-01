# %%
import os
import sys
import numpy as np
from functions import calculate_area_scores_and_plot, get_area_masks, convert_to_date, generate_map_from_labels, generate_patches, get_MCAOidx
import pandas as pd
import xarray as xr
import json

def import_labeled_data(data_loc, year_threshold=None, filename="image_results_filip_500.json"):
    print("importing labeled data...")
    """
    Import labeled data
    - base directory, 
    - year threshold (optional) image_id contains the given string 
    - filename of .json containing data (optional)
    """

    with open(f"{data_loc}labeled_data/{filename}", "r") as f:
        data = json.load(f)["data"]["image_results"]
    labeled_data = pd.json_normalize(data)

    if year_threshold != None:
        filtered_data = labeled_data[labeled_data["image_id"].str.contains(f"_{year_threshold}")]
        labeled_data = filtered_data
        # filtered_data = labeled_data[labeled_data["user_id"] == "Filip_500_profession_metorology-position_phd-1736256271078"]
    return labeled_data


def import_modis_swath_data(modis_folder, files):
    print("importing modis data...")
    """
    - modis_folder: Directory of files
    - files: Files containing data
    """

    dates = []
    times = [] 
    x_cao = [] 
    masks_cao = [] 
    lon_lats_cao = []

    for file in files:
        dat_tmp = np.load(modis_folder + file)
        date = file.split(".")[1][1:]
        time = file.split(".")[2]
        # if (date, time) in zip(dates_old, times_old) or (date, str(int(time)+5)) in zip(dates_old, times_old):
        dates.append(date)
        times.append(time)
        x_cao.append(dat_tmp["data"])
        masks_cao.append(np.where(dat_tmp["mask"]>0.1, 0, 1))#np.where(dat_tmp["mask"]==1, 0, 1))
        lon_lats_cao.append(np.array([dat_tmp["lon"], dat_tmp["lat"]]))
    
    return dates, times, x_cao, masks_cao, lon_lats_cao




def find_matching_modis_and_label_data(labeled_data, dates, times, x_cao, masks_cao, lon_lats_cao, files):
    print("get matching data...")
    """
    Check if modis and label data matches. Returns only data found both places.
    - labeled_data
    - dates
    - times
    - x_cao
    - masks_cao
    - lon_lats_cao
    - files
    """
    index_list = []
    for idx in range(len(x_cao)):

        if f"MOD021KM.A{dates[idx]}.{times[idx]}" in np.array(labeled_data["image_id"].str.split("/").str[1].str.split("_").str[0]):
            index_list.append(idx)
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
            
    index_list_new = range(len(x_cao))

    return dates_new, times_new, x_cao_new, masks_cao_new, lon_lats_cao_new, index_list_new




def get_patch_lonlat(lon_lats_cao, patch_size, strides):#, lon_min, lon_max, lat_min, lat_max):
    print("Extract patch longitude and latitudes...")
    """
    Get average longitude and latitude of patches of lon_lats_cao data
    - lon_lats_cao: including lon and lat
    - patch_size: size of patches in pixels
    - strides: striding of patch extraction
    """

    lon_patch_mean = []
    lat_patch_mean = []
    # masks = []
    for i in range(len(lon_lats_cao)):
        lons_i = lon_lats_cao[i][0]
        lats_i = lon_lats_cao[i][1]
        shp = lon_lats_cao[i][0].shape
        stide_shape0 = (shp[0] - patch_size) // strides + 1
        stide_shape1 = (shp[1] - patch_size) // strides + 1

        lon_mean = np.zeros((stide_shape0, stide_shape1))
        lat_mean = np.zeros((stide_shape0, stide_shape1))

        for j in range(stide_shape0):
            for k in range(stide_shape1):
                j_start = j*strides 
                j_end   = j*strides + patch_size
                k_start = k*strides 
                k_end   = k*strides + patch_size
                lon_mean[j,k] = np.mean(lons_i[j_start:j_end, k_start:k_end])
                lat_mean[j,k] = np.mean(lats_i[j_start:j_end, k_start:k_end])

        # mask = np.where((lon_mean < lon_max) & (lon_mean >= lon_min) & (lat_mean < lat_max) & (lat_mean >= lat_min))
        lon_patch_mean.append(lon_mean)
        lat_patch_mean.append(lat_mean)
        # masks.append(mask)
    return lon_patch_mean, lat_patch_mean#, masks




def get_merra_model_areas(dates, times, lon_mesh_merra, lat_mesh_merra, lon_patch_mean, lat_patch_mean, data_loc, merra_folder, T_pressure, M_threshold, masks_lon_lat):
    print("get merra areas...")
    lookup_table = {"U": "MERRA2.wind_at_950hpa", 
                        "V": "MERRA2.wind_at_950hpa",
                        "AIRDENS": "MERRA2_400.inst3_3d_aer_Nv",
                        "SO4": "MERRA2_400.inst3_3d_aer_Nv",
                        "SS001": "MERRA2_400.inst3_3d_aer_Nv",
                        "SS002": "MERRA2_400.inst3_3d_aer_Nv",
                        "SS003": "MERRA2_400.inst3_3d_aer_Nv",
                        "SS004": "MERRA2_400.inst3_3d_aer_Nv",
                        "SS005": "MERRA2_400.inst3_3d_aer_Nv",
                        "CLDTMP": "MERRA2_400.tavg1_2d_slv_Nx",
                        "CLDPRS": "MERRA2_400.tavg1_2d_slv_Nx",
                        "PS": "MERRA2_400.tavg1_2d_slv_Nx",
                        "T2M": "MERRA2_400.tavg1_2d_slv_Nx",
                        "TS": "MERRA2_400.tavg1_2d_slv_Nx",
                        "T850": "MERRA2_400.tavg1_2d_slv_Nx",
                        "U10M": "MERRA2_400.tavg1_2d_slv_Nx",
                        "V10M": "MERRA2_400.tavg1_2d_slv_Nx",
                        "ZLCL": "MERRA2_400.tavg1_2d_slv_Nx",
                        "TQL": "MERRA2_400.tavg1_2d_slv_Nx",
                        "TQV": "MERRA2_400.tavg1_2d_slv_Nx",
                        "TQI": "MERRA2_400.tavg1_2d_slv_Nx",
                        "QL":"MERRA2_400.inst3_3d_asm_Np",
                        "QI":"MERRA2_400.inst3_3d_asm_Np",
                        "T":"MERRA2_400.inst3_3d_asm_Np",
                        "H":"MERRA2_400.inst3_3d_asm_Np",
                        "PBLH": "MERRA2_400.tavg1_2d_flx_Nx",
                        "PRECTOT": "MERRA2_400.tavg1_2d_flx_Nx",
                        "PRECTOTCORR": "MERRA2_400.tavg1_2d_flx_Nx"
                        }

    file = lookup_table["TS"].split(".")[1]
    file_T = lookup_table["T"].split(".")[1]

    # ds_water_mask=xr.open_dataset(f"{data_loc}/land_sea_ice_mask/nimbus/with_lonlat/NSIDC0051_SEAICE_PS_N25km_20200302_v2.0.nc")
    # coords_lowres = np.vstack((ds_water_mask.lat.values.ravel(), ds_water_mask.lon.values.ravel())).T
    # tree = cKDTree(coords_lowres)
    # tree = cKDTree(coords_lowres)
    # coords_highres = np.column_stack((lat_mesh_merra.ravel(), lon_mesh_merra.ravel()))
    # distances, indices = tree.query(coords_highres, k=1,  eps=0.5)

    model_areas = []

    for i, (date, time) in enumerate(zip(dates, times)):
        date_mmdd = convert_to_date(date)
        ds = xr.open_dataset(f"{data_loc}land_sea_ice_mask/nimbus/NSIDC0051_SEAICE_PS_N25km_{date_mmdd}_v2.0.nc")
        if "F13_ICECON" in ds:
            full_water_mask = ds.F13_ICECON.values.ravel()
        else:
            full_water_mask = ds.F17_ICECON.values.ravel()
        # mask_sea_ice = full_water_mask[indices].reshape(lon_mesh_merra.shape[:2])
        # mask_sea_ice = np.where(mask_sea_ice>0.1, 0, 1).astype(bool)
        # date = convert_to_date(date)
        print(f"{i}/{len(dates)}", date, end="\r")
        dic = {}

        ds = xr.open_mfdataset(f"{merra_folder}/{date_mmdd[:4]}/*{file}.{date_mmdd}.SUB.nc")
        if T_pressure == 850:
            ds_Th = ds.T850
        else:
            dsT = xr.open_mfdataset(f"{merra_folder}/{date_mmdd[:4]}/*{file_T}.{date_mmdd}.SUB.nc")
            ds_Th = dsT.T.sel(lev=T_pressure)
        
            # Get 3hourly data to match with ds_Th
            ds = ds.isel(time=slice(None, None, 3))

        time_sel = f"{date_mmdd[:4]}-{date_mmdd[4:6]}-{date_mmdd[6:]}T{time}"

        ds = ds.sel(time=time_sel, method="nearest")
        ds_Th = ds_Th.sel(time=time_sel, method="nearest")


        M = get_MCAOidx(ds.TS.values, ds_Th.values, ds.PS.values*1e-2, T_pressure)
        mask = np.where(M >= M_threshold, 1.0, 0.0)
        
        lons_i = lon_patch_mean[i]
        lats_i = lat_patch_mean[i]
        mask_i = np.zeros_like(lons_i)

        mask_lon_lat = masks_lon_lat[i]
        for j in range(lons_i.shape[0]):
            for k in range(lons_i.shape[1]):

                lon_diff = (lon_mesh_merra - lons_i[j,k])*np.sin(np.radians(lats_i[j,k]))
                lat_diff = (lat_mesh_merra - lats_i[j,k])
                distance_array = np.sqrt(lon_diff**2 + lat_diff**2)

                lat_index, lon_index = np.unravel_index(np.argmin(distance_array), distance_array.shape)
                mask_i[j,k] = mask[lat_index, lon_index]

        mask_i[~mask_lon_lat] = np.nan
        model_areas.append(mask_i)
    return model_areas


# %%
def main():
    print("Remember to set right directories and parameters in main() function")
    data_loc = "//" # Base directory for data used and generated
    merra_folder = "//" # Base directory for MERRA data
    modis_folder = "labeling_data/full_record_500/" # directory of MODIS npz files that have been labeled
    score_save_folder = f"{data_loc}models/merra/scores/" # Folder for saving score file

    try:
        year_threshold = sys.argv[1]
        save_addon = f"_full_record_500_{year_threshold}"
    except:
        print("Either change code or add command line argument for year thresholds \nExample: python calculate_scores_merra.py 00-05")
        year_threshold = None
        save_addon = ""

    # Some parameters for score calculation 
    patch_size = 128 # size of patch
    strides = 64 # stride of patch extraction
    T_pressure = 850 # hPa
    M_thresholds = np.arange(0, 6.25, 0.25) #[1, 1.25, 1.5, 1.75, 2, 2.25] # K
    
    # Get any MERRA ds for creating a lon lat mesh
    lon_merra = xr.open_dataset(f"{merra_folder}/2020/MERRA2_400.tavg1_2d_slv_Nx.20200829.SUB.nc").lon.values
    lat_merra = xr.open_dataset(f"{merra_folder}/2020/MERRA2_400.tavg1_2d_slv_Nx.20200829.SUB.nc").lat.values
    lon_mesh_merra, lat_mesh_merra = np.meshgrid(lon_merra, lat_merra)

    # Import labeled data
    labeled_data = import_labeled_data(data_loc, year_threshold)
    
    # Import modis swaths and associated data
    files = [f for f in os.listdir(modis_folder) if f.endswith(".npz")]
    dates, times, x_cao, masks_cao, lon_lats_cao = import_modis_swath_data(modis_folder, files)

    # Extract the files matching across modis swaths and labeled data
    dates, times, x_cao, masks_cao, lon_lats_cao, index_list = find_matching_modis_and_label_data(labeled_data, dates, times, x_cao, masks_cao, lon_lats_cao, files)

    from autoencoder import SimpleAutoencoder
    autoencoder_predict = SimpleAutoencoder(len([31]), patch_size, patch_size)

    patches_cao, all_lon_patches_cao, all_lat_patches_cao, starts_cao, ends_cao, shapes_cao, n_patches_tot_cao, indices_cao = generate_patches([x for x in x_cao],
                                                                                                                                                masks_cao,
                                                                                                                                                lon_lats_cao,
                                                                                                                                                1,
                                                                                                                                                0,
                                                                                                                                                autoencoder_predict,
                                                                                                                                                strides=[1, strides, strides,1])
    
    labels = np.ones(patches_cao)

    mask_maps = []
    for i in range(len(x_cao)):
        mask_map = generate_map_from_labels(labels, starts_cao[i], ends_cao[i], shapes_cao[i], indices_cao[i], 0, n_patches_tot_cao[i], patch_size, stride=strides)
        mask_maps.append(mask_map.atype(bool))


    # Get the labeled areas
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

    # Mean longitude and latitudes of each patch of lon_lats_cao
    lon_patch_mean, lat_patch_mean = get_patch_lonlat(lon_lats_cao, patch_size, strides)

    # Get the model areas
    for M_threshold in M_thresholds:
        model_run_name = f"merra_T{T_pressure}_Mthr{M_threshold}"
        
        model_areas = get_merra_model_areas(dates,
                                            times,
                                            lon_mesh_merra,
                                            lat_mesh_merra,
                                            lon_patch_mean,
                                            lat_patch_mean,
                                            data_loc,
                                            merra_folder,
                                            T_pressure,
                                            M_threshold, 
                                            mask_maps)
        
        ######################### BUg IN SCORE CALCULATION WHERE MISSING LABELED AREA IS STILL APPENDED 

        # Calculate the scores and save the score file
        area_scores = calculate_area_scores_and_plot(model_areas, labeled_areas, dates, times)
        if not os.path.exists(score_save_folder):
            os.makedirs(score_save_folder)
        np.save(score_save_folder + f"area_scores_{model_run_name}_res{strides}{save_addon}", area_scores)

        print("\n\n", model_run_name)
        # Print Some score calculation examples
        tot_points = np.sum(area_scores["tot_points"]) 
        print("True:", np.sum(area_scores["area_true_prediction_scores"]) / tot_points)
        TP = np.sum(area_scores["area_true_positive_scores"]) / tot_points
        FP = np.sum(area_scores["area_false_positive_scores"]) / tot_points
        TN = np.sum(area_scores["area_true_negative_scores"]) / tot_points
        FN = np.sum(area_scores["area_false_negative_scores"]) / tot_points
        precission = TP / (TP + FP)
        TPR = TP / (TP + FN)
        MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        print("TP:", TP )
        print("FP:", FP )
        print("TN:", TN )
        print("FN:", FN )
        print("F1:",  2* (TPR * precission) / (precission + TPR))
        print("MCC:", MCC)

if __name__=="__main__":
    main()


