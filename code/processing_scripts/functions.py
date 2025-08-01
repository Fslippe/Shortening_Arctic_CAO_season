import matplotlib.pyplot as plt
#from skimage import measure
import numpy as np
import pyproj
import cartopy.crs as ccrs
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from scipy import ndimage  
import tensorflow as tf
import tqdm 

# from autoencoder import *
import datetime
import pyproj
from shapely.geometry import Point, Polygon, LineString
# from plot_functions import *
import os
import joblib
from tensorflow.keras.models import load_model
import matplotlib
# from functions import load_and_predict_encoder, process_pred_info_all_labels, find_closest_indices_merra, get_cluster_and_label_lists, dump_clustering, process_label_maps, calculate_area_scores_and_plot, calculate_scores_and_plot, process_model_area_mask, process_model_masks, get_valid_lons_lats_2, get_valid_lons_lats, find_closest_indices, get_area_mask, get_closed_open_area_mask, get_area_masks, get_closed_open_masks, get_area_and_border_mask, gaussian_brush, apply_brush, bresenham_line, interpolate_coords, generate_xy_grid, generate_hist_map, is_closer_to_any_point, convert_to_day_of_year, convert_to_date, generate_date_list, convert_to_standard_date, generate_map_from_labels, generate_map_from_lon_lats, generate_map_from_patches, reconstruct_from_patches, shuffle_in_unison, process_image, generate_patches_parallel, generate_patches, get_patches_of_img_cao, perpendicular_distance, douglas_peucker, simplify_line, remove_labels_from_size_thresholds
geodesic = pyproj.Geod(ellps='WGS84')

# from extract_modis_data import extract_1km_data, process_key, combine_images_based_on_time, extract_250m_data, replace_out_of_bounds_with_nearest, process_file, process_hdf_file, process_npy_file, append_data, normalize_data, create_nc_from_extracted_data, parse_date
# from functions import load_and_predict_encoder, process_pred_info_all_labels, find_closest_indices_merra, get_cluster_and_label_lists, dump_clustering, process_label_maps, calculate_area_scores_and_plot, calculate_scores_and_plot, process_model_area_mask, process_model_masks, get_valid_lons_lats_2, get_valid_lons_lats, find_closest_indices, get_area_mask, get_closed_open_area_mask, get_area_masks, get_closed_open_masks, get_area_and_border_mask, gaussian_brush, apply_brush, bresenham_line, interpolate_coords, generate_xy_grid, generate_hist_map, is_closer_to_any_point, convert_to_day_of_year, convert_to_date, generate_date_list, convert_to_standard_date, generate_map_from_labels, generate_map_from_lon_lats, generate_map_from_patches, reconstruct_from_patches, shuffle_in_unison, process_image, generate_patches_parallel, generate_patches, get_patches_of_img_cao, perpendicular_distance, douglas_peucker, simplify_line, remove_labels_from_size_thresholds
# from plot_functions import make_variable_histogram, plot_filtered_map, save_img_with_labels, plot_hist_map, plot_monthly_boxplots, plot_double_hist_map, plot_img_cluster_mask, plot_map_with_nearest_neighbors, plot_map_with_boundaries_in_projection
# from reanalysis_functions import get_histogram_from_var_and_index, get_potential_temperature, get_theta, get_MCAOidx, get_LTS, get_qs, get_moist_lapse_rate, get_pressure_height, get_EIS, get_rf_dataframe, align_time_coordinates, extract_var_at_idx, extract_var_at_indices, extract_var_at_indices_old
# from autoencoder import SobelFilterLayer, SimpleAutoencoder
# from comparison_plots_functions import get_data_mask_for_plotting, get_merra_mask, plot_comparisons
# from histogram_functions import get_monthly_observations, plot_monthly_barplots, create_region_path, get_dict_list_for_labels, get_linregress_grid_old, get_monthly_fraction_grid, get_monthly_average_coverage, get_linregress_grid, plot_trend_maps, get_counts_from_dict_list, extract_result_dict, extract_n_obs_per_day_xarray, extract_result_dict_same_grid, extract_result_dict_xarray, calculate_daily_coverage_for_months, calculate_daily_coverage_for_months_xr, calculate_daily_averages_for_months_xr, calculate_daily_coverage_for_season, mk_test, mk_test_for_months, seasonal_mannkendall_test, find_repeats, dijk, theil_sen_seasonal, theil_sen_multi_season, extract_result_dict_on_same_grid, extract_result_dict_merra, extract_merra_dict_list, filter_dates_and_files, load_dict_list, get_closest_indices, compute_geodesic_distance, get_hist_counts_month, get_hist_counts_month_in_year, get_percentages, get_percentages_new

#load_and_predict_encoder, process_label_maps, calculate_area_scores_and_plot, process_model_area_mask, get_area_masks, generate_patches
#calculate_area_scores_and_plot, get_area_masks, convert_to_date, generate_map_from_labels, generate_patches 
#convert_to_date
#process_label_maps, process_model_area_mask, get_area_masks, generate_patches
#process_pred_info_all_labels, process_label_maps, generate_xy_grid, generate_patches
#process_label_maps, generate_patches
#generate_map_from_labels
#generate_date_list
#process_label_maps, convert_to_day_of_year, generate_patches
# generate_xy_grid

def load_and_predict_encoder(patch_size, last_filter, patches_cao):
    """
    Loads a pre-trained encoder model based on the specified patch size and last filter size,
    and uses it to predict the encoded representation of the provided input patches.

    Args:
        patch_size (int): The size of the patch used for training the encoder model.
        last_filter (int): Determines which encoder model to load, should be one of [128, 64, 32].
        patches_cao (numpy.ndarray): The input data containing patches to encode.

    Returns:
        numpy.ndarray: The flattened encoded representations of the input patches.
    """
    if last_filter == 128:
        encoder = load_model(f"/uio/kant/geo-geofag-u1/fslippe/data/models/patch_size{patch_size}/filter128/encoder_dnb_l95_z50_ps128_f128_1e3_201812-202312.h5")
    elif last_filter == 64:
        encoder = load_model(f"/uio/kant/geo-geofag-u1/fslippe/data/models/patch_size{patch_size}/filter64/encoder_dnb_l95_z50_ps128_f64_1e3_201812-202312_epoch_500.h5")
    elif last_filter == 32:
        encoder = load_model(f"/uio/kant/geo-geofag-u1/fslippe/data/models/patch_size{patch_size}/filter32/encoder_dnb_l95_z50_ps128_f32_1e3_201812-202312.h5")

    encoded_patches_cao = encoder.predict(patches_cao)
    encoded_patches_flat_cao = encoded_patches_cao.reshape(encoded_patches_cao.shape[0], -1)

    return encoded_patches_flat_cao

def get_theta(T, P, P0=1000):
    theta = T * (P0 / P) ** (0.286)
    return theta 


def get_MCAOidx(Tskn, T_p, Ps, p):
    theta_p = get_theta(T_p, p)
    thetaskn = get_theta(Tskn, Ps)
    return thetaskn - theta_p  


def process_pred_info_all_labels(args):
    date_cao, mod_min_cao, lon_map, lat_map, label_map, all_labels, lon_mesh, lat_mesh = args

    datetime_obj = datetime.datetime.strptime(
        date_cao + str(mod_min_cao).zfill(4), "%Y%j%H%M")
    formatted_datetime_str = datetime_obj.strftime('%Y-%m-%dT%H:%M')

    combined_dict = {}
    for label in all_labels:
        swath_map_pred_mask = np.isin(label_map, label)
        lon_pred = lon_map[swath_map_pred_mask]
        lat_pred = lat_map[swath_map_pred_mask]

        idx_pred = find_closest_indices_merra(lon_pred, lat_pred, lon_mesh, lat_mesh)
        combined_dict.update({f"label_{label}": idx_pred})
    
    dict_list_base = {"date": convert_to_date(date_cao),
                    "date_day": date_cao,
                    "datetime": formatted_datetime_str}
    
    dict_list = {**dict_list_base, **combined_dict}
    
    return dict_list
    


def find_closest_indices_merra(lon, lat, lon_mesh, lat_mesh):
    # Initialize empty lists for the closest indices
    unique_indices = set()

    # Iterate over the values in lon and lat
    for lon, lat in zip(lon, lat):
        # Calculate the Euclidean distance between the current point and all the points in lon_mesh and lat_mesh
        distances = np.sqrt((lon_mesh - lon)**2 + (lat_mesh - lat)**2)

        # Find the minimum distance and its index
        min_distance_index = np.unravel_index(
            distances.argmin(), distances.shape)

        # Append the indices to the respective lists
        unique_indices.add(min_distance_index)

    return np.array(list(unique_indices))


def get_cluster_and_label_lists(patch_load_name, patch_size, last_filter, n_K_list, encoded_patches_flat, encoded_patches_flat_cao):
    cluster_list = []
    label_list = []
    for n_K in [10, 11, 12, 13, 14, 15, 16]:
        cluster = dump_clustering(
            patch_load_name, patch_size, last_filter, n_K, encoded_patches_flat)
        labels = cluster.predict(encoded_patches_flat_cao)
        cluster_list.append(cluster)
        label_list.append(labels)

    return cluster_list, label_list


def dump_clustering(patch_load_name, patch_size, filter, n_K, encoded_patches_flat):
    cluster = KMeans(n_K, init='k-means++',
                     random_state=42).fit(encoded_patches_flat)
    file_path = "/uio/kant/geo-geofag-u1/fslippe/data/models/patch_size%s/filter%s/clustering/cluster_%s_filter%s_K%s.pkl" % (
        patch_size, filter, patch_load_name, filter, n_K)
    directory = os.path.dirname(file_path)

    # Create the directory if it does not already exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Check if the file already existspatch_load_name
    counter = 1
    while os.path.exists(file_path):
        # If the file exists, modify the file path by adding a suffix
        file_path = "/uio/kant/geo-geofag-u1/fslippe/data/models/patch_size%s/filter%s/clustering/cluster_%s_filter%s_K%s_%d.pkl" % (
            patch_size, filter, patch_load_name, filter, n_K, counter)
        counter += 1

    # Save the model with the modified file path
    joblib.dump(cluster, file_path)
    return cluster


def process_label_maps(labels,
                       all_lon_patches,
                       all_lat_patches,
                       starts_cao,
                       ends_cao,
                       shapes_cao,
                       indices_cao,
                       global_max,
                       n_patches_tot_cao,
                       patch_size,
                       strides,
                       label_1,
                       label_2,
                       size_thr_1=0,
                       size_thr_2=0):
    
    def calculate_patch_mean(patches):
        if patches.ndim == 2:
            return np.mean(np.expand_dims(patches, axis=0), axis=(1, 2))
        else:
            return np.mean(patches, axis=(1, 2))

    pat_lon = [calculate_patch_mean(patch) for patch in all_lon_patches]
    pat_lat = [calculate_patch_mean(patch) for patch in all_lat_patches]
    pat_lon = np.concatenate(np.array(pat_lon, dtype=object), axis=0)
    pat_lat = np.concatenate(np.array(pat_lat, dtype=object), axis=0)

    label_map = np.empty(len(starts_cao), dtype=object)
    lon_map = np.empty(len(starts_cao), dtype=object)
    lat_map = np.empty(len(starts_cao), dtype=object)

    index_list = range(len(starts_cao))

    for i in index_list:
        label_map[i] = generate_map_from_labels(
            labels, starts_cao[i], ends_cao[i], shapes_cao[i], indices_cao[i], global_max, n_patches_tot_cao[i], patch_size, strides)
        if size_thr_1 and size_thr_2:
            label_map[i] = remove_labels_from_size_thresholds(
                label_map[i], label_1, label_2, size_thr_1=size_thr_1, size_thr_2=size_thr_2)

        lon_map[i] = generate_map_from_lon_lats(
            pat_lon, starts_cao[i], ends_cao[i], shapes_cao[i], indices_cao[i], global_max, n_patches_tot_cao[i], patch_size, strides)
        lat_map[i] = generate_map_from_lon_lats(
            pat_lat, starts_cao[i], ends_cao[i], shapes_cao[i], indices_cao[i], global_max, n_patches_tot_cao[i], patch_size, strides)

    return label_map, lon_map, lat_map


def calculate_area_scores_and_plot( model_areas, labeled_areas, dates, times, plot=False):
    tot_points_list = []

    area_scores = []  # To store the area and border scores
    weighted_area_scores = []
    area_true_positive_scores = []
    area_false_positive_scores = []
    area_true_negative_scores = []
    area_false_negative_scores = []
    area_true_prediction_scores = []
    area_false_prediction_scores = []
    area_tot_labeled_list = []
    date_list = []
    time_list = []

   
    for (m_area, l_area, date, time) in zip(model_areas, labeled_areas, dates, times):
        if l_area.ndim > 1: 
            date_list.append(date)
            time_list.append(time)

            masked_m_area = m_area
            masked_l_area = np.where(np.isnan(m_area), np.nan, l_area)
            tot_area_points = np.sum(~np.isnan(m_area))
            tot_points_list.append(tot_area_points)
            area_tot_labeled = np.nansum(np.where(masked_l_area > 0, 1, 0))
            area_tot_labeled_list.append(area_tot_labeled)

            area_false_positives = np.nansum(
                np.where((masked_m_area == 1) & (masked_l_area == 0), 1, 0))
            area_false_positive_scores.append(area_false_positives)

            area_true_positives = np.nansum(
                np.where((masked_m_area == 1) & (masked_l_area > 0), 1, 0))

            area_true_positive_scores.append(area_true_positives)

            area_true_negatives = np.nansum(
                np.where((masked_m_area == 0) & (masked_l_area == 0), 1, 0))
            area_true_negative_scores.append(area_true_negatives)

            area_false_negatives = np.nansum(
                np.where((masked_m_area == 0) & (masked_l_area > 0), 1, 0))
            area_false_negative_scores.append(area_false_negatives)

            area_true_predictions = np.nansum(np.where(((masked_m_area == 1) & (
                masked_l_area > 0)) | ((masked_m_area == 0) & (masked_l_area == 0)), 1, 0))
            area_true_prediction_scores.append(area_true_predictions)

            area_false_predictions = np.nansum(np.where(((masked_m_area == 1) & (
                masked_l_area == 0)) | ((masked_m_area == 0) & (masked_l_area > 0)), 1, 0))
            area_false_prediction_scores.append(area_false_predictions)

            area_diff = np.abs(m_area - l_area)
            area_score = 1 - np.nanmean(area_diff)

            area_agreement = np.abs(l_area - 0.5)*10
            area_scores.append(area_score)
            weighted_area_scores.append(1 - np.nanmean(area_diff * area_agreement))


    all_area_scores = {"date": date_list, 
                       "time": time_list,
                       "tot_points": tot_points_list,
                       "area_tot_labeled": area_tot_labeled_list,
                       "area_scores": area_scores,
                       "weighted_area_scores": weighted_area_scores,
                       "area_true_positive_scores": area_true_positive_scores,
                       "area_false_positive_scores": area_false_positive_scores,
                       "area_true_negative_scores": area_true_negative_scores,
                       "area_false_negative_scores": area_false_negative_scores,
                       "area_true_prediction_scores": area_true_prediction_scores,
                       "area_false_prediction_scores": area_false_prediction_scores
                       }


    return all_area_scores

def calculate_scores_and_plot(model_boundaries, model_areas, labeled_boundaries, labeled_areas, plot=False):
    tot_points_list = []

    area_scores = []  # To store the area and border scores
    weighted_area_scores = []
    area_true_positive_scores = []
    area_false_positive_scores = []
    area_true_negative_scores = []
    area_false_negative_scores = []
    area_true_prediction_scores = []
    area_false_prediction_scores = []
    area_tot_labeled_list = []

    border_scores = []  # To store the area and border scores
    weighted_border_scores = []
    border_true_positive_scores = []
    border_false_positive_scores = []
    border_true_negative_scores = []
    border_false_negative_scores = []
    border_true_prediction_scores = []
    border_false_prediction_scores = []
    border_tot_labeled_list = []
    
    for (m_border, m_area, l_border, l_area) in zip(model_boundaries, model_areas, labeled_boundaries, labeled_areas):
        tot_labeled_area_points = np.where(l_area > 0, 1, 0)
        tot_nonlabeled_area_points = np.where(l_area == 0, 1, 0)
        masked_m_area = m_area
        masked_l_area = np.where(np.isnan(m_area), np.nan, l_area)

        tot_area_points = np.sum(~np.isnan(m_area))
        tot_points_list.append(tot_area_points)
        area_tot_labeled = np.nansum(np.where(masked_l_area > 0, 1, 0))
        area_tot_labeled_list.append(area_tot_labeled)

        area_false_positives = np.nansum(
            np.where((masked_m_area == 1) & (masked_l_area == 0), 1, 0))
        area_false_positive_scores.append(area_false_positives)

        area_true_positives = np.nansum(
            np.where((masked_m_area == 1) & (masked_l_area > 0), 1, 0))

        area_true_positive_scores.append(area_true_positives)

        area_true_negatives = np.nansum(
            np.where((masked_m_area == 0) & (masked_l_area == 0), 1, 0))
        area_true_negative_scores.append(area_true_negatives)

        area_false_negatives = np.nansum(
            np.where((masked_m_area == 0) & (masked_l_area > 0), 1, 0))
        area_false_negative_scores.append(area_false_negatives)

        area_true_predictions = np.nansum(np.where(((masked_m_area == 1) & (
            masked_l_area > 0)) | ((masked_m_area == 0) & (masked_l_area == 0)), 1, 0))
        area_true_prediction_scores.append(area_true_predictions)

        area_false_predictions = np.nansum(np.where(((masked_m_area == 1) & (
            masked_l_area == 0)) | ((masked_m_area == 0) & (masked_l_area > 0)), 1, 0))
        area_false_prediction_scores.append(area_false_predictions)

        area_diff = np.abs(m_area - l_area)
        area_score = 1 - np.nanmean(area_diff)

        area_agreement = np.abs(l_area - 0.5)*10
        area_scores.append(area_score)
        weighted_area_scores.append(1 - np.nanmean(area_diff * area_agreement))

        masked_m_border = m_border
        masked_l_border = np.where(np.isnan(m_border), np.nan, l_border)

        border_false_positives = np.nansum(
            np.where((masked_m_border > 0) & (masked_l_border == 0), 1, 0))
        border_false_positive_scores.append(border_false_positives)

        border_true_positives = np.nansum(
            np.where((masked_m_border > 0) & (masked_l_border > 0), 1, 0))
        border_true_positive_scores.append(border_true_positives)

        border_true_negatives = np.nansum(
            np.where((masked_m_border == 0) & (masked_l_border == 0), 1, 0))
        border_true_negative_scores.append(border_true_negatives)

        border_false_negatives = np.nansum(
            np.where((masked_m_border == 0) & (masked_l_border > 0), 1, 0))
        border_false_negative_scores.append(border_false_negatives)

        border_true_predictions = np.nansum(np.where(((masked_m_border > 0) & (
            masked_l_border > 0)) | ((masked_m_border == 0) & (masked_l_border == 0)), 1, 0))
        border_true_prediction_scores.append(border_true_predictions)

        border_false_predictions = np.nansum(np.where(((masked_m_border > 0) & (
            masked_l_border == 0)) | ((masked_m_border == 0) & (masked_l_border > 0)), 1, 0))
        border_false_prediction_scores.append(border_false_predictions)

        border_tot_labeled = np.nansum(np.where(masked_l_border > 0, 1, 0))
        border_tot_labeled_list.append(border_tot_labeled)


        """OLD border score calcualation"""
        # max_boundary = np.max(l_border)
        # border_diff = np.abs(m_border - l_border)
        # border_score = 1 - np.nanmean(border_diff)
        # border_scores.append(border_score)

        # border_agreement = np.abs(l_border - 0.5)*10
        # weighted_border_scores.append(
        #     1 - np.nanmean(border_diff * border_score))

        """NEW border score calculation"""
        m_preds = m_border == 1
        tot_border_preds = np.where(m_preds)[0].shape[0]
        l_yes_no_border = np.where(l_border > 0, 1, 0)
        border_weighted_frac = np.where(m_preds, abs(m_border - l_border) / tot_border_preds, np.nan)
        border_frac = np.where(m_preds, abs(m_border - l_yes_no_border) / tot_border_preds, np.nan)
        
        border_scores.append(1- np.nansum(border_frac))
        weighted_border_scores.append(1- np.nansum(border_weighted_frac))


        if plot:
            # fig, axs = plt.subplots(1, 2)
            # axs[0].imshow(m_area)
            # axs[1].imshow(l_area)
            # plt.show()

            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(m_border)
            axs[1].imshow(l_border)
            plt.show()

            # fig, axs = plt.subplots(1, 2)
            # cb1 = axs[0].imshow(np.abs(m_area - l_area))
            # plt.colorbar(cb1, ax=axs[0])
            # cb2 = axs[1].imshow(np.abs(m_border - l_border))
            # plt.colorbar(cb2, ax=axs[1])
            # plt.show()

    all_area_scores = {"tot_points": tot_points_list,
                       "area_tot_labeled": area_tot_labeled_list,
                       "area_scores": area_scores,
                       "weighted_area_scores": weighted_area_scores,
                       "area_true_positive_scores": area_true_positive_scores,
                       "area_false_positive_scores": area_false_positive_scores,
                       "area_true_negative_scores": area_true_negative_scores,
                       "area_false_negative_scores": area_false_negative_scores,
                       "area_true_prediction_scores": area_true_prediction_scores,
                       "area_false_prediction_scores": area_false_prediction_scores
                       }

    all_border_scores = {"tot_points": tot_points_list,
                         "border_tot_labeled": border_tot_labeled_list,
                         "border_scores": border_scores,
                         "weighted_border_scores": weighted_border_scores,
                         "border_true_positive_scores": border_true_positive_scores,
                         "border_false_positive_scores": border_false_positive_scores,
                         "border_true_negative_scores": border_true_negative_scores,
                         "border_false_negative_scores": border_false_negative_scores,
                         "border_true_prediction_scores": border_true_prediction_scores,
                         "border_false_prediction_scores": border_false_prediction_scores
                         }

    return all_area_scores, all_border_scores


def process_model_area_mask(index_list, lon_map, lat_map, indices_cao, label_map, label_1, label_2,  plot=False):
    model_areas = []

      
    for i in index_list:
        area_mask = np.zeros_like(lon_map[i], dtype=np.float)

        valid_pos = indices_cao[i].numpy()
        valid_mask = np.full(area_mask.shape, False)
        valid_mask.flat[valid_pos] = True
        
        area_mask = np.where(np.isin(label_map[i], label_1) | np.isin(label_map[i], label_2), 1, 0).astype(np.float)
        # area_mask = np.where((label_map[i] == label_1) | (
        #     label_map[i] == label_2), 1, 0).astype(np.float)
        area_mask[~valid_mask] = np.nan

        model_areas.append(area_mask)

        if plot:
            fig, axs = plt.subplots(1, 1, figsize=[10, 10])
            axs.imshow(area_mask)
            # axs[1].imshow(boundary_mask)
            plt.show()

    return  model_areas

def process_model_masks(index_list, lon_map, lat_map, valid_lons, valid_lats, indices_cao, label_map, label_1, label_2, brush=True, plot=False):
    if brush:
        brush_mask = gaussian_brush(width=5, height=5, sigma=1.2, strength=1)
    model_boundaries = []
    model_areas = []

    for i in index_list:
        boundary_mask = np.zeros_like(lon_map[i], dtype=np.float)
        closest_indices = find_closest_indices(
            lon_map[i], lat_map[i], valid_lons[i], valid_lats[i])
        for (x, y) in closest_indices:
            if brush:
                apply_brush(boundary_mask, y, x, brush_mask)
            else:
                boundary_mask[x, y] = 1  # Directly mark the position if brush is False

        # This should be a flattened list or 1D np.ndarray of valid indices
        valid_pos = indices_cao[i].numpy()
        # Start with a mask of False (invalid) values
        valid_mask = np.full(boundary_mask.shape, False)
        # Set positions defined by valid_pos to True (valid)
        valid_mask.flat[valid_pos] = True
        # Set invalid positions in boundary_mask to np.nan
        boundary_mask[~valid_mask] = np.nan

        area_mask = np.where((label_map[i] == label_1) | (
            label_map[i] == label_2), 1, 0).astype(np.float)
        area_mask[~valid_mask] = np.nan

        model_boundaries.append(boundary_mask)
        model_areas.append(area_mask)

        if plot:
            fig, axs = plt.subplots(1, 2, figsize=[10, 10])
            axs[0].imshow(area_mask)
            axs[1].imshow(boundary_mask)
            plt.show()

    return model_boundaries, model_areas




def find_closest_indices(grid_lons, grid_lats, lons_list, lats_list):
    index_list = []

    for lon, lat in zip(lons_list, lats_list):
        min_distance = None
        closest_index = None

        for i in range(grid_lons.shape[0]):
            for j in range(grid_lons.shape[1]):
                if not np.isnan(grid_lons[i,j]): 
                    _, _, distance = geodesic.inv(
                        lon, lat, grid_lons[i, j], grid_lats[i, j])

                    if min_distance is None or distance < min_distance:
                        min_distance = distance
                        closest_index = (i, j)
        index_list.append(closest_index)

    return index_list




def get_area_mask(boundary_coordinates, mask_shape, scale_factor_x, scale_factor_y, subpixel_resolution=10, threshold=0.5):
    # Ensure boundary coordinates are lists of tuples
    boundary_coordinates = list(map(tuple, boundary_coordinates))

    polygon = Polygon(boundary_coordinates)
    
    minx, miny, maxx, maxy = polygon.bounds
    minx, miny, maxx, maxy = map(int, [minx * scale_factor_x, miny * scale_factor_y, maxx * scale_factor_x, maxy * scale_factor_y])

    mask = np.full(mask_shape, False)

    # Define the step size for sub-pixels
    step = 1.0 / subpixel_resolution

    for x in range(minx, maxx + 1):
        for y in range(miny, maxy + 1):
            inside_count = 0
            for i in range(subpixel_resolution):
                for j in range(subpixel_resolution):
                    sub_x = x + i * step
                    sub_y = y + j * step
                    # Scale back to original coordinates to check containment in the original polygon
                    back_x = sub_x / scale_factor_x
                    back_y = sub_y / scale_factor_y
                    point = Point(back_x, back_y)
                    if polygon.contains(point) or polygon.touches(point):
                        inside_count += 1

            # Calculate fraction of subpixels inside the polygon
            fraction_inside = inside_count / (subpixel_resolution ** 2)
            if fraction_inside >= threshold:
                if 0 <= x < mask_shape[1] and 0 <= y < mask_shape[0]:
                    mask[y, x] = True

    return mask



def get_area_masks(x_cao, dates, times, masks_cao, df, reduction, patch_size=128, index_list=None, plot=False, min_outside=None, max_outside=None, subpixel_resolution=10):
    """
    Function to get the area masks for the given index list and plot the results if specified.
    - x_cao: The input images
    - dates: The dates of the images
    - times: The times of the images
    - masks_cao: The masks of the images
    - df: The dataframe containing the area and border lines
    - reduction: The reduction factor used for the images
    - patch_size: The size of the patches
    - index_list: The list of indices to process
    - plot: Whether to plot the results
    - min_outside: The minimum value to consider outside the area
    - max_outside: The maximum value to consider outside the area
    - subpixel_resolution: The resolution to use for sub-pixel interpolation
    """

    downscaled_areas = []
    for idx in index_list:
        print(f"{idx}/{len(index_list)}", end="\r")
        arr_appended = False

        extracted_rows = df[df["image_id"].str.split(
            "/").str[1].str.split("_").str[0] == f"MOD021KM.A{dates[idx]}.{times[idx]}"]
        if len(extracted_rows) >= 1:
            if plot:
                fig, axs = plt.subplots(1, 4, figsize=[35, 20])

            interpolated_border = []
            interpolated_area_i = []

            for i in range(len(extracted_rows)):
                interpolated_area = []
                interpolated_area_mask = []
                test = []

                di = extracted_rows.iloc[i]
                date_img = str(di["image_id"].split("/")[1].split(".")[1][1:])
                time_img = int(di["image_id"].split("/")[1].split(".")[2].split("_")[0])
                area_lines = np.array(di["data.areaLines"], dtype=object)
                border_lines = np.array(di["data.borderLines"], dtype=object)

                # if min_outside and area_lines.ndim == 3:
                    
                #     area_lines[:,:,0] = np.where(area_lines[:,:,0] < min_outside, min_outside, area_lines[:,:,0])
                #     area_lines[:,:,0] = np.where(area_lines[:,:,0] > max_outside, max_outside, area_lines[:,:,0])
                #     area_lines[:,:,0] -= min_outside

                    # border_lines[:,:,0] = np.where(border_lines[:,:,0] < min_outside, min_outside, border_lines[:,:,0])
                    # border_lines[:,:,0] = np.where(border_lines[:,:,0] > max_outside, max_outside, border_lines[:,:,0])
                    # border_lines[:,:,0] -= min_outside

                reduced_height = (x_cao[idx].shape[0] - patch_size) // reduction + 1
                reduced_width = (x_cao[idx].shape[1] - patch_size) // reduction + 1
                scale_factor_x = reduced_width / x_cao[idx].shape[1]
                scale_factor_y = reduced_height / x_cao[idx].shape[0]
                n_areas = len(area_lines)

                if n_areas > 0:
                    for j in range(n_areas):
                        area = np.array(area_lines[j])

                        if min_outside and area.ndim == 2:
                            area[:,0] = np.where(area[:,0] < min_outside, 0, area[:,0]-min_outside)
                            area[:,0] = np.where(area[:,0] > max_outside, max_outside, area[:,0]-min_outside)
                            # area[:,0] -= min_outside

                        interpolated_area_boundary = interpolate_coords(
                            area, connect_first_last=True)
                        scaled_boundary_coordinates = np.copy(interpolated_area_boundary.astype(float))
                        scaled_boundary_coordinates[:, 0] *= scale_factor_x
                        scaled_boundary_coordinates[:, 1] *= scale_factor_y
                        # area_mask = get_area_mask(scaled_boundary_coordinates, (reduced_height, reduced_width))
                        area_mask = get_area_mask(interpolated_area_boundary, (reduced_height, reduced_width), scale_factor_x, scale_factor_y, subpixel_resolution, threshold=0.5)
                        # interpolated_area.append(scaled_boundary_coordinates)
                        interpolated_area.append(interpolated_area_boundary)
                        boundary_coordinates = np.where(scaled_boundary_coordinates < 0.5, 0, scaled_boundary_coordinates).astype(int)

                        test.append(boundary_coordinates)
                        interpolated_area_mask.append(area_mask)

                    interpolated_sum = np.sum(interpolated_area_mask, axis=0)
                    interpolated_area_i.append(np.where(interpolated_sum > 1, 1, interpolated_sum))
                else:
                    interpolated_area_i.append(np.zeros((reduced_height, reduced_width)))
                if plot:
                    axs[0].imshow(x_cao[idx], cmap="gray_r")
                    # axs[0].imshow(x_cao[idx][::128,::128], cmap="gray_r")

                    for k in range(len(interpolated_area)):
                        # axs[0].scatter(interpolated_area[k].T[0] // reduction, interpolated_area[k].T[1] // reduction, s=0.05, color="r")
                        # axs[0].imshow(interpolated_area[k], alpha=0.8/len(extracted_rows), cmap="Reds")
                        axs[0].fill(interpolated_area[k].T[0], interpolated_area[k].T[1],
                                    alpha=0.8/len(extracted_rows), color="r")

                n_borders = len(border_lines)
                if n_borders > 0:
                    for j in range(n_borders):
                        border = np.array(border_lines[j])
                        interpolated_border.append(interpolate_coords(
                            border.astype(float), connect_first_last=False))

                if plot:
                    axs[1].imshow(x_cao[idx][::128, ::128], cmap="gray_r")

                    # axs[1].imshow(x_cao[idx][::128,::128], cmap="gray_r")

                    for k in range(len(interpolated_area)):
                        # axs[1].scatter(interpolated_area[k].T[0] // reduction, interpolated_area[k].T[1] // reduction, s=0.05, color="r")
                        # axs[1].imshow(interpolated_area[k], alpha=0.8/len(extracted_rows), cmap="Reds")
                        axs[1].fill(test[k].T[0], test[k].T[1],
                                    alpha=0.8/len(extracted_rows), color="r")

            # Final areas
            interpolated_sum_i = np.sum(
                interpolated_area_i, axis=0) / len(extracted_rows)
            downscaled_areas.append(interpolated_sum_i)
            arr_appended = True

        if not arr_appended:
            downscaled_areas.append(np.array([np.nan]))

    if plot:
        axs[2].imshow(downscaled_areas[0])

    return downscaled_areas



def get_area_and_border_mask(x_cao, dates, times, masks_cao, df, reduction, patch_size=128, index_list=None, plot=False):
    downscaled_areas = []
    downscaled_borders = []

    for idx in index_list:
        extracted_rows = df[df["image_id"].str.split(
            "/").str[1].str.split("_").str[0] == f"MOD021KM.A{dates[idx]}.{times[idx]}"]
        if len(extracted_rows) > 1:
            if plot:
                fig, axs = plt.subplots(1, 3, figsize=[35, 20])

            interpolated_border = []
            interpolated_area_i = []
            for i in range(len(extracted_rows)):
                interpolated_area = []
                interpolated_area_mask = []
                di = extracted_rows.iloc[i]
                date_img = str(di["image_id"].split("/")[1].split(".")[1][1:])
                time_img = int(di["image_id"].split("/")[1].split(".")[2].split("_")[0])
                area_lines = np.array(di["data.areaLines"])
                border_lines = np.array(di["data.borderLines"], dtype=object)
                reduced_height = (x_cao[idx].shape[0] - patch_size) // reduction + 1
                reduced_width = (x_cao[idx].shape[1] - patch_size) // reduction + 1
                scale_factor_y = reduced_height / x_cao[idx].shape[0]
                scale_factor_x = reduced_width / x_cao[idx].shape[1]

                n_areas = len(area_lines)

                if n_areas > 0:
                    for j in range(n_areas):
                        area = np.array(area_lines[j])
                        interpolated_area_boundary = interpolate_coords(
                            area, connect_first_last=True)
                        scaled_boundary_coordinates = np.copy(interpolated_area_boundary.astype(float))
                        scaled_boundary_coordinates[:, 0] *= scale_factor_x
                        scaled_boundary_coordinates[:, 1] *= scale_factor_y
                        area_mask = get_area_mask(scaled_boundary_coordinates, (reduced_height, reduced_width))

                        interpolated_area.append(interpolated_area_boundary)
                        interpolated_area_mask.append(area_mask)

                    interpolated_sum = np.sum(interpolated_area_mask, axis=0)
                    interpolated_area_i.append(np.where(interpolated_sum > 1, 1, interpolated_sum))
                else:
                    interpolated_area_i.append(np.zeros((reduced_height, reduced_width)))
                if plot:
                    axs[0].imshow(x_cao[idx], cmap="gray_r")
                    for k in range(len(interpolated_area)):
                        # axs[0].scatter(interpolated_area[k].T[0] // reduction, interpolated_area[k].T[1] // reduction, s=0.05, color="r")
                        # axs[0].imshow(interpolated_area[k], alpha=0.8/len(extracted_rows), cmap="Reds")
                        axs[0].fill(interpolated_area[k].T[0], interpolated_area[k].T[1],
                                    alpha=0.8/len(extracted_rows), color="r")

                n_borders = len(border_lines)
                if n_borders > 0:
                    for j in range(n_borders):
                        border = np.array(border_lines[j])
                        interpolated_border.append(interpolate_coords(
                            border.astype(float), connect_first_last=False))

                if plot:
                    axs[1].imshow(x_cao[idx], cmap="gray_r")
                    for k in range(len(interpolated_border)):
                        axs[1].scatter(interpolated_border[k].T[0],
                                       interpolated_border[k].T[1], s=0.5, color="r")

            # Final areas
            interpolated_sum_i = np.sum(
                interpolated_area_i, axis=0) / len(extracted_rows)
            downscaled_areas.append(interpolated_sum_i)

            # adjust width, height, and sigma as needed
            brush = gaussian_brush(width=200, height=200,
                                   sigma=64, strength=1/len(extracted_rows))

            tot_border = np.zeros(x_cao[idx].shape[:2])
            tot_border_reduced = np.zeros((reduced_height, reduced_width))

            for border_coords in interpolated_border:
                border_mask = np.zeros(x_cao[idx].shape[:2])
                for x, y in border_coords:
                    apply_brush(border_mask, int(x), int(y), brush)
                tot_border += border_mask

            for i in range(reduced_height):
                for j in range(reduced_width):
                    tot_border_reduced[i, j] = np.mean(
                        tot_border[i * reduction: (i + 1) * reduction, j * reduction: (j + 1) * reduction])
            downscaled_borders.append(tot_border_reduced)

            if plot:
                cb = axs[2].imshow(tot_border_reduced, vmin=0, vmax=1)
                plt.colorbar(cb)
                plt.show()
        plt.show()
    return downscaled_areas, downscaled_borders


def gaussian_brush(width=5, height=5, sigma=1.0, strength=1):
    """
    Create a 2D Gaussian brush centered in the middle of the width and height.
    """
    x, y = np.meshgrid(np.linspace(-width/2, width/2, width),
                       np.linspace(-height/2, height/2, height))
    d = np.sqrt(x*x + y*y)
    g = strength*np.exp(-(d**2 / (2.0 * sigma**2)))
    return g


def apply_brush(mask, x, y, brush):
    """
    Apply the given brush to the mask at position x, y.
    """
    half_width = brush.shape[1] // 2
    half_height = brush.shape[0] // 2

    col_start = max(0, x - half_width)
    col_end = col_start + brush.shape[1]

    row_start = max(0, y - half_height)
    row_end = row_start + brush.shape[0]

    if col_end > mask.shape[1]:
        col_end = mask.shape[1]
        col_start = col_end - brush.shape[1]

    if row_end > mask.shape[0]:
        row_end = mask.shape[0]
        row_start = row_end - brush.shape[0]

    mask[row_start:row_end, col_start:col_end] = np.where(
        mask[row_start:row_end, col_start:col_end] < brush, brush, mask[row_start:row_end, col_start:col_end])
    # mask[row_start:row_end, col_start:col_end] = mask[row_start:row_end, col_start:col_end] + brush
   # mask[row_start:row_end, col_start:col_end] = brush

    return mask


def bresenham_line(x0, y0, x1, y1):
    """Bresenham's Line Algorithm to generate points between start and end."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


def interpolate_coords(coords, connect_first_last):
    """Interpolate between points in coords if they are not neighbors."""
    interpolated = []
    for i in range(len(coords) - 1):
        start = coords[i]
        end = coords[i + 1]
        if start[0] < 0:
            start[0] = 0
        if start[1] < 0:
            start[1] = 0
        if end[0] < 0:
            end[0] = 0
        if end[1] < 0:
            end[1] = 0

        # Check if points are neighbors
        if max(abs(round(start[0]) - round(end[0])), abs(round(start[1]) - round(end[1]))) > 1:
            interpolated.extend(bresenham_line(
                round((start[0])), round((start[1])), round((end[0])), round((end[1]))))
        else:
            interpolated.append((round(start[0]), round(start[1])))

    interpolated.append(coords[-1])  # Add the last point

    if connect_first_last:
        start = (coords[-1])
        end = (coords[0])
        # Check if points are neighbors
        if max(abs(round(start[0]) - round(end[0])), abs(round(start[1]) - round(end[1]))) > 1:
            interpolated.extend(bresenham_line(
                round((start[0])), round((start[1])), round((end[0])), round((end[1]))))
        else:
            interpolated.append((round(start[0]), round(start[1])))

    return np.array(interpolated)


def generate_xy_grid(x_extent=[-2.2e6, 2.2e6], y_extent=[-3.6e6, -0.5e6], grid_resolution=128e3):
    x_grid, y_grid = np.meshgrid(np.arange(x_extent[0], x_extent[1], grid_resolution),
                                 np.arange(y_extent[0], y_extent[1], grid_resolution))
    return x_grid, y_grid


def generate_hist_map(n_patches_tot,
                      indices,
                      labels,
                      starts,
                      ends,
                      shapes,
                      all_lon_patches,
                      all_lat_patches,
                      dates,
                      desired_label,
                      size_threshold,
                      patch_size,
                      global_max,
                      projection=ccrs.Stereographic(central_latitude=90),
                      grid_resolution=128e3):

    # Generate grid to add counts on

    x_grid, y_grid = generate_xy_grid(grid_resolution=grid_resolution)
    # Initialize the count matrix
    counts = np.zeros_like(x_grid)

    # Create a KDTree for faster nearest neighbor search
    tree = cKDTree(list(zip(x_grid.ravel(), y_grid.ravel())))

    # This will track which dates have been counted for each grid cell
    dates_counted = {}

    s = 0
    # Run through all images
    for i in range(len(dates)):
        # Generate lon lat maps
        height, width = shapes[i]
        reduced_height = height // patch_size
        reduced_width = width // patch_size

        current_lon = np.empty((n_patches_tot[i], patch_size, patch_size))
        current_lon[np.squeeze(indices[i].numpy())] = all_lon_patches[i]
        lon_map = np.reshape(current_lon, (reduced_height,
                             reduced_width, patch_size, patch_size))

        current_lat = np.empty((n_patches_tot[i], patch_size, patch_size))
        current_lat[np.squeeze(indices[i].numpy())] = all_lat_patches[i]
        lat_map = np.reshape(current_lat, (reduced_height,
                             reduced_width, patch_size, patch_size))

        # Get label map

        label_map = generate_map_from_labels(
            labels, starts[i], ends[i], shapes[i], indices[i], global_max, n_patches_tot[i], patch_size)

        binary_map = np.isin(label_map, desired_label)

        # Label connected components, considering diagonal connections
        """USE OF DIAGONAL CONNECTIONS"""
        structure = ndimage.generate_binary_structure(2, 2)
        labeled_map, num_features = ndimage.label(
            binary_map, structure=structure)

        """NO DIAGONAL CONNECTIONS:"""
        # labeled_map, num_features = ndimage.label(binary_map)

        # Measure sizes of connected components
        region_sizes = ndimage.sum(
            binary_map, labeled_map, range(num_features + 1))

        # Iterate through each region and check if its size exceeds the threshold
        for region_idx, region_size in enumerate(region_sizes):
            if region_size >= size_threshold:
                # Get the indices of the region
                region_coordinates = np.where(labeled_map == region_idx)

                # Convert to projected coordinates
                x_proj, y_proj = projection.transform_points(ccrs.PlateCarree(),
                                                             lon_map[region_coordinates].ravel(
                ),
                    lat_map[region_coordinates].ravel())[:, :2].T
                s += 1

                # Query the KDTree for nearest grid points
                _, idxs = tree.query(list(zip(x_proj, y_proj)))

                # Check and Increment the counts based on date condition
                for idx in idxs:
                    if idx not in dates_counted:
                        dates_counted[idx] = set()
                    if dates[i] not in dates_counted[idx]:
                        counts.ravel()[idx] += 1
                        dates_counted[idx].add(dates[i])

    return x_grid, y_grid, counts






def is_closer_to_any_point(original_lon, original_lat, new_lon, new_lat, existing_lons, existing_lats):
    # Check if the new point is closer to any existing point
    original_distance = geodesic.inv(
        new_lon, new_lat, original_lon, original_lat)[2]

    for (existing_lon, existing_lat) in zip(existing_lons, existing_lats):
        distance = geodesic.inv(
            new_lon, new_lat, existing_lon, existing_lat)[2]
        # Extract the distance from the result
        if distance < original_distance:  # You may adjust this threshold
            return True
    return False


def convert_to_day_of_year(date_str):
    # Parse the date
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    # Convert to datetime object
    date_obj = datetime.datetime(year, month, day)

    # Get the day of the year
    day_of_year = date_obj.timetuple().tm_yday

    # Return in the desired format
    # Using :03d to ensure it's a 3-digit number
    return f"{year}{day_of_year:03d}"


def convert_to_date(day_of_year_str):
    # Parse the day of the year
    year = int(day_of_year_str[:4])
    day_of_year = int(day_of_year_str[4:])

    # Convert to datetime object
    date_obj = datetime.datetime(year, 1, 1) + \
        datetime.timedelta(day_of_year - 1)

    # Format the date as "YYYYMMDD"
    date_str = date_obj.strftime("%Y%m%d")

    # Return the formatted date
    return date_str


def generate_date_list(start, end):
    start_date = datetime.datetime.strptime(start, '%Y%m%d')
    end_date = datetime.datetime.strptime(end, '%Y%m%d')

    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(convert_to_day_of_year(
            current_date.strftime('%Y%m%d')))
        current_date += datetime.timedelta(days=1)
    return date_list


def convert_to_standard_date(date_str):
    # Parse the date
    year = int(date_str[:4])
    day_of_year = int(date_str[4:])

    # Convert to datetime.datetime object
    # Using day_of_year - 1 because datetime.timedelta is 0-indexed
    date_obj = datetime.datetime(year, 1, 1) + \
        datetime.timedelta(days=day_of_year - 1)

    # Return in the desired format
    return date_obj.strftime('%Y%m%d')


def generate_map_from_labels(labels, start, end, shape, idx, global_max, n_patches, patch_size, stride=None):
    # Calculate the dimensions of the reduced resolution array
    height, width = shape

    if stride is None or stride == patch_size:
        reduced_height = height // patch_size
        reduced_width = width // patch_size
    else:
        reduced_height = (height - patch_size) // stride + 1
        reduced_width = (width - patch_size) // stride + 1

    # Generate an empty map with all values set to global_max + 1
    cluster_map = np.full((reduced_height, reduced_width),
                          global_max, dtype=labels.dtype)

    # Get the indices corresponding to the patches
    patch_indices = np.squeeze(idx.numpy())

    # Ensure the provided indices are within the expected range
    valid_indices = patch_indices < n_patches
    patch_indices = patch_indices[valid_indices]

    # Set the labels for the patches with valid indices
    cluster_map.flat[patch_indices] = labels[start:end][valid_indices]

    return cluster_map


def generate_map_from_lon_lats(lon_lats, start, end, shape, idx, global_max, n_patches, patch_size, stride=None):
    height, width = shape

    if stride is None or stride == patch_size:
        reduced_height = height // patch_size
        reduced_width = width // patch_size
    else:
        reduced_height = (height - patch_size) // stride + 1
        reduced_width = (width - patch_size) // stride + 1

    # Generate an empty map with all values set to global_max + 1
    lon_lats_map = np.full((reduced_height, reduced_width),
                           np.nan, dtype=lon_lats.dtype)

    # Get the indices corresponding to the patches
    patch_indices = np.squeeze(idx.numpy())

    # Ensure the provided indices are within the expected range
    valid_indices = patch_indices < n_patches
    patch_indices = patch_indices[valid_indices]

    # Set the labels for the patches with valid indices
    lon_lats_map.flat[patch_indices] = lon_lats[start:end][valid_indices]

    return lon_lats_map


def generate_map_from_patches(patches, start, end, shape, patch_size, idx):
    num_patches_y, num_patches_x = shape[0] // patch_size, shape[1] // patch_size
    reduced_height, reduced_width = num_patches_y * \
        patch_size, num_patches_x * patch_size

    # Create an empty map of the reduced resolution
    reconstructed_image = np.zeros((reduced_height, reduced_width))

    # Extract the patches corresponding to this image
    image_patches = patches[start:end]

    for i in range(len(image_patches)):
        if i >= len(idx):
            break

        y = int(idx[i][0] // num_patches_x) * patch_size
        x = int(idx[i][0] % num_patches_x) * patch_size

        # Place the patch in the correct position
        reconstructed_image[y:y+patch_size, x:x+patch_size] = image_patches[i]

    return reconstructed_image


def reconstruct_from_patches(patches, shapes, starts, ends, patch_size):
    reconstructed_images = []

    for i, shape in enumerate(shapes):
        # Create an empty image of the shape
        reconstructed_image = np.zeros(
            (shape[0], shape[1], patches[0].shape[2]))

        # Extract the patches corresponding to this image
        image_patches = patches[starts[i]:ends[i]]

        # Place each patch into the empty image
        patch_idx = 0
        for y in range(0, shape[0], patch_size):
            for x in range(0, shape[1], patch_size):
                reconstructed_image[y:y+patch_size, x:x +
                                    patch_size, :] = image_patches[patch_idx]
                patch_idx += 1

        # Append the reconstructed image to the list
        reconstructed_images.append(reconstructed_image)

    return reconstructed_images


def shuffle_in_unison(*args):
    rng_state = np.random.get_state()
    for array in args:
        np.random.set_state(rng_state)
        np.random.shuffle(array)

from joblib import Parallel, delayed

def process_image(args):
    try:


        with tf.device('/CPU:0'):   

            image, mask, lon_lat, autoencoder, strides, lon_lat_min_max, min_vals, max_vals, mask_threshold = args
            patches, idx, n_patches, lon, lat = autoencoder.extract_patches(image,
                                                                            mask,
                                                                            mask_threshold=mask_threshold,
                                                                            lon_lat=lon_lat,
                                                                            extract_lon_lat=True,
                                                                            strides=strides,
                                                                            lon_lat_min_max=lon_lat_min_max)
            patches = (patches - min_vals) / (max_vals - min_vals)
        return patches, lon, lat, len(patches), n_patches, idx
    except Exception as e:
        print(f"Error processing image: {e}")
        return None  # Return None or handle as needed

def generate_patches_parallel(x, masks, lon_lats, max_vals, min_vals, autoencoder, strides=[None, None, None, None], lon_lat_min_max=[-35, 45, 60, 82], mask_threshold=0.95, workers=None):
    starts = []
    ends = []
    shapes = [image.shape[0:2] for image in x]
    start = 0

        
    args = [(image, mask, lon_lat, autoencoder, strides, lon_lat_min_max,
             min_vals, max_vals, mask_threshold) for image, mask, lon_lat in zip(x, masks, lon_lats)]
    if workers == None:
        if len(args) < 64:
            workers = len(args)
        else:
            workers = 64
    with tf.device('/CPU:0'):   

        results = Parallel(n_jobs=workers, backend='loky')(
            delayed(process_image)(arg) for arg in tqdm(args, total=len(args), desc="Processing"))
        # Log the completion
        print("Parallel processing completed.")
        # Filter out None results
        results = [r for r in results if r is not None]

        all_patches, all_lon_patches, all_lat_patches, lens, n_patches_tot, indices = zip(*results)

        patches = np.concatenate(all_patches, axis=0)
        starts = np.cumsum([0] + list(lens[:-1]))
        ends = np.cumsum(lens)

    return patches, all_lon_patches, all_lat_patches, starts, ends, shapes, n_patches_tot, indices



def generate_patches(x, masks, lon_lats, max_vals, min_vals, autoencoder, strides=[None, None, None, None], lon_lat_min_max=[-55, 65, 55, 82], mask_threshold=0.95, mask_lon_lats=True):
    all_patches = []
    all_lon_patches = []
    all_lat_patches = []

    starts = []
    ends = []
    shapes = []
    start = 0
    n_patches_tot = []
    indices = []

    # encoder = load_model("/uio/kant/geo-geofag-u1/fslippe/data/models/winter_2020_21_band(6,20,29)_encoder")
    # normalized_patches = np.concatenate([autoencoder.extract_patches(n_d) for n_d in normalized_data], axis=0)

    i = 0
    tot = len(x)
    for (image, mask, lon_lat) in zip(x, masks, lon_lats):
        print(f"{i}/{tot}", end="\r")
        shapes.append(image.shape[0:2])
        patches, idx, n_patches, lon, lat = autoencoder.extract_patches(image,
                                                                        mask,
                                                                        mask_threshold=mask_threshold,
                                                                        lon_lat=lon_lat,
                                                                        extract_lon_lat=True,
                                                                        strides=strides,
                                                                        lon_lat_min_max=lon_lat_min_max,
                                                                        mask_lon_lats=mask_lon_lats)  
        
        # Assuming this function extracts and reshapes patches for a single image
        # patches = autoencoder_predict.extract_patches(image)  # Assuming this function extracts and reshapes patches for a single image
        # n_patches = len(patches)

        all_patches.append(patches)
        all_lon_patches.append(lon)
        all_lat_patches.append(lat)

        starts.append(start)
        ends.append(start + len(patches))
        n_patches_tot.append(n_patches)
        indices.append(idx)
        start += len(patches)
        i += 1
    # Stack filtered patches from all images
    patches = (np.concatenate(all_patches, axis=0) -
               min_vals) / (max_vals - min_vals)

    return patches, all_lon_patches, all_lat_patches, starts, ends, shapes, n_patches_tot, indices


def get_patches_of_img_cao(labels, patches, starts, ends, shapes, indices, global_max, n_patches_tot, desired_label, size_threshold, n,  patch_size):
    """
    Find pictures with regions of patches of a desired label of sizes higher than given threshold 
    """
    patches_w = []

    for i in range(n):
        label_map = generate_map_from_labels(
            labels, starts[i], ends[i], shapes[i], indices[i], global_max, n_patches_tot[i], patch_size)

        binary_map = (label_map == desired_label)

        # Label connected components
        labeled_map, num_features = ndimage.label(binary_map)

        # Measure sizes of connected components
        region_sizes = ndimage.sum(
            binary_map, labeled_map, range(num_features + 1))
        # Iterate through each region and check if its size exceeds the threshold

        for region_idx, region_size in enumerate(region_sizes):
            if region_size > size_threshold:
                patches_w.append(patches[starts[i]:ends[i]])
    patches_w = np.concatenate(patches_w)
    return patches_w


def perpendicular_distance(point, line_start, line_end):
    """
    Calculate the perpendicular distance from a point to a line defined by two endpoints.
    """
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end

    numerator = np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

    return numerator / denominator if denominator != 0 else 0


def douglas_peucker(point_list, epsilon):
    """
    Douglas-Peucker algorithm for line simplification.
    Returns the simplified coordinates and the indices of points inside the epsilon.
    """
    dmax = 0
    index = 0
    end = len(point_list)
    tot_indices = []
    for i in range(2, end - 1):
        d = perpendicular_distance(
            point_list[i], point_list[0], point_list[end - 1])
        if d > dmax:
            index = i
            dmax = d

    result_list = []
    indices_inside_epsilon = [False] * len(point_list)

    if dmax > epsilon:
        rec_results1, rec_indices1 = douglas_peucker(
            point_list[:index + 1], epsilon)
        rec_results2, rec_indices2 = douglas_peucker(
            point_list[index:], epsilon)

        # Build the result list
        result_list = rec_results1[:-1] + rec_results2
        indices_inside_epsilon[:index + 1] = rec_indices1
        indices_inside_epsilon[index:] = rec_indices2
    else:
        result_list = [point_list[0], point_list[end - 1]]

    return result_list, indices_inside_epsilon


def simplify_line(coords, tolerance):
    line = LineString(coords)
    simplified_line = line.simplify(tolerance, preserve_topology=False)
    return np.array(simplified_line.xy).T





def remove_labels_from_size_thresholds(m, label1, label2, size_thr_1, size_thr_2):

    if size_thr_1:
        m_max = np.max(m)
        binary_map = np.isin(m, label1)
        labeled_map, num_features = ndimage.label(binary_map)
        region_sizes = ndimage.sum(
            binary_map, labeled_map, range(num_features + 1))

        # Loop through each region and check if the region size is below the thr
        # Skipping background (label 0)
        for region_label in range(1, num_features + 1):
            if region_sizes[region_label] < size_thr_1:
                # Set the pixels of this region to the maximum value of m
                m[labeled_map == region_label] = m_max
    if size_thr_2:
        m_max = np.max(m)
        binary_map = np.isin(m, label2)
        labeled_map, num_features = ndimage.label(binary_map)
        region_sizes = ndimage.sum(
            binary_map, labeled_map, range(num_features + 1))

        # Loop through each region and check if the region size is below the thr
        # Skipping background (label 0)
        for region_label in range(1, num_features + 1):
            if region_sizes[region_label] < size_thr_2:
                # Set the pixels of this region to the maximum value of m
                m[labeled_map == region_label] = m_max

    return m



# def compute_boundary_coordinates_between_labels(m, lon_map, lat_map, label1, label2):
#     lons = []
#     lats = []
#     highest_confidence = -1  # Initialize to a low value

#     for i in range(m.shape[0]):
#         for j in range(m.shape[1]):
#             if m[i, j] == label1:
#                 neighbors = [
#                     (i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)
#                 ]

#                 for ni, nj in neighbors:
#                     if 0 <= ni < m.shape[0] and 0 <= nj < m.shape[1]:
#                         if m[ni, nj] == label2:
#                             # Calculate confidence (e.g., based on distance)
#                             confidence = (lon_map[i, j] - lon_map[ni, nj])**2 + (lat_map[i, j] - lat_map[ni, nj])**2

#                             if confidence > highest_confidence:
#                                 highest_confidence = confidence
#                                 # Store lon and lat values for the boundary
#                                 interp_lon = (lon_map[i, j] + lon_map[ni, nj]) / 2
#                                 interp_lat = (lat_map[i, j] + lat_map[ni, nj]) / 2
#                                 lons = [interp_lon]
#                                 lats = [interp_lat]
#     return lons, lats


