import numpy as np 
from pyproj import Geod
from scipy.spatial import cKDTree
import datetime
from scipy.stats import theilslopes, kendalltau
import matplotlib.pyplot as plt
import glob
from statsmodels.tsa.stattools import acf
from scipy.stats.mstats import kendalltau_seasonal
from matplotlib.patches import Patch

import numpy.ma as ma 
from matplotlib.colors import BoundaryNorm
from calendar import monthrange
import xarray as xr 
import cartopy.crs as ccrs
from scipy.stats import distributions
from collections import defaultdict
from matplotlib.colors import ListedColormap
import matplotlib.path as mpath
import matplotlib.patches as mpatches
# set np random seed:
import pandas as pd
np.random.seed(42)
import matplotlib

def get_theta(T, P, P0=1000):
    theta = T * (P0 / P) ** (0.286)
    return theta 

def get_MCAOidx(Tskn, T_p, Ps, p):
    theta_p = get_theta(T_p, p)
    thetaskn = get_theta(Tskn, Ps)
    return thetaskn - theta_p  

# get_monthly_observations, plot_monthly_barplots, create_region_path, get_dict_list_for_labels, extract_result_dict, extract_n_obs_per_day_xarray, extract_result_dict_same_grid, extract_result_dict_xarray, calculate_daily_coverage_for_months, mk_test_for_months, theil_sen_multi_season, extract_merra_dict_list, load_dict_list, get_closest_indices
def get_monthly_observations(result_dict, years, months):
    months_str = [str(t).zfill(2) for t in months]
    years_str = [str(t) for t in years]
    grid_shape = result_dict[[key for key in result_dict.keys()][0]].shape
    array_list = np.empty((len(years), len(months), grid_shape[0], grid_shape[1]))
    # result_dict is a dictionary with keys on the form "yyyymmdd"
    # Running through every year in years and month in months to get a percentage of cloud cover for each month
    for i, year in enumerate(years_str):
        for j, month in enumerate(months_str):
            # number of days in current month
            days_in_month = monthrange(int(year), int(month))[1]
            yearmonth_sum = np.zeros(grid_shape)
            measurements = np.zeros(grid_shape)
            yearmonth_sum[:] = np.nan
            days_per_gridpoint = np.zeros(grid_shape)
            # for loop using days_in_month to get the cloud cover for each day in the month
            for k, day in enumerate(range(1, days_in_month+1)):
                date = f"{year}{month}{str(day).zfill(2)}"
                date = str(date)
                if date in result_dict.keys():
                    measurements = np.isfinite(result_dict[date]) 
                    days_per_gridpoint += measurements

            array_list[i, j] = days_per_gridpoint / days_in_month
    return array_list

def plot_monthly_barplots(month_names, array_lists, model_names, alpha=0.05, figsize=None, legend=True, rotate_labels=False):
    if isinstance(figsize, list):
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
    else:
        fig, ax = plt.subplots(figsize=(2*len(month_names), 3), dpi=300)
    
    index = np.arange(len(month_names))
    bar_width = 0.2
    bar_spacing = 0.28
    n_models = len(array_lists)
    
    # Calculate average occurrences for each model
    avg_occurrences = [[np.nanmean(arr)*100 for arr in array_list] for array_list in array_lists]

    # Determine colors for each model
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ymax = np.max([max(occurrences) for occurrences in avg_occurrences]) + 2.5
    
    for pos, (occurrences, model_name) in enumerate(zip(avg_occurrences, model_names)):
        positions_model = index - bar_spacing * (n_models-1)/2 + bar_spacing*pos
        ax.bar(positions_model, occurrences, width=bar_width, color=colors[pos], label=model_name)

    ax.set_xticks(index-0.5)

    ax.set_xticklabels([])
    ax.set_yticks([0, 25, 50])

    # rotation = 45 if rotate_labels else 0
    # for i, month in enumerate(month_names):
    #     ax.text(index[i], -0.005, month, ha='center', va='top', rotation=rotation, fontsize=16, color='black')
    
    ax.set_ylim([0, ymax])
    ax.set_xlim([-0.5, len(index)-0.5])
    ax.set_facecolor("white")
    
    if legend:
        legend_elements = [Patch(facecolor=colors[i], edgecolor='w', label=model_names[i]) for i in range(n_models)]
        ax.legend(handles=legend_elements, loc='best', title='Model')
    
    # ax.set_ylabel("coverage\n[%]")
    plt.tight_layout()
    return fig

def create_region_path(ax, vertices, num_pts=100, edgecolor='red', linewidth=4):
    # Interpolate points along the edges
    lons = np.concatenate([np.linspace(vertices[i, 0], vertices[i+1, 0], num_pts) for i in range(vertices.shape[0] - 1)])
    lats = np.concatenate([np.linspace(vertices[i, 1], vertices[i+1, 1], num_pts) for i in range(vertices.shape[0] - 1)])

    # Close the polygon loop by appending the first vertex
    lons = np.append(lons, vertices[0, 0])
    lats = np.append(lats, vertices[0, 1])

    # Create a path from these vertices
    path = mpath.Path(np.vstack((lons, lats)).T)

    # Create a patch from the path
    patch = mpatches.PathPatch(path, facecolor='none', edgecolor=edgecolor, linewidth=linewidth, transform=ccrs.PlateCarree())
    ax.add_patch(patch)


def get_dict_list_for_labels(dict_list, labels, nK):
    """
    Function to accuire dict_list for given labels 
    """
    for i, di in enumerate(dict_list):
        di["idx_cao"] = np.empty((0, 2), dtype=int)
        di["idx_nocao"] = np.empty((0, 2), dtype=int)
        for K in range(nK):
            if K in labels:
                if di[f"label_{K}"].ndim > 1:
                    di["idx_cao"] = np.append(di["idx_cao"], di[f"label_{K}"], axis=0) 
            else:
                if di[f"label_{K}"].ndim > 1:
                    di["idx_nocao"] = np.append(di["idx_nocao"], di[f"label_{K}"], axis=0) 
        dict_list[i] = di 

    return dict_list




def extract_result_dict(dict_list, x_grid_coarse, closest_indices, distances_closest, first_date="20000301", last_date="20291231"):
    grouped_dict = defaultdict(list)

    # Group dictionaries by 'date' key

    for item in dict_list:
        grouped_dict[item['date']].append(item)

    # Convert defaultdict to a regular dict if needed
    grouped_dict = dict(grouped_dict)

    result_dict = {}

    for key in grouped_dict.keys():
        counts_cao = np.zeros_like(x_grid_coarse)
        counts_nocao = np.zeros_like(x_grid_coarse)
        if int(key) > int(last_date) or int(key) < int(first_date):
            continue
        for i, di in enumerate(grouped_dict[key]):
            # Extract coordinates
            cord_x_cao = di["idx_cao"][:, 0]
            cord_y_cao = di["idx_cao"][:, 1]
            
            cord_x_nocao = di["idx_nocao"][:, 0]
            cord_y_nocao = di["idx_nocao"][:, 1]

            # Find closest coarse indices
            closest_x_cao = closest_indices[:, :, 0][cord_x_cao, cord_y_cao]
            closest_y_cao = closest_indices[:, :, 1][cord_x_cao, cord_y_cao]
            
            closest_x_nocao = closest_indices[:, :, 0][cord_x_nocao, cord_y_nocao]
            closest_y_nocao = closest_indices[:, :, 1][cord_x_nocao, cord_y_nocao]

            # Find distances
            dist_cao = distances_closest[cord_x_cao, cord_y_cao]
            dist_nocao = distances_closest[cord_x_nocao, cord_y_nocao]

            # Use sets to manage coordinates
            cao_coordinates_set = set(zip(closest_x_cao, closest_y_cao))
            nocao_coordinates_set = set(zip(closest_x_nocao, closest_y_nocao))

            cao_coordinates = np.array([closest_x_cao, closest_y_cao])

            for x_nocao, y_nocao in zip(cord_x_nocao, cord_y_nocao):
                closest_x_nocao_i = closest_indices[:, :, 0][x_nocao, y_nocao]
                closest_y_nocao_i = closest_indices[:, :, 1][x_nocao, y_nocao]
                matches = np.logical_and(cao_coordinates[0,:] == closest_x_nocao_i, cao_coordinates[1,:] == closest_y_nocao_i)
                idx = np.where(matches)[0]
                if len(idx) > 0:
                    dist_nocao_i = distances_closest[x_nocao, y_nocao]
                    dist_cao_i = distances_closest[cord_x_cao[idx], cord_y_cao[idx]] 

                    distcao_farther = np.where(dist_nocao_i < dist_cao_i)
                    distcao_shorter = np.where(dist_nocao_i > dist_cao_i)

                    if len(distcao_farther[0]) > len(distcao_shorter[0]):
                        counts_nocao[closest_x_nocao_i, closest_y_nocao_i] += 1

                    elif len(distcao_farther[0]) < len(distcao_shorter[0]):
                        counts_cao[closest_x_nocao_i, closest_y_nocao_i] += 1
                    else:
                        rand_n = np.random.randint(0,2)
                        if rand_n == 0:
                            counts_nocao[closest_x_nocao_i, closest_y_nocao_i] += 1
                        else:   
                            counts_cao[closest_x_nocao_i, closest_y_nocao_i] += 1

                    # if np.any(dist_nocao_i < dist_cao_i):
                    #     counts_nocao[closest_x_nocao_i, closest_y_nocao_i] += 1
                    # else:
                    #     counts_cao[closest_x_nocao_i, closest_y_nocao_i] += 1
                else:
                    counts_nocao[closest_x_nocao_i, closest_y_nocao_i] += 1

            # Loop through cao points and handle non-conflicting points
            for x_cao, y_cao in zip(closest_x_cao, closest_y_cao):
                if (x_cao, y_cao) not in nocao_coordinates_set:
                    counts_cao[x_cao, y_cao] += 1


        # Calculate percentages
        total_counts = counts_cao + counts_nocao
        cao_perc = counts_cao / total_counts
        nocao_perc = counts_nocao / total_counts
        
        # Generate result based on random numbers and percentages
        random_numbers = np.random.rand(*counts_cao.shape)
        result = np.where((np.isnan(cao_perc) & np.isnan(nocao_perc)), np.nan, (random_numbers <= cao_perc).astype(int))
        
        # Update result dictionary
        result_dict[key] = result
    
    return result_dict


def extract_n_obs_per_day_xarray(dict_list, lon_grid, lat_grid, first_date="20000301", last_date="20291231"):
    np.random.seed(42)

    grouped_dict = defaultdict(list)

    # Group dictionaries by 'date' key

    for item in dict_list:
        grouped_dict[item['date']].append(item)

    # Convert defaultdict to a regular dict if needed
    grouped_dict = dict(grouped_dict)

    result_dict = {}

    for key in grouped_dict.keys():
        if int(key) > int(last_date) or int(key) < int(first_date):
            continue

        mask_obs = np.zeros_like(lon_grid)
        
        for i, di in enumerate(grouped_dict[key]):
            # Extract coordinates
            cord_x_cao = di["idx_cao"][:, 0]
            cord_y_cao = di["idx_cao"][:, 1]
            
            cord_x_nocao = di["idx_nocao"][:, 0]
            cord_y_nocao = di["idx_nocao"][:, 1]
    
            mask_obs[cord_x_cao, cord_y_cao] +=1
            mask_obs[cord_x_nocao, cord_y_nocao] +=1
            

        result_dict[key] = mask_obs


    dates = [datetime.datetime.strptime(date, "%Y%m%d") for date in result_dict.keys()]
    data = np.stack(list(result_dict.values()))
    sorted_indices = sorted(range(len(dates)), key=lambda i: dates[i])
    sorted_dates = [dates[i] for i in sorted_indices]
    sorted_data = np.stack([list(result_dict.values())[i] for i in sorted_indices])

    result_xarray = xr.DataArray(
        sorted_data,
        coords={
            "time": sorted_dates,
            "lat": (("lat_dim", "lon_dim"), lat_grid),
            "lon": (("lat_dim", "lon_dim"), lon_grid)
        },
        dims=["time", "lat_dim", "lon_dim"]
    )

    # Rename dimensions for clarity
    result_xarray = result_xarray.rename({"lat_dim": "y", "lon_dim": "x"})

    return result_xarray

def extract_result_dict_same_grid(dict_list, x_grid_coarse, first_date="20000301", last_date="20291231"):
    np.random.seed(42)

    grouped_dict = defaultdict(list)

    # Group dictionaries by 'date' key

    for item in dict_list:
        grouped_dict[item['date']].append(item)

    # Convert defaultdict to a regular dict if needed
    grouped_dict = dict(grouped_dict)

    result_dict = {}

    for key in grouped_dict.keys():
        if int(key) > int(last_date) or int(key) < int(first_date):
            continue

        mask_cao = np.zeros_like(x_grid_coarse)
        mask_nocao = np.zeros_like(x_grid_coarse)
        
        for i, di in enumerate(grouped_dict[key]):
            # Extract coordinates
            cord_x_cao = di["idx_cao"][:, 0]
            cord_y_cao = di["idx_cao"][:, 1]
            
            cord_x_nocao = di["idx_nocao"][:, 0]
            cord_y_nocao = di["idx_nocao"][:, 1]
    
            mask_cao[cord_x_cao, cord_y_cao] +=1
            mask_nocao[cord_x_nocao, cord_y_nocao] +=1
            
        final_arr = np.where((mask_cao >= 1) & (mask_nocao < 1), 1, 0)
        final_arr = np.where((mask_cao >= 1) & (mask_nocao >= 1), mask_cao / (mask_cao + mask_nocao), final_arr)
        final_arr = np.where((mask_cao == 0) & (mask_nocao == 0), np.nan, final_arr)


        # Generate result based on random numbers and percentages
        random_numbers = np.random.rand(*x_grid_coarse.shape)
        result = np.where((np.isnan(final_arr)), np.nan, (random_numbers <= final_arr).astype(int))
        # Update result dictionary
        result_dict[key] = result
    
    return result_dict


def extract_result_dict_xarray(dict_list, lon_grid, lat_grid, closest_indices=None, distances_closest=None, first_date="20000301", last_date="20291231", same_grid=False):
    if same_grid:
        print("Expecting dict_list being on SAME grid as lon lat")
        result_dict = extract_result_dict_same_grid(dict_list, lon_grid, first_date, last_date)
        
    else:
        print("Expecting dict_list being on DIFFERENT grid than lon lat")
        result_dict = extract_result_dict(dict_list, lon_grid, closest_indices, distances_closest, first_date, last_date)

    # Convert result_dict to xarray DataArray
    dates = [datetime.datetime.strptime(date, "%Y%m%d") for date in result_dict.keys()]
    data = np.stack(list(result_dict.values()))
    sorted_indices = sorted(range(len(dates)), key=lambda i: dates[i])
    sorted_dates = [dates[i] for i in sorted_indices]
    sorted_data = np.stack([list(result_dict.values())[i] for i in sorted_indices])

    result_xarray = xr.DataArray(
        sorted_data,
        coords={
            "time": sorted_dates,
            "lat": (("lat_dim", "lon_dim"), lat_grid),
            "lon": (("lat_dim", "lon_dim"), lon_grid)
        },
        dims=["time", "lat_dim", "lon_dim"]
    )

    # Rename dimensions for clarity
    result_xarray = result_xarray.rename({"lat_dim": "y", "lon_dim": "x"})
    return result_xarray






def calculate_daily_coverage_for_months(ds, years, months, region_mask=None):
    arr = np.empty((len(years), len(months)), dtype=object)
    date_range = pd.date_range(start=f"2000-03-01", end=f"2025-02-28")
    ds_complete = ds.reindex(time=date_range)
    for i, year in enumerate(years):
        data_year = ds_complete.where(ds_complete['time.year'].isin(year), drop=True)
        for j, month in enumerate(months):
            data_month = data_year.where(data_year['time.month'].isin(month), drop=True)
            arr[i, j] = np.nanmean(data_month.values[:, region_mask[0], region_mask[1]], axis=1)#data_month.values[:, region_all[0], region_all[1]].mean(axis=0)
    array_list = []
    for i in range(arr.shape[1]):
        array_list.append(np.concatenate(arr[:, i]))
    return array_list, arr


def mk_test(data, alpha=0.95, x=None):
    y = data #*100
    if x is None:
        x = np.arange(0, len(y))
    else:
        x = np.array(x)
    mask = np.isfinite(y)

    # Theil-Sen slope detrend
    ts_slope, intercept, slope_low, slope_high = theilslopes(y[mask], x[mask], alpha=alpha)
    # plt.plot(x, y)
    # plt.show()
    # intercept =  np.median(y[mask] - ts_slope * x[mask]) 
    y_detrended = y[mask] - ts_slope * x[mask]

    # Autocorrelation
    acf_values = acf(y_detrended, nlags=1, fft=False)
    lag_1_ac = acf_values[1]

    # Prewhitening - Adjust time series if autocorrelation is significant
    if abs(lag_1_ac) > 0.1:  # Consider a threshold to decide on prewhitening
        y_pw = y_detrended - lag_1_ac * np.roll(y_detrended, 1)
        y_pw[0] = y_detrended[0]  # Keep first element unchanged
    else:
        y_pw = y_detrended 
    
    # add trend on the prewhitened data 
    y_pw = y_pw + ts_slope * x[mask]

    # Mann-Kendall test on prewhitened data
    if ts_slope != 0:
        tau, p_value = kendalltau(x[mask], y_pw)
    else:
        p_value = 1
    return ts_slope, intercept, p_value, slope_low, slope_high

def mk_test_for_months(data, months, alpha=0.95, x=None):
    n = len(months)
    if x is None:
        x = [None]*len(data)
    else:
        x = np.array(x)
    slopes = []
    slopes_low = []
    slopes_high = []

    intecepts = []
    p_values = [] 
    for idx in range(n):
        # data = np.concatenate(data[:, idx])
        slope, intercept, p_value, slope_low, slope_high = mk_test(data[idx], alpha, x[idx])

        slopes.append(slope)
        slopes_low.append(slope_low)
        slopes_high.append(slope_high)
        
        intecepts.append(intercept)
        p_values.append(p_value)
    
    return np.array(slopes), np.array(intecepts), np.array(p_values), np.array(slopes_low), np.array(slopes_high)

def seasonal_mannkendall_test(arr_YM, slope):
    
    
    arr_flat = arr_YM.flatten()
    arr_pw = np.zeros_like(arr_flat, dtype=object)#flatten()
    arr_concat = np.concatenate(arr_flat)
    x = np.arange(len(arr_concat))
    y_detrended = arr_concat - slope * x
    y_detrend_masked = y_detrended[np.isfinite(y_detrended)] 

    acf_values = acf(y_detrend_masked, nlags=1, fft=False)
    lag_1_ac = acf_values[1]

    # Prewhitening - Adjust time series if autocorrelation is significant
    y_pw = y_detrended - lag_1_ac * np.roll(y_detrended, 1)
    y_pw[0] = y_detrended[0]  # Keep first element unchanged

    # add trend on the prewhitened data 
    y_pw = y_pw + slope * x
    lens = [len(ar) for ar in arr_flat]

    pos = 0 
    for i in range(len(lens)):
        arr_pw[i] = y_pw[pos:pos + lens[i]]
        pos += lens[i]

    array_list = [np.concatenate(arr_pw.reshape(arr_YM.shape)[:, i]) for i in range(arr_YM.shape[1])]


    max_length = max([len(a) for a in array_list])

    # Pad arrays with NaNs to the max length
    padded_arrays = [np.pad(a, (0, max_length - len(a)), constant_values=np.nan) for a in array_list]

    # Convert to masked array
    y =  ma.vstack([ma.masked_invalid(pad_a) for pad_a in padded_arrays]).T
    seasonal = kendalltau_seasonal(y)

    return seasonal 

def find_repeats(arr):
    # This function assumes it may clobber its input.
    if len(arr) == 0:
        return np.array(0, np.float64), np.array(0, np.intp)

    # XXX This cast was previously needed for the Fortran implementation,
    # should we ditch it?
    arr = np.asarray(arr, np.float64).ravel()
    arr.sort()

    # Taken from NumPy 1.9's np.unique.
    change = np.concatenate(([True], arr[1:] != arr[:-1]))
    unique = arr[change]
    change_idx = np.concatenate(np.nonzero(change) + ([arr.size],))
    freq = np.diff(change_idx)
    atleast2 = freq > 1
    return unique[atleast2], freq[atleast2]

def dijk(x, yi):
    n = yi.shape[0]
    dy = yi - yi[:, np.newaxis]
    dx = x - x[:, np.newaxis]
    # we only want unique pairs of distinct indices
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return dy[mask] / dx[mask]




def theil_sen_seasonal(y_obj, alpha=0.95, x_obj=None):
    print(len(y_obj))
    slopes_i = []
    sigsq = 0 
    for i in range(len(y_obj)):
        if x_obj is None:
            x = np.arange(len(y_obj[i]))
        else:
            x = x_obj[i]
        mask = np.isfinite(y_obj[i])
        slopes_i.append(dijk(x[mask], y_obj[i][mask]))
        
        _, nxreps = find_repeats(x[mask])
        _, nyreps = find_repeats(y_obj[i][mask])
        ny = len(y_obj[i])            # n in Sen (1968)

        # Equation 2.6 in Sen (1968) summed for each season 
        sigsq += 1/18. * (ny * (ny-1) * (2*ny+5) -
                        sum(k * (k-1) * (2*k + 5) for k in nxreps) -
                        sum(k * (k-1) * (2*k + 5) for k in nyreps))
        
    y = np.concatenate(y_obj)
    if x_obj is None:
        x = np.arange(len(y))
    else:
        x = np.concatenate(x_obj)
    mask = np.isfinite(y)
    y = y[mask]
    x = x[mask]

    slopes = np.concatenate(slopes_i)
    medslope = np.median(slopes)
    medinter = np.median(y) - medslope * np.median(x)

    slopes.sort()

    if alpha > 0.5:
        alpha = 1. - alpha

    z = distributions.norm.ppf(alpha / 2.)
    # This implements (2.6) from Sen (1968)
        
     
    # Find the confidence interval indices in `slopes`
    
    nt = len(slopes)       # N in Sen (1968)
    sigma = np.sqrt(sigsq)
    Ru = min(int(np.round((nt - z*sigma)/2.)), len(slopes)-1)
    Rl = max(int(np.round((nt + z*sigma)/2.)) - 1, 0)

    delta = slopes[[Rl, Ru]]

    return medslope, medinter, delta

def theil_sen_multi_season(arr_YM):
    season_slopes = []
    season_intercepts = []
    season_ci = []
    p_values = []
    arr_season_list = [np.concatenate(arr_YM[:, i]) for i in range(arr_YM.shape[1])]
    for i in range(3):
        arr_season = arr_YM[:, i*3 : i*3+3]

        slope, inter, ci = theil_sen_seasonal(arr_season_list[i*3 : i*3+3], alpha=0.95)
     

        mk_result = seasonal_mannkendall_test(arr_season, slope) 
        p_glob_indep = mk_result["global p-value (indep)"]
        p_glob_dep = mk_result["global p-value (dep)"]
        p_season = mk_result["seasonal p-value"]
        p_values.append(p_glob_dep)
        season_slopes.append(slope)
        season_intercepts.append(inter)
        season_ci.append(ci)

    season_ci = np.stack(season_ci)

    return season_slopes, season_intercepts, season_ci[:,0], season_ci[:,1], p_values




def extract_merra_dict_list(dates,
                             merra_folder,
                             M_threshold = 4,
                             T_pressure=850, # hPa
                             data_loc="/uio/kant/geo-geofag-u1/fslippe/data/"):
    
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

    # dates = ["20190101"]
    merra_dates  = []
    merra_dict_list = []


    file = lookup_table["TS"].split(".")[1]
    file_T = lookup_table["T"].split(".")[1]

    ds_water_mask=xr.open_dataset(f"{data_loc}/land_sea_ice_mask/nimbus/with_lonlat/NSIDC0051_SEAICE_PS_N25km_20200302_v2.0.nc")
    coords_lowres = np.vstack((ds_water_mask.lat.values.ravel(), ds_water_mask.lon.values.ravel())).T
    tree = cKDTree(coords_lowres)
    ds_lon_lat = xr.open_dataset(f"{merra_folder}/2000/MERRA2_200.inst3_3d_asm_Np.20000903.SUB.nc")
    lat = ds_lon_lat.lat.values
    lon = ds_lon_lat.lon.values
    lon_highres, lat_highres = np.meshgrid(lon, lat)
    coords_highres = np.column_stack((lat_highres.ravel(), lon_highres.ravel()))
    distances, indices = tree.query(coords_highres, k=1,  eps=0.5)

    for date in dates:
        try:
            ds = xr.open_dataset(f"{data_loc}land_sea_ice_mask/nimbus/NSIDC0051_SEAICE_PS_N25km_{date}_v2.0.nc")
        except:
            print("using sea ice mask from 20241231")
            ds = xr.open_dataset(f"{data_loc}land_sea_ice_mask/nimbus/NSIDC0051_SEAICE_PS_N25km_20241231_v2.0.nc")

        if "F13_ICECON" in ds:
            full_water_mask = ds.F13_ICECON.values.ravel()
        else:
            full_water_mask = ds.F17_ICECON.values.ravel()
        mask_sea_ice = full_water_mask[indices].reshape(lon_highres.shape[:2])
        mask_sea_ice = np.where(mask_sea_ice>0.1, 0, 1).astype(bool)
        # date = convert_to_date(date)
        print(date, end="\r")
        dic = {}

        ds = xr.open_mfdataset(f"{merra_folder}/{date[:4]}/*{file}.{date}.SUB.nc")
        if T_pressure == 850:
            ds_Th = ds.T850
        else:
            dsT = xr.open_mfdataset(f"{merra_folder}/{date[:4]}/*{file_T}.{date}.SUB.nc")
            ds_Th = dsT.T.sel(lev=T_pressure)
        
            # Get 3hourly data to match with ds_Th
            ds = ds.isel(time=slice(None, None, 3))
        
        M = get_MCAOidx(ds.TS.values, ds_Th.values, ds.PS.values*1e-2, T_pressure).mean(axis=0)
        M[~mask_sea_ice] = np.nan 
        mask = np.where(M >= M_threshold)
        mask = np.array([mask[0], mask[1]]).T
        mask_nocao = np.where(M < M_threshold)
        mask_nocao = np.array([mask_nocao[0], mask_nocao[1]]).T
        dic["datetime"] = f"{date[:4]}-{date[4:6]}-{date[6:]}T00:00"
        dic["date"] = date
        dic["idx_cao"] = mask
        dic["idx_nocao"] = mask_nocao
        merra_dict_list.append(dic)
        merra_dates.append(date)

    return merra_dict_list, merra_dates



def load_dict_list(years, # [2019, 2020, 2021, 2022, 2023, 2024],
                    data_loc, # "/mn/vann/fslippe/modis/MOD02_npz/",
                    model_run_name, #
                    dict_ext_model_run_name, # 
                    cao_threshold=0, # 
                    patch_threshold=0,
                    last_filter=128,
                    nK=14,
                    stride=64,
                    addon="",
                    labels=None):

    time_periods = []
    for year in years:
        time_periods.append(f"{year}0101-{year}0531")
        time_periods.append(f"{year}0901-{year}1231")

    if cao_threshold >= 30:
        dict_list = np.load(f"{data_loc}model_pred_info/filter{last_filter}/{model_run_name}_filter{last_filter}_nK{nK}/caothr{cao_threshold}/sizethr{patch_threshold}/dict_filter{last_filter}_nK{nK}_caothr{cao_threshold}_sizethr_{patch_threshold}_stride{stride}{addon}.npy", allow_pickle=True)
    else:
        combined_dict_list = []# /uio/kant/geo-geofag-u1/fslippe/data/model_pred_info/filter128/dnb_ice01_l95_z50_ps128_band31_filter128_nK7/caothr0/sizethr15/labels_3_6
        try:
            for time_period in time_periods:
                if labels != None:
                    label_str = "labels" 

                    for l in labels :
                        label_str = label_str + f"_{l}" 

                    path_with_wildcard = f"{data_loc}model_pred_info/filter{last_filter}/{model_run_name}_filter{last_filter}_nK{nK}/caothr{cao_threshold}/sizethr{patch_threshold}/{label_str}/dict_all_labels_{dict_ext_model_run_name}_filter{last_filter}_nK{nK}_caothr{cao_threshold}_sizethr{patch_threshold}_stride{stride}{addon}_{time_period}*"
                else:
                    path_with_wildcard = f"{data_loc}model_pred_info/filter{last_filter}/{model_run_name}_filter{last_filter}_nK{nK}/caothr{cao_threshold}/sizethr{patch_threshold}/dict_all_labels_{dict_ext_model_run_name}_filter{last_filter}_nK{nK}_caothr{cao_threshold}_sizethr{patch_threshold}_stride{stride}{addon}_{time_period}*"
                # print(path_with_wildcard)
                files_at_current_threshold = glob.glob(path_with_wildcard)
                # print(path_with_wildcard)
                # print(files_at_current_threshold)
                
                for file in files_at_current_threshold:
                    print(file)
                    combined_dict_list.append(np.load(file, allow_pickle=True))
            dict_list = np.concatenate(combined_dict_list)

        except Exception as e:
            print("An error occurred:", e)
            for time_period in time_periods:
                path_with_wildcard = f"{data_loc}model_pred_info/filter{last_filter}/{model_run_name}_filter{last_filter}_nK{nK}/caothr{cao_threshold}/sizethr{patch_threshold}/dict_all_labels_{dict_ext_model_run_name}filter{last_filter}_nK{nK}_caothr{cao_threshold}_sizethr_{patch_threshold}_stride{stride}{addon}_{time_period}*"
                files_at_current_threshold = glob.glob(path_with_wildcard)
                
                for file in files_at_current_threshold:
                    combined_dict_list.append(np.load(file, allow_pickle=True))
            
            dict_list = np.concatenate(combined_dict_list)
    
    return dict_list, time_periods



def get_closest_indices(x_grid, y_grid, x_grid_coarse, y_grid_coarse):
    # Flatten the coarse grids to 1D arrays
    x_grid_coarse_flat = x_grid_coarse.flatten()
    y_grid_coarse_flat = y_grid_coarse.flatten()

    # Initialize arrays to store closest indices and distances with the shape of fine grids
    closest_indices = np.zeros((x_grid.shape[0], x_grid.shape[1], 2), dtype=int)
    closest_distances = np.full((x_grid.shape[0], x_grid.shape[1]), np.inf)
    
    # Initialize Geod object with WGS84 ellipsoid
    geodesic = Geod(ellps='WGS84')

    # Function to compute geodesic distance using pyproj
    def compute_geodesic_distance(lon1, lat1, lon2, lat2):
        _, _, distance = geodesic.inv(lon1, lat1, lon2, lat2)
        return distance

    # Find the closest indices and corresponding distances
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            lon_f, lat_f = x_grid[i, j], y_grid[i, j]
            min_distance = float('inf')
            closest_index = -1

            # Iterate through all the coarse grid points to find the closest one
            for k, (lon_c, lat_c) in enumerate(zip(x_grid_coarse_flat, y_grid_coarse_flat)):
                distance = compute_geodesic_distance(lon_f, lat_f, lon_c, lat_c)

                if distance < min_distance:
                    min_distance = distance
                    closest_index = k
            
            # Convert the flat index back to 2D indices and store the closest distance
            closest_indices[i, j, 0] = np.unravel_index(closest_index, x_grid_coarse.shape)[0]
            closest_indices[i, j, 1] = np.unravel_index(closest_index, x_grid_coarse.shape)[1]
            closest_distances[i, j] = min_distance

    return closest_indices, closest_distances

def get_hist_counts_month(month, dict_list, x_grid, y_grid, coarse=False, closest_indices=None, x_grid_coarse=None, y_grid_coarse=None):
    dates_counted = {}
    counts = np.zeros_like(x_grid_coarse)
    tree = cKDTree(list(zip(x_grid_coarse.ravel(), y_grid_coarse.ravel())))
    if coarse:
        counts = np.zeros_like(x_grid_coarse)
        tree = cKDTree(list(zip(x_grid_coarse.ravel(), y_grid_coarse.ravel())))
    else:
        counts = np.zeros_like(x_grid)
        tree = cKDTree(list(zip(x_grid.ravel(), y_grid.ravel())))
    
    for i in range(len(dict_list)):
    # for i in range(1):
        date = dict_list[i]["date"]
        dtime_str = dict_list[i]["datetime"]
        dtime = datetime.datetime.strptime(dtime_str, '%Y-%m-%dT%H:%M')
        
        if dtime.month == month:
            if dict_list[i]["idx_cao"].ndim == 2:
                cord_x = dict_list[i]["idx_cao"][:,0]
                cord_y = dict_list[i]["idx_cao"][:,1]
            else:
                continue
            if "idx_open" in dict_list[i]:
                if dict_list[i]["idx_open"].ndim == 2:
                    cord_x = np.append(cord_x, dict_list[i]["idx_open"][:,0])
                    cord_y = np.append(cord_y, dict_list[i]["idx_open"][:,1])
            if len(cord_x) > 0:
                if coarse:
                    cord_x_reduced = closest_indices[:,:,0][np.array(cord_x), np.array(cord_y)]
                    cord_y_reduced = closest_indices[:,:,1][np.array(cord_x), np.array(cord_y)]

                    lat_i = y_grid_coarse[cord_x_reduced, cord_y_reduced]
                    lon_i = x_grid_coarse[cord_x_reduced, cord_y_reduced]
                else:
                    lat_i = y_grid[cord_x, cord_y]
                    lon_i = x_grid[cord_x, cord_y]

                x_proj, y_proj = lon_i.ravel(), lat_i.ravel()
                _, idxs = tree.query(list(zip(x_proj, y_proj)))
                
                for idx in idxs:
                    # counts.ravel()[idx] += 1
                    if idx not in dates_counted:
                        dates_counted[idx] = set()
                    if date not in dates_counted[idx]:
                        counts.ravel()[idx] += 1
                        dates_counted[idx].add(date)
    return counts






def get_hist_counts_month_in_year(month, year, dict_list, x_grid, y_grid, coarse=False, closest_indices=None, x_grid_coarse=None, y_grid_coarse=None):
    
    dates_counted = {}
    dates_counted_fram = {}
    if coarse:
        counts = np.zeros_like(x_grid_coarse)
        tree = cKDTree(list(zip(x_grid_coarse.ravel(), y_grid_coarse.ravel())))
    else:
        counts = np.zeros_like(x_grid)
        tree = cKDTree(list(zip(x_grid.ravel(), y_grid.ravel())))
    
    for i in range(len(dict_list)):
    # for i in range(1):
        date = dict_list[i]["date"]
        dtime_str = dict_list[i]["datetime"]
        dtime = datetime.datetime.strptime(dtime_str, '%Y-%m-%dT%H:%M')
        
        if dtime.month == month and dtime.year == year:
            if dict_list[i]["idx_cao"].ndim == 2:
                cord_x = dict_list[i]["idx_cao"][:,0]
                cord_y = dict_list[i]["idx_cao"][:,1]
            else:
                continue
            # if dict_list[i]["idx_open"].ndim == 2:
            #     cord_x = np.append(cord_x, dict_list[i]["idx_open"][:,0])
            #     cord_y = np.append(cord_y, dict_list[i]["idx_open"][:,1])

            if coarse:
                cord_x_reduced = closest_indices[:,:,0][cord_x, cord_y]
                cord_y_reduced = closest_indices[:,:,1][cord_x, cord_y]

                lat_i = y_grid_coarse[cord_x_reduced, cord_y_reduced]
                lon_i = x_grid_coarse[cord_x_reduced, cord_y_reduced]
            else:
                lat_i = y_grid[cord_x, cord_y]
                lon_i = x_grid[cord_x, cord_y]

            x_proj, y_proj = lon_i.ravel(), lat_i.ravel()
            _, idxs = tree.query(list(zip(x_proj, y_proj)))
            
            for idx in idxs:
                # counts.ravel()[idx] += 1
                if idx not in dates_counted:
                    dates_counted[idx] = set()
                if date not in dates_counted[idx]:
                    counts.ravel()[idx] += 1
                    dates_counted[idx].add(date)

    return counts




def plot_hist_map(x_grid,
                 y_grid,
                 counts,
                 tot_days,
                 projection,
                 title="Percentage of time with predicted CAO",
                 extent=[-50,50,55,84],
                 levels=10,
                 cmap="turbo",
                 lon_min=-55,
                 lon_max = 65,
                 lat_min=55,
                 lat_max = 82,
                 vertex_coords = [
                                    [-55, 55],  # Bottom left corner
                                    [16.5, 55],  # Bottom right corner before the turn upward
                                    [16.5, 67.5],  # Top right corner after the turn upward
                                    [65, 67.5],  # Top right corner after the turn rightward
                                    [65, 82],  # Top right corner
                                    [-55, 82]  # Top left corner
                ],
                plot_type="contourf"):
    

    fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(12, 7), dpi=300)

    plt.title(title)
    ax.set_extent(extent, ccrs.PlateCarree())  # Set extent to focus on the Arctic
    #new_cmap = ListedColormap(['white'] + [plt.get_cmap(cmap)(i) for i in range(plt.get_cmap(cmap).N)])
    try:
        turbo = plt.cm.turbo(np.linspace(0, 1, levels -1))
    except:
        turbo = plt.cm.turbo(levels)

    white = np.array([1, 1, 1, 1])  # RGBA values for white
    turbo_with_white = ListedColormap(np.vstack([white, turbo]))

    lon_grid_2d = x_grid
    lat_grid_2d = y_grid
    # Check if the points fall within the geographic bounds
    inside_bounds_mask = (lon_grid_2d >= lon_min) & (lon_grid_2d <= lon_max) & \
                        (lat_grid_2d >= lat_min) & (lat_grid_2d <= lat_max)

    # Create a 2D mask from the 1D mask, matching the original grid shape
    inside_bounds_mask_2d = inside_bounds_mask.reshape(x_grid.shape)

    # Mask the data array, setting points outside the region to np.nan
    masked_data = np.where(~inside_bounds_mask_2d, np.nan, counts / tot_days * 100)
    masked_x = np.where(~inside_bounds_mask_2d, np.nan, x_grid)
    masked_y = np.where(~inside_bounds_mask_2d, np.nan, y_grid)


    if plot_type == "contourf":
        c = ax.contourf(masked_x, masked_y, np.where(masked_data == 0 , np.nan, masked_data), transform=ccrs.PlateCarree(), levels=levels, cmap=cmap)#,set_under='white', extend="max")
    if plot_type == "pcolormesh":
        norm = BoundaryNorm(boundaries=levels, ncolors=256)
        c = ax.pcolormesh(x_grid, y_grid, np.where(masked_data == 0 , np.nan, masked_data), transform=ccrs.PlateCarree(),  cmap=cmap, norm=norm)#,set_under='white', extend="max")


    # ax.add_feature(cfeature.LAND, edgecolor='black')
    # ax.add_feature(cfeature.OCEAN)
    # ax.add_feature(cfeature.COASTLINE)
    ax.coastlines()

    cbar = plt.colorbar(c, ax=ax, orientation='vertical', label='Mean CAO frequency [%]')
    cbar.set_ticks([int(i) for i in cbar.get_ticks()])
    gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    gl.ylabels_right = False
    gl.xlabels_bottom = False

    # Define the limits of the "rectangle"


    # Define the number of points for smoothness
    num_pts = 100

    # Create arrays of latitudes and longitudes
    # lon_min, lon_max = -34.5, 46
    # lat_min, lat_max = 60.3, 82
    # vertex_coords = [
    #     [lon_min, lat_min],  # Bottom left corner
    #     [16.5, lat_min],  # Bottom right corner before the turn upward
    #     [16.5, 67.5],  # Top right corner after the turn upward
    #     [lon_max, 67.5],  # Top right corner after the turn rightward
    #     [lon_max, lat_max],  # Top right corner
    #     [lon_min, lat_max]  # Top left corner
    # ]

    # Convert coordinate lists into numpy arrays
    vertices = np.array(vertex_coords)

    # Interpolate points along the edges of the polygon for smooth transition
    num_pts = 100
    lons = np.concatenate([
        np.linspace(vertices[i, 0], vertices[i+1, 0], num_pts)
        for i in range(vertices.shape[0] - 1)
    ])
    lats = np.concatenate([
        np.linspace(vertices[i, 1], vertices[i+1, 1], num_pts)
        for i in range(vertices.shape[0] - 1)
    ])

    # Close the polygon loop by appending the first vertex at the end
    lons = np.append(lons, vertices[0, 0])
    lats = np.append(lats, vertices[0, 1])

    # Create a path of the "rectangle"
    path = mpath.Path(np.vstack((lons, lats)).T)

    # Create a patch from the path
    patch = matplotlib.patches.PathPatch(path, facecolor='none',
                                        edgecolor='black', linewidth=10, transform=ccrs.PlateCarree())

    # Add the patch to the Axes
    ax.add_patch(patch)

    return fig, ax



def plot_monthly_boxplots(month_names, slopes, intercepts, p_values, percentiles_low, percentiles_high, array_lists, model_names, alpha=0.05, figsize=None, legend=True, rotate_labels=True):
    if isinstance(figsize, list):
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
    else:
        fig, ax = plt.subplots(figsize=(2*len(month_names), 7), dpi=300)
    index = np.arange(len(month_names))

    box_spacing = 0.26
    
    # CAOnet slope values and percentiles
    
    # Calculate relative slopes for CAOnet
    pos = 0 
    n_models = len(slopes)

    # index =  index - box_spacing * n_models/2
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']#["#A60628", "skyblue", "#ff7f0e"]
    ymax = []
    for slope, intercept, p_value, ci_lower, ci_upper, array_list in zip(slopes, intercepts, p_values, percentiles_low, percentiles_high, array_lists):
        positions_model = index - box_spacing * (n_models-1)/2 + box_spacing*pos 

        relative_slopes = [sl / inter * len(arr) * 100 if inter != 0 else 0 for sl, inter, arr in zip(slope, intercept, array_list)]
        relative_ci_lower = [ci / inter * len(arr) * 100 if inter != 0 else 0 for ci, inter, arr in zip(ci_lower, intercept, array_list)]
        relative_ci_upper = [ci / inter * len(arr) * 100 if inter != 0 else 0 for ci, inter, arr in zip(ci_upper, intercept, array_list)]

        # Determine alpha values based on significance
        alpha_model = [1 if p <= alpha else 0.2 for p in p_value]

        # Plotting the slopes as points and adding error bars for the confidence intervals
        for i in range(len(alpha_model)):
            # ax.text(positions_model[i]-0.15 , relative_slopes[i], f"{np.nanmean(array_list[i])*100:.1f}%", ha='center', va='bottom', fontsize=10)
            ax.errorbar(positions_model[i], relative_slopes[i], yerr=[[relative_slopes[i] - relative_ci_lower[i]], [relative_ci_upper[i] - relative_slopes[i]]], fmt='o', capthick=3.5, capsize=8, color=colors[pos], alpha=alpha_model[i], label='CAOnet' if i == 0 else "", lw=3.5, markersize=8)

        # MERRA slope values and percentiles
        ymax.append(np.nanmax([abs(np.array(relative_ci_lower)), abs(np.array(relative_ci_upper))]))
        pos+=1
        
    # Determine alpha values based on significance
    ymax = np.max(ymax)
    ax.set_xticks(index-0.5)
    ymax += 0.05*ymax


    ax.set_ylim([-ymax, ymax])
    ax.set_xlim([-0.5, len(index)-0.5])

    ax.set_xticklabels([])
    rotation = 45 if rotate_labels else 0
    for i, month in enumerate(month_names):
        ax.text(index[i], -ymax-0.05*ymax, month, ha='center', va='top', rotation=0, fontsize=16, color='black')
    ax.axhline(0, color='black', linewidth=0.8)
    # ax.set_title("Monthly Trends with Significance (Relative Slopes with Confidence Intervals)")
    ax.set_facecolor("white")
    from matplotlib.patches import Patch
    
    if legend:
        ax.set_ylabel("25 year change [%]")
        legend_elements = [ ]
        for i in range(n_models):
            legend_elements.append(Patch(facecolor=colors[i], edgecolor='w', label=model_names[i]))

        ax.legend(handles=legend_elements, loc='upper left', title='Model')

    plt.tight_layout()
    return fig, ax


def plot_double_hist_map(x_grid, y_grid, count_list, tot_days, projection,
                         extent=[-50, 50, 55, 84], levels=10, cmap="turbo",
                         lon_min=-55, lon_max=65, lat_min=55, lat_max=82,
                         vertex_coords=None, plot_type="contourf", colorbar_fontsize=12):

    if vertex_coords is None:
        vertex_coords = [
            [-55, 55],
            [16.5, 55],
            [16.5, 67.5],
            [65, 67.5],
            [65, 82],
            [-55, 82]
        ]

    fig, axs = plt.subplots(1, len(count_list), subplot_kw={'projection': projection}, figsize=(8*len(count_list), 7), dpi=300)
    i = 0
    for ax, counts in zip(axs, count_list):
        ax.set_extent(extent, ccrs.PlateCarree())

        try:
            turbo = plt.cm.turbo(np.linspace(0, 1, levels - 1))
        except:
            turbo = plt.cm.turbo(levels)

        white = np.array([1, 1, 1, 1])
        turbo_with_white = ListedColormap(np.vstack([white, turbo]))

        lon_grid_2d = x_grid
        lat_grid_2d = y_grid
        inside_bounds_mask = (lon_grid_2d >= lon_min) & (lon_grid_2d <= lon_max) & \
                             (lat_grid_2d >= lat_min) & (lat_grid_2d <= lat_max)

        inside_bounds_mask_2d = inside_bounds_mask.reshape(x_grid.shape)
        masked_data = np.where(~inside_bounds_mask_2d, np.nan, counts / tot_days * 100)
        masked_x = np.where(~inside_bounds_mask_2d, np.nan, x_grid)
        masked_y = np.where(~inside_bounds_mask_2d, np.nan, y_grid)

        if plot_type == "contourf":
            c = ax.contourf(masked_x, masked_y, np.where(masked_data == 0, np.nan, masked_data),
                            transform=ccrs.PlateCarree(), levels=levels, cmap=cmap)
        elif plot_type == "pcolormesh":
            norm = BoundaryNorm(boundaries=levels, ncolors=256)
            c = ax.pcolormesh(x_grid, y_grid, np.where(masked_data == 0, np.nan, masked_data),
                              transform=ccrs.PlateCarree(), cmap=cmap, norm=norm)

        ax.coastlines()
        # if i == len(count_list)-1:
        #     cbar = plt.colorbar(c, ax=ax, orientation='vertical', label='[%]')
        #     cbar.set_ticks([int(i) for i in cbar.get_ticks()])
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.ylabels_right = False
        gl.xlabels_bottom = False

        vertices = np.array(vertex_coords)
        num_pts = 100
        lons = np.concatenate([
            np.linspace(vertices[i, 0], vertices[i+1, 0], num_pts)
            for i in range(vertices.shape[0] - 1)
        ])
        lats = np.concatenate([
            np.linspace(vertices[i, 1], vertices[i+1, 1], num_pts)
            for i in range(vertices.shape[0] - 1)
        ])

        lons = np.append(lons, vertices[0, 0])
        lats = np.append(lats, vertices[0, 1])

        path = mpath.Path(np.vstack((lons, lats)).T)
        patch = matplotlib.patches.PathPatch(path, facecolor='none', edgecolor='black',
                                             linewidth=10, transform=ccrs.PlateCarree())
        ax.add_patch(patch)
        i+=1

    cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.03]) 
    print("CHANGE")
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')#, aspect=2000)
    cbar.set_label("Mean CAO frequency [%]", fontsize=colorbar_fontsize)
    cbar.ax.tick_params(labelsize=colorbar_fontsize)
    plt.subplots_adjust(bottom=0.4, top=1.2)
    fig.tight_layout(rect=(0, 0.16, 1, 0.93))

    # plt.tight_layout()
    return fig, axs