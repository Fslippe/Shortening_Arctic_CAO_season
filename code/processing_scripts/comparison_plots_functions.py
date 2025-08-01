import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs  
from matplotlib.colors import Normalize
#tf.config.threading.set_inter_op_parallelism_threads(1)
from code.model_training.functions import convert_to_date

import matplotlib.patches as mpatches


def get_theta(T, P, P0=1000):
    theta = T * (P0 / P) ** (0.286)
    return theta 

def get_MCAOidx(Tskn, T_p, Ps, p):
    theta_p = get_theta(T_p, p)
    thetaskn = get_theta(Tskn, Ps)
    return thetaskn - theta_p  

def get_data_mask_for_plotting(data_area, label_map, xi_shape, patch_size, strides, n_K):
    cao_index = np.where(data_area == 1)
    data_mask = np.ones(xi_shape)*-1

    for k in range(len(cao_index[0])):
        if cao_index[0][k] == 0: 
            row_start = cao_index[0][k] * strides 
        else:
            if label_map[cao_index[0][k] -1, cao_index[1][k]] == n_K:
                row_start = cao_index[0][k] * strides 
            else:
                row_start = cao_index[0][k] * strides + 64 -strides//2

        if cao_index[0][k] == (data_mask.shape[0] - patch_size) // strides :
            row_end = cao_index[0][k] * strides + 128
        else:
            if label_map[cao_index[0][k] + 1, cao_index[1][k]] == n_K:
                row_end = cao_index[0][k] * strides + 128
            else:
                row_end = cao_index[0][k] * strides + 64+strides//2

        if cao_index[1][k] == 0: 
            col_start = cao_index[1][k] * strides 
        else:
            if label_map[cao_index[0][k] , cao_index[1][k] - 1] == n_K:
                col_start = cao_index[1][k] * strides 
            else:
                col_start = cao_index[1][k] * strides + 64 -strides//2

        if cao_index[1][k] == (data_mask.shape[1] - patch_size) // strides :
            col_end = cao_index[1][k] * strides + 128
        else:
            if label_map[cao_index[0][k] , cao_index[1][k] + 1] == n_K:
                col_end = cao_index[1][k] * strides + 128
            else:
                col_end = cao_index[1][k] * strides + 64+strides//2
        data_mask[row_start:row_end, col_start:col_end] = 1


    data_mask = np.where(data_mask < 0, np.nan, 0.95)

    return data_mask

def get_merra_mask(date, time, lon_map, lat_map, model_areas, M_threshold, merra_folder):
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
                "TS": "MERRA2_400.tavg1_2d_slv_Nx",
                "QL":"MERRA2_400.inst3_3d_asm_Np",
                "QI":"MERRA2_400.inst3_3d_asm_Np",
                "T":"MERRA2_400.inst3_3d_asm_Np",
                "H":"MERRA2_400.inst3_3d_asm_Np",
                "PBLH": "MERRA2_400.tavg1_2d_flx_Nx",
                "PRECTOT": "MERRA2_400.tavg1_2d_flx_Nx",
                "PRECTOTCORR": "MERRA2_400.tavg1_2d_flx_Nx"
                }
    
    if len(date) == 7:
        converted_date = convert_to_date(date)
    elif len(date) == 8:
        converted_date = date
    print(converted_date)
    var = "TS"
    file_type = lookup_table[var].split(".")[1]
    Tskn =  xr.open_mfdataset(f"{merra_folder}{converted_date[:4]}/*{file_type}.{converted_date}.SUB.nc").TS
    lon_grid, lat_grid = np.meshgrid(Tskn.lon.values, Tskn.lat.values)

    var = "T850"
    file_type = lookup_table[var].split(".")[1]
    T850 =  xr.open_mfdataset(f"{merra_folder}{converted_date[:4]}/*{file_type}.{converted_date}.SUB.nc").T850
    var = "PS"
    file_type = lookup_table[var].split(".")[1]
    Ps =  xr.open_mfdataset(f"{merra_folder}{converted_date[:4]}/*{file_type}.{converted_date}.SUB.nc").PS
    var_arr = get_MCAOidx(Tskn, T850, Ps/100, 850)
    var_da = xr.DataArray(var_arr) 
    
    time_sel = converted_date + str(time).zfill(4)
    M_idx = var_da.sel(time=time_sel, method='nearest') > M_threshold
    coords_lowres = np.vstack((lat_grid.ravel(), lon_grid.ravel())).T
    tree = cKDTree(coords_lowres)
    coords_highres = np.column_stack((lat_map.ravel(), lon_map.ravel()))
    distances, merra_indices = tree.query(coords_highres, k=1,  eps=0.5)
    merra_indices = np.where(merra_indices == coords_lowres.shape[0], -1, merra_indices)
    merra_mask = M_idx.values.ravel()[merra_indices]#
    merra_mask = np.where(merra_indices==-1, np.nan, merra_mask).reshape(lat_map.shape[:2])
    merra_mask = np.where(np.isnan(model_areas), np.nan, merra_mask)

    return merra_mask



matplotlib.rcParams.update({'font.size': 20})

def plot_comparisons(x, lon_map, lat_map, label_map, model_areas, labeled_areas, dates, mod_min, lon_lats, M_threshold, patch_size,strides, n_K, merra_folder, save_dir="/uio/kant/geo-geofag-u1/fslippe/master_project/paper/figures/comparisons"):
    max_val = 7.2
    min_val = 1.5
    for i in range(len(x)):
        xi_shape = x[i].shape[:2]
        labeled_area = np.where(np.isnan(model_areas[i]), np.nan, labeled_areas[i])

        merra_mask = get_merra_mask(dates[i], mod_min[i], lon_map[i], lat_map[i], model_areas[i], M_threshold, merra_folder)
        merra_true_positive = np.where((merra_mask == 1) &  (labeled_area == 1), 1, 0) 
        merra_false_positive = np.where((merra_mask == 1) & (labeled_area != 1), 1, 0)    #np.where((model_diff != 0) & np.isfinite(model_areas[i]) & np.isnan(labeled_area), 1, 0) 
        merra_false_negative = np.where((merra_mask != 1) & (labeled_area == 1), 1, 0)

        merra0_mask = get_merra_mask(dates[i], mod_min[i], lon_map[i], lat_map[i], model_areas[i], 0, merra_folder)
        merra0_true_positive = np.where((merra0_mask == 1) &  (labeled_area == 1), 1, 0) 
        merra0_false_positive = np.where((merra0_mask == 1) & (labeled_area != 1), 1, 0)    #np.where((model_diff != 0) & np.isfinite(model_areas[i]) & np.isnan(labeled_area), 1, 0) 
        merra0_false_negative = np.where((merra0_mask != 1) & (labeled_area == 1), 1, 0)

        model_true_positive = np.where((model_areas[i] == 1) &  (labeled_area == 1), 1, 0) 
        model_false_positive = np.where((model_areas[i] == 1) & (labeled_area != 1), 1, 0)    #np.where((model_diff != 0) & np.isfinite(model_areas[i]) & np.isnan(labeled_area), 1, 0) 
        model_false_negative = np.where((model_areas[i] != 1) & (labeled_area == 1), 1, 0)   #np.where((model_diff != 0) & np.isnan(model_areas[i]) & np.isfinite(labeled_area), 1, 0) 


        labeled_label_mask = get_data_mask_for_plotting(labeled_area, label_map[i], xi_shape, patch_size, strides, n_K)
        merra_TP_mask =  get_data_mask_for_plotting(merra_true_positive, label_map[i], xi_shape, patch_size, strides, n_K)
        merra_FP_mask =  get_data_mask_for_plotting(merra_false_positive, label_map[i], xi_shape, patch_size, strides, n_K)
        merra_FN_mask =  get_data_mask_for_plotting(merra_false_negative, label_map[i], xi_shape, patch_size, strides, n_K)

        merra_tot_mask = np.where(np.isfinite(merra_TP_mask), 1, np.where(np.isfinite(merra_FP_mask), 0, np.where(np.isfinite(merra_FN_mask), 0.5, np.nan)))

        merra0_TP_mask =  get_data_mask_for_plotting(merra0_true_positive, label_map[i], xi_shape, patch_size, strides, n_K)
        merra0_FP_mask =  get_data_mask_for_plotting(merra0_false_positive, label_map[i], xi_shape, patch_size, strides, n_K)
        merra0_FN_mask =  get_data_mask_for_plotting(merra0_false_negative, label_map[i], xi_shape, patch_size, strides, n_K)

        merra0_tot_mask = np.where(np.isfinite(merra0_TP_mask), 1, np.where(np.isfinite(merra0_FP_mask), 0, np.where(np.isfinite(merra0_FN_mask), 0.5, np.nan)))

        model_TP_mask = get_data_mask_for_plotting(model_true_positive, label_map[i], xi_shape, patch_size, strides, n_K)
        model_FP_mask = get_data_mask_for_plotting(model_false_positive, label_map[i], xi_shape, patch_size, strides, n_K)
        model_FN_mask =  get_data_mask_for_plotting(model_false_negative, label_map[i], xi_shape, patch_size, strides, n_K)
        model_tot_mask = np.where(np.isfinite(model_TP_mask), 1, np.where(np.isfinite(model_FP_mask), 0, np.where(np.isfinite(model_FN_mask), 0.5, np.nan)))

        fig, test_ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()})

        # Plot the data to get the extent
        cb = test_ax.pcolormesh(lon_lats[i][0], lon_lats[i][1], x[i], cmap='gray_r', transform=ccrs.PlateCarree())
        extent = test_ax.get_extent(crs=ccrs.NorthPolarStereo())
        plt.close(fig)  # Close the test figure as it is no longer needed

        # Calculate the aspect ratio
        lon_min, lon_max, lat_min, lat_max = extent
        aspect_ratio = (lon_max - lon_min) / (lat_max - lat_min)
        base_width = 6  # Base width per plot
        base_height = base_width / aspect_ratio  # Height adjusted for aspect ratio

        ncols = 4
        fig_width = base_width * ncols
        fig_height = base_height + 0.2*base_height

        # Create the main figure with the calculated size
        fig, axs = plt.subplots(1, ncols, subplot_kw={'projection': ccrs.NorthPolarStereo()}, figsize=(fig_width, fig_height), dpi=200)

        combined_str = str(dates[i]) + str(mod_min[i]).zfill(4)
        datetime_obj = datetime.datetime.strptime(combined_str, "%Y%j%H%M")
        datetime_obj = str(datetime_obj).replace(" ", "_")
        plot_skip_idx = 1
        cb = axs[0].pcolormesh(lon_lats[i][0][::plot_skip_idx,::plot_skip_idx], lon_lats[i][1][::plot_skip_idx,::plot_skip_idx], x[i][::plot_skip_idx,::plot_skip_idx], vmin=min_val, vmax=max_val, cmap='gray_r', transform=ccrs.PlateCarree())
        cb = axs[1].pcolormesh(lon_lats[i][0][::plot_skip_idx,::plot_skip_idx], lon_lats[i][1][::plot_skip_idx,::plot_skip_idx], x[i][::plot_skip_idx,::plot_skip_idx], vmin=min_val, vmax=max_val, cmap='gray_r', transform=ccrs.PlateCarree())
        cb = axs[2].pcolormesh(lon_lats[i][0][::plot_skip_idx,::plot_skip_idx], lon_lats[i][1][::plot_skip_idx,::plot_skip_idx], x[i][::plot_skip_idx,::plot_skip_idx], vmin=min_val, vmax=max_val, cmap='gray_r', transform=ccrs.PlateCarree())
        cb = axs[3].pcolormesh(lon_lats[i][0][::plot_skip_idx,::plot_skip_idx], lon_lats[i][1][::plot_skip_idx,::plot_skip_idx], x[i][::plot_skip_idx,::plot_skip_idx], vmin=min_val, vmax=max_val, cmap='gray_r', transform=ccrs.PlateCarree())

        # axs[0].set_title("(a)")
        # axs[1].set_title("(b)")
        # axs[2].set_title("(c)")

        cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.02]) 
        cbar = fig.colorbar(cb, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("Radiance band 31 [W m-2 µm-1 sr-1]")

        RdBu = matplotlib.colormaps.get_cmap('RdBu')
        custom_cmap = ListedColormap(["tab:red","#ff7f0e", RdBu(0.95)])

        norm_binary = Normalize(vmin=0, vmax=1)
        image = axs[0].pcolormesh(lon_lats[i][0][::plot_skip_idx,::plot_skip_idx], lon_lats[i][1][::plot_skip_idx,::plot_skip_idx], labeled_label_mask[::plot_skip_idx,::plot_skip_idx],  transform=ccrs.PlateCarree(),
                        cmap=custom_cmap, zorder=2, alpha=0.4, norm=norm_binary)
        image = axs[1].pcolormesh(lon_lats[i][0][::plot_skip_idx,::plot_skip_idx], lon_lats[i][1][::plot_skip_idx,::plot_skip_idx], model_tot_mask[::plot_skip_idx,::plot_skip_idx],  transform=ccrs.PlateCarree(),
                        cmap=custom_cmap, zorder=2, alpha=0.4, norm=norm_binary)
        image = axs[2].pcolormesh(lon_lats[i][0][::plot_skip_idx,::plot_skip_idx], lon_lats[i][1][::plot_skip_idx,::plot_skip_idx], merra_tot_mask[::plot_skip_idx,::plot_skip_idx],  transform=ccrs.PlateCarree(),
                        cmap=custom_cmap, zorder=2, alpha=0.4, norm=norm_binary)
        image = axs[3].pcolormesh(lon_lats[i][0][::plot_skip_idx,::plot_skip_idx], lon_lats[i][1][::plot_skip_idx,::plot_skip_idx], merra0_tot_mask[::plot_skip_idx,::plot_skip_idx],  transform=ccrs.PlateCarree(),
                        cmap=custom_cmap, zorder=2, alpha=0.4, norm=norm_binary)

        patch1 = mpatches.Patch(color=matplotlib.colormaps.get_cmap(custom_cmap)(0.95), label='True Positive', alpha=0.8)
        patch2 = mpatches.Patch(color=matplotlib.colormaps.get_cmap(custom_cmap)(0), label='False Positive', alpha=0.8)
        patch3 = mpatches.Patch(color=matplotlib.colormaps.get_cmap(custom_cmap)(0.5), label='False Negative', alpha=0.8)

        # patch2 = mpatches.Patch(color='tab:red', label='Closed cells', alpha=0.8)
        patch_label = mpatches.Patch(color=matplotlib.colormaps.get_cmap(custom_cmap)(0.95), label='Label', alpha=0.8)

        axs[0].legend(title="Human",handles=[patch_label], loc="upper left")
        axs[0].coastlines()
        axs[0].gridlines()
        axs[0].set_facecolor("white")
        y_extent_diff = extent[3] - extent[2]
        # axs[0].text(extent[0], extent[3]-y_extent_diff*0.045, "(a)", fontsize=40)
        # axs[1].text(extent[0], extent[3]-y_extent_diff*0.045, "(b)", fontsize=40)
        # axs[2].text(extent[0], extent[3]-y_extent_diff*0.045, "(c)", fontsize=40)
        # axs[3].text(extent[0], extent[3]-y_extent_diff*0.045, "(d)", fontsize=40)




        axs[1].legend(title="CAOnet",handles=[patch1, patch2, patch3], loc="upper left")
        axs[1].coastlines()
        axs[1].gridlines()
        axs[1].set_facecolor("white")


        extent_from_axis1 = axs[0].get_extent(crs=ccrs.NorthPolarStereo())
        axs[2].set_extent(extent_from_axis1, crs=ccrs.NorthPolarStereo())
        axs[2].legend(title=r"$M_{3.75}$",handles=[patch1, patch2, patch3], loc="upper left")
        axs[2].coastlines()
        axs[2].gridlines()
        axs[2].set_facecolor("white")

        axs[3].set_extent(extent_from_axis1, crs=ccrs.NorthPolarStereo())
        axs[3].legend(title=r"$M_{0}$", handles=[patch1, patch2, patch3], loc="upper left")
        axs[3].coastlines()
        axs[3].gridlines()
        axs[3].set_facecolor("white")

        plt.subplots_adjust(bottom=0.4, top=1.2)

        # fig.tight_layout(rect=(0, 0.13, 1, 0.97))
        fig.tight_layout(rect=(0, 0.16, 1, 0.93))
        # date_final = datetime_obj.split(" ")[0] + "T" + datetime_obj.split(" ")[1]
        print(f"{save_dir}/CAO_pred_comparison_Mthr{M_threshold}_{datetime_obj}_skip{plot_skip_idx}.png")
        fig.savefig(f"{save_dir}/CAO_pred_comparison_Mthr{M_threshold}_{datetime_obj}_skip{plot_skip_idx}.png")
        print("saved fig:", i)
        plt.show()
        # plt.close()