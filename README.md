# Project Overview
[![DOI](https://zenodo.org/badge/1030253417.svg)](https://doi.org/10.5281/zenodo.16680335)
This repository contains the code and resources necessary for reproducing the results of this project, along with relevant data files, models, and documentation.

## Repository Structure

- **code/**: All code needed for reproducing the results of this project.
  - **interactive_notebook/**: Code containing a notebook for reproducing some of the plots of this project.
  - **processing_scripts/**: Scripts used for various tasks, including model training and pre-/post-processing.
  
- **data/**: Contains CAO masks for CAOnet and MERRA-2 in .nc file format, necessary for using the interactive notebook.
  
- **models/**: 
  - Autoencoder (encoder/decoder)
  - Max/min values for data normalization
  - Best-performing K-means model for clustering

- **LICENSE**: The License for this repository.

- **requirements.txt**: Lists all Python packages used in the production of this project. Python version: 3.9.16. The code is likely to work with other versions, with minimal adjustments required.

## Description of Scripts

### Processing Scripts in `code/processing_scripts`

The scripts are ordered chronologically based on their requirements in the processing chain:

1. **extract_modis_data.py**: Functions for extracting MODIS .hdf or processed .npz files; includes band extraction, swath combination, and masking.
   
2. **extract_npy_from_hdf.py**: Reduces dataset size by extracting single band .npz files from full MODIS .hdf files; the .npz files are used for further analysis.
   
3. **autoencoder.py**: Class definition for the autoencoder and code for patch extraction.
   
4. **save_train_test_data.py**: Saves train and test data for a specified year (via command line argument).
   
5. **write_patches_to_tf.ipynb**: Notebook for saving train and test data into .tfrecord files for autoencoder training on GPU.
   
6. **write_to_tf.py**: Supporting functions for .tfrecord extraction.
   
7. **train_autoencoder.py**: Trains the autoencoder for any given structure and input data; customizable parameters and extraction directories included.
   
8. **train_Kmeans.py**: Trains K-means with varying numbers of clusters, utilizing the trained autoencoder, and saves the K-means models.
   
9. **test_clusters.ipynb**: Notebook to evaluate the performance of any given CAOnet model on sample images; also used for defining CAO-aligning clusters for data labeling.
   
10. **generate_labeling_dataset_full_record.ipynb**: Generates a labeling dataset containing CAO, non-CAO, random, and mixed data subsets (500 images created).
    
11. **comparison_plots.ipynb**: Produces comparison plots between hand-labeled data, CAOnet, $$M_{3.75}$$, and $$M_0$$.
    
12. **comparison_plots_functions.py**: Supporting functions for the comparison plots notebook.
    
13. **calculate_CAOnet_scores.py**: Calculates scores for a specific CAOnet model.
    
14. **calculate_merra_scores.py**: Calculates scores for a specified M-threshold using MERRA-2 reanalysis data.
    
15. **find_best_model.ipynb**: Utilizes calculated scores to plot confusion matrices and identify the best-performing models.
    
16. **extract_dict_list_all_labels.py**: Extracts a CAOnet CAO mask dataset.
    
17. **CAO_masks_to_nc.ipynb**: Extracts the .nc file from the CAOnet CAO mask dataset.

### Supporting Functions

- **functions.py**: Contains general-purpose functions.
  
- **trends_and_climatology_functions.py**: Functions for trends and climatology analysis.
