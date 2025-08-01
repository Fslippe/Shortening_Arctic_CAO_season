# %%
import os
import socket

os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.cluster import KMeans
import joblib
import numpy as np

# %%
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





# %%
print("Importing model...")

data_loc = "/location_of_data_for_loading_and_saving_models"
band_str = "31"
model_run_name = f"dnb_ice01_l95_z50_ps128_band{band_str}"

max_vals = np.load(f"{data_loc}models/patch_size128/filter128/max_val_{model_run_name}_2019-2023.npy")
min_vals = np.load(f"{data_loc}models/patch_size128/filter128/min_val_{model_run_name}_2019-2023.npy")
encoder = load_model(f"{data_loc}models/patch_size128/filter128/encoder_{model_run_name}.h5")

# %%
print("Extracting data...")

train_data_folder = "data_to_traiz"
files = [f for f in os.listdir(train_data_folder) if f.endswith(".npz")]
data_list = []
for file in files:
    if "test" in file:
        di = np.load(train_data_folder + file)["test_patches"]
    elif "train" in file:
        di = np.load(train_data_folder + file)["train_patches"]
    data_list.append(di)

all_data = np.concatenate(data_list)#.shape
all_data_scaled = (all_data - min_vals) / (max_vals - min_vals)#.max()


# %%
# val_data = np.load("/scratch/fslippe/modis/training_data/tf_data/dnb_ice01_l95_z50_ps128_band31/normalized_valpatches_dnb_ice01_l95_z50_ps128_band31.npy")
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

print("encoding...")
with tf.device('/CPU:0'):   
    encoded_patches_val = encoder.predict(all_data_scaled[:,:,:,0])                                       
    encoded_patches_flat_val = encoded_patches_val.reshape(encoded_patches_val.shape[0], -1)  

# %%
patch_size = 128 
last_filter = 128 
n_Ks = [7, 8, 9, 11, 12, 13, 14, 15, 16]
print("running K-means loop...")
for n_K in n_Ks:
    file_path = f"{data_loc}models/patch_size{patch_size}/filter{last_filter}/clustering/K{n_K}/cluster_{model_run_name}_filter{last_filter}_K{n_K}_full5years.pkl" 
    os.makedirs(f"{data_loc}models/patch_size{patch_size}/filter{last_filter}/clustering/K{n_K}" , exist_ok=True)
    cluster = KMeans(n_K, init='k-means++',
                        random_state=n_K).fit(encoded_patches_flat_val)
    joblib.dump(cluster, file_path)
    print(f"Finished K-means with {n_K} clusters")

# %%
