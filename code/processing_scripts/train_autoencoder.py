import tensorflow as tf
from autoencoder import  SimpleAutoencoder
from keras.callbacks import LearningRateScheduler
import numpy as np 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, LearningRateScheduler
import os
import pickle
import sys 
from keras import backend as K

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

class CustomLearningRateScheduler(Callback):
    """Custom learning rate scheduler that runs after ReduceLROnPlateau."""

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Only reduce the lr at the start of epoch 15 or below
        if epoch == 14:
            K.set_value(self.model.optimizer.lr, 1e-4)

class CustomModelCheckpoint(Callback):
    def __init__(self, model, autoencoder, save_folder, model_run_name, save_freq):
        self.model = model
        self.autoencoder = autoencoder
        self.save_folder = save_folder
        self.model_run_name = model_run_name
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            # Save the complete model
            self.model.save(f"{self.save_folder}autoencoder_{self.model_run_name}_epoch_{epoch+1}.h5")
            # Save the individual encoder and decoder
            self.autoencoder.encoder.save(f"{self.save_folder}encoder_{self.model_run_name}_epoch_{epoch+1}.h5")
            self.autoencoder.decoder.save(f"{self.save_folder}decoder_{self.model_run_name}_epoch_{epoch+1}.h5")


def parse_function(example_proto, patch_size=64):
    # Define the feature description needed to decode the TFRecord
    feature_description = {
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'patch': tf.io.FixedLenFeature([], tf.string),
    }

    # Parse the input `tf.train.Example` proto using the dictionary above
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    # Decode the patch
    depth = parsed_example['depth']
    decoded_patch = tf.io.decode_raw(parsed_example['patch'], tf.float32)
    decoded_patch = tf.reshape(decoded_patch, (patch_size, patch_size, depth))
    return decoded_patch

def input_target_map_fn(patch):
    return (patch, patch)

def scheduler(epoch):
    if epoch < 1:
        return 1e-3
    else:
        return 1e-4


def main():
    #### Define parameters
    bands = [31]  
    band_str = ["_" + str(b) for b in bands]
    band_str = "".join(band_str) 
    scale_type = "normalized"
  
    # example for patch_size 384
    patch_size = 384 
    filters = [16, 32, 64, 128]
    batch_size = 32 
    buffer_size = 50000#total_records #patches_per_file * num_files

    print(f'Patch size is set to: {patch_size}')
    print(f'Filters is set to: {filters}')

    #### Define load and save names
    patch_load_name = f"dnb_l95_z50_ps384_band31"
    train_data_folder = "/folder_containing_tfrecords" 
    model_run_name = f"dnb_ice01_l95_z50_ps128_a1_band{band_str}_{len(filters)}layer_filters_{filters[0]}-{filters[-1]}" #%(patch_size, filters[-1], "201812", "202312")
    
    save_folder = f"/where_to_save_the_model/"


    #### prepare files
    print(f"{train_data_folder}/{scale_type}_trainingpatches_{patch_load_name}*.tfrecord")
    file_pattern = f"{train_data_folder}/{scale_type}_trainingpatches_{patch_load_name}*.tfrecord"
    files = tf.data.Dataset.list_files(file_pattern)
   
    num_files = len(tf.io.gfile.glob(file_pattern))
    print("Number of tfrecord files:", num_files)

    dataset = files.interleave(
    lambda x: tf.data.TFRecordDataset(x)
            .map(lambda item: parse_function(item, patch_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .map(input_target_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE),
    cycle_length=20,  # number of files read concurrently
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Set up model 
    autoencoder = SimpleAutoencoder(len(bands), patch_size, patch_size, filters=filters)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model = autoencoder.model(optimizer=optimizer, loss="combined")

    print("Preparing dataset prefetch")
    # Train the model on dataset
    total_records = sum(1 for _ in dataset) #534460
    total_records -= total_records % batch_size 
    print(total_records)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    steps_per_epoch = total_records // batch_size #patches_per_file * num_files // batch_size
    print("Finished dataset prefetch")

    print("Loading validation data")
    skip_every_val_patch = 1
    val_data = np.load(f"{train_data_folder}/{scale_type}_valpatches_{patch_load_name}.npy")[::skip_every_val_patch]

    print("Finished loading validation data")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    # val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_data))



    custom_lr_scheduler = CustomLearningRateScheduler()

    lr_schedule = ReduceLROnPlateau(
                                    monitor='val_loss', 
                                    factor=0.2, 
                                    patience=20, 
                                    verbose=1, 
                                    mode='auto',
                                    min_delta=0.00001,
                                    min_lr=5e-6
                                    )


    custom_checkpoint_callback = CustomModelCheckpoint(model, autoencoder, save_folder, model_run_name, save_freq=50)

    early_stopping = EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True, min_delta=0.000001)
    print("Finished all preparations")
    print("Starting training")
    history = model.fit(
            dataset,  
            epochs=1000,
            steps_per_epoch=steps_per_epoch,  
            validation_data=(val_data,val_data),  

            callbacks=[early_stopping, custom_lr_scheduler, lr_schedule, custom_checkpoint_callback]
    )

    # Save models
    model.save(f"{save_folder}autoencoder_{model_run_name}.h5")
    autoencoder.encoder.save(f"{save_folder}encoder_{model_run_name}.h5")
    autoencoder.decoder.save(f"{save_folder}decoder_{model_run_name}.h5")


    with open(f'{save_folder}/training_history_{model_run_name}.pkl' , 'wb') as f:
        pickle.dump(history.history, f)

if __name__ == "__main__":
    main()