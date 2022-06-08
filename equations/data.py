import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

data_dir = pathlib.Path("../raw_data/data")

def create_dataset(data_dir= pathlib.Path("../raw_data/data"), batch_size = 32,img_height = 45, img_width = 45):
    """
    Return a train set and a val set
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    return train_ds, val_ds



def normalize(train_ds): #### problem
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    return image_batch, labels_batch

def whitening_images(img):
    ret, new_image = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
    return new_image
