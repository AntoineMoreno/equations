import numpy as np
import tensorflow as tf
import numpy as np
import os
import pathlib

data_dir = pathlib.Path("../raw_data/data")

def create_dataset(batch_size = 32,img_height = 45, img_width = 45):
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

def get_class_name(train_ds):
    return tuple(train_ds.class_names)

#normalize
#print figures

def create_model():
    num_classes = 82
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])
    return model

def compile_model(model):
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

def normalize(train_ds):
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    return image_batch, labels_batch

def fit_model(model):
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3
    ).history
    return hist

def plot_graph(hist):
    plt.figure()
    plt.ylabel("Loss (training and validation)")
    plt.xlabel("Training Steps")
    plt.ylim([0,2])
    plt.plot(hist["loss"])
    plt.plot(hist["val_loss"])

    plt.figure()
    plt.ylabel("Accuracy (training and validation)")
    plt.xlabel("Training Steps")
    plt.ylim([0,1])
    plt.plot(hist["accuracy"])
    plt.plot(hist["val_accuracy"])

def make_prediction(model, img, class_names):
    prediction_scores= model.predict(np.expand_dims(img, axis=0))
    predicted_index = np.argmax(prediction_scores)
    print("Predicted label: " + class_names[predicted_index])
