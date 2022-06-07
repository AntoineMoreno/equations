import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib


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

def make_prediction(model, img, class_names):
    prediction_scores= model.predict(np.expand_dims(img, axis=0))
    predicted_index = np.argmax(prediction_scores)
    print("Predicted label: " + class_names[predicted_index])


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

def get_class_name(train_ds):
    return tuple(train_ds.class_names)

def fit_model(model, train_ds, val_ds, epochs):
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    ).history
    return hist
