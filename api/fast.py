from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image

import cv2
from tensorflow.keras import models


import joblib



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"message": "Hello"}


@app.get("/predict")
def predict(img_given):

    #ici handle image to put in pipeline
    #loaded_model = joblib.load('model.joblib')
    #result = loaded_model.predict(X_pred)

    return {'result' : img_given}



@app.post("/equations")
def get_image(file: UploadFile = File(...)):
    image = np.array(Image.open(file.file))

    loaded_model = models.load_model("modelDL")
    image_3 = np.expand_dims(image, -1)

    resized_3  =  cv2.cvtColor(image_3,cv2.COLOR_GRAY2RGB)

    prediction_scores = loaded_model.predict(np.expand_dims(resized_3, axis=0))
    predicted_index = np.argmax(prediction_scores)

    return {'result' : int(predicted_index)}
