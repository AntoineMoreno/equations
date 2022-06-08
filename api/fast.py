from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import cv2
from tensorflow.keras import models

from equations.train import make_prediction
from equations.test_data import test_data
from equations.utils import to_latex, give_classes



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

@app.post("/equations")
def get_image(file: UploadFile = File(...)):
    #print(type(file.file))
    image = np.array(Image.open(file.file))

    list_images=test_data(image)

    class_names = give_classes()

    loaded_model = models.load_model("modelDL2-MT")

    a = []
    for image in list_images:
        a.append(to_latex(make_prediction(loaded_model, image, class_names)))
    make_prediction(loaded_model, list_images[0], class_names)
    b=''.join(a)

    print("OUI OUI OUI",len(list_images))

    return {'code LaTeX' : b}
