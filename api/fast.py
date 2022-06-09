from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image

import tensorflow as tf

from equations.train import make_prediction
from equations.test_data import test_data, test_data_with_positions
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
    positions = test_data_with_positions(image)

    class_names = give_classes()

    loaded_model = tf.keras.models.load_model("math_model_v1")

    elements_latex = []
    for image in list_images:
        elements_latex.append(to_latex(make_prediction(loaded_model, image, class_names)))

    positions_elements = []
    for position in positions:
        positions_elements.append(position[1])

    equa_final = []
    ignored_elements = []
    for i in range(len(elements_latex)):
        if positions_elements[i] == 'exponent':
            equa_final.append("^{"+f"{elements_latex[i]}"+"}")
        elif positions_elements[i] == 'index':
            equa_final.append("_{"+f"{elements_latex[i]}"+"}")
        elif positions_elements[i] == 'root':
            j=1
            radicandsss = []
            while (i+j)<len(elements_latex) and (positions_elements[i+j]== 'radicand' or positions_elements[i+j]== 'radicand-exp'or positions_elements[i+j]== 'radicand-index'):
                if positions_elements[i+j] == 'radicand-exp':
                    radicandsss.append("^{"+f"{elements_latex[i+j]}"+"}")
                elif positions_elements[i+j] == 'radicand-index':
                    radicandsss.append("_{"+f"{elements_latex[i+j]}"+"}")
                else:
                    radicandsss.append(elements_latex[i+j])
                j+=1
            r=''.join(radicandsss)
            equa_final.append("\\sqrt{"+r+"}")
        #elif positions_elements[i] == 'radicand':
        #    equa_final.append("_{"+f"{elements_latex[i]}"+"}")
        elif (positions_elements[i] == 'radicand' or positions_elements[i] == 'radicand-exp' or positions_elements[i] == 'radicand-index'):
            ignored_elements.append(elements_latex[i])
        else:
            equa_final.append(elements_latex[i])
       # elif image[1] == 'index':
            #d = f"_{{c}}"
            #elif image[1] == 'root':
            #   d = f"\sqrt{{c}}"
            #elif image[1] == 'radicand':
        #else:
            #d=c

    b=''.join(equa_final)

    print("Nombre contours: ",len(list_images))
    print("Resultat modele: ",elements_latex)
    print("Resultat position: ",positions_elements)


    return {'code LaTeX' : b}
