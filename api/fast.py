from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def index():
    return {"message": "Hello"}


@app.get("/predict")
def predict(img_given):

    #ici handle image to put in pipeline
    #loaded_model = joblib.load('model.joblib')
    #result = loaded_model.predict(X_pred)

    return {'result' : img_given}
