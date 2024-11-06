from fastapi import FastAPI
import mlflow
import random

app = FastAPI()

mlflow.set_tracking_uri('http://127.0.0.1:8080')

# MLflow registered model name
model_name = "tracking-quickstart"

def load_model():
    """Function to load the model from MLflow"""
    return mlflow.sklearn.load_model(f"models:/{model_name}/latest")


model = load_model()
next_model = model
p = 0.8

@app.post("/predict")
async def predict(request: dict):
    if random.random() < p:
        prediction = model.predict(request["features"])
        model_used = "current"
    else:
        prediction = next_model.predict(request["features"])
        model_used = "next"

    return {"model_used": model_used, "prediction": prediction.tolist()}


@app.post("/update-model")
async def update_model():
    global next_model
    next_model = load_model()

    return {"message": "Next model updated successfully."}


@app.post("/accept-next-model")
async def accept_next_model():
    global model
    model = next_model

    return {"message": "Next model accepted as current model."}
