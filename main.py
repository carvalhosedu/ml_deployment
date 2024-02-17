from fastapi import Body, FastAPI
import data_handler
from typing import Any
import json

# rodar a nossa api
# uvicorn main:api --reload

api = FastAPI()

@api.get("/hello_world")
def hello_world():
    return {"Hello": "World"}

@api.get("/get_titanic_data")
def get_titanic():
    dados = data_handler.load_data()
    dados_json = dados.to_json(orient="records")
    return dados_json

@api.get("/get_all_predictions")
def get_all_predictions():
    predictions = data_handler.get_all_predictions()
    return predictions

@api.post("/save_prediction")
def save_prediction(passageiro_json: Any = Body(None)):
    passageiro = json.loads(passageiro_json)
    data_handler.save_prediction(passageiro)
    return {"status": "success"}

@api.post("/predict")
def predict(passageiro_json: Any = Body(None)):
    passageiro = json.loads(passageiro_json)
    survived = data_handler.survival_predictor(passageiro)
    return survived