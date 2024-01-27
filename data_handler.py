import pandas as pd
import json

def load_data():
    # Load data from csv file
    dados = pd.read_csv('./data/titanic.csv')
    return dados

def get_all_predictions():
    data = None
    with open('./data/predictions.json', 'r') as json_file:
        data = json.load(json_file)
    return data

def save_prediction(passageiro):
    data = get_all_predictions()

    data.append(passageiro)

    with open('./data/predictions.json', 'w') as json_file:
        json.dump(data, json_file)