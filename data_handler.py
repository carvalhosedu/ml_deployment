import pandas as pd
import json
import pickle

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

P_CLASS_MAP = {
    '1st': 1,
    '2nd': 2,
    '3rd': 3
}
SEX_MAP = {
    'Male': 0,
    'Female': 1
}
EMBARKED_MAP = {
    'Cherbourg': 1,
    'Queenstown': 2,
    'Southampton': 3
}

def survival_predictor(passageiro):
    passageiro['Pclass'] = P_CLASS_MAP[passageiro['Pclass']]
    passageiro['Sex'] = SEX_MAP[passageiro['Sex']]
    passageiro['Embarked'] = EMBARKED_MAP[passageiro['Embarked']]

    values = pd.DataFrame([passageiro])

    # carregando modelo de predição
    model = pickle.load(open('./models/model.pkl', 'rb'))

    results = model.predict(values)

    result = None

    if len(results) == 1:
        result = int(results[0])

    return result