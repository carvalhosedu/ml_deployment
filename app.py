import streamlit as st
import data_handler
import util
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Verificando senha dentro de secrets para dar continuidade a execução do app
if not util.check_password():
    st.stop()

# carregando dados do csv
dados = data_handler.load_data()

# carregando modelo de predição
model = pickle.load(open('./models/model.pkl', 'rb'))

data_analyses_on = st.toggle('Mostrar gráficos')

if(data_analyses_on):
    # mostrando os dados do csv
    st.dataframe(dados.head())
    st.header('Histograma das idades')
    # da para usar o echarts para plotar os graficos, tem muitas opções
    fig = plt.figure()
    plt.hist(dados.Age, bins=30)
    plt.xlabel('Idade')
    plt.ylabel('Quantidade')
    st.pyplot(fig)

    st.header('Sobreviventes')
    st.bar_chart(dados.Survived.value_counts())

st.header('Preditor de sobrevivência')
# nova linha com 3 colunas
col1, col2, col3 = st.columns(3)

# primeiro select box
with col1:
    classes = ["1st", "2nd", "3rd"]
    p_class = st.selectbox("Ticket Class", classes)

st.write(f"Ticket class: {p_class}")

# segundo select box
with col2:
    classes = ['Male', 'Female']
    sex = st.selectbox('Sex', classes)

# input number
with col3:
    age = st.number_input('Age in years', min_value=0, max_value=100, step=1)

# nova linha com 3 colunas
col1, col2, col3 = st.columns([2,2,1])

with col1:
    sib_sp = st.number_input('Number of Siblings / Spouses aboard', min_value=0, max_value=10, step=1)

with col2:
    par_ch = st.number_input('Number of Parents / Children aboard', min_value=0, max_value=10, step=1)

with col3:
    fare = st.number_input('Passenger Fare', min_value=0, max_value=1000)

# nova linha com 2 colunas
col1, col2 = st.columns(2)

with col1:
    classes = ['Cherbourg', 'Queenstown', 'Southampton']
    embarked = st.selectbox('Port of Embarkation', classes)

with col2:
    submit = st.button('Verificar')

# mapeando os valores de classe dos passageiros
p_class_map = {
    '1st': 1,
    '2nd': 2,
    '3rd': 3
}
sex_map = {
    'Male': 0,
    'Female': 1
}
embarked_map = {
    'Cherbourg': 1,
    'Queenstown': 2,
    'Southampton': 3
}

if submit or 'survived' in st.session_state:
    passageiro = {
        'Pclass': p_class_map[p_class],
        'Sex': sex_map[sex],
        'Age': age,
        'SibSp': sib_sp,
        'Parch': par_ch,
        'Fare': fare,
        'Embarked': embarked_map[embarked]
    }

    # st.write(passageiro)

    values = pd.DataFrame([passageiro])
    # st.dataframe(values)

    results = model.predict(values)

    if len(results) == 1:
        survided = int(results[0])

        if survided == 1:
            st.subheader('Passageiro Sobreviveu')
            if 'survived' not in st.session_state:
                st.balloons()
        else:
            st.subheader('Passageiro Não sobreviveu')
            if 'survived' not in st.session_state:
                st.snow()

        st.session_state['survived'] = survided


    if passageiro and 'survived' in st.session_state:
        st.write('A predição está correta?')

        col1, col2, col3 = st.columns([1,1,5])

        with col1:
            correct_prediction = st.button('👍')
        with col2:
            wrong_prediction = st.button('👎')

        if correct_prediction or wrong_prediction:
            message = "Muito obrigado pelo feedback"
            if wrong_prediction:
                message = ", irems usar esses dados parar melhorar nosso modelo"

            if correct_prediction:
                passageiro['CorrectPrediction'] = True
            if wrong_prediction:
                passageiro['CorrectPrediction'] = False

            passageiro['Survived'] = st.session_state['survived']

            st.write(message)

            data_handler.save_prediction(passageiro)

        col1, col2, col3 = st.columns(3)

        with col2:
            new_test = st.button("Iniciar nova análise")

            if new_test and 'survived' in st.session_state:
                del st.session_state['survived']
                st.rerun()

accuracy_predictions_on = st.toggle('Exibir acurácia')

if accuracy_predictions_on:
    # pega todas as predições salvas no JSON
    predictions = data_handler.get_all_predictions()
    # salva o número total de predições realizadas
    num_total_predictions = len(predictions)

    # calcula o número de predições corretas e salva os resultados conforme as predições foram sendo realizadas
    accuracy_hist = [0]
    # salva o numero de predições corretas
    correct_predictions = 0
    # percorre cada uma das predições, salvando o total móvel e o número de predições corretas
    for index, passageiro in enumerate(predictions):
        total = index + 1
        if passageiro['CorrectPrediction'] == True:
            correct_predictions += 1

        # calcula a acurracia movel
        temp_accuracy = correct_predictions / total if total else 0
        # salva o valor na lista de historico de acuracias
        accuracy_hist.append(round(temp_accuracy, 2))

    # calcula a acuracia atual
    accuracy = correct_predictions / num_total_predictions if num_total_predictions else 0

    # exibe a acuracia atual para o usuário
    st.metric(label='Acurácia', value=round(accuracy, 2))
    # TODO: usar o attr delta do st.metric para exibir a diferença na variação da acurácia

    # exibe o histórico da acurácia
    st.subheader("Histórico de acurácia")
    st.line_chart(accuracy_hist)




