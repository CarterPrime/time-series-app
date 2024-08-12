import streamlit as st
import joblib
import pandas as pd

# Charger les modèles
model_prophet = joblib.load('model_prophet.pkl')
model_arma = joblib.load('model_arma.pkl')
model_arima = joblib.load('model_arima.pkl')
model_sarima = joblib.load('model_sarima.pkl')

# Créer une interface utilisateur avec Streamlit
st.title("Prédictions de Séries Temporelles")

# Sélectionner le modèle
model_choice = st.selectbox("Choisissez le modèle", ["Prophet", "ARMA", "ARIMA", "SARIMA"])

# Entrée pour la date de prédiction
date_input = st.date_input("Sélectionnez une date", value=pd.to_datetime("2021-01-01"))

# Bouton pour faire une prédiction
if st.button("Faire une prédiction"):
    if model_choice == "Prophet":
        future = model_prophet.make_future_dataframe(periods=1, freq='D')
        forecast = model_prophet.predict(future)
        prediction = forecast.loc[forecast['ds'] == date_input.strftime('%Y-%m-%d'), 'yhat']
        if not prediction.empty:
            st.write(f"Prédiction pour la date {date_input}: {prediction.values[0]}")
        else:
            st.write("Pas de prédiction disponible pour cette date.")

    elif model_choice == "ARMA":
        # Prédiction ARMA (logique simplifiée)
        prediction = model_arma.forecast(steps=1)[0]
        st.write(f"Prédiction ARMA pour la date {date_input}: {prediction}")

    elif model_choice == "ARIMA":
        # Prédiction ARIMA (logique simplifiée)
        prediction = model_arima.forecast(steps=1)[0]
        st.write(f"Prédiction ARIMA pour la date {date_input}: {prediction}")

    elif model_choice == "SARIMA":
        # Prédiction SARIMA (logique simplifiée)
        prediction = model_sarima.forecast(steps=1)[0]
        st.write(f"Prédiction SARIMA pour la date {date_input}: {prediction}")
