"""
Interface Streamlit 
"""

# DEPENDENCIAS #
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# CONFIGURAÇÕES DA PAGINA #
st.set_page_config(
    page_title='Previsões de Cancelamento',
    page_icon='🔮',
    layout='centered',
)

st.title('Previsão de Cancelamento de Serviços')
st.markdown("Preencha os dados do cliente abaixo para saber se ele tem risco de cancelar o serviço.")

# ARTEFATOS #
@st.cache_resource
def load_artifacts():
    model = joblib.load('./src/models/churn_model.pkl')
    scaler = joblib.load('./src/models/scaler.pkl')
    encoders = joblib.load('./src/models/encoders.pkl')
    
    return model, scaler, encoders

try:
    model, scaler, encoders = load_artifacts()
    artifacts_ok=True
except FileNotFoundError:
    artifacts_ok=False

if not artifacts_ok:
    st.error('Arquivos do modelo não encontrados.')
    st.stop()

# FORMULARIO #
st.subheader('Dados do Cliente')

col1, col2 = st.columns(2)

with col1:
    gender          = st.selectbox("Gênero", ["Male", "Female"])
    senior_citizen  = st.selectbox("Cliente Sênior?", ["Não", "Sim"])
    partner         = st.selectbox("Tem parceiro(a)?", ["Yes", "No"])
    dependents      = st.selectbox("Tem dependentes?", ["Yes", "No"])
    tenure          = st.slider("Tempo de contrato (meses)", 0, 72, 12)
    phone_service   = st.selectbox("Serviço telefônico?", ["Yes", "No"])
    multiple_lines  = st.selectbox("Múltiplas linhas?", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Serviço de internet", ["DSL", "Fiber optic", "No"])
 
with col2:
    online_security  = st.selectbox("Segurança online?", ["No", "Yes", "No internet service"])
    online_backup    = st.selectbox("Backup online?", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Proteção de dispositivo?", ["No", "Yes", "No internet service"])
    tech_support     = st.selectbox("Suporte técnico?", ["No", "Yes", "No internet service"])
    streaming_tv     = st.selectbox("Streaming TV?", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming filmes?", ["No", "Yes", "No internet service"])
    contract         = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Fatura sem papel?", ["Yes", "No"])
    payment_method   = st.selectbox(
        "Método de pagamento",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges  = st.number_input("Cobranças mensais ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
    total_charges    = st.number_input("Cobranças totais ($)", min_value=0.0, max_value=10000.0, value=monthly_charges * tenure, step=1.0)