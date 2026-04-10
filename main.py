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
