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

# FUNC AUXILIAR #
## PREPROCESSING ##
def preprocess(input_dict, encoders, scaler):
    df = pd.DataFrame([input_dict])
 
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
 
    # feature engineering
    df["AvgCharges"]  = df["TotalCharges"] / (df["tenure"] + 1)
    df["NewCustomer"] = (df["tenure"] < 12).astype(int)
 
    # encoding
    for col, le in encoders.items():
        if col in df.columns:
            val = df[col].iloc[0]
            if val not in le.classes_:
                val = le.classes_[0]
            df[col] = le.transform([val])
 
    # ordem das colunas = treino
    feature_order = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges", "AvgCharges", "NewCustomer"
    ]
    df = df[feature_order]
 
    # scaling
    df_scaled = scaler.transform(df)
 
    return df_scaled

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
    online_security  = st.selectbox("Segurança online?", ["No", "Yes", "No internet service"])
 
with col2:
    online_backup    = st.selectbox("Backup online?", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Proteção de dispositivo?", ["No", "Yes", "No internet service"])
    tech_support     = st.selectbox("Suporte técnico?", ["No", "Yes", "No internet service"])
    streaming_tv     = st.selectbox("Streaming TV?", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming filmes?", ["No", "Yes", "No internet service"])
    contract         = st.selectbox("Tipo de contrato", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Fatura sem papel?", ["Yes", "No"])
    # payment_method   = st.selectbox(
    #     "Método de pagamento",
    #     ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    # )
    monthly_charges  = st.number_input("Cobranças mensais ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
    total_charges    = st.number_input("Cobranças totais ($)", min_value=0.0, max_value=10000.0, value=monthly_charges * tenure, step=1.0)

payment_method   = st.selectbox(
        "Método de pagamento",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

# PREVISOES #
st.divider()
if st.button('Gerar Previsões', type='primary', use_container_width=True):
    
    # inputs
    senior_value = 1 if senior_citizen == 'Sim' else 0
    input_data = {
        "gender":            gender,
        "SeniorCitizen":     senior_value,
        "Partner":           partner,
        "Dependents":        dependents,
        "tenure":            tenure,
        "PhoneService":      phone_service,
        "MultipleLines":     multiple_lines,
        "InternetService":   internet_service,
        "OnlineSecurity":    online_security,
        "OnlineBackup":      online_backup,
        "DeviceProtection":  device_protection,
        "TechSupport":       tech_support,
        "StreamingTV":       streaming_tv,
        "StreamingMovies":   streaming_movies,
        "Contract":          contract,
        "PaperlessBilling":  paperless_billing,
        "PaymentMethod":     payment_method,
        "MonthlyCharges":    monthly_charges,
        "TotalCharges":      total_charges,
    }
    
    
    # preprocess
    X_input = preprocess(input_data, encoders, scaler)

    # pred
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]

    # exibir previsao
    st.subheader('Resultado')

    if prediction == 1:
        st.error(f'Alto risco de churn com {probability:.1%} de probabilidade de cancelamento.')
        st.markdown("""
           Este cliente tem perfil de cancelamento. 
                    
           Considere ações de retenção como ofertas personalizadas ou contato proativo.
        """, text_alignment='center', )
    else:
        st.success(f'Baixo risco de churn com {probability:.1%} de probabilidade de cancelamento. ')
        st.markdown('Este cliente apresenta baixa probabilidade de cancelar o serviço. ')

    st.space('medium')
    st.progress(
        float(probability), 
        text='Risco de Cancelamento.',

    )

    

# FOOTER #
st.divider()
st.caption("Modelo: Regressão Logística treinada no dataset IBM Telco Customer Churn.", text_alignment='center')

