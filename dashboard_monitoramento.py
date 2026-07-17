import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Dashboard de Manutenção Preditiva",
    layout="wide"
)

st.title("Dashboard de Monitoramento — Manutenção Preditiva")
st.write(
    "Aplicação complementar para estimar risco de falha de máquinas a partir do modelo salvo com joblib."
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "modelo_final_manutencao_preditiva.joblib"
DEFAULT_DATA_PATH = ROOT / "data" / "manutencao_preditiva.csv"

COLUNAS_VAZAMENTO = [
    "falha_maquina",
    "falha_twf",
    "falha_hdf",
    "falha_pwf",
    "falha_osf",
    "falha_rnf"
]
COLUNAS_IDENTIFICACAO = ["udi", "id_produto"]

@st.cache_resource
def carregar_modelo():
    return joblib.load(MODEL_PATH)

def preparar_dados(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    dados = df.copy()

    for col in dados.select_dtypes(include="number").columns:
        dados[col] = dados[col].fillna(dados[col].median())

    for col in dados.select_dtypes(exclude="number").columns:
        if not dados[col].mode().empty:
            dados[col] = dados[col].fillna(dados[col].mode()[0])
        else:
            dados[col] = dados[col].fillna("desconhecido")

    if "potencia_operacional" not in dados.columns:
        dados["potencia_operacional"] = dados["velocidade_rotacao_rpm"] * dados["torque_nm"]

    if "delta_temperatura" not in dados.columns:
        dados["delta_temperatura"] = dados["temperatura_processo_k"] - dados["temperatura_ar_k"]

    dados = dados.drop(columns=[c for c in COLUNAS_VAZAMENTO + COLUNAS_IDENTIFICACAO if c in dados.columns], errors="ignore")

    for feature in features:
        if feature not in dados.columns:
            dados[feature] = 0

    return dados[features]

artefato = carregar_modelo()
pipeline = artefato["pipeline"]
threshold = artefato["threshold"]
features = artefato["features"]
metricas = artefato["metrics"]

st.sidebar.header("Configurações")
threshold_dashboard = st.sidebar.slider(
    "Threshold de alerta de falha",
    min_value=0.05,
    max_value=0.95,
    value=float(threshold),
    step=0.05
)

uploaded_file = st.sidebar.file_uploader("Enviar novo CSV para monitoramento", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(DEFAULT_DATA_PATH)

X_monitoramento = preparar_dados(df, features)

probabilidades = pipeline.predict_proba(X_monitoramento)[:, 1]
predicoes = (probabilidades >= threshold_dashboard).astype(int)

resultado = df.copy()
resultado["probabilidade_falha"] = probabilidades
resultado["alerta_falha"] = predicoes

col1, col2, col3, col4 = st.columns(4)

col1.metric("Registros monitorados", len(resultado))
col2.metric("Alertas de falha", int(resultado["alerta_falha"].sum()))
col3.metric("Taxa de alerta", f"{resultado['alerta_falha'].mean() * 100:.2f}%")
col4.metric("Threshold ativo", f"{threshold_dashboard:.2f}")

st.subheader("Métricas do modelo salvo")
st.dataframe(pd.DataFrame([metricas]), use_container_width=True)

st.subheader("Distribuição dos alertas")
st.bar_chart(resultado["alerta_falha"].value_counts().rename(index={0: "Normal", 1: "Alerta"}))

st.subheader("Top 20 máquinas com maior risco previsto")
colunas_exibir = [c for c in ["udi", "id_produto", "tipo", "probabilidade_falha", "alerta_falha"] if c in resultado.columns]
top_risco = resultado.sort_values("probabilidade_falha", ascending=False).head(20)
st.dataframe(top_risco[colunas_exibir], use_container_width=True)

st.subheader("Base monitorada com previsões")
st.dataframe(resultado, use_container_width=True)

csv = resultado.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Baixar previsões em CSV",
    data=csv,
    file_name="previsoes_manutencao_preditiva.csv",
    mime="text/csv"
)