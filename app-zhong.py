import streamlit as st
import joblib
import numpy as np
import pandas as pd

# =========================================================
# 页面配置
# =========================================================
st.set_page_config(
    page_title="PABC 抗压强度预测系统",
    page_icon="🧱",
    layout="wide"
)

# =========================================================
# 全局样式：白底 + 深蓝论文风格
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9fc;
        color: #1f2937;
    }

    .block-container {
        max-width: 1280px;
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
        height: 0rem;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: none;
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #0b2e59;
        margin-bottom: 0.35rem;
        letter-spacing: 0.5px;
    }

    .sub-text {
        color: #4b5563;
        font-size: 1.05rem;
        line-height: 1.75;
        margin-bottom: 1.2rem;
    }

    .white-card {
        background: #ffffff;
        border: 1px solid #dbe3ee;
        border-radius: 16px;
        padding: 22px 22px 18px 22px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
        margin-bottom: 18px;
    }

    .section-title {
        color: #0b2e59;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }

    .stNumberInput label {
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.98rem !important;
    }

    div[data-baseweb="input"] {
        background: #ffffff !important;
        border: 1.4px solid #cbd5e1 !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }

    div[data-baseweb="input"]:focus-within {
        border: 1.6px solid #0b2e59 !important;
        box-shadow: 0 0 0 2px rgba(11, 46, 89, 0.10) !important;
    }

    div[data-baseweb="input"] input {
        color: #111827 !important;
        background: #ffffff !important;
        font-weight: 500 !important;
    }

    div[data-baseweb="input"] button {
        color: #0b2e59 !important;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0b2e59 0%, #174a84 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        height: 3.2em;
        font-size: 1.02rem;
        font-weight: 700;
        box-shadow: 0 6px 16px rgba(23, 74, 132, 0.22);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #174a84 0%, #215c9d 100%) !important;
        color: #ffffff !important;
    }

    .result-card {
        background: linear-gradient(135deg, #eef9f0 0%, #f7fcf8 100%);
        border: 1px solid #b7dfc0;
        border-radius: 16px;
        padding: 18px 22px;
        margin-top: 6px;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.08);
    }

    .result-title {
        color: #166534;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .result-value {
        color: #14532d;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.2;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    details {
        background: #ffffff;
        border: 1px solid #dbe3ee;
        border-radius: 12px;
        padding: 0.4rem 0.8rem;
    }

    .stDataFrame {
        border: 1px solid #d9e2ec;
        border-radius: 10px;
        overflow: hidden;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

MODEL_PATH = "best_model.pkl"
FEATURE_COLS = ['Cement', 'Sand', 'Water', 'SA', 'EP', 'BF', 'HRWR', 'DP', 'T']
EPS = 1e-8

# =========================================================
# 加载模型
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# =========================================================
# 特征工程（保留，但不在界面显示）
# =========================================================
def transform_temperature(T: np.ndarray) -> np.ndarray:
    return np.asarray(T, dtype=float) / 800.0


def add_comprehensive_ratio_features(X: np.ndarray, feature_cols) -> np.ndarray:
    X = np.array(X, dtype=float)

    col_indices = {col: feature_cols.index(col) for col in feature_cols}

    Cement = X[:, col_indices['Cement']]
    Sand   = X[:, col_indices['Sand']]
    Water  = X[:, col_indices['Water']]
    SA     = X[:, col_indices['SA']]
    EP     = X[:, col_indices['EP']]
    BF     = X[:, col_indices['BF']]
    HRWR   = X[:, col_indices['HRWR']]
    DP     = X[:, col_indices['DP']]

    Binder = Cement + SA + DP + HRWR
    TotalMass = Water + Cement + Sand + SA + EP + BF + HRWR + DP

    ratio_features = [
        Water / (Cement + EPS),
        SA / (Cement + EPS),
        DP / (Cement + EPS),
        HRWR / (Cement + EPS),
        EP / (Sand + EPS),
        Water / (Binder + EPS),
        Sand / (Binder + EPS),
        TotalMass,
        Water / (TotalMass + EPS),
        BF / (Binder + EPS),
        Sand / (Sand + BF + EPS),
        (Sand + BF) / (Binder + EPS)
    ]

    ratio_features_array = np.column_stack(ratio_features)
    X_extended = np.hstack([X, ratio_features_array])

    return X_extended


def add_temperature_nonlinear_features(X: np.ndarray, temp_col_idx: int) -> np.ndarray:
    X = np.array(X, dtype=float)

    T_prime = X[:, temp_col_idx]
    T_prime_squared = T_prime ** 2
    logT = np.log1p(T_prime)
    invT = 1.0 / (T_prime + EPS)

    X_extended = np.hstack([
        X[:, :temp_col_idx + 1],
        T_prime_squared.reshape(-1, 1),
        logT.reshape(-1, 1),
        invT.reshape(-1, 1),
        X[:, temp_col_idx + 1:]
    ])

    return X_extended


def preprocess_input(cement, sand, water, sa, ep, bf, hrwr, dp, t) -> np.ndarray:
    X = np.array([[cement, sand, water, sa, ep, bf, hrwr, dp, t]], dtype=float)
    temp_col_idx = FEATURE_COLS.index('T')
    X[:, temp_col_idx] = transform_temperature(X[:, temp_col_idx])
    X = add_comprehensive_ratio_features(X, FEATURE_COLS)
    X = add_temperature_nonlinear_features(X, temp_col_idx=temp_col_idx)
    return X


def predict_strength(cement, sand, water, sa, ep, bf, hrwr, dp, t):
    X = preprocess_input(cement, sand, water, sa, ep, bf, hrwr, dp, t)
    y_pred = model.predict(X)[0]
    return float(y_pred)


# =========================================================
# 页面标题
# =========================================================
st.markdown('<div class="main-title">PABC 抗压强度预测系统</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-text">请输入材料配合比参数及温度条件，以获得 PABC 抗压强度预测结果。</div>',
    unsafe_allow_html=True
)

# =========================================================
# 侧边栏（仅显示你需要保留的内容）
# =========================================================
with st.sidebar:
    st.markdown("## 模型信息")
    st.write("**模型类型：** LightGBM")
    st.write("**预测目标：** 抗压强度")
    st.write("**输入变量：** 9个原始特征")

    st.markdown("## 说明")
    st.write("**所有材料用量单位应与训练数据保持一致。**")
    st.write("**温度单位：** °C")
    st.write("**预测结果单位：** MPa")

# =========================================================
# 输入区
# =========================================================
st.markdown('<div class="white-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">输入参数</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    cement = st.number_input("水泥 (g)", min_value=0.0, value=250.0, step=1.0)
    sa = st.number_input("SA (g)", min_value=0.0, value=20.0, step=1.0)
    hrwr = st.number_input("HRWR (g)", min_value=0.0, value=5.0, step=0.1)

with col2:
    sand = st.number_input("砂 (g)", min_value=0.0, value=180.0, step=1.0)
    ep = st.number_input("EP (g)", min_value=0.0, value=15.0, step=1.0)
    dp = st.number_input("DP (g)", min_value=0.0, value=10.0, step=1.0)

with col3:
    water = st.number_input("水 (g)", min_value=0.0, value=125.0, step=1.0)
    bf = st.number_input("BF (g)", min_value=0.0, value=0.0, step=1.0)
    t = st.number_input("温度 (°C)", min_value=0.0, value=20.0, step=1.0)

predict_button = st.button("预测抗压强度")
st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# 预测结果
# =========================================================
if predict_button:
    try:
        pred = predict_strength(cement, sand, water, sa, ep, bf, hrwr, dp, t)

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">预测结果</div>
                <div class="result-value">{pred:.3f} MPa</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"预测失败：{e}")
