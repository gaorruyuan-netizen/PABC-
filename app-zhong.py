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
        background-color: #ffffff;
        color: #1f2937;
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    header[data-testid="stHeader"] {
        background: #ffffff;
        border-bottom: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] {
        background-color: #f7f9fc;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] * {
        color: #1f2937 !important;
    }

    h1, h2, h3 {
        color: #0b2e59 !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px;
    }

    p, span, label, div {
        color: #1f2937;
    }

    .stNumberInput label,
    .stTextInput label,
    .stSelectbox label,
    .stTextArea label {
        color: #374151 !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border: 1px solid #cfd8e3 !important;
        border-radius: 10px !important;
        box-shadow: none !important;
    }

    div[data-baseweb="input"]:focus-within {
        border: 1px solid #0b2e59 !important;
        box-shadow: 0 0 0 1px #0b2e59 !important;
    }

    div[data-baseweb="input"] input {
        background-color: #ffffff !important;
        color: #111827 !important;
        font-weight: 500;
    }

    div[data-baseweb="input"] button {
        color: #0b2e59 !important;
    }

    .stButton > button {
        width: 100%;
        background-color: #0b2e59 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        height: 3.2em;
        font-size: 16px;
        font-weight: 600;
        transition: 0.2s ease-in-out;
    }

    .stButton > button:hover {
        background-color: #174a84 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(11, 46, 89, 0.18);
    }

    div[data-testid="stAlert"] {
        border-radius: 10px !important;
    }

    details {
        background: #ffffff;
        border: 1px solid #d9e2ec;
        border-radius: 10px;
        padding: 0.35rem 0.75rem;
    }

    .stDataFrame {
        border: 1px solid #d9e2ec;
        border-radius: 10px;
        overflow: hidden;
    }

    .custom-card {
        background-color: #f8fafc;
        border: 1px solid #d9e2ec;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 10px;
    }

    .intro-text {
        font-size: 1.02rem;
        line-height: 1.75;
        color: #374151;
        margin-top: -6px;
        margin-bottom: 18px;
    }

    .metric-box {
        background: linear-gradient(135deg, #edf7ed 0%, #f6fbf6 100%);
        border: 1px solid #b7dfc0;
        border-radius: 12px;
        padding: 16px 18px;
        margin-top: 10px;
        margin-bottom: 8px;
    }

    .metric-title {
        color: #166534 !important;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .metric-value {
        color: #14532d !important;
        font-size: 1.55rem;
        font-weight: 700;
    }

    .section-gap {
        margin-top: 0.3rem;
    }

    small {
        color: #6b7280 !important;
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
# 特征工程
# =========================================================
def transform_temperature(T: np.ndarray, method: str = "divide") -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if method == "divide":
        return T / 800.0
    elif method == "log":
        return np.log1p(T)
    else:
        raise ValueError(f"未知的温度变换方法: {method}")


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

    ratio_features = []

    ratio_features.append(Water / (Cement + EPS))
    ratio_features.append(SA / (Cement + EPS))
    ratio_features.append(DP / (Cement + EPS))
    ratio_features.append(HRWR / (Cement + EPS))
    ratio_features.append(EP / (Sand + EPS))

    ratio_features.append(Water / (Binder + EPS))
    ratio_features.append(Sand / (Binder + EPS))

    ratio_features.append(TotalMass)
    ratio_features.append(Water / (TotalMass + EPS))

    ratio_features.append(BF / (Binder + EPS))
    ratio_features.append(Sand / (Sand + BF + EPS))
    ratio_features.append((Sand + BF) / (Binder + EPS))

    ratio_features_array = np.column_stack(ratio_features)
    X_extended = np.hstack([X, ratio_features_array])

    return X_extended


def add_temperature_nonlinear_features(
    X: np.ndarray,
    temp_col_idx: int
):
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
    X[:, temp_col_idx] = transform_temperature(X[:, temp_col_idx], method='divide')

    X = add_comprehensive_ratio_features(X, FEATURE_COLS)
    X = add_temperature_nonlinear_features(X, temp_col_idx=temp_col_idx)

    return X


def predict_strength(cement, sand, water, sa, ep, bf, hrwr, dp, t):
    X = preprocess_input(cement, sand, water, sa, ep, bf, hrwr, dp, t)
    y_pred = model.predict(X)[0]
    return float(y_pred), X


# =========================================================
# 页面标题与说明
# =========================================================
st.title("PABC 抗压强度预测系统")
st.markdown(
    """
    <div class="intro-text">
    本应用基于训练完成的 <b>LightGBM</b> 模型，对 <b>PABC 抗压强度</b>进行快速预测。
    请输入材料配合比参数及温度条件，以获得实时预测结果。
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# 侧边栏
# =========================================================
with st.sidebar:
    st.header("模型信息")
    st.write("**模型类型：** LightGBM")
    st.write("**预测目标：** 抗压强度")
    st.write("**输入变量：** 9个原始特征")
    st.write("**温度变换：** T / 800")
    st.write("**附加特征工程：** 比例特征 + 温度非线性特征")

    st.header("说明")
    st.write("- 所有材料用量单位应与训练数据保持一致。")
    st.write("- 温度单位：°C")
    st.write("- 预测结果单位：MPa")

# =========================================================
# 主界面
# =========================================================
col1, col2 = st.columns([1.35, 1], gap="large")

with col1:
    st.subheader("输入参数")

    c1, c2, c3 = st.columns(3)
    with c1:
        cement = st.number_input("水泥 (g)", min_value=0.0, value=250.0, step=1.0)
        sa = st.number_input("SA (g)", min_value=0.0, value=20.0, step=1.0)
        hrwr = st.number_input("HRWR (g)", min_value=0.0, value=5.0, step=0.1)

    with c2:
        sand = st.number_input("砂 (g)", min_value=0.0, value=180.0, step=1.0)
        ep = st.number_input("EP (g)", min_value=0.0, value=15.0, step=1.0)
        dp = st.number_input("DP (g)", min_value=0.0, value=10.0, step=1.0)

    with c3:
        water = st.number_input("水 (g)", min_value=0.0, value=125.0, step=1.0)
        bf = st.number_input("BF (g)", min_value=0.0, value=0.0, step=1.0)
        t = st.number_input("温度 (°C)", min_value=0.0, value=20.0, step=1.0)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    predict_button = st.button("预测抗压强度", use_container_width=True)

with col2:
    st.subheader("系统说明")
    st.markdown(
        """
        <div class="custom-card">
        本界面集成了训练完成的 LightGBM 模型，并采用与模型开发阶段一致的特征工程流程，
        包括温度归一化、比例特征构建以及温度非线性特征扩展，从而保证离线训练与在线预测过程的一致性。
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# 预测结果
# =========================================================
if predict_button:
    try:
        pred, X_processed = predict_strength(
            cement, sand, water, sa, ep, bf, hrwr, dp, t
        )

        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-title">预测结果</div>
                <div class="metric-value">{pred:.3f} MPa</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("查看处理后的特征向量"):
            feature_names = [
                "Cement", "Sand", "Water", "SA", "EP", "BF", "HRWR", "DP", "T'",
                "T'^2", "logT", "invT",
                "W/C", "SA/Cement", "DP/Cement", "HRWR/Cement", "EP/Sand",
                "Water/Binder", "Sand/Binder", "TotalMass", "WaterFraction",
                "BF/Binder", "Fine/TotalAgg", "Agg/Binder"
            ]
            df = pd.DataFrame(X_processed, columns=feature_names)
            st.dataframe(df, use_container_width=True)

        raw_input_df = pd.DataFrame([{
            "水泥": cement,
            "砂": sand,
            "水": water,
            "SA": sa,
            "EP": ep,
            "BF": bf,
            "HRWR": hrwr,
            "DP": dp,
            "温度 (°C)": t
        }])

        st.subheader("输入参数汇总")
        st.dataframe(raw_input_df, use_container_width=True)

    except Exception as e:
        st.error(f"预测失败：{e}")