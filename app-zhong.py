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
    }

    .stNumberInput label {
        color: #374151 !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border: 1px solid #cfd8e3 !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="input"] input {
        background-color: #ffffff !important;
        color: #111827 !important;
    }

    .stButton > button {
        width: 100%;
        background-color: #0b2e59 !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        height: 3.2em;
        font-size: 16px;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #174a84 !important;
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
    }

    .metric-value {
        color: #14532d !important;
        font-size: 1.55rem;
        font-weight: 700;
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
# 特征工程（必须与训练时一致）
# =========================================================
def transform_temperature(T):
    return T / 800.0


def preprocess_input(cement, sand, water, sa, ep, bf, hrwr, dp, t):
    X = np.array([[cement, sand, water, sa, ep, bf, hrwr, dp, t]], dtype=float)
    X[:, -1] = transform_temperature(X[:, -1])
    return X


def predict_strength(cement, sand, water, sa, ep, bf, hrwr, dp, t):
    X = preprocess_input(cement, sand, water, sa, ep, bf, hrwr, dp, t)
    y_pred = model.predict(X)[0]
    return float(y_pred), X


# =========================================================
# 页面标题
# =========================================================
st.title("PABC 抗压强度预测系统")
st.write("请输入材料配合比参数及温度条件，以获得抗压强度预测结果。")

# =========================================================
# 侧边栏
# =========================================================
with st.sidebar:
    st.header("模型信息")
    st.write("模型类型：LightGBM")
    st.write("预测目标：抗压强度")
    st.write("输入变量：9个原始特征")
    st.write("温度处理方式：T / 800")

    st.header("说明")
    st.write("所有材料用量单位应与训练数据保持一致。")
    st.write("温度单位：°C")
    st.write("预测结果单位：MPa")

# =========================================================
# 输入区域
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    cement = st.number_input("水泥 (g)", 0.0, 1000.0, 250.0)
    sa = st.number_input("SA (g)", 0.0, 200.0, 20.0)
    hrwr = st.number_input("HRWR (g)", 0.0, 50.0, 5.0)

with col2:
    sand = st.number_input("砂 (g)", 0.0, 1000.0, 180.0)
    ep = st.number_input("EP (g)", 0.0, 200.0, 15.0)
    dp = st.number_input("DP (g)", 0.0, 200.0, 10.0)

with col3:
    water = st.number_input("水 (g)", 0.0, 500.0, 125.0)
    bf = st.number_input("BF (g)", 0.0, 200.0, 0.0)
    t = st.number_input("温度 (°C)", 0.0, 800.0, 20.0)

predict_button = st.button("预测抗压强度")

# =========================================================
# 预测
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

