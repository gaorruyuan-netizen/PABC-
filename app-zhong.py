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
# 白色论文风格界面
# =========================================================
st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
}
h1, h2, h3 {
    color: #0b2e59;
    font-weight: 700;
}
.stButton>button {
    background-color: #0b2e59;
    color: white;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
    font-weight: 600;
}
.metric-box {
    background: #edf7ed;
    border: 1px solid #b7dfc0;
    border-radius: 12px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

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
# 特征工程（必须保留！！！）
# =========================================================
def transform_temperature(T):
    return T / 800.0

def add_features(X):
    Cement, Sand, Water, SA, EP, BF, HRWR, DP, T = X.T

    Binder = Cement + SA + DP + HRWR
    TotalMass = Water + Cement + Sand + SA + EP + BF + HRWR + DP

    features = [
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
        (Sand + BF) / (Binder + EPS),
    ]

    T_prime = T
    T2 = T_prime ** 2
    logT = np.log1p(T_prime)
    invT = 1 / (T_prime + EPS)

    X_new = np.column_stack([X, T2, logT, invT] + features)
    return X_new

def preprocess_input(cement, sand, water, sa, ep, bf, hrwr, dp, t):
    X = np.array([[cement, sand, water, sa, ep, bf, hrwr, dp, t]])
    X[:, -1] = transform_temperature(X[:, -1])
    X = add_features(X)
    return X

def predict_strength(cement, sand, water, sa, ep, bf, hrwr, dp, t):
    X = preprocess_input(cement, sand, water, sa, ep, bf, hrwr, dp, t)
    y_pred = model.predict(X)[0]
    return y_pred

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

    st.header("说明")
    st.write("所有材料用量单位应与训练数据保持一致。")
    st.write("温度单位：°C")
    st.write("预测结果单位：MPa")

# =========================================================
# 输入
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
        pred = predict_strength(
            cement, sand, water, sa, ep, bf, hrwr, dp, t
        )

        st.markdown(f"""
        <div class="metric-box">
        <h3>预测结果：{pred:.3f} MPa</h3>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"预测失败：{e}")
