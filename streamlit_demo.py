import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Dự đoán điểm cuối kỳ", page_icon="🎓", layout="wide")

DEFAULT_DATA_PATH = Path("TRAIN2.xlsx")

st.title("Demo dự đoán điểm cuối kỳ")
st.write("Ứng dụng dùng mô hình hồi quy tuyến tính để dự đoán điểm cuối kỳ từ điểm giữa kỳ.")

@st.cache_data
def load_data_from_file(file_or_path):
    df = pd.read_excel(file_or_path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

@st.cache_data
def prepare_data(df, feature_col, target_col):
    data = df[[target_col, feature_col]].copy()
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data[feature_col] = pd.to_numeric(data[feature_col], errors="coerce")
    data = data.dropna().reset_index(drop=True)
    data[target_col] = data[target_col].clip(0, 10)
    data[feature_col] = data[feature_col].clip(0, 10)
    return data

def train_model(data, feature_col, target_col, test_size, random_state):
    X = data[[feature_col]]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model_eval = LinearRegression()
    model_eval.fit(X_train, y_train)
    y_pred = model_eval.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²": r2_score(y_test, y_pred),
    }

    final_model = LinearRegression()
    final_model.fit(X, y)

    return final_model, metrics, (X_train, X_test, y_train, y_test, y_pred)

def draw_regression_plot(data, model, feature_col, target_col):
    x_line = np.linspace(data[feature_col].min(), data[feature_col].max(), 200).reshape(-1, 1)
    y_line = model.predict(pd.DataFrame(x_line, columns=[feature_col]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(data[feature_col], data[target_col], alpha=0.55, label="Dữ liệu thật")
    ax.plot(x_line, y_line, linewidth=2, label="Đường dự đoán")
    ax.set_xlabel("Điểm giữa kỳ")
    ax.set_ylabel("Điểm cuối kỳ")
    ax.set_title("Đường hồi quy tuyến tính")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig

def draw_actual_vs_predicted(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, y_pred, alpha=0.7)
    min_score = min(y_test.min(), y_pred.min())
    max_score = max(y_test.max(), y_pred.max())
    ax.plot([min_score, max_score], [min_score, max_score], linestyle="--")
    ax.set_xlabel("Điểm thật")
    ax.set_ylabel("Điểm dự đoán")
    ax.set_title("Thực tế vs dự đoán")
    ax.grid(True, alpha=0.3)
    return fig

def draw_computation_graph(b0, b1):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    boxes = [
        (0.08, 0.5, "Input\nmidterm = x"),
        (0.32, 0.5, f"Multiply\n{b1:.4f} × x"),
        (0.56, 0.5, f"Add bias\n+ {b0:.4f}"),
        (0.80, 0.5, "Output\nfinal_pred"),
    ]

    for x, y, text in boxes:
        ax.text(
            x, y, text,
            ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", linewidth=1.5),
        )

    for start, end in [(0.17, 0.25), (0.41, 0.49), (0.65, 0.73)]:
        ax.annotate("", xy=(end, 0.5), xytext=(start, 0.5), arrowprops=dict(arrowstyle="->", lw=1.8))

    ax.set_title("Đồ thị tính toán của mô hình", pad=20)
    return fig

with st.sidebar:
    st.header("Cấu hình dữ liệu")
    uploaded_file = st.file_uploader("Tải file dataset .xlsx", type=["xlsx"])
    test_size = st.slider("Tỉ lệ test", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

try:
    if uploaded_file is not None:
        df = load_data_from_file(uploaded_file)
    elif DEFAULT_DATA_PATH.exists():
        df = load_data_from_file(DEFAULT_DATA_PATH)
    else:
        st.error("Không tìm thấy TRAIN2.xlsx. Hãy đặt file cùng thư mục với app.py hoặc upload file ở sidebar.")
        st.stop()
except Exception as exc:
    st.error(f"Không đọc được file Excel: {exc}")
    st.stop()

st.subheader("1. Xem trước dữ liệu")
st.dataframe(df.head(10), use_container_width=True)

columns = list(df.columns)

def default_col(name, fallback_index):
    return name if name in columns else columns[min(fallback_index, len(columns) - 1)]

col1, col2 = st.columns(2)
with col1:
    feature_col = st.selectbox("Cột điểm giữa kỳ / biến đầu vào X", columns, index=columns.index(default_col("midterm", 1)))
with col2:
    target_col = st.selectbox("Cột điểm cuối kỳ / biến cần dự đoán y", columns, index=columns.index(default_col("final", 0)))

if feature_col == target_col:
    st.error("Cột đầu vào và cột cần dự đoán phải khác nhau.")
    st.stop()

data = prepare_data(df, feature_col, target_col)
if len(data) < 10:
    st.error("Dữ liệu hợp lệ quá ít để train model.")
    st.stop()

model, metrics, eval_data = train_model(data, feature_col, target_col, test_size, int(random_state))
X_train, X_test, y_train, y_test, y_pred = eval_data

b0 = float(model.intercept_)
b1 = float(model.coef_[0])

st.subheader("2. Công thức mô hình")
st.latex(r"\hat{y} = \beta_0 + \beta_1x")
st.code(f"final_pred = {b0:.6f} + {b1:.6f} * midterm", language="text")

metric_cols = st.columns(3)
metric_cols[0].metric("MAE", f"{metrics['MAE']:.4f}")
metric_cols[1].metric("RMSE", f"{metrics['RMSE']:.4f}")
metric_cols[2].metric("R²", f"{metrics['R²']:.6f}")

st.subheader("3. Dự đoán điểm cuối kỳ")
midterm_score = st.slider("Nhập điểm giữa kỳ", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
prediction = float(model.predict(pd.DataFrame({feature_col: [midterm_score]}))[0])
prediction = float(np.clip(prediction, 0, 10))
st.success(f"Điểm cuối kỳ dự đoán: {prediction:.2f}")

st.subheader("4. Đồ thị")
left, right = st.columns(2)
with left:
    st.pyplot(draw_regression_plot(data, model, feature_col, target_col), use_container_width=True)
with right:
    st.pyplot(draw_actual_vs_predicted(y_test, y_pred), use_container_width=True)

st.subheader("5. Đồ thị tính toán tương ứng")
st.pyplot(draw_computation_graph(b0, b1), use_container_width=True)

with st.expander("Thông tin dữ liệu sau xử lý"):
    st.write(f"Số dòng hợp lệ: {len(data)}")
    st.dataframe(data.describe(), use_container_width=True)