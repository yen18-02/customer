import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# === 1. Đọc và xử lý dữ liệu ===
@st.cache_data
def load_data():
    uploaded_file = st.file_uploader("📁 Tải lên file CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        st.warning("⚠️ Vui lòng tải lên file CSV để tiếp tục.")
        st.stop()
    # Cột không xử lý số
    non_numeric_cols = ["CUST_ID", "COMMENT", "ITEM"]
    for col in df.columns:
        if col not in non_numeric_cols:
            df[col] = df[col].apply(clean_currency)

    df.dropna(inplace=True)
    return df

# === 2. Streamlit UI ===
st.set_page_config(page_title="Phân tích tâm lý khách hàng", layout="wide")
st.title("🔍 Phân tích tâm lý khách hàng")
df = load_data()

# === 3. Phân cụm KMeans ===
st.subheader("2️⃣ Phân cụm khách hàng (KMeans)")

non_numeric_cols = ["CUST_ID", "COMMENT", "ITEM"]
X = df.drop(columns=non_numeric_cols)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df["CLUSTER"] = clusters

# === 4. Hiển thị kết quả ===
st.write("### Kết quả phân cụm khách hàng")
st.dataframe(df[["CUST_ID", "CLUSTER", "COMMENT", "ITEM"]])

# === 5. Biểu đồ phân cụm ===
st.write("### Phân bố số lượng khách hàng theo cụm")
cluster_counts = df["CLUSTER"].value_counts().sort_index()
fig, ax = plt.subplots()
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, ax=ax)
ax.set_xlabel("Cụm")
ax.set_ylabel("Số lượng khách hàng")
ax.set_title("Phân bố theo cụm")
st.pyplot(fig)
