import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from wordcloud import WordCloud

st.set_page_config(page_title="Phân tích tâm lý người tiêu dùng", layout="wide")

st.title("📊 Phân Tích Tâm Lý Người Tiêu Dùng")
st.markdown("Phân tích từ đánh giá sản phẩm trực tuyến để hiểu hành vi khách hàng.")

# --- Upload ---
uploaded_file = st.file_uploader("Tải file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("1️⃣ Tổng quan dữ liệu")
    st.write(df.head())

    # --- Xử lý chi tiêu và phân cụm ---
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if "AMOUNT" in df.columns:
        df["AMOUNT"] = df["AMOUNT"].astype(int)
    elif numeric_cols:
        df["AMOUNT"] = df[numeric_cols[0]]

    st.subheader("2️⃣ Phân cụm khách hàng (KMeans)")
    X = df[["AMOUNT"]]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)

    cluster_count = df["Cluster"].value_counts().sort_index()
    fig1, ax1 = plt.subplots()
    cluster_count.plot(kind="bar", ax=ax1)
    ax1.set_title("Số lượng khách hàng theo cụm")
    ax1.set_xlabel("Cụm")
    ax1.set_ylabel("Số lượng")
    st.pyplot(fig1)

    # --- Bất thường (outliers) ---
    st.subheader("3️⃣ Khách hàng bất thường (Isolation Forest)")
    iso = IsolationForest(contamination=0.05)
    df["Outlier"] = iso.fit_predict(X)
    outliers = df[df["Outlier"] == -1]
    st.write("🔍 Khách hàng chi tiêu bất thường:")
    st.dataframe(outliers[["CUST_ID", "AMOUNT"]])

    # --- Trích xuất từ khóa từ comment ---
    st.subheader("4️⃣ Từ khóa nổi bật trong đánh giá")
    if "COMMENT" in df.columns:
        comments = " ".join(df["COMMENT"].dropna().astype(str).tolist())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(comments)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.imshow(wordcloud, interpolation='bilinear')
        ax2.axis("off")
        st.pyplot(fig2)
    else:
        st.warning("Không có cột COMMENT để tạo wordcloud.")

    # --- Kết luận ---
    st.subheader("5️⃣ Kết luận & Đề xuất")
    st.markdown("""
    - **Các cụm khách hàng** có hành vi chi tiêu khác biệt rõ rệt.
    - **Khách hàng bất thường** có mức chi tiêu vượt xa trung bình.
    - **Từ khóa trong đánh giá** phản ánh nhu cầu về chất lượng, giá cả, dịch vụ.
    - **Đề xuất**: tập trung vào phân khúc hóa khách hàng, cải thiện sản phẩm theo insight thu được.
    """)

else:
    st.info("Hãy tải lên file CSV để bắt đầu phân tích.")
