import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
import io

st.set_page_config(page_title="Phân Tích Tâm Lý Người Tiêu Dùng", layout="wide")
st.title("📊 Phân Tích Tâm Lý Người Tiêu Dùng")

st.markdown("""
<style>
    .css-1d391kg {padding-top: 1rem;}
    .reportview-container .markdown-text-container { font-family: 'Arial'; font-size:16px; }
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Tải lên file CSV dữ liệu đã làm sạch", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("1. Dữ liệu ban đầu")
    st.dataframe(df.head())

    st.subheader("2. Phân tích cảm xúc từ đánh giá")
    def get_sentiment(text):
        if pd.isnull(text):
            return 0
        return TextBlob(str(text)).sentiment.polarity

    df["SENTIMENT"] = df["COMMENT"].apply(get_sentiment)
    st.write("Hoàn tất phân tích cảm xúc.")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    X = df[numeric_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("3. Phân cụm khách hàng (KMeans)")
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['CLUSTER'] = kmeans.fit_predict(X_scaled)
    st.bar_chart(df['CLUSTER'].value_counts())

    st.subheader("4. Phát hiện khách hàng bất thường (Isolation Forest)")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['OUTLIER'] = iso_forest.fit_predict(X_scaled)
    df['OUTLIER'] = df['OUTLIER'].map({1: 'Bình thường', -1: 'Bất thường'})
    st.dataframe(df[df['OUTLIER'] == 'Bất thường'][["CUST_ID", "OUTLIER"]])

    st.subheader("5. Từ khóa nổi bật trong đánh giá")
    vectorizer = CountVectorizer(stop_words='english', max_features=10)
    X_words = vectorizer.fit_transform(df['COMMENT'].dropna().astype(str))
    keywords = vectorizer.get_feature_names_out()
    st.write(", ".join(keywords))

    st.subheader("6. Gợi ý chiến lược theo cụm khách hàng")
    def product_strategy(cluster_id):
        if cluster_id == 0:
            return "Tập trung cải tiến dịch vụ hậu mãi và hỗ trợ khách hàng"
        elif cluster_id == 1:
            return "Cải tiến chất lượng sản phẩm, giá hợp lý"
        else:
            return "Đẩy mạnh quảng cáo, tập trung vào điểm mạnh sản phẩm"

    df["GỢI_Ý_CHIẾN_LƯỢC"] = df["CLUSTER"].apply(product_strategy)
    st.dataframe(df[["CUST_ID", "CLUSTER", "GỢI_Ý_CHIẾN_LƯỢC"]].head(10))

    st.subheader("7. Tải xuống kết quả phân tích")
    output_csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Tải file kết quả (.csv)",
        data=output_csv,
        file_name='ket_qua_phan_tich.csv',
        mime='text/csv'
    )
else:
    st.warning("Vui lòng tải lên file CSV để bắt đầu phân tích.")
