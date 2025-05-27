import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import IsolationForest
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import io
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

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

    # Phân tích cảm xúc
    st.subheader("2. Phân tích cảm xúc từ đánh giá")
    def get_sentiment(text):
        if pd.isnull(text):
            return 0
        return TextBlob(str(text)).sentiment.polarity

    df["SENTIMENT"] = df["COMMENT"].apply(get_sentiment)
    st.write("Hoàn tất phân tích cảm xúc.")

    # TF-IDF
    st.subheader("3. Phân tích TF-IDF")
    tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['COMMENT'].fillna('').astype(str))
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    st.write("Từ khóa phổ biến (TF-IDF):")
    st.write(tfidf_df.sum().sort_values(ascending=False).head(10))

    # WordCloud
    st.subheader("4. WordCloud từ bình luận")
    text = " ".join(df['COMMENT'].dropna().astype(str))
    wordcloud = WordCloud(width=800, height=400).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)

    # LDA
    #st.subheader("5. Phân tích chủ đề (LDA)")
    #tokenized_docs = df['COMMENT'].dropna().apply(lambda x: word_tokenize(str(x).lower()))
    #dictionary = corpora.Dictionary(tokenized_docs)
    #corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    #lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=10)
    #topics = lda_model.print_topics(num_words=5)
    #for i, topic in topics:
    #    st.write(f"Chủ đề {i+1}: {topic}")

    # PCA
    st.subheader("6. Giảm chiều PCA và trực quan hóa")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.select_dtypes(include=np.number).fillna(0))
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]
    fig, ax = plt.subplots()
    sns.scatterplot(x='PCA1', y='PCA2', data=df, hue='SENTIMENT', ax=ax)
    st.pyplot(fig)

    # KMeans
    st.subheader("7. Phân cụm khách hàng (KMeans)")
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['CLUSTER'] = kmeans.fit_predict(scaled_data)
    st.bar_chart(df['CLUSTER'].value_counts())

    # Isolation Forest
    st.subheader("8. Phát hiện khách hàng bất thường")
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['OUTLIER'] = iso_forest.fit_predict(scaled_data)
    df['OUTLIER'] = df['OUTLIER'].map({1: 'Bình thường', -1: 'Bất thường'})
    st.dataframe(df[df['OUTLIER'] == 'Bất thường'][['CUST_ID', 'OUTLIER']])

    # Chiến lược
    st.subheader("9. Gợi ý chiến lược theo cụm khách hàng")
    def product_strategy(cluster_id):
        if cluster_id == 0:
            return "Tập trung cải tiến dịch vụ hậu mãi và hỗ trợ khách hàng"
        elif cluster_id == 1:
            return "Cải tiến chất lượng sản phẩm, giá hợp lý"
        else:
            return "Đẩy mạnh quảng cáo, tập trung vào điểm mạnh sản phẩm"

    df["GỢI_Ý_CHIẾN_LƯỢC"] = df["CLUSTER"].apply(product_strategy)
    st.dataframe(df[["CUST_ID", "CLUSTER", "GỢI_Ý_CHIẾN_LƯỢC"]].head(10))

    st.subheader("10. Tải xuống kết quả")
    output_csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Tải file kết quả (.csv)",
        data=output_csv,
        file_name='ket_qua_phan_tich.csv',
        mime='text/csv'
    )

else:
    st.warning("Vui lòng tải lên file CSV để bắt đầu phân tích.")
