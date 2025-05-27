import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.set_page_config(page_title="Phân tích tâm lý khách hàng", layout="wide")

# 1. Giới thiệu
st.title("🧠 Phân tích tâm lý khách hàng từ dữ liệu chi tiêu")
st.markdown("""
Ứng dụng giúp bạn:
- Xem dữ liệu khách hàng
- Thống kê mô tả
- Phân tích cụm (KMeans) để hiểu hành vi khách hàng
- Đề xuất chiến lược Marketing tương ứng
""")

# 2. Upload file
uploaded_file = st.file_uploader("📂 Tải lên file dữ liệu CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # loại bỏ khoảng trắng tên cột

    # Kiểm tra cột cần thiết
    if "Số tiền (VNĐ)" not in df.columns:
        st.error("❌ Không tìm thấy cột 'Số tiền (VNĐ)' trong dữ liệu.")
        st.stop()

    # 3. Hiển thị dữ liệu
    st.subheader("📋 Dữ liệu đầu vào")
    st.dataframe(df.head())

    # 4. Thống kê mô tả
    st.subheader("📊 Thống kê mô tả")
    st.write(df.describe())

    # 5. Biểu đồ phân bố
    st.subheader("📈 Biểu đồ phân bố 'Số tiền (VNĐ)'")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["Số tiền (VNĐ)"], bins=20, kde=True, ax=ax1)
    st.pyplot(fig1)

    # 6. Phân cụm KMeans
    st.subheader("🔎 Phân cụm khách hàng (KMeans)")
    X = df[["Số tiền (VNĐ)"]].dropna()

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X)

    st.write("✅ Dữ liệu sau khi phân cụm:")
    st.dataframe(df[["Số tiền (VNĐ)", "Cluster"]])

    # 7. Biểu đồ cụm
    st.subheader("🎯 Biểu đồ phân cụm khách hàng")
    fig2, ax2 = plt.subplots()
    colors = ['red', 'green', 'blue']
    for i in range(3):
        cluster_data = df[df["Cluster"] == i]
        ax2.scatter(cluster_data.index, cluster_data["Số tiền (VNĐ)"], label=f"Nhóm {i}", color=colors[i])
    ax2.set_title("Phân cụm theo số tiền chi tiêu")
    ax2.set_xlabel("Khách hàng")
    ax2.set_ylabel("Số tiền (VNĐ)")
    ax2.legend()
    st.pyplot(fig2)

    # 8. Gợi ý chiến lược
    st.subheader("💡 Gợi ý chiến lược marketing")
    strategies = {
        0: "Nhóm chi tiêu thấp - Khuyến mãi nhỏ, chương trình khách hàng mới.",
        1: "Nhóm chi tiêu trung bình - Ưu đãi định kỳ, tích điểm đổi quà.",
        2: "Nhóm chi tiêu cao - Ưu đãi VIP, dịch vụ cá nhân hóa, tri ân đặc biệt."
    }

    for i in range(3):
        st.markdown(f"**🎯 Nhóm {i}:** {strategies[i]}")

else:
    st.info("📝 Vui lòng tải lên file dữ liệu để bắt đầu.")
