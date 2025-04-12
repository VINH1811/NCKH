import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import base64
# Thêm import numpy nếu chưa có
import numpy as np

# Load dữ liệu gốc để tính tỷ lệ
file_path = "german_credit_data.csv"  # Đảm bảo file này tồn tại trong thư mục làm việc
try:
    df = pd.read_csv(file_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
except FileNotFoundError:
    st.error("File 'german_credit_data.csv' không tồn tại. Vui lòng cung cấp file dữ liệu để tính tỷ lệ rủi ro từng đặc trưng.")
    df = None

# Tính tỷ lệ rủi ro xấu cho từng đặc trưng
def calculate_risk_rates(df, feature):
    if df is None:
        return {}
    risk_rates = df.groupby(feature)["Risk"].value_counts(normalize=True).unstack().fillna(0)
    risk_rates["Bad_Rate"] = risk_rates["bad"] * 100
    return risk_rates["Bad_Rate"].to_dict()

# Tính tỷ lệ rủi ro cho từng đặc trưng
if df is not None:
    age_risk_dict = calculate_risk_rates(df, "Age")
    job_risk_dict = calculate_risk_rates(df, "Job")
    credit_amount_risk_dict = calculate_risk_rates(df, "Credit amount")
    duration_risk_dict = calculate_risk_rates(df, "Duration")
    sex_risk_dict = calculate_risk_rates(df, "Sex")
    housing_risk_dict = calculate_risk_rates(df, "Housing")
    saving_risk_dict = calculate_risk_rates(df, "Saving accounts")
    checking_risk_dict = calculate_risk_rates(df, "Checking account")
    purpose_risk_dict = calculate_risk_rates(df, "Purpose")
else:
    age_risk_dict = job_risk_dict = credit_amount_risk_dict = duration_risk_dict = sex_risk_dict = housing_risk_dict = saving_risk_dict = checking_risk_dict = purpose_risk_dict = {}
# ========== CẤU HÌNH TRANG ==========
st.set_page_config(
    page_title="CreditRisk AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_image_base64(path):
    if not os.path.exists(path):
        st.error(f"Ảnh không tồn tại: {path}")
        return ""
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load ảnh nền từ local và encode base64
bg1 = local_image_base64("ad2.jpg")
bg2 = local_image_base64("risk2.jpg")
bg3 = local_image_base64("chiuu.jpg")

# ========== CẤU HÌNH TRANG ==========
# Chèn CSS vào ứng dụng
st.markdown(f"""
<style>
/* Import font Lato */
@import url("https://fonts.googleapis.com/css?family=Lato:400,700");

:root {{
    --primary: #2568FB;
    --secondary: #F8F9FA;
    --text: #2C3E50;
    --success: #28B463;
    --danger: #E74C3C;
}}

body, html {{
    margin: 0;
    padding: 0;
    height: 100vh;
    overflow: hidden;
    font-family: "Lato", sans-serif;
    color: white;
}}

/* Sidebar tùy chỉnh */
[data-testid="stSidebar"] {{
    background: transparent;
    color: white;
    z-index: 10;
    height: 100vh;
    background-color:rgba(240, 240, 240, 0);
    backdrop-filter: blur(15px);
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
}}

/* Tăng độ tương phản cho chữ trong sidebar */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] label {{
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
}}

/* Tùy chỉnh tiêu đề trong sidebar */
[data-testid="stSidebar"] h1 {{
    color: white;
    text-align: center;
    font-size: 2em;
    margin-bottom: 20px;
   
}}

/* Tùy chỉnh nhãn của radio button */
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    color: white;
    font-size: 1.2em;
}}

/* Đảm bảo các phần tử con của label cũng có màu trắng */
[data-testid="stSidebar"] [data-testid="stRadio"] label div {{
    color: white !important;
}}

/* Button style với hiệu ứng hover chuyển màu đỏ */
.stButton {{
    position: relative;
    display: flex;
    justify-content: center; /* Căn giữa nút trong cột */
    margin: 10px;
}}

.stButton > button {{
    width: 200px;
    color: white;
    font-family: Helvetica, sans-serif;
    font-weight: bold;
    font-size: 20px; /* Giảm kích thước chữ */
    text-align: center;
    text-decoration: none;
    background-color:rgba(240, 240, 240, 0);
    backdrop-filter: blur(15px);
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    display: block;
    position: relative;
    padding: 15px 24px; /* Tăng padding */
    -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
    text-shadow: 0px 1px 0px #000;
    -webkit-border-radius: 5px;
    -moz-border-radius: 5px;
    border-radius: 5px;
    z-index: 4;
    transition: background-color 0.3s ease;
    white-space: nowrap; /* Ngăn nội dung xuống dòng */
    height : 100px
}}

.stButton > button:hover {{
    background-color: var(--danger);
    color: black;
}}

/* Đảm bảo không có khoảng cách thừa phía trên nút */
.stButton {{
    margin-top: 0;
}}
/* Thẻ thông tin */
.custom-card {{
    border-radius: 12px;
    padding: 20px;
    background: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 20px;
    border: 1px solid #EDF2F7;
    z-index: 3;
}}

/* Slideshow nền toàn màn hình */
.slideshow-background {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-size: cover;
    background-position: center;
    z-index: 1;
    animation: slide 9s infinite, zoom 3s infinite;
}}

/* Keyframes chuyển ảnh từ base64 */
@keyframes slide {{
    0%, 30% {{
        background-image: url("data:image/jpg;base64,{bg1}");
        opacity: 1;
    }}
    33.33%, 63.33% {{
        background-image: url("data:image/jpg;base64,{bg2}");
        opacity: 1;
    }}
    66.66%, 96.66% {{
        background-image: url("data:image/jpg;base64,{bg3}");
        opacity: 1;
    }}
    31%, 32%, 64.66%, 65.66%, 97.66%, 99% {{
        opacity: 0;
    }}
}}

/* Hiệu ứng zoom nhẹ */
@keyframes zoom {{
    0% {{ transform: scale(1); }}
    50% {{ transform: scale(1.1); }}
    100% {{ transform: scale(1); }}
}}

/* Overlay làm mờ nền */
.overlay {{
    background: rgba(0, 0, 0, 0.3);
    position: fixed;
    top: 0; left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 2;
}}

/* Nội dung trung tâm */
.content {{
    position: relative;
    z-index: 3;
    color: white;
    text-align: center;
    padding: 50px 20px;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}}

.content h1 {{
    font-size: 3.5em;
    margin-bottom: 10px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}}

.content p {{
    font-size: 1.5em;
    opacity: 0.9;
    margin-bottom: 20px;
    max-width: 600px;
}}

.button-container {{
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    justify-content: center;
    z-index: 4;
}}
</style>
""", unsafe_allow_html=True)

# ========== HÀM TIỆN ÍCH ==========
@st.cache_resource
def load_model():
    return joblib.load("credit_risk_model.pkl")

@st.cache_resource
def load_preprocessor():
    return joblib.load("preprocessor.pkl")

def save_prediction(data):
    # Kiểm tra xem file có tồn tại không
    if not os.path.exists("history.json"):
        with open("history.json", "w") as f:
            json.dump([], f)  # Tạo file mới với mảng rỗng

    # Đọc file history.json
    try:
        with open("history.json", "r") as f:
            # Kiểm tra file rỗng
            content = f.read().strip()
            if not content:  # Nếu file rỗng, gán history là mảng rỗng
                history = []
            else:
                f.seek(0)  # Quay lại đầu file để đọc lại
                history = json.load(f)  # Thử đọc JSON
    except json.JSONDecodeError:
        # Nếu file không hợp lệ, ghi đè bằng mảng rỗng
        st.warning("File history.json không hợp lệ, sẽ tạo lại file mới.")
        history = []
        with open("history.json", "w") as f:
            json.dump(history, f)

    # Thêm dữ liệu mới vào history
    history.append(data)

    # Ghi lại file, chỉ giữ 10 bản ghi gần nhất
    with open("history.json", "w") as f:
        json.dump(history[-10:], f)
        
def load_history():
    if not os.path.exists("history.json"):
        return []
    with open("history.json", "r") as f:
        return json.load(f)

# ========== ĐIỀU HƯỚNG ==========
# Khởi tạo trạng thái trang
if "page" not in st.session_state:
    st.session_state.page = "🏠 Trang chủ"

# Cập nhật giá trị page từ st.session_state
page = st.session_state.page

# Điều hướng
with st.sidebar:
    st.markdown("<h1 style='color: white; text-align: center;'>🏦 CreditRisk</h1>", unsafe_allow_html=True)
    page = st.radio(
        "Menu",
        ["🏠 Trang chủ", "📝 Phân tích mới", "🕒 Lịch sử"],
        label_visibility="collapsed"
    )
    st.session_state.page = page  # Cập nhật trạng thái khi người dùng chọn từ sidebar

# ========== TRANG CHỦ (SLIDESHOW FULLSCREEN TỰ ĐỘNG) ==========
if page == "🏠 Trang chủ":
    # Chèn nền và overlay
    st.markdown("""
    <div class="slideshow-background"></div>
    <div class="overlay"></div>
    <div class="content">
        <h1>Chào mừng bạn đến với chương trình phát hiện rủi ro tín dụng trong ngân hàng</h1>
        <p>Ứng dụng phân tích rủi ro tín dụng thông minh giúp bạn đưa ra quyết định chính xác và nhanh chóng.</p>
        <h4> Sản phẩm thuộc về Khoa Công Nghệ Thông Tin </h4>
    """, unsafe_allow_html=True)

    # Dùng container và columns để căn chỉnh nút
    with st.container():
        col1, col2 = st.columns(2, gap="medium") # Tăng khoảng cách giữa hai cột
        with col1:
            if st.button("🔮 Bắt đầu phân tích", key="analyze_btn"):
                st.session_state.page = "📝 Phân tích mới"
        with col2:
            if st.button("🕒 Xem lịch sử", key="history_btn"):
                st.session_state.page = "🕒 Lịch sử"

    st.markdown("""
    </div>
    """, unsafe_allow_html=True)

# ========== PHẦN PHÂN TÍCH MỚI ==========
elif page == "📝 Phân tích mới":
    # Tùy chỉnh CSS từ code mới
    st.markdown("""
        <style>
        
        .main {
            background-color: #F7F9FC;
            background-image : url("anh2.jpg")
        }
        
        .stButton>button {
            background-color: #2E86C1;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #1B4F72;
        }
        .stSlider .st-dn {background-color: #2E86C1;}
        .stRadio>label {font-size: 16px;}
        .stSelectbox>label {font-size: 16px;}
        .stNumberInput>label {font-size: 16px;}
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 220px;
            background-color: #566573;
            color: white;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -110px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
    """, unsafe_allow_html=True)

    # Load mô hình và bộ tiền xử lý
    model = load_model()
    preprocessor = load_preprocessor()

    # Header
    st.markdown("<h1 style='text-align: center; color: #2E86C1; font-family: Arial;'>🔍 Dự Đoán Rủi Ro Tín Dụng</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #566573; font-family: Arial;'>Phân tích khả năng hoàn trả khoản vay một cách nhanh chóng và chính xác</h4>", unsafe_allow_html=True)

    # Nhập dữ liệu khách hàng
    st.markdown("---")
    st.markdown("<h3 style='color: #2E86C1; font-family: Arial;'>📋 Nhập thông tin khách hàng</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        with st.expander("Thông tin cá nhân", expanded=True):
            age = st.slider("📆 Tuổi", 18, 100, 30, help="Chọn tuổi của khách hàng")
            sex = st.radio("🚻 Giới tính", ["Nam", "Nữ"], horizontal=True)
            sex = "male" if sex == "Nam" else "female"
            job = st.selectbox("👔 Loại công việc", ["Không có kỹ năng & không cư trú", "Không có kỹ năng & cư trú", "Có kỹ năng", "Rất có kỹ năng"])
            job_mapping = {"Không có kỹ năng & không cư trú": 0, "Không có kỹ năng & cư trú": 1, "Có kỹ năng": 2, "Rất có kỹ năng": 3}
            job = job_mapping[job]

    with col2:
        with st.expander("Thông tin tài chính & mục đích vay", expanded=True):
            credit_amount = st.number_input("💵 Khoản vay (USD)", min_value=500, max_value=50000, value=10000, step=100)
            duration = st.slider("🕒 Thời hạn vay (tháng)", 6, 72, 24)
            purpose = st.selectbox("🎯 Mục đích vay", ["Mua ô tô", "Mua nội thất/trang thiết bị", "Mua radio/TV", "Mua thiết bị gia dụng", "Sửa chữa", "Giáo dục", "Kinh doanh", "Du lịch/Khác"])
            purpose_mapping = {"Mua ô tô": "car", "Mua nội thất/trang thiết bị": "furniture/equipment", "Mua radio/TV": "radio/TV", "Mua thiết bị gia dụng": "domestic appliances",
                               "Sửa chữa": "repairs", "Giáo dục": "education", "Kinh doanh": "business", "Du lịch/Khác": "vacation/others"}
            purpose = purpose_mapping[purpose]

    col3, col4 = st.columns([1, 1], gap="large")
    with col3:
        with st.expander("Tình trạng nhà ở", expanded=True):
            housing = st.selectbox("🏠 Hình thức nhà ở", ["Sở hữu", "Thuê", "Miễn phí"])
            housing_mapping = {"Sở hữu": "own", "Thuê": "rent", "Miễn phí": "free"}
            housing = housing_mapping[housing]

    with col4:
        with st.expander("Tài khoản ngân hàng", expanded=True):
            st.markdown("""
                <div class="tooltip">
                    💰 Tài khoản tiết kiệm
                    <span class="tooltiptext">Không có: 0 USD<br>Ít: 1-500 USD<br>Trung bình: 501-1000 USD<br>Khá nhiều: 1001-5000 USD<br>Nhiều: >5000 USD</span>
                </div>
            """, unsafe_allow_html=True)
            saving_accounts = st.selectbox("", ["Không có", "Ít", "Trung bình", "Khá nhiều", "Nhiều"], key="savings")
            saving_mapping = {"Không có": "NA", "Ít": "little", "Trung bình": "moderate", "Khá nhiều": "quite rich", "Nhiều": "rich"}
            saving_accounts = saving_mapping[saving_accounts]

            st.markdown("""
                <div class="tooltip">
                    🏦 Tài khoản vãng lai
                    <span class="tooltiptext">Không có: 0 USD<br>Ít: 1-200 USD<br>Trung bình: 201-500 USD<br>Nhiều: >500 USD</span>
                </div>
            """, unsafe_allow_html=True)
            checking_account = st.selectbox("", ["Không có", "Ít", "Trung bình", "Nhiều"], key="checking")
            checking_mapping = {"Không có": "NA", "Ít": "little", "Trung bình": "moderate", "Nhiều": "rich"}
            checking_account = checking_mapping[checking_account]

    # Nút dự đoán
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button("📌 Dự đoán ngay", key="predict_button"):
        with st.spinner("⏳ Đang phân tích dữ liệu..."):
            input_data = pd.DataFrame([{
                "Age": age,
                "Job": job,
                "Credit amount": credit_amount,
                "Duration": duration,
                "Sex": sex,
                "Housing": housing,
                "Saving accounts": saving_accounts,
                "Checking account": checking_account,
                "Purpose": purpose
            }])
            input_transformed = preprocessor.transform(input_data)
            prediction = model.predict_proba(input_transformed)[:, 1]
            risk_score = prediction[0]

            # Lưu vào lịch sử (từ test.py)
            result_data = {
                "input": input_data.to_dict("records")[0],
                "risk_score": float(risk_score),
                "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M")
            }
            save_prediction(result_data)
            st.session_state.current_result = result_data

        # Hiển thị kết quả chi tiết từng đặc trưng
        st.markdown("---")
        st.markdown("<h3 style='color: #2E86C1; font-family: Arial;'>🔍 Phân tích rủi ro từng đặc trưng</h3>", unsafe_allow_html=True)
        if df is None:
            st.warning("Không thể tính tỷ lệ rủi ro từng đặc trưng do thiếu dữ liệu 'german_credit_data.csv'.")
        else:
            feature_contributions = {
                "Tuổi": {"Giá trị": f"{age} tuổi", "Tỷ lệ rủi ro xấu": f"{age_risk_dict.get(age, 0):.2f}%"},
                "Giới tính": {"Giá trị": "Nam" if sex == "male" else "Nữ", "Tỷ lệ rủi ro xấu": f"{sex_risk_dict.get(sex, 0):.2f}%"},
                "Công việc": {"Giá trị": list(job_mapping.keys())[list(job_mapping.values()).index(job)], "Tỷ lệ rủi ro xấu": f"{job_risk_dict.get(job, 0):.2f}%"},
                "Khoản vay": {"Giá trị": f"{credit_amount:,} DM", "Tỷ lệ rủi ro xấu": f"{credit_amount_risk_dict.get(credit_amount, 0):.2f}%"},
                "Thời hạn": {"Giá trị": f"{duration} tháng", "Tỷ lệ rủi ro xấu": f"{duration_risk_dict.get(duration, 0):.2f}%"},
                "Nhà ở": {"Giá trị": list(housing_mapping.keys())[list(housing_mapping.values()).index(housing)], "Tỷ lệ rủi ro xấu": f"{housing_risk_dict.get(housing, 0):.2f}%"},
                "Tài khoản tiết kiệm": {"Giá trị": list(saving_mapping.keys())[list(saving_mapping.values()).index(saving_accounts)], "Tỷ lệ rủi ro xấu": f"{saving_risk_dict.get(saving_accounts, 0):.2f}%"},
                "Tài khoản vãng lai": {"Giá trị": list(checking_mapping.keys())[list(checking_mapping.values()).index(checking_account)], "Tỷ lệ rủi ro xấu": f"{checking_risk_dict.get(checking_account, 0):.2f}%"},
                "Mục đích vay": {"Giá trị": list(purpose_mapping.keys())[list(purpose_mapping.values()).index(purpose)], "Tỷ lệ rủi ro xấu": f"{purpose_risk_dict.get(purpose, 0):.2f}%"}
            }
            feature_df = pd.DataFrame.from_dict(feature_contributions, orient="index")
            st.table(feature_df.style.set_properties(**{'background-color': '#ECF0F1', 'border-color': '#D5DBDB', 'padding': '8px', 'text-align': 'center'}))

        # Hiển thị kết quả tổng hợp
        st.markdown("---")
        st.markdown("<h3 style='color: #2E86C1; font-family: Arial;'>🔹 Kết quả Dự Đoán Tổng hợp</h3>", unsafe_allow_html=True)
        col_result1, col_result2 = st.columns([1, 2])
        with col_result1:
            if risk_score > 0.5:
                st.error(f"⚠️ **Nguy cơ tín dụng xấu: {risk_score:.2%}**")
            else:
                st.success(f"✅ **Khả năng hoàn trả tốt: {risk_score:.2%}**")
        with col_result2:
            st.markdown("<p style='color: #566573; font-family: Arial;'>Xác suất này được tính dựa trên mô hình với dữ liệu đầu vào.</p>", unsafe_allow_html=True)

        # Biểu đồ trực quan
        st.markdown("---")
        st.markdown("<h3 style='color: #2E86C1; font-family: Arial;'>📊 Phân tích rủi ro</h3>", unsafe_allow_html=True)
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                title={"text": "Nguy cơ tín dụng xấu (%)", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#E74C3C" if risk_score > 0.5 else "#28B463"},
                    "steps": [
                        {"range": [0, 50], "color": "#D5F5E3"},
                        {"range": [50, 100], "color": "#FADBD8"}
                    ],
                    "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": 50}
                }
            ))
            st.plotly_chart(fig1, use_container_width=True)

        with col_chart2:
            labels = ["Hoàn trả tốt", "Nợ xấu"]
            values = [1 - risk_score, risk_score]
            fig3 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
            fig3.update_traces(marker=dict(colors=["#28B463", "#E74C3C"]))
            fig3.update_layout(title="Tỷ lệ rủi ro tín dụng", title_x=0.5)
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
# ========== PHẦN LỊCH SỬ ==========
elif page == "🕒 Lịch sử":
    st.markdown("<h1 style='color: var(--text); text-align: center;'>🕒 Lịch sử phân tích</h1>", unsafe_allow_html=True)
    history = load_history()
    if not history:
        st.warning("Chưa có lịch sử phân tích")
    else:
        for idx, item in enumerate(reversed(history)):
            risk_score = item["risk_score"]
            with st.expander(f"Phân tích #{len(history)-idx} - {item['timestamp']} - {risk_score:.1%}"):
                col1, col2 = st.columns(2)
                with col1:
                    for key, value in item["input"].items():
                        st.write(f"{key}: {value}")
                with col2:
                    st.markdown(f"Rủi ro: **{risk_score:.1%}** {'(Cao)' if risk_score > 0.5 else '(Thấp)'}")

# Footer
st.markdown("<div style='text-align: center; color: var(--text); opacity: 0.7; margin-top: 50px;'>© 2025 CreditRisk AI</div>", unsafe_allow_html=True)
