import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

# Đọc dữ liệu
file_path = "german_credit_data.csv"  # Thay thế bằng đường dẫn thực tế
df = pd.read_csv(file_path)

# Xóa cột không cần thiết
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Xác định cột số và cột phân loại
cot_phan_loai = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
cot_so = ["Age", "Job", "Credit amount", "Duration"]

# Xử lý giá trị khuyết thiếu
bo_sung_du_lieu_phan_loai = SimpleImputer(strategy="constant", fill_value="unknown")
bo_sung_du_lieu_so = SimpleImputer(strategy="median")

# Chuẩn hóa dữ liệu số
bo_chuan_hoa = MinMaxScaler()

# Mã hóa biến phân loại
bien_doi_phan_loai = Pipeline([
    ("bo_sung", bo_sung_du_lieu_phan_loai),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Xây dựng bộ biến đổi cho dữ liệu số
bien_doi_so = Pipeline([
    ("bo_sung", bo_sung_du_lieu_so),
    ("chuan_hoa", bo_chuan_hoa)
])

# Kết hợp xử lý dữ liệu số và phân loại
xu_ly_truoc = ColumnTransformer([
    ("so", bien_doi_so, cot_so),
    ("phan_loai", bien_doi_phan_loai, cot_phan_loai)
])

# Mã hóa nhãn mục tiêu
ma_hoa_nhan = LabelEncoder()
df["Risk"] = ma_hoa_nhan.fit_transform(df["Risk"])

# Chia tập dữ liệu
X = df.drop(columns=["Risk"])
y = df["Risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit và transform dữ liệu
X_train = xu_ly_truoc.fit_transform(X_train)
X_test = xu_ly_truoc.transform(X_test)

# Lưu transformer và LabelEncoder để sử dụng sau này
joblib.dump(xu_ly_truoc, "preprocessor.pkl")
joblib.dump(ma_hoa_nhan, "label_encoder.pkl")

# Tìm tham số tối ưu cho XGBoost
param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7]
}

grid_search = GridSearchCV(XGBClassifier(eval_metric="logloss", random_state=42, use_label_encoder=False),
                           param_grid=param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("🎯 Tìm thấy tham số tối ưu:", grid_search.best_params_)

# Huấn luyện mô hình với tham số tối ưu
mo_hinh = XGBClassifier(**grid_search.best_params_, eval_metric="logloss", random_state=42, use_label_encoder=False)
mo_hinh.fit(X_train, y_train)

# Dự đoán trên tập test
y_du_doan = mo_hinh.predict(X_test)

# Đánh giá mô hình
do_chinh_xac = accuracy_score(y_test, y_du_doan)
bao_cao = classification_report(y_test, y_du_doan, target_names=ma_hoa_nhan.classes_)

print("\n🎯 Đánh giá mô hình:")
print(f"🎯 Độ chính xác: {do_chinh_xac:.4f}")
print("📊 Báo cáo phân loại:\n", bao_cao)

# Lưu mô hình
joblib.dump(mo_hinh, "credit_risk_model.pkl")
print("💾 Mô hình đã được lưu thành công!")

# Kiểm tra mô hình sau khi lưu
mo_hinh_tai = joblib.load("credit_risk_model.pkl")
du_doan_moi = mo_hinh_tai.predict(X_test)
print(f"✅ Kiểm tra mô hình đã lưu: {accuracy_score(y_test, du_doan_moi):.4f}")
