import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def train_and_save_model(feature_csv, model_path):
    if not os.path.exists(feature_csv):
        print(f"[!] Feature file not found: {feature_csv}")
        return

    try:
        df = pd.read_csv(feature_csv)
    except Exception as e:
        print(f"[!] Error reading {feature_csv}: {e}")
        return

    if df.empty:
        print("[!] Feature CSV is empty. Cannot train model.")
        return

    # Định nghĩa các cột đặc trưng
    feature_columns = ['tempo', 'chroma_stft', 'rmse', 'spectral_centroid',
                       'spectral_bandwidth', 'rolloff', 'zero_crossing_rate'] + \
                      [f'mfcc{i+1}' for i in range(20)]

    # Kiểm tra xem các cột đặc trưng có tồn tại không
    missing_columns = [col for col in feature_columns if col not in df.columns]
    if missing_columns:
        print(f"[!] Error: Missing columns in {feature_csv}: {missing_columns}")
        return

    # Lấy X (đặc trưng) và y (nhãn)
    X = df[feature_columns].copy()  # Tạo bản sao để tránh SettingWithCopyWarning
    y = df['label']

    # Kiểm tra và làm sạch dữ liệu
    for col in feature_columns:
        X.loc[:, col] = pd.to_numeric(X[col], errors='coerce')

    # Loại bỏ các hàng có giá trị NaN
    X = X.dropna()
    y = y[X.index]  # Đồng bộ nhãn với dữ liệu đã làm sạch

    if X.empty:
        print("[!] Error: No valid data after cleaning. Check features.csv.")
        return

    # Chia dữ liệu thành tập train và test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Huấn luyện mô hình
    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)

    # Đánh giá mô hình
    y_pred = model.predict(X_test_scaled)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # Lưu mô hình và scaler
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({'model': model, 'scaler': scaler}, model_path)
        print(f"[+] Model saved to {model_path}")
    except Exception as e:
        print(f"[!] Error saving {model_path}: {e}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    feature_csv = os.path.join(base_dir, 'features', 'features.csv')
    model_path = os.path.join(base_dir, 'model', 'model.pkl')
    train_and_save_model(feature_csv, model_path)