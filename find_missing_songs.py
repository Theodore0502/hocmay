import pandas as pd
import os

def find_missing_songs(label_csv, feature_csv):
    """Tìm các file có trong labels.csv nhưng không có trong features.csv."""
    if not os.path.exists(label_csv):
        print(f"[!] {label_csv} không tồn tại.")
        return

    if not os.path.exists(feature_csv):
        print(f"[!] {feature_csv} không tồn tại.")
        return

    try:
        labels_df = pd.read_csv(label_csv)
        features_df = pd.read_csv(feature_csv)
    except Exception as e:
        print(f"[!] Lỗi đọc file CSV: {e}")
        return

    # Lấy danh sách filename từ labels.csv
    label_files = set(labels_df['filename'].str.lower())
    # Lấy danh sách filename từ features.csv (dựa trên nhãn, giả sử nhãn khớp với file)
    feature_labels = features_df['label']
    feature_files = set(labels_df[labels_df['label'].isin(feature_labels)]['filename'].str.lower())

    # Tìm các file có trong labels.csv nhưng không có trong features.csv
    missing_files = label_files - feature_files

    if missing_files:
        print("\n=== Các file bị lỗi hoặc không được xử lý ===")
        for file in missing_files:
            print(f"- {file}")
        print(f"Tổng cộng: {len(missing_files)} file bị thiếu.")
    else:
        print("[+] Không có file nào bị thiếu trong features.csv.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    label_csv = os.path.join(base_dir, 'data', 'labels.csv')
    feature_csv = os.path.join(base_dir, 'features', 'features.csv')
    find_missing_songs(label_csv, feature_csv)