import os
import pandas as pd

def generate_labels(data_dir, output_csv):
    """Tạo file labels.csv từ các thư mục thể loại."""
    genres = ['Jazz', 'Phonk', 'Pop']
    data = {'filename': [], 'label': []}

    for genre in genres:
        genre_path = os.path.join(data_dir, genre)
        if os.path.exists(genre_path):
            wav_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
            for wav_file in wav_files:
                data['filename'].append(wav_file)
                data['label'].append(genre.lower())
        else:
            print(f"[!] Thư mục {genre_path} không tồn tại.")

    if not data['filename']:
        print("[!] Không tìm thấy file WAV nào.")
        return

    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[+] Đã tạo {output_csv} với {len(df)} bài hát.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    label_csv = os.path.join(base_dir, 'data', 'labels.csv')
    generate_labels(data_dir, label_csv)