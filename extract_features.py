import os
import pandas as pd
import numpy as np
import librosa
import logging

# Thiết lập logging
logging.basicConfig(filename='music_classifier.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def validate_wav(file_path):
    """Kiểm tra file WAV có hợp lệ và đủ dài không."""
    logging.debug(f"Validating WAV file: {file_path}")
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / sr
        logging.debug(f"WAV file {file_path} loaded, sample rate: {sr}, duration: {duration}s, samples: {len(y)}")
        if duration < 3.0:  # Yêu cầu tối thiểu 3 giây
            error_msg = f"WAV file {file_path} too short: {duration}s"
            logging.warning(error_msg)
            print(f"[!] {error_msg}")
            return False
        logging.info(f"WAV file {file_path} is valid, duration: {duration}s")
        return True
    except Exception as e:
        error_msg = f"Invalid WAV file {file_path}: {str(e)}"
        logging.error(error_msg)
        print(f"[!] {error_msg}")
        return False

def extract_features(file_path, duration=30):
    """Trích xuất đặc trưng âm thanh từ file WAV."""
    logging.debug(f"Extracting features from: {file_path}")
    try:
        y, sr = librosa.load(file_path, sr=None, duration=duration)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        rmse = np.mean(librosa.feature.rms(y=y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs, axis=1)

        if len(mfccs_mean) != 20:
            logging.error(f"Invalid MFCC length for {file_path}: {len(mfccs_mean)}")
            return None

        features = [
            float(tempo),
            float(chroma_stft),
            float(rmse),
            float(spectral_centroid),
            float(spectral_bandwidth),
            float(rolloff),
            float(zero_crossing_rate)
        ]
        features.extend(float(x) for x in mfccs_mean)

        if len(features) != 27:
            logging.error(f"Invalid feature length for {file_path}: {len(features)}")
            return None

        logging.info(f"Successfully extracted features for {file_path}, length: {len(features)}")
        return np.array(features, dtype=np.float32)
    except Exception as e:
        error_msg = f"Error processing {file_path}: {str(e)}"
        logging.error(error_msg)
        print(f"[!] {error_msg}")
        return None

def generate_feature_csv(data_folder, label_csv, output_csv):
    logging.info("Starting feature extraction...")
    print("[+] Starting feature extraction...")

    if not os.path.exists(label_csv):
        error_msg = f"Error: {label_csv} not found. Ensure labels.csv exists in data directory."
        logging.error(error_msg)
        print(f"[!] {error_msg}")
        return

    try:
        labels_df = pd.read_csv(label_csv)
        logging.info(f"Read {label_csv} successfully, found {len(labels_df)} files")
        print(f"[+] Found {len(labels_df)} files in {label_csv}")
    except Exception as e:
        error_msg = f"Error reading {label_csv}: {e}"
        logging.error(error_msg)
        print(f"[!] {error_msg}")
        return

    if labels_df.empty:
        error_msg = f"Error: {label_csv} is empty. Ensure data directories contain WAV files."
        logging.error(error_msg)
        print(f"[!] {error_msg}")
        return

    feature_list = []
    label_list = []

    for index, row in labels_df.iterrows():
        filename = row['filename']
        label = row['label']
        print(f"[+] Processing file: {filename} (label: {label})")
        logging.info(f"Processing file: {filename} (label: {label})")

        # Tìm file trong toàn bộ thư mục con của data_folder
        file_path = None
        for root, _, files in os.walk(data_folder):
            if filename in files:
                file_path = os.path.join(root, filename)
                break

        if file_path is None:
            error_msg = f"File not found: {filename} in {data_folder}"
            logging.warning(error_msg)
            print(f"[!] {error_msg}")
            continue

        print(f"[+] Found file at: {file_path}")
        logging.info(f"Found file at: {file_path}")

        if not validate_wav(file_path):
            error_msg = f"Invalid WAV file: {filename}"
            logging.warning(error_msg)
            print(f"[!] {error_msg}")
            continue

        print(f"[+] Extracting features for: {filename}")
        logging.info(f"Extracting features for: {filename}")
        features = extract_features(file_path)
        if features is not None:
            feature_list.append(features)
            label_list.append(label)
            print(f"[+] Successfully extracted features for {filename}")
            logging.info(f"Successfully extracted features for {filename}")
        else:
            error_msg = f"Failed to extract features for {filename}"
            logging.warning(error_msg)
            print(f"[!] {error_msg}")

    if feature_list:
        feature_columns = ['tempo', 'chroma_stft', 'rmse', 'spectral_centroid',
                          'spectral_bandwidth', 'rolloff', 'zero_crossing_rate'] + \
                         [f'mfcc{i+1}' for i in range(20)]
        features_df = pd.DataFrame(feature_list, columns=feature_columns)
        features_df['label'] = label_list
        try:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            features_df.to_csv(output_csv, index=False)
            print(f"[+] Đã lưu {len(feature_list)} features vào {output_csv}")
            logging.info(f"Saved {len(feature_list)} features to {output_csv}")
        except Exception as e:
            error_msg = f"Error saving {output_csv}: {e}"
            logging.error(error_msg)
            print(f"[!] {error_msg}")
    else:
        error_msg = "No features extracted. Check file paths, WAV validity, or music_classifier.log for errors."
        logging.error(error_msg)
        print(f"[!] {error_msg}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    label_csv = os.path.join(base_dir, 'data', 'labels.csv')
    feature_csv = os.path.join(base_dir, 'features', 'features.csv')

    generate_feature_csv(data_dir, label_csv, feature_csv)