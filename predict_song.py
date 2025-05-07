import os
import numpy as np
import pandas as pd
import librosa
import joblib

def extract_features_for_prediction(file_path, duration=30):
    """Trích xuất đặc trưng âm thanh từ file WAV, khớp với huấn luyện."""
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
            print(f"[!] Invalid MFCC length for {file_path}: {len(mfccs_mean)}")
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
            print(f"[!] Invalid feature length for {file_path}: {len(features)}")
            return None

        return np.array(features, dtype=np.float32)
    except Exception as e:
        print(f"[!] Error processing {file_path}: {e}")
        return None

def predict_song(audio_path, model_path):
    if not os.path.isfile(audio_path):
        print(f"[!] Audio file not found: {audio_path}")
        return

    if not os.path.isfile(model_path):
        print(f"[!] Model file not found: {model_path}")
        return

    # Trích xuất đặc trưng
    features = extract_features_for_prediction(audio_path)
    if features is None:
        print(f"[!] Failed to extract features for {audio_path}")
        return

    # Tải mô hình và scaler
    try:
        model_data = joblib.load(model_path)
        model = model_data['model']
        scaler = model_data['scaler']
    except Exception as e:
        print(f"[!] Error loading model {model_path}: {e}")
        return

    # Chuyển đặc trưng thành DataFrame với tên cột
    feature_columns = ['tempo', 'chroma_stft', 'rmse', 'spectral_centroid',
                       'spectral_bandwidth', 'rolloff', 'zero_crossing_rate'] + \
                      [f'mfcc{i+1}' for i in range(20)]
    features_df = pd.DataFrame([features], columns=feature_columns)

    # Chuẩn hóa đặc trưng
    features_scaled = scaler.transform(features_df)

    # Dự đoán
    prediction = model.predict(features_scaled)
    print(f"\nPredicted genre: {prediction[0]}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    audio_path = input("Enter path to WAV file: ")
    model_path = os.path.join(base_dir, 'model', 'model.pkl')
    predict_song(audio_path, model_path)