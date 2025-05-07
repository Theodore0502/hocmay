from extract_features import generate_feature_csv
from train_model import train_and_save_model
from predict_song import predict_song
import os

def menu():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, 'data')
    label_csv = os.path.join(base_dir, 'data', 'labels.csv')  # Sửa đường dẫn
    output_csv = os.path.join(base_dir, 'features', 'features.csv')
    model_path = os.path.join(base_dir, 'model', 'model.pkl')

    while True:
        print("\n=== MUSIC GENRE CLASSIFIER ===")
        print("1. Extract features")
        print("2. Train model")
        print("3. Predict genre")
        print("0. Exit")

        choice = input("Select option: ")

        if choice == '1':
            generate_feature_csv(
                data_folder=data_folder,
                label_csv=label_csv,
                output_csv=output_csv
            )
        elif choice == '2':
            train_and_save_model(
                feature_csv=output_csv,
                model_path=model_path
            )
        elif choice == '3':
            song_path = input("Enter path to WAV file: ")
            predict_song(
                audio_path=song_path,
                model_path=model_path
            )
        elif choice == '0':
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    menu()