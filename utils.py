import os

def get_audio_files(folder_path):
    audio_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    return audio_files
