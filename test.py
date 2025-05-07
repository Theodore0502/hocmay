import librosa
files = [
    "D:\\music_classification\\data\\Jazz\\jazz_Alone.wav",
    "D:\\music_classification\\data\\Phonk\\phonk_LOVELYBASTARDS.wav",
    "D:\\music_classification\\data\\Pop\\pop_APT.wav"
]
for file_path in files:
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / sr
        print(f"{file_path}: duration={duration}s, sample_rate={sr}")
    except Exception as e:
        print(f"{file_path}: Error={str(e)}")