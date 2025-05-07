import os

def count_songs(data_dir):
    """Đếm số lượng bài hát theo thể loại trong thư mục dữ liệu."""
    genres = ['Jazz', 'Phonk', 'Pop']
    song_counts = {genre: 0 for genre in genres}

    for genre in genres:
        genre_path = os.path.join(data_dir, genre)
        if os.path.exists(genre_path):
            # Đếm file WAV trong thư mục
            wav_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
            song_counts[genre] = len(wav_files)
        else:
            print(f"[!] Thư mục {genre_path} không tồn tại.")

    # In kết quả
    print("\n=== Số lượng bài hát theo thể loại ===")
    for genre, count in song_counts.items():
        print(f"{genre}: {count} bài")
    print(f"Tổng cộng: {sum(song_counts.values())} bài")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    count_songs(data_dir)