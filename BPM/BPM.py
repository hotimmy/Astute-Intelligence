import librosa
import librosa.display

# 載入音訊並進行節奏檢測
y, sr = librosa.load("audio_fast.mp3")
tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

# 節拍時間點
#beat_times = librosa.frames_to_time(beats, sr=sr)
print(f"Tempo: {tempo} BPM")
#print(f"Beat Times: {beat_times}")