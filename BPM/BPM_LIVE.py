import sounddevice as sd
import numpy as np
import librosa
import collections
bpm_history = collections.deque(maxlen=5)  # 儲存最近 5 個 BPM 值


# 參數設置
sr = 22050  # 音訊取樣率
buffer_duration = 4  # 每次處理的音訊秒數
hop_length = 512  # FFT hop 長度
threshold = 0.2  # 能量閾值，用來過濾無聲音訊段

# 選擇音訊設備
print(sd.query_devices())
device_index = int(input("請輸入想要使用的音訊設備編號（根據 `sd.query_devices()` 的結果）："))



#def detect_beats(audio_buffer):
#    tempo, beats = librosa.beat.beat_track(y=audio_buffer, sr=sr, hop_length=hop_length)
#    bpm_history.append(tempo)
#    avg_bpm = np.mean(bpm_history).item()  # 將平均值轉為標量
#    print(f"Smoothed BPM: {avg_bpm:.2f}")
#    return avg_bpm, beats


def detect_beats(audio_buffer):
    # 使用 librosa 偵測節拍
    tempo, beats = librosa.beat.beat_track(y=audio_buffer, sr=sr, hop_length=hop_length)
    print(f"Tempo: {tempo} BPM, Beats Detected: {len(beats)}")
    return tempo, beats

def audio_callback(indata, frames, time, status):
    # 將音訊數據轉為一維並標準化
    audio_buffer = indata[:, 0]
    
    # 檢查音訊能量以過濾靜音段
    if np.max(np.abs(audio_buffer)) > threshold:
        # 偵測節拍
        detect_beats(audio_buffer)

# 開始錄製並進行即時處理
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sr, blocksize=int(sr * buffer_duration), device=device_index):
    print("Real-time beat detection started. Press Ctrl+C to stop.")
    try:
        while True:
            sd.sleep(1000)  # 每秒檢查一次音訊片段
    except KeyboardInterrupt:
        print("Real-time beat detection stopped.")
