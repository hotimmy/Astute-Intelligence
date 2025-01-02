from typing import Type
from tensorflow.keras.models import load_model
import sounddevice as sd
import numpy as np
import librosa
import collections
import cv2
import threading
from tensorflow import keras
bpm_history = collections.deque(maxlen=5)


# 儲存最近 5 個 BPM 值
AIpath = input("Enter AI file path:")
network = keras.models.Sequential()
network = load_model(AIpath)

def GetMFCC(y):
  mfcc = np.array(librosa.feature.mfcc(y=y, sr=320000))
  #mfcc = np.array(cv2.resize(mfcc, (resolution,20)))
  #mfcc = cv2.cvtColor(mfcc, cv2.COLOR_GRAY2BGR)
  return mfcc


def GetMelspectrogram(y):
  melspectrogram = np.array(librosa.feature.melspectrogram(y=y, sr=320000))
  #melspectrogram = np.array(cv2.resize(melspectrogram, (resolution,128)))
  #melspectrogram = cv2.cvtColor(melspectrogram, cv2.COLOR_GRAY2BGR)
  return melspectrogram


def GetChromaVector(y):
  chroma = np.array(librosa.feature.chroma_stft(y=y, sr=320000))
  #chroma = np.array(cv2.resize(chroma, (resolution,12)))
  #chroma = cv2.cvtColor(chroma, cv2.COLOR_GRAY2BGR)
  return chroma


def NParraymap(x,arrayMin,arrayMax):
    return (x + (-(arrayMin))) / (arrayMax-arrayMin) * 255


def GetTonnetz(y):
  tonnetz = np.array(librosa.feature.tonnetz(y=y, sr=320000), dtype='float32')
  #tonnetz = np.array(cv2.resize(tonnetz, (resolution,6)))
  #arrayMax = np.max(tonnetz)
  #arrayMin = np.min(tonnetz)
  #tonnetz = np.round(NParraymap(tonnetz,arrayMin,arrayMax))
  #print(tonnetz.shape)
  #print(tonnetz)
  #cv2.cvtColor(tonnetz, cv2.COLOR_BGRA2RGB)
  #tonnetz = cv2.cvtColor(tonnetz, cv2.COLOR_GRAY2BGR)
  return tonnetz


def get_feature(y):
  # Extracting MFCC feature
  mfcc = GetMFCC(y)
  #print(mfcc.shape)
  #mfcc_mean = mfcc.mean(axis=1)
  #mfcc_min = mfcc.min(axis=1)
  #mfcc_max = mfcc.max(axis=1)
  #mfcc_feature = np.concatenate( (mfcc_mean, mfcc_min, mfcc_max) )

  # Extracting Mel Spectrogram feature
  melspectrogram = GetMelspectrogram(y)
  #print(melspectrogram.shape)
  #melspectrogram_mean = melspectrogram.mean(axis=1)
  #melspectrogram_min = melspectrogram.min(axis=1)
  #melspectrogram_max = melspectrogram.max(axis=1)
  #melspectrogram_feature = np.concatenate( (melspectrogram_mean, melspectrogram_min, melspectrogram_max) )

  # Extracting chroma vector feature
  chroma = GetChromaVector(y)
 # print(chroma.shape)
  #chroma_mean = chroma.mean(axis=1)
  #chroma_min = chroma.min(axis=1)
  #chroma_max = chroma.max(axis=1)
  #chroma_feature = np.concatenate( (chroma_mean, chroma_min, chroma_max) )

  # Extracting tonnetz feature
  tntz = GetTonnetz(y)
  #print(tntz.shape)
  #tntz_mean = tntz.mean(axis=1)
  #tntz_min = tntz.min(axis=1)
  #tntz_max = tntz.max(axis=1)
  #tntz_feature = np.concatenate( (tntz_mean, tntz_min, tntz_max) ) 
  
  #feature = np.concatenate( (chroma_feature, melspectrogram_feature, mfcc_feature, tntz_feature) )
  feature = np.concatenate((mfcc, melspectrogram, chroma, tntz))
  #feature = cv2.resize(feature, (128,166))
  feature = cv2.cvtColor(feature, cv2.COLOR_GRAY2BGR)
  return feature

def ProcessData(nameFunc):
    global Soundarrays
    global dataNow
    dataNow = []
    while True:
        if type(dataNow) == np.ndarray:
            Datacache = dataNow
            #print(Datacache)
            #print(Datacache.shape)
            feature = get_feature(Datacache)
            #print(feature)
            #print(feature.shape)
            Soundarrays = feature

def AIpredict(model, name):
    global Soundarrays
    global output
    output = -1
    Soundarrays = []
    PredictData = []
    while True:
        if type(Soundarrays) == np.ndarray:
            if len(PredictData) < 4:
                PredictData.append(Soundarrays)
            elif len(PredictData) == 4:
                PredictData = np.array((PredictData[1],PredictData[2],PredictData[3],Soundarrays))
                model = keras.models.Sequential()
                prediction = model.predict(PredictData)
                prediction = np.add(np.add(np.add(prediction[0],prediction[1]),prediction[2]),prediction[3])
                output = prediction.tolist().index(np.max(prediction))
                print(output)

    return


# 參數設置
sr = 320000  # 音訊取樣率
buffer_duration = 0.3  # 每次處理的音訊秒數
hop_length = 512  # FFT hop 長度
threshold = 0.2  # 能量閾值，用來過濾無聲音訊段
thread1 = threading.Thread(target=ProcessData, args=("Thread A",))
thread2 = threading.Thread(target=AIpredict, args=(network,"Thread B"))
thread1.start()
thread2.start()
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
    global dataNow
    dataNow = np.reshape(indata, [indata.shape[0]])
    

    

# 開始錄製並進行即時處理
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sr, blocksize=int(sr * buffer_duration), device=device_index):
    print("Real-time beat detection started. Press Ctrl+C to stop.")
    try:
        while True:
            sd.sleep(1000)  # 每秒檢查一次音訊片段
    except KeyboardInterrupt:
        print("Real-time beat detection stopped.")
