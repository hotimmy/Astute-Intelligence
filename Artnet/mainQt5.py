from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QTextEdit, QComboBox, QCheckBox, QGridLayout, QFileDialog
)
from PyQt5.QtCore import Qt

from TArtnet import StupidArtnet
from SignalGenerator import SignalGenerator as sg
import time
import sys
import select
import threading
import warnings
warnings.filterwarnings("ignore",category=UserWarning)
from pynput.keyboard import Key, Listener
#from keras.saving.save import load_model
import sounddevice as sd
import numpy as np
import librosa
import collections
import cv2
from tensorflow import keras
import os

# Initialize Art-Net and Signal Generator
target_ip = '172.20.10.3'  # Target IP
universe = 0                  # Universe number
packet_size = 512             # Packet size
frame_rate = 40               # Frame rate (Hz)
manual_input = "0"

sg = sg()

# Create StupidArtnet object
artnet = StupidArtnet(target_ip, universe, packet_size, frame_rate, True, True)

# Check initialization
print(artnet)

# Define light data list, each element is a bytearray of length packet_size
packet = bytearray([0] * packet_size)

# Define lights
B1, B2, B3 = 1, 2, 3
PN1, PN2 = 17, 19
P1, P2, P3, P4, P5, P6 = 33, 49, 65, 81, 97, 113
#P1, P2, P3, P4, P5, P6 = 90, 105, 400, 410, 120, 135

# Initialize the neural network
#AIpath = input("Enter AI file path:").split('"')[1]
#network = keras.models.Sequential()
#network = load_model(AIpath)

# Initialize the neural network
# 獲取腳本所在目錄
script_dir = os.path.dirname(os.path.abspath(__file__))

# 檔案路徑
AIpath = os.path.join(script_dir, "model.h5")
try:
    network = keras.models.Sequential()
    network.predict(np.zeros((3,3,3)))
    network = keras.models.load_model(AIpath)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

print(sd.query_devices())
lines = str(sd.query_devices()).splitlines()
# 遍歷每一行，找出包含 "BlackHole" 的行
for line_number, line in enumerate(lines):
    if "BlackHole" in line:
        device_index = line_number
print(device_index)

# Define global variables for audio processing
sr = 320000  # Sampling rate
sr1 = 44100
buffer_duration1 = 0.3  # Duration of audio chunks for AI processing
buffer_duration2 = 2  # Duration of audio chunks for BPM detection
hop_length = 512  # FFT hop length
threshold = 0.2  # Energy threshold for filtering silent segments
dataNow = []
Soundarrays = []



def isStart(indata, frames, time, status):
    global started
    cacheStart = librosa.feature.mfcc(y=indata, sr=sr)
    started = indata
with sd.InputStream(callback=isStart, channels=1, samplerate=sr, blocksize=int(sr * buffer_duration1), device=device_index):
    global started
    started = -1
    print("Starting Librosa services...")
    while True:
        if type(started) == np.ndarray:
            print("Librosa started.")
            break


# Feature extraction functions (MFCC, MelSpectrogram, etc.)
def GetMFCC(y):
    return np.array(librosa.feature.mfcc(y=y, sr=320000)) #給0.3秒的音訊 得到位元率32000的頻譜圖

def GetMelspectrogram(y):
    return np.array(librosa.feature.melspectrogram(y=y, sr=320000)) #給0.3秒的音訊 得到位元率32000的頻譜圖

def GetChromaVector(y):
    return np.array(librosa.feature.chroma_stft(y=y, sr=320000)) #給0.3秒的音訊 得到位元率32000的光譜圖

def GetTonnetz(y):
    return np.array(librosa.feature.tonnetz(y=y, sr=320000), dtype='float32') #給0.3秒的音訊 得到位元率32000的音調圖

def get_feature(y): 
    mfcc = GetMFCC(y)
    melspectrogram = GetMelspectrogram(y)
    chroma = GetChromaVector(y)
    tntz = GetTonnetz(y)
    #return cv2.resize(np.concatenate((mfcc, melspectrogram, chroma, tntz)), (94,83))
    return cv2.resize(cv2.cvtColor(np.concatenate((mfcc, melspectrogram, chroma, tntz)), cv2.COLOR_GRAY2BGR), (94,83)) #根據上述圖片 直接將四張圖拼起來 曲得特徵圖

def ProcessData(nameFunc):
    global Soundarrays
    global dataNow
    global control_mode
    control_mode = -1
    dataNow = []
    while True:
        if type(dataNow) == np.ndarray:
            #print("Process data")
            Datacache = dataNow #防止處理時 dataNow 變更資料
            feature = get_feature(Datacache)
            Soundarrays = feature

def AIpredict(model, name):
    global Soundarrays
    global output
    output = -1
    predictions = []
    Soundarrays = []
    PredictData = []
    print("AI start")
    while True:
        if type(Soundarrays) == np.ndarray:
            if len(PredictData) < 4:
                PredictData.append(Soundarrays)
            elif len(PredictData) == 4: #開始預測
                PredictData = np.array((PredictData[1], PredictData[2], PredictData[3], Soundarrays)) #PredictData 更新
                prediction = model.predict(PredictData,verbose=0)
                prediction = np.add(np.add(np.add(prediction[0], prediction[1]), prediction[2]), prediction[3]) #把機率值加起來
                prediction_num = prediction.tolist().index(np.max(prediction)) #取得最大項
                if len(predictions) < 6:
                    predictions.append(prediction_num)
                    #print(predictions)
                elif len(predictions) == 6:
                    if control_mode == "AI":
                        print(predictions)
                    predictions = [predictions[1],predictions[2],predictions[3],predictions[4],predictions[5],prediction_num]
                
                    if predictions[0] == predictions[1] and predictions[1] == predictions[2] and predictions[2] == predictions[3] and predictions[3] == predictions[4] and predictions[4] == predictions[5]:
                	    
                        output = prediction_num
                
                #print(output)


def audio_callback(indata, frames, time, status):
    global dataNow
    dataNow = np.reshape(indata, [indata.shape[0]])

def audio_callback_for_BPM(indata, frames, time, status):
    # 將音訊數據轉為一維並標準化
    #print("callback")
    audio_buffer = indata[:, 0]
    
    # 檢查音訊能量以過濾靜音段
    if np.max(np.abs(audio_buffer)) > threshold:
        # 偵測節拍
        detect_beats(audio_buffer)

def BPMaudio():
    with sd.InputStream(callback=audio_callback_for_BPM, channels=1, samplerate=sr1, blocksize=int(sr1 * buffer_duration2), device=device_index):
        print("Real-time audio processing started. Press Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("Real-time audio processing stopped.")

# Real-time audio recording and processing
def audio():
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sr, blocksize=int(sr * buffer_duration1), device=device_index):
        print("Real-time audio processing started. Press Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("Real-time audio processing stopped.")

def detect_beats(audio_buffer):
    global BPM
    # 使用 librosa 偵測節拍
    tempo, beats = librosa.beat.beat_track(y=audio_buffer, sr=sr1, hop_length=hop_length)
    print(f"Tempo: {tempo} BPM, Beats Detected: {len(beats)}")
    BPM = tempo

#def check_input():
    #if select.select([sys.stdin], [], [], 0)[0]:
    #    return sys.stdin.readline().strip()
    #return None

def check_input():
    global manual_input
    while True:
        manual_input = sys.stdin.readline().strip()

def monitor_control_mode():
    """Monitor for user input to switch control modes."""
    global control_mode
    global manual_input
    manual_input = "0"
    manual_input1 = 0
    control_mode = "manual"
    while True:
        if manual_input != manual_input1:
            manual_input1 = manual_input
            if manual_input1 == "s":
                control_mode = "manual"
                print(f"Control mode switched to: {control_mode}")
            elif manual_input1 == "r":
                control_mode = "AI"
            
            time.sleep(0.001)
        else:
            pass

def set_color():
    global color
    if color == 0:
        for i in [P1,P2,P4]:
            artnet.set_single_value(i+1, 98)
            artnet.set_single_value(i+2, 234)
            artnet.set_single_value(i+3, 255)
        for i in [P3,P5,P6]:
            artnet.set_single_value(i+1, 255)
            artnet.set_single_value(i+2, 79)
            artnet.set_single_value(i+3, 255)
    elif color == 1:
        for i in [P1,P2,P3,P4,P5,P6]:
            artnet.set_single_value(i+1, 251)
            artnet.set_single_value(i+2, 179)
            artnet.set_single_value(i+3, 35)
    else:
        for i in [P1,P2,P3,P4,P5,P6]:
            artnet.set_single_value(i+1, 0)
            artnet.set_single_value(i+2, 0)
            artnet.set_single_value(i+3, 255)


def control_lights():
    global output
    global BPM
    global control_mode
    global color
    control_mode = "manual"
    output = -1
    user_input = -1
    manual_input2 = -1
    BPM = 120
    color = 0
    set_color()
    artnet.start()
    artnet.set(packet)
    effect = False

    for i in [B1,B2,B3]:
        artnet.set_single_value(i, 255)

    while True:
        if control_mode == "AI":
            if str(output) == user_input:
                pass
            else:
                user_input = str(output) 

                if user_input =="bon":
                    for i in [B1,B2,B3]:
                        artnet.set_single_value(i,255)

                elif user_input =="boff":
                    for i in [B1,B2,B3]:
                        artnet.set_single_value(i,0)


                elif user_input == "pnon":
                    print("PN on")
                    artnet.set_single_value(PN1, 255)
                    artnet.set_single_value(PN1+1, 255)
                    artnet.set_single_value(PN2, 255)
                    artnet.set_single_value(PN2+1, 255)


                elif user_input == "pnoff":
                    print("PN off")
                    artnet.set_single_value(PN1, 0)
                    artnet.set_single_value(PN1+1, 0)
                    artnet.set_single_value(PN2, 0)
                    artnet.set_single_value(PN2+1, 0)


                elif user_input == "color":
                    print("color default")
                    for i in [P1,P2,P3,P4,P5,P6]:
                        artnet.set_single_value(i+1, 251)
                        artnet.set_single_value(i+2, 179)
                        artnet.set_single_value(i+3, 35)


                elif user_input == "8":
                    print("9 P on")
                    set_color()
                    for i in [P1,P2,P3,P4,P5,P6]:
                        artnet.set_single_value(i, 255)
                    effect = False

                elif user_input == "9":
                    print("10 P off")
                    set_color()
                    for i in [P1,P2,P3,P4,P5,P6]:
                        artnet.set_single_value(i, 0)
                    effect = False

                elif user_input == "0":
                    #漸進漸出 bpm
                    print("1 漸進漸出 bpm")
                    set_color()
                    sg.set_generator(P1, "sin", BPM, frame_rate, offset=0)
                    sg.set_generator(P2, "sin", BPM, frame_rate, offset=0.2)
                    sg.set_generator(P3, "sin", BPM, frame_rate, offset=0.4)
                    sg.set_generator(P4, "sin", BPM, frame_rate, offset=0.6)
                    sg.set_generator(P5, "sin", BPM, frame_rate, offset=0.8)
                    sg.set_generator(P6, "sin", BPM, frame_rate, offset=1) 
                    effect = True

                elif user_input == "1":

                    #Can-Can bpm
                    print("2 Can-Can bpm")
                    set_color()
                    sg.set_generator(P1, "square", BPM, frame_rate, offset=0)
                    sg.set_generator(P2, "square", BPM, frame_rate, offset=0)
                    sg.set_generator(P4, "square", BPM, frame_rate, offset=0)
                    sg.set_generator(P3, "square", BPM, frame_rate, offset=60/BPM)
                    sg.set_generator(P5, "square", BPM, frame_rate, offset=60/BPM)
                    sg.set_generator(P6, "square", BPM, frame_rate, offset=60/BPM)
                    effect = True

                elif user_input == "2":
                    #Can-Can 柔 bpm
                    print("3 Can-Can 柔 bpm")
                    set_color()
                    sg.set_generator(P1, "sin", BPM, frame_rate, offset=0)
                    sg.set_generator(P2, "sin", BPM, frame_rate, offset=0)
                    sg.set_generator(P4, "sin", BPM, frame_rate, offset=0)
                    sg.set_generator(P3, "sin", BPM, frame_rate, offset=30/BPM)
                    sg.set_generator(P5, "sin", BPM, frame_rate, offset=30/BPM)
                    sg.set_generator(P6, "sin", BPM, frame_rate, offset=30/BPM) 
                    effect = True

                elif user_input == "3":
                    #Rain bpm
                    print("4 Rain bpm")
                    set_color()
                    sg.set_generator(P1, "rain", BPM, frame_rate, offset=0)
                    sg.set_generator(P2, "rain", BPM, frame_rate, offset=BPM/30/6*1 )
                    sg.set_generator(P3, "rain", BPM, frame_rate, offset=BPM/30/6*3 )
                    sg.set_generator(P4, "rain", BPM, frame_rate, offset=BPM/30/6*4 )
                    sg.set_generator(P5, "rain", BPM, frame_rate, offset=BPM/30/6*4 )
                    sg.set_generator(P6, "rain", BPM, frame_rate, offset=BPM/30/6*5 ) 
                    effect = True

                elif user_input == "4":
                    #快閃
                    print("5 快閃")
                    set_color()
                    flash = 612
                    sg.set_generator(P1, "square", flash, frame_rate, offset=0)
                    sg.set_generator(P2, "square", flash, frame_rate, offset=0.1)
                    sg.set_generator(P3, "square", flash, frame_rate, offset=0.2)
                    sg.set_generator(P4, "square", flash, frame_rate, offset=0.3)
                    sg.set_generator(P5, "square", flash, frame_rate, offset=0.4)
                    sg.set_generator(P6, "square", flash, frame_rate, offset=0.5) 
                    effect = True

                elif user_input == "5":
                    #白光
                    print("6 白光")
                    for i in [P1,P2,P3,P4,P5,P6]:
                        artnet.set_single_value(i+1, 255)
                        artnet.set_single_value(i+2, 255)
                        artnet.set_single_value(i+3, 255)
                    effect = False

                elif user_input == "6":
                    #漸進漸出 慢
                    print("7 漸進漸出 慢")
                    set_color()
                    slow = 60
                    sg.set_generator(P1, "sin", slow, frame_rate, offset=0)
                    sg.set_generator(P2, "sin", slow, frame_rate, offset=0.2)
                    sg.set_generator(P3, "sin", slow, frame_rate, offset=0.4)
                    sg.set_generator(P4, "sin", slow, frame_rate, offset=0.6)
                    sg.set_generator(P5, "sin", slow, frame_rate, offset=0.8)
                    sg.set_generator(P6, "sin", slow, frame_rate, offset=1) 
                    effect = True

                elif user_input == "7":
                    #漸進漸出 快
                    print("8 漸進漸出 快")
                    set_color()
                    fast = 100
                    sg.set_generator(P1, "sin", fast, frame_rate, offset=0)
                    sg.set_generator(P2, "sin", fast, frame_rate, offset=0.2)
                    sg.set_generator(P3, "sin", fast, frame_rate, offset=0.4)
                    sg.set_generator(P4, "sin", fast, frame_rate, offset=0.6)
                    sg.set_generator(P5, "sin", fast, frame_rate, offset=0.8)
                    sg.set_generator(P6, "sin", fast, frame_rate, offset=1) 
                    effect = True
                else:
                    #print("No output")
                    None

            if effect == True:
                for i in [P1,P2,P3,P4,P5,P6]:
                    artnet.set_single_value(i,sg.signal_generator(i))

            # 保持40Hz頻率
            artnet.show()
            time.sleep(1 / frame_rate)
        elif control_mode == "manual":
            # Manual control logic
            #print("Manual control enabled. Enter commands:")
            
            if manual_input:
                if manual_input != manual_input2:
                    manual_input2 = manual_input
                    if manual_input =="bon":
                        for i in [B1,B2,B3]:
                            artnet.set_single_value(i,255)

                    elif manual_input =="boff":
                        for i in [B1,B2,B3]:
                            artnet.set_single_value(i,0)


                    elif manual_input == "pnon":
                        print("PN on")
                        artnet.set_single_value(PN1, 255)
                        artnet.set_single_value(PN1+1, 255)
                        artnet.set_single_value(PN2, 255)
                        artnet.set_single_value(PN2+1, 255)


                    elif manual_input == "pnoff":
                        print("PN off")
                        artnet.set_single_value(PN1, 0)
                        artnet.set_single_value(PN1+1, 0)
                        artnet.set_single_value(PN2, 0)
                        artnet.set_single_value(PN2+1, 0)


                    elif manual_input == "9":
                        print("9 P on")
                        set_color()
                        for i in [P1,P2,P3,P4,P5,P6]:
                            artnet.set_single_value(i, 255)
                        effect = False

                    elif manual_input == "10":
                        print("10 P off")
                        set_color()
                        for i in [P1,P2,P3,P4,P5,P6]:
                            artnet.set_single_value(i, 0)
                        effect = False

                    elif manual_input == "1":
                        #漸進漸出 bpm
                        print("1 漸進漸出 bpm")
                        set_color()
                        sg.set_generator(P1, "sin", BPM, frame_rate, offset=0)
                        sg.set_generator(P2, "sin", BPM, frame_rate, offset=0.2)
                        sg.set_generator(P3, "sin", BPM, frame_rate, offset=0.4)
                        sg.set_generator(P4, "sin", BPM, frame_rate, offset=0.6)
                        sg.set_generator(P5, "sin", BPM, frame_rate, offset=0.8)
                        sg.set_generator(P6, "sin", BPM, frame_rate, offset=1) 
                        effect = True

                    elif manual_input == "2":
                        #Can-Can bpm
                        print("2 Can-Can bpm")
                        set_color()
                        sg.set_generator(P1, "square", BPM, frame_rate, offset=0)
                        sg.set_generator(P2, "square", BPM, frame_rate, offset=0)
                        sg.set_generator(P4, "square", BPM, frame_rate, offset=0)
                        sg.set_generator(P3, "square", BPM, frame_rate, offset=60/BPM)
                        sg.set_generator(P5, "square", BPM, frame_rate, offset=60/BPM)
                        sg.set_generator(P6, "square", BPM, frame_rate, offset=60/BPM)
                        effect = True

                    elif manual_input == "3":
                        #Can-Can 柔 bpm
                        print("3 Can-Can 柔 bpm")
                        set_color()
                        sg.set_generator(P1, "sin", BPM, frame_rate, offset=0)
                        sg.set_generator(P2, "sin", BPM, frame_rate, offset=0)
                        sg.set_generator(P4, "sin", BPM, frame_rate, offset=0)
                        sg.set_generator(P3, "sin", BPM, frame_rate, offset=30/BPM)
                        sg.set_generator(P5, "sin", BPM, frame_rate, offset=30/BPM)
                        sg.set_generator(P6, "sin", BPM, frame_rate, offset=30/BPM) 
                        effect = True

                    elif manual_input == "4":
                        #Rain bpm
                        print("4 Rain bpm")
                        set_color()
                        sg.set_generator(P1, "rain", BPM, frame_rate, offset=0)
                        sg.set_generator(P2, "rain", BPM, frame_rate, offset=BPM/30/6*1 )
                        sg.set_generator(P3, "rain", BPM, frame_rate, offset=BPM/30/6*3 )
                        sg.set_generator(P4, "rain", BPM, frame_rate, offset=BPM/30/6*4 )
                        sg.set_generator(P5, "rain", BPM, frame_rate, offset=BPM/30/6*4 )
                        sg.set_generator(P6, "rain", BPM, frame_rate, offset=BPM/30/6*5 ) 
                        effect = True

                    elif manual_input == "5":
                        #快閃
                        print("5 快閃")
                        set_color()
                        flash = 612
                        sg.set_generator(P1, "square", flash, frame_rate, offset=0)
                        sg.set_generator(P2, "square", flash, frame_rate, offset=0.1)
                        sg.set_generator(P3, "square", flash, frame_rate, offset=0.2)
                        sg.set_generator(P4, "square", flash, frame_rate, offset=0.3)
                        sg.set_generator(P5, "square", flash, frame_rate, offset=0.4)
                        sg.set_generator(P6, "square", flash, frame_rate, offset=0.5) 
                        effect = True

                    elif manual_input == "6":
                        #白光
                        print("6 白光")
                        for i in [P1,P2,P3,P4,P5,P6]:
                            artnet.set_single_value(i+1, 255)
                            artnet.set_single_value(i+2, 255)
                            artnet.set_single_value(i+3, 255)
                        effect = False
                    elif manual_input == "7":
                        #漸進漸出 慢
                        print("7 漸進漸出 慢")
                        set_color()
                        slow = 60
                        sg.set_generator(P1, "sin", slow, frame_rate, offset=0)
                        sg.set_generator(P2, "sin", slow, frame_rate, offset=0.2)
                        sg.set_generator(P3, "sin", slow, frame_rate, offset=0.4)
                        sg.set_generator(P4, "sin", slow, frame_rate, offset=0.6)
                        sg.set_generator(P5, "sin", slow, frame_rate, offset=0.8)
                        sg.set_generator(P6, "sin", slow, frame_rate, offset=1) 
                        effect = True

                    elif manual_input == "8":
                        #漸進漸出 快
                        print("8 漸進漸出 快")
                        set_color()
                        fast = 100
                        sg.set_generator(P1, "sin", fast, frame_rate, offset=0)
                        sg.set_generator(P2, "sin", fast, frame_rate, offset=0.2)
                        sg.set_generator(P3, "sin", fast, frame_rate, offset=0.4)
                        sg.set_generator(P4, "sin", fast, frame_rate, offset=0.6)
                        sg.set_generator(P5, "sin", fast, frame_rate, offset=0.8)
                        sg.set_generator(P6, "sin", fast, frame_rate, offset=1) 
                        effect = True
                    else:
                        #print("No output")
                        None

            if effect == True:
                for i in [P1,P2,P3,P4,P5,P6]:
                    artnet.set_single_value(i,sg.signal_generator(i))
            artnet.show()
            time.sleep(1 / frame_rate)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Control Panel")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()

        # 1. Switch (AI/Manual)
        switch_layout = QHBoxLayout()
        self.switch_label = QLabel("AI / Manual:")
        self.switch_toggle = QCheckBox()
        self.switch_toggle.setChecked(False)  # Default to manual
        self.switch_toggle.stateChanged.connect(self.update_control_model)  # Connect to state change handler
        switch_layout.addWidget(self.switch_label)
        switch_layout.addWidget(self.switch_toggle)
        main_layout.addLayout(switch_layout)

        # 2. Terminal display
        self.terminal_display = QTextEdit()
        self.terminal_display.setPlaceholderText("Terminal output...")
        main_layout.addWidget(self.terminal_display)

        # 3. Input and selection area
        input_layout = QVBoxLayout()

        # 3.1 IP input
        ip_layout = QHBoxLayout()
        ip_label = QLabel("Text Input IP:")
        self.ip_input = QLineEdit()
        ip_layout.addWidget(ip_label)
        ip_layout.addWidget(self.ip_input)
        input_layout.addLayout(ip_layout)

        # 3.2 Audio device selection
        audio_layout = QHBoxLayout()
        audio_label = QLabel("Audio Device:")
        self.audio_combo = QComboBox()
        self.audio_combo.addItems(["Device 1", "Device 2", "Device 3"])
        audio_layout.addWidget(audio_label)
        audio_layout.addWidget(self.audio_combo)
        input_layout.addLayout(audio_layout)

        # 3.3 AI file chooser
        file_layout = QHBoxLayout()
        file_label = QLabel("AI File:")
        self.file_button = QPushButton("Choose File")
        self.file_button.clicked.connect(self.choose_file)
        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_button)
        input_layout.addLayout(file_layout)

        main_layout.addLayout(input_layout)

        # 4. Effect buttons area
        effect_layout = QGridLayout()
        for i in range(1, 11):
            button = QPushButton(f"Effect {i}")
            button.clicked.connect(self.effect_clicked)
            effect_layout.addWidget(button, (i - 1) // 5, (i - 1) % 5)
        main_layout.addLayout(effect_layout)

        self.setLayout(main_layout)

    def update_control_model(self):
        # 更新全域變數 control_mode 根據複選框的狀態
        global control_mode  # 使用 global 關鍵字來修改全域變數
        if self.switch_toggle.isChecked():
            control_mode = "AI"  # 如果勾選，設定為 AI 模式
        else:
            control_mode = "manual"  # 如果未勾選，設定為手動模式

        # 顯示當前模式
        self.terminal_display.append(f"Control mode: {control_mode}")

    def effect_clicked(self):
        global control_mode, manual_input  # 使用 global 關鍵字來修改全域變數

        sender = self.sender()
        effect_number = str(sender.text().split()[-1])  # 取得效果的數字
        
        # 將 control_mode 設為 Manual，並更新 manual_input
        control_mode = "manual"
        manual_input = effect_number

        # 同步更新 QCheckBox 的狀態
        self.switch_toggle.setChecked(False)  # 在手動模式下，複選框為未勾選

        # 顯示當前效果和模式
        self.terminal_display.append(f"Effect {effect_number} selected")
        self.terminal_display.append(f"Control mode: {control_mode}")
        self.terminal_display.append(f"Manual input: {manual_input}")

    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select AI File", "", "All Files (*)")
        if file_path:
            self.terminal_display.append(f"Selected file: {file_path}")


# Threading setup
thread1 = threading.Thread(target=ProcessData, args=("Thread A",))
thread2 = threading.Thread(target=AIpredict, args=(network, "Thread B"))
thread3 = threading.Thread(target=control_lights, args=())
thread4 = threading.Thread(target=BPMaudio, args=())
thread5 = threading.Thread(target=monitor_control_mode)
thread6 = threading.Thread(target=check_input)
thread7 = threading.Thread(target=audio)


# Start threads

thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()
thread6.start()
thread7.start()


import sys
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())


