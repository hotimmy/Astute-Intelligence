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
# Initialize Art-Net and Signal Generator
target_ip = '192.168.50.141'  # Target IP
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
AIpath = input("Enter AI file path:").strip('"')
try:
    network = keras.models.Sequential()
    network.predict(np.zeros((3,3,3)))
    network = keras.models.load_model(AIpath)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

print(sd.query_devices())
device_index = int(input("Enter the device index: "))

# Define global variables for audio processing
sr = 320000  # Sampling rate
sr1 = 44100
buffer_duration1 = 0.3  # Duration of audio chunks for AI processing
buffer_duration2 = 2  # Duration of audio chunks for BPM detection
hop_length = 512  # FFT hop length
threshold = 0.2  # Energy threshold for filtering silent segments
dataNow = []
Soundarrays = []

"""
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
"""

def NParraymap(x):
    return np.array((x - np.min(x)) / (np.max(x)-np.min(x)) * 255, dtype='uint8')

def GetMFCC(y):
    return np.array(librosa.feature.mfcc(y=y,n_mfcc=100 , sr=320000)) #給0.3秒的音訊 得到位元率32000的頻譜圖

def GetMelspectrogram(y):
    return np.array(librosa.feature.melspectrogram(y=y, sr=320000)) #給0.3秒的音訊 得到位元率32000的頻譜圖

def GetChromaVector(y):
    return np.array(librosa.feature.chroma_stft(y=y, sr=320000, n_chroma=24)) #給0.3秒的音訊 得到位元率32000的光譜圖


def get_feature(y): 
    mfcc = GetMFCC(y)
    melspectrogram = GetMelspectrogram(y)
    chroma = GetChromaVector(y)
    #tntz = GetTonnetz(y)
    return np.reshape(cv2.resize(np.concatenate((mfcc, melspectrogram, chroma)), (188,252)), (252,188,1))

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

def BPMaudio():
    with sd.InputStream(callback=audio_callback_for_BPM, channels=1, samplerate=sr1, blocksize=int(sr1 * buffer_duration2), device=device_index):
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

def audio_callback_for_BPM(indata, frames, time, status):
    # 將音訊數據轉為一維並標準化
    #print("callback")
    audio_buffer = indata[:, 0]
    
    # 檢查音訊能量以過濾靜音段
    if np.max(np.abs(audio_buffer)) > threshold:
        # 偵測節拍
        detect_beats(audio_buffer)

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

        

def control_lights():
    global output
    global BPM
    global control_mode
    control_mode = "manual"
    output = -1
    user_input = -1
    manual_input2 = -1
    BPM = 120
    artnet.start()
    artnet.set(packet)
    effect = False
    for i in [P1,P2,P4]:
        artnet.set_single_value(i+1, 98)
        artnet.set_single_value(i+2, 234)
        artnet.set_single_value(i+3, 255)
        #artnet.set_single_value(i-1, 255) # for 大堂
        #artnet.set_single_value(i-4, 127) # for 大堂
        #artnet.set_single_value(i-5, 127) # for 大堂
    for i in [P3,P5,P6]:
        artnet.set_single_value(i+1, 255)
        artnet.set_single_value(i+2, 79)
        artnet.set_single_value(i+3, 255)


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
                    for i in [P1,P2,P3,P4,P5,P6]:
                        artnet.set_single_value(i, 255)
                    effect = False

                elif user_input == "9":
                    print("10 P off")
                    for i in [P1,P2,P3,P4,P5,P6]:
                        artnet.set_single_value(i, 0)
                    effect = False

                elif user_input == "0":
                    #漸進漸出 bpm
                    print("1 漸進漸出 bpm")
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
                    flash = 600
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
                    slow = 60
                    sg.set_generator(P1, "sin", slow, frame_rate, offset=0)
                    sg.set_generator(P2, "sin", slow, frame_rate, offset=0.2)
                    sg.set_generator(P3, "sin", slow, frame_rate, offset=0.4)
                    sg.set_generator(P4, "sin", slow, frame_rate, offset=0.6)
                    sg.set_generator(P5, "sin", slow, frame_rate, offset=0.8)
                    sg.set_generator(P6, "sin", slow, frame_rate, offset=1) 
                    effect = True

                elif user_input == "6":
                    #漸進漸出 慢
                    print("7 漸進漸出 慢")
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


                    elif manual_input == "color":
                        print("color default")
                        for i in [P1,P2,P3,P4,P5,P6]:
                            artnet.set_single_value(i+1, 251)
                            artnet.set_single_value(i+2, 179)
                            artnet.set_single_value(i+3, 35)


                    elif manual_input == "9":
                        print("9 P on")
                        for i in [P1,P2,P3,P4,P5,P6]:
                            artnet.set_single_value(i, 255)
                        effect = False

                    elif manual_input == "10":
                        print("10 P off")
                        for i in [P1,P2,P3,P4,P5,P6]:
                            artnet.set_single_value(i, 0)
                        effect = False

                    elif manual_input == "1":
                        #漸進漸出 bpm
                        print("1 漸進漸出 bpm")
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
                        flash = 600
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
                        slow = 60
                        sg.set_generator(P1, "sin", slow, frame_rate, offset=0)
                        sg.set_generator(P2, "sin", slow, frame_rate, offset=0.2)
                        sg.set_generator(P3, "sin", slow, frame_rate, offset=0.4)
                        sg.set_generator(P4, "sin", slow, frame_rate, offset=0.6)
                        sg.set_generator(P5, "sin", slow, frame_rate, offset=0.8)
                        sg.set_generator(P6, "sin", slow, frame_rate, offset=1) 
                        effect = True

                    elif manual_input == "7":
                        #漸進漸出 慢
                        print("7 漸進漸出 慢")
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

# Threading setup
thread1 = threading.Thread(target=ProcessData, args=("Thread A",))
thread2 = threading.Thread(target=AIpredict, args=(network, "Thread B"))
thread3 = threading.Thread(target=control_lights, args=())
thread4 = threading.Thread(target=BPMaudio, args=())
thread5 = threading.Thread(target=monitor_control_mode)
thread6 = threading.Thread(target=check_input)

# Start threads
thread1.start()
thread2.start()
thread3.start()
thread4.start()
thread5.start()
thread6.start()


# Real-time audio recording and processing

with sd.InputStream(callback=audio_callback, channels=1, samplerate=sr, blocksize=int(sr * buffer_duration1), device=device_index):
    print("Real-time audio processing started. Press Ctrl+C to stop.")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("Real-time audio processing stopped.")