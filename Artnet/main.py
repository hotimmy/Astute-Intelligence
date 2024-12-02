from TestArtnet import StupidArtnet
from SignalGenerator import SignalGenerator as sg
import time
import sys
import select
import threading
from tensorflow.keras.models import load_model
#from keras.saving.save import load_model
import sounddevice as sd
import numpy as np
import librosa
import collections
import cv2
from tensorflow import keras

# Initialize Art-Net and Signal Generator
target_ip = '169.254.44.100'  # Target IP
universe = 0                  # Universe number
packet_size = 512             # Packet size
frame_rate = 40               # Frame rate (Hz)


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

# Initialize the neural network
#AIpath = input("Enter AI file path:").split('"')[1]
#network = keras.models.Sequential()
#network = load_model(AIpath)

# Initialize the neural network
AIpath = input("Enter AI file path:").strip()
try:
    network = keras.models.Sequential()
    network = load_model(AIpath)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

print(sd.query_devices())

# Define global variables for audio processing
sr = 320000  # Sampling rate
buffer_duration1 = 0.3  # Duration of audio chunks for AI processing
buffer_duration2 = 2  # Duration of audio chunks for BPM detection
hop_length = 512  # FFT hop length
threshold = 0.2  # Energy threshold for filtering silent segments
dataNow = []
Soundarrays = []

# Define global variables for BPM processing
bpm_buffer = collections.deque(maxlen=int(sr * buffer_duration2))  # Buffer for BPM detection

# Feature extraction functions (MFCC, MelSpectrogram, etc.)
def GetMFCC(y):
    return np.array(librosa.feature.mfcc(y=y, sr=320000))

def GetMelspectrogram(y):
    return np.array(librosa.feature.melspectrogram(y=y, sr=320000))

def GetChromaVector(y):
    return np.array(librosa.feature.chroma_stft(y=y, sr=320000))

def GetTonnetz(y):
    return np.array(librosa.feature.tonnetz(y=y, sr=320000), dtype='float32')

def get_feature(y):
    mfcc = GetMFCC(y)
    melspectrogram = GetMelspectrogram(y)
    chroma = GetChromaVector(y)
    tntz = GetTonnetz(y)
    return cv2.cvtColor(np.concatenate((mfcc, melspectrogram, chroma, tntz)), cv2.COLOR_GRAY2BGR)

def ProcessData(nameFunc):
    global Soundarrays
    global dataNow
    dataNow = []
    while True:
        if type(dataNow) == np.ndarray:
            Datacache = dataNow
            feature = get_feature(Datacache)
            Soundarrays = feature

def AIpredict(model, name):
    global Soundarrays
    global output
    output = -1
    Soundarrays = []
    PredictData = []
    print("AI start")
    while True:
        if type(Soundarrays) == np.ndarray:
            if len(PredictData) < 4:
                PredictData.append(Soundarrays)
            elif len(PredictData) == 4:
                PredictData = np.array((PredictData[1], PredictData[2], PredictData[3], Soundarrays))
                prediction = model.predict(PredictData,verbose=0)
                prediction = np.add(np.add(np.add(prediction[0], prediction[1]), prediction[2]), prediction[3])
                output = prediction.tolist().index(np.max(prediction))
                #print(output)
        

def audio_callback(indata, frames, time, status):
    global dataNow
    dataNow = np.reshape(indata, [indata.shape[0]])

def detect_beats(audio_buffer):
    tempo, beats = librosa.beat.beat_track(y=audio_buffer, sr=sr, hop_length=hop_length)
    return tempo, beats

def BPMProcessor():
    bpm_buffer = collections.deque(maxlen=int(sr * buffer_duration2))  # Buffer for BPM detection
    while True:
        if len(bpm_buffer) == int(sr * buffer_duration2):  # Ensure enough data is present
            audio_data = np.array(bpm_buffer)
            tempo, beats = detect_beats(audio_data)
            print(f"Detected BPM: {tempo}")
        else:
            continue  # Wait until enough data is available

def audio_callback_for_BPM(indata, frames, time, status):
    global bpm_buffer
    audio_data = indata[:, 0]
    bpm_buffer.extend(audio_data)

def check_input():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip()
    return None

def control_lights():
    global output
    output = -1
    user_input = -1
    BPM = 120
    artnet.start()
    artnet.set(packet)
    effect = False
    while True:

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
                artnet.set_single_value(PN1, 63)
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
                print("P on")
                for i in [P1,P2,P3,P4,P5,P6]:
                    artnet.set_single_value(i, 255)
                effect = False

            elif user_input == "9":
                print("P off")
                for i in [P1,P2,P3,P4,P5,P6]:
                    artnet.set_single_value(i, 0)
                effect = False

            elif user_input == "0":
                #漸進漸出 bpm
                print("漸進漸出 bpm")
                sg.set_generator(P1, "sin", BPM, frame_rate, offset=0)
                sg.set_generator(P2, "sin", BPM, frame_rate, offset=0.2)
                sg.set_generator(P3, "sin", BPM, frame_rate, offset=0.4)
                sg.set_generator(P4, "sin", BPM, frame_rate, offset=0.6)
                sg.set_generator(P5, "sin", BPM, frame_rate, offset=0.8)
                sg.set_generator(P6, "sin", BPM, frame_rate, offset=1) 
                effect = True

            elif user_input == "1":

                #Can-Can bpm
                print("Can-Can bpm")
                sg.set_generator(P1, "square", BPM, frame_rate, offset=0)
                sg.set_generator(P2, "square", BPM, frame_rate, offset=0)
                sg.set_generator(P4, "square", BPM, frame_rate, offset=0)
                sg.set_generator(P3, "square", BPM, frame_rate, offset=60/BPM)
                sg.set_generator(P5, "square", BPM, frame_rate, offset=60/BPM)
                sg.set_generator(P6, "square", BPM, frame_rate, offset=60/BPM)
                effect = True

            elif user_input == "2":
                #Can-Can 柔 bpm
                print("Can-Can 柔 bpm")
                sg.set_generator(P1, "sin", BPM, frame_rate, offset=0)
                sg.set_generator(P2, "sin", BPM, frame_rate, offset=0)
                sg.set_generator(P3, "sin", BPM, frame_rate, offset=0)
                sg.set_generator(P4, "sin", BPM, frame_rate, offset=30/BPM)
                sg.set_generator(P5, "sin", BPM, frame_rate, offset=30/BPM)
                sg.set_generator(P6, "sin", BPM, frame_rate, offset=30/BPM) 
                effect = True

            elif user_input == "3":
                #Rain bpm
                print("Rain bpm")
                sg.set_generator(P1, "rain", BPM, frame_rate, offset=0)
                sg.set_generator(P2, "rain", BPM, frame_rate, offset=BPM/30/6*1 )
                sg.set_generator(P3, "rain", BPM, frame_rate, offset=BPM/30/6*3 )
                sg.set_generator(P4, "rain", BPM, frame_rate, offset=BPM/30/6*4 )
                sg.set_generator(P5, "rain", BPM, frame_rate, offset=BPM/30/6*4 )
                sg.set_generator(P6, "rain", BPM, frame_rate, offset=BPM/30/6*5 ) 
                effect = True

            elif user_input == "4":
                #快閃
                print("快閃")
                flash = 600
                sg.set_generator(P1, "square", flash, frame_rate, offset=0)
                sg.set_generator(P2, "square", flash, frame_rate, offset=0.1)
                sg.set_generator(P3, "square", flash, frame_rate, offset=0.2)
                sg.set_generator(P4, "square", flash, frame_rate, offset=0.3)
                sg.set_generator(P5, "square", flash, frame_rate, offset=0.4)
                sg.set_generator(P6, "square", flash, frame_rate, offset=0.5) 
                effect = True
            
            elif user_input == "5":
                #漸進漸出 慢
                print("漸進漸出 慢")
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
                print("漸進漸出 慢")
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
                print("漸進漸出 快")
                fast = 120
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

# Threading setup
thread1 = threading.Thread(target=ProcessData, args=("Thread A",))
thread2 = threading.Thread(target=AIpredict, args=(network, "Thread B"))
#thread3 = threading.Thread(target=BPMProcessor, args=())
thread4 = threading.Thread(target=control_lights, args=())

# Start threads
thread1.start()
thread2.start()
#thread3.start()
thread4.start()

# Real-time audio recording and processing
device_index = int(input("Enter the device index: "))
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sr, blocksize=int(sr * buffer_duration1), device=device_index):
    print("Real-time audio processing started. Press Ctrl+C to stop.")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("Real-time audio processing stopped.")

with sd.InputStream(callback=audio_callback_for_BPM, channels=1, samplerate=sr, blocksize=int(sr * buffer_duration2), device=device_index):
    print("Real-time audio processing started. Press Ctrl+C to stop.")
    try:
        while True:
            sd.sleep(1000)
    except KeyboardInterrupt:
        print("Real-time audio processing stopped.")