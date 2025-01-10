import platform
from keras.activations import selu, sigmoid
import keras
import keras.layers
from keras.optimizers.optimizer_experimental.adamw import AdamW  ; print(platform.platform()); import sys; print("Python", sys.version); import numpy; print("NumPy", numpy.__version__); import scipy; print("SciPy", scipy.__version__); import librosa; print("librosa", librosa.__version__); librosa.show_versions();
import librosa
from keras.metrics import AUC, Accuracy, Precision, sparse_categorical_accuracy
from tensorflow._api.v2.compat.v1 import ConfigProto, Session
from tqdm.keras import TqdmCallback
import math
import os
from keras.losses import BinaryCrossentropy, CategoricalCrossentropy, KLDivergence, MeanSquaredError, binary_crossentropy, sparse_categorical_crossentropy
from keras.models import Sequential, load_model, save_model
from keras.optimizers import SGD, Adam
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
import numpy as np
from tensorflow import keras as tfkrs
from keras.layers import BatchNormalization, Input, InputLayer, LeakyReLU, Normalization, Conv1D, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling1D, GlobalAveragePooling2D, MaxPooling1D, MaxPooling2D, UpSampling2D
import tensorflow as tf
import keras as krs

from matplotlib import pyplot
from datetime import datetime
import time
import cv2
from mutagen.mp3 import MP3
from keras.backend import set_session
from keras.backend import clear_session
from keras.backend import get_session
import gc
#from jax import jit
#import jax.numpy as jnp
#from numba import jit
import itertools
from keras import backend as K
import warnings
#warnings.filterwarnings("ignore",category=UserWarning)
#warnings.filterwarnings("ignore")
#os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
os.environ["TF_CPP_VMODULE"]="gpu_process_state=10,gpu_cudamallocasync_allocator=10"
print(krs.__version__)
print(tf.config.list_physical_devices('GPU'))
 
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.visible_device_list = "0"
set_session(Session(config=config))

def add_convlayers(inputs, filters, kernelSize):
    x = keras.layers.Conv2D(filters = filters, kernel_size = kernelSize, strides = (1, 1), padding = "valid" if (1,1) == kernelSize else "same")(inputs)
    #x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha = 0.1)(x)
    return x

def addResidualBlock(inputs, filters, looptime):
    x = inputs
    for i in range(looptime):
        x = add_convlayers(x, filters, kernelSize=(1,1))
        x = add_convlayers(x, filters*2, kernelSize=(3,3))
        x = keras.layers.Add()([inputs, x])
        print(x)

    return x

def complex_net(inputs = None):
    x = add_convlayers(inputs, filters = 32, kernelSize = (3, 3))
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (2, 2), padding = "same")(x)
    
    x = addResidualBlock(x, filters = 32, looptime = 1)
    x = Conv2D(filters = 128, kernel_size = (3, 3), strides = (2, 2), padding = "same")(x)
    
    x = addResidualBlock(x, filters = 64, looptime = 2)
    x = Conv2D(filters = 256, kernel_size = (3, 3), strides = (2, 2), padding = "same")(x)
    
    # 4xmipmap
    x4x = addResidualBlock(x, filters = 128, looptime = 8)
    x = Conv2D(filters = 512, kernel_size = (3, 3), strides = (2, 2), padding = "same")(x4x)
    
    # 2xmipmap
    x2x = addResidualBlock(x, filters = 256, looptime = 8)
    x = Conv2D(filters = 1024, kernel_size = (3, 3), strides = (2, 2), padding = "same")(x2x)
    
    # 1xmipmap
    x1x = addResidualBlock(x, filters = 512, looptime = 4)
    
    return x1x, x2x, x4x

def conv2Block(inputs, filters):
    x = add_convlayers(inputs, filters, kernelSize = (1, 1))
    x = add_convlayers(x, filters * 2, kernelSize = (3, 3))
    x = add_convlayers(x, filters, kernelSize = (1, 1))
    x = add_convlayers(x, filters * 2, kernelSize = (3, 3))
    x = add_convlayers(x, filters, kernelSize = (1, 1))
    return x

def neck(inputs = None):
    x1x, x2x, x4x = inputs
    feature = conv2Block(x1x, 512)
    feature = add_convlayers(feature, filters = 256, kernelSize = (1, 1))
    feature = UpSampling2D(size = (2, 2), interpolation = "bilinear")(feature)
    feature = keras.layers.Concatenate(axis = -1)([feature, x2x])
    
    x2x = conv2Block(feature, 256)
    
    feature = add_convlayers(x2x, filters = 128, kernelSize = (1, 1))
    feature = UpSampling2D(size = (2, 2), interpolation = "bilinear")(feature)
    feature = keras.layers.Concatenate(axis = -1)([feature, x4x])
    
    x4x = conv2Block(feature, 128)
    
    return x1x, x2x, x4x

def head(inputs, filters):
    x1x, x2x, x4x = inputs
    x1x = add_convlayers(x1x, 1024, kernelSize = (3, 3))
    x1x = add_convlayers(x1x, filters, kernelSize = (1, 1))
    
    x2x = add_convlayers(x2x, 512, kernelSize = (3, 3))
    x2x = add_convlayers(x2x, filters, kernelSize = (1, 1))
    
    x4x = add_convlayers(x4x, 256, kernelSize = (3, 3))
    x4x = add_convlayers(x4x, filters, kernelSize = (1, 1))
    print(x4x)
    print(x2x)
    print(x1x)
    return x1x, x2x, x4x

def custom_model(lenlabels):
    image = Input(shape=(94,94,1), name="input")

    x1 = add_convlayers(inputs=image, filters=16, kernelSize=(3,3))
    x2 = addResidualBlock(x1, filters=8, looptime=1)
    x2 = Dropout(0.15)(x2)
    x = Conv2D(filters = 32, kernel_size = (3, 3), strides = (2, 2), padding = "same")(x2)
    x3 = add_convlayers(inputs=x, filters=32, kernelSize=(3,3))
    x4 = addResidualBlock(x3, filters=16, looptime=2)
    x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (2, 2), padding = "same")(x4)
    x5 = add_convlayers(inputs=x, filters=64, kernelSize=(3,3))
    x6 = addResidualBlock(x5, filters=32, looptime=3)
    x = Conv2D(filters = 128, kernel_size = (3, 3), strides = (2, 2), padding = "same")(x6)
    x7 = add_convlayers(inputs=x, filters=128, kernelSize=(3,3))
    x8 = addResidualBlock(x7, filters=64, looptime=4)
    x = Conv2D(filters = 256, kernel_size = (3, 3), strides = (2, 2), padding = "same")(x8)
    x9 = add_convlayers(inputs=x, filters=256, kernelSize=(3,3))
    x = MaxPooling2D(pool_size=(2,2))(x9)
    x10 = Conv2D(filters = 1024, kernel_size = (3, 3), strides = (2, 2))(x)
    #x = BatchNormalization()(x10)
    x = LeakyReLU(alpha = 0.1)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048)(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = Dense(lenlabels)(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    model = keras.Model(inputs = image,
                        outputs = x,
                        name = "notyolov3")
    print(model.summary())
    return model



def yolomodel(lenlabels):
    image = Input(shape = (94, 94, 1), name = "input")

    x1x, x2x, x4x = complex_net(inputs = image)
    x1x, x2x, x4x = neck([x1x, x2x, x4x])
    x1x, x2x, x4x = head([x1x, x2x, x4x], filters = 75)
    x = Conv2D(128,(2,2), strides=(1,1))(x4x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.1)(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(256,(2,2), strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.1)(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(1024,(2,2), strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha = 0.1)(x)
    #x = MaxPooling2D(pool_size=(2,2))(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048)(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = Dense(lenlabels)(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    model = keras.Model(inputs = image,
                        outputs = [x],
                        name = "yolov3")
    model.summary()
    print(model.predict(np.ones([1,94,94,1])))
    return model
    
def reset_keras():
    sess = get_session()
    for i in range(0,20): #暴力重置記憶體，不知道為什麼，它能運作
        clear_session()
        sess.close()
    sess = get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted



def NParraymap(x):#normalize data
    return (x - np.min(x)) / (np.max(x)-np.min(x))

#取頻譜圖

def GetMFCC(path, offset, duration, resolution):
  y, sr = librosa.load(path, offset=offset, duration=duration, sr=320000)
  mfcc = np.array(librosa.feature.mfcc(y=y, sr=sr))
  #mfcc = NParraymap(mfcc)
  #mfcc = np.array(cv2.resize(mfcc, (resolution,20)))
  #mfcc = cv2.cvtColor(mfcc, cv2.COLOR_GRAY2BGR)
  return mfcc


def GetMelspectrogram(path, offset, duration, resolution):
  y, sr = librosa.load(path, offset=offset, duration=duration, sr=320000)
  melspectrogram = np.array(librosa.feature.melspectrogram(y=y, sr=sr))
  #melspectrogram = NParraymap(melspectrogram)
  #melspectrogram = np.array(cv2.resize(melspectrogram, (resolution,128)))
  #melspectrogram = cv2.cvtColor(melspectrogram, cv2.COLOR_GRAY2BGR)
  return melspectrogram


def GetChromaVector(path, offset, duration, resolution):
  y, sr = librosa.load(path, offset=offset, duration=duration, sr=320000)
  chroma = np.array(librosa.feature.chroma_stft(y=y, sr=sr))
  #chroma = NParraymap(chroma)
  #chroma = np.array(cv2.resize(chroma, (resolution,12)))
  #chroma = cv2.cvtColor(chroma, cv2.COLOR_GRAY2BGR)
  return chroma


def GetTonnetz(path, offset, duration, resolution):
  y, sr = librosa.load(path, offset=offset, duration=duration, sr=320000)
  tonnetz = np.array(librosa.feature.tonnetz(y=y, sr=sr), dtype='float32')
  #tonnetz = NParraymap(tonnetz)
  #tonnetz = np.array(cv2.resize(tonnetz, (resolution,6)))
  #arrayMax = np.max(tonnetz)
  #arrayMin = np.min(tonnetz)
  #tonnetz = np.round(NParraymap(tonnetz,arrayMin,arrayMax))
  #print(tonnetz.shape)
  #print(tonnetz)
  #cv2.cvtColor(tonnetz, cv2.COLOR_BGRA2RGB)
  #tonnetz = cv2.cvtColor(tonnetz, cv2.COLOR_GRAY2BGR)
  return tonnetz

#串頻譜圖，改解析度以確保可以被模型fit

def get_feature(path, offset, duration, resolution):
    print(path)
    # Extracting MFCC feature
    mfcc = GetMFCC(path, offset, duration, resolution)
    #print(mfcc.shape)
    #mfcc_mean = mfcc.mean(axis=1)
    #mfcc_min = mfcc.min(axis=1)
    #mfcc_max = mfcc.max(axis=1)
    #mfcc_feature = np.concatenate( (mfcc_mean, mfcc_min, mfcc_max) )

    # Extracting Mel Spectrogram feature
    melspectrogram = GetMelspectrogram(path, offset, duration, resolution)
    #print(melspectrogram.shape)
    #melspectrogram_mean = melspectrogram.mean(axis=1)
    #melspectrogram_min = melspectrogram.min(axis=1)
    #melspectrogram_max = melspectrogram.max(axis=1)
    #melspectrogram_feature = np.concatenate( (melspectrogram_mean, melspectrogram_min, melspectrogram_max) )

    # Extracting chroma vector feature
    chroma = GetChromaVector(path, offset, duration, resolution)
    #print(chroma.shape)
    #chroma_mean = chroma.mean(axis=1)
    #chroma_min = chroma.min(axis=1)
    #chroma_max = chroma.max(axis=1)
    #chroma_feature = np.concatenate( (chroma_mean, chroma_min, chroma_max) )

    # Extracting tonnetz feature
    tntz = GetTonnetz(path, offset, duration, resolution)
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

#def acc_append(array1, array2):
#    return jnp.append(array1, jnp.array([array2]), axis=0)

#取得資料，資料前處理

def generateFeature():
    timecache = time.time()
    directory = 'data'
    duration = eval(input("Enter duration:"))
    resolution = round(duration * 100)
    genres = ['effect1','effect2','effect3','effect4','effect5','effect6','effect7','effect8','effect9','effect10']
    features = []
    labels = []
    #取得檔案列表
    def get_all_file_in_genre(directory, genre):
        result = os.listdir(directory + "/" + genre)
        return map(lambda file : os.path.join(directory, genre, file), result)
    
    file_list = [get_all_file_in_genre(directory, genre) for genre in genres ]
    
    file_list = list(itertools.chain(*file_list))
    #取檔案長度，類別
    def file_to_feature_label_list(file_path): #duration, resolution=round(duration * 100)):
        print("getting feature for " + "data")
        music = MP3(file_path)
        length = music.info.length
        index = genres.index(file_path.split('\\')[1])
        return[length, index]
        #return \
        #    [ get_feature(file_path2, i * duration, duration, resolution) for i in range(math.floor(length / duration)) for file_path2 in file_list], \
        #    [ np.eye(len(genres))[index] for _ in range(math.floor(length / duration)) ]
        # features.append(get_feature(file_path, offset, duration, resolution))
        # for i in range(1, math.floor(length/duration)):
        #     timecache = time.time()
        #     offset = i * duration
        #     labelList = []
        #     features = acc_append(jnp.array(features), get_feature(file_path, offset, duration, resolution))
        #     print(features.shape)
        #     label = [1 if idx == genres.index(genre) else 0 for idx in range(len(genres))]
        #     labels = np.append(labels, [label], axis = 0)
        #     print(time.time() - timecache)
    #features = [file_to_feature_label_list(file_path, duration) for file_path in file_list]
    
    #切成0.xxx秒一份，產頻譜圖，串接資料及類別（答案）
    for file_path in file_list:
        length, index = file_to_feature_label_list(file_path=file_path)
        if type(features) == list:
            features = np.array([get_feature(file_path, i * duration, duration, resolution) for i in range(math.floor(length / duration))])
        else:
            features = np.concatenate([features, [get_feature(file_path, i * duration, duration, resolution) for i in range(math.floor(length / duration))]], axis = 0)
        if type(labels) == list:
            labels = np.array([ np.eye(len(genres))[index] for _ in range(math.floor(length / duration)) ])
        else:
            labels = np.concatenate([labels, [ np.eye(len(genres))[index] for _ in range(math.floor(length / duration)) ]], axis = 0)
    
        print(np.array(features).shape)
        print(np.array(labels).shape)
    #features = list(itertools.chain(*features))
    print(np.array(features).shape)
    #labels = list(itertools.chain(*labels))
    print(np.array(labels).shape)
    
    # for genre in genres:
    #     print("getting feature for genre " + genre)
    #     for file in os.listdir(directory+"/"+genre):
    #         file_path = os.path.join(directory, genre, file)
    #         print("getting feature for " + file_path)
    #         music = MP3(file_path)
    #         length = music.info.length
    #         offset = 0
    #         labelList = []
            
    #         if type(features) == list:
    #             features.append(get_feature(file_path, offset, duration, resolution))
    #         else:
    #             features = acc_append(jnp.array(features), get_feature(file_path, offset, duration, resolution))
            
    #         label = [1 if idx == genres.index(genre) else 0 for idx in range(len(genres))]
            
    #         if type(labels) == list:
    #             labels.append(label)
    #         else:
    #             labels = np.append(labels, [label], axis = 0)
            
    #         for i in range(1, math.floor(length/duration)):
    #             timecache = time.time()
    #             offset = i * duration
    #             labelList = []
    #             features = acc_append(jnp.array(features), get_feature(file_path, offset, duration, resolution))
    #             print(features.shape)
    #             label = [1 if idx == genres.index(genre) else 0 for idx in range(len(genres))]
    #             labels = np.append(labels, [label], axis = 0)
    #             print(time.time() - timecache)

    time_now = datetime.now() #儲存資料
    years = time_now.year
    months = time_now.month
    days = time_now.day
    hours = time_now.hour
    minutes = time_now.minute
    secs = time_now.second
    np.save(f'feature/feature{years}{months}{days}{hours}{minutes}{secs}.npy', features)
    np.save(f'feature/label{years}{months}{days}{hours}{minutes}{secs}.npy', labels)
    print(f"used {time.time()-timecache} seconds")
    return features, labels
        
def generateFeatureExpiremental():
    timecache = time.time()
    directory = 'data'
    duration = eval(input("Enter duration:"))
    resolution = round(duration * 100)
    genres = ['effect1','effect2','effect3','effect4','effect5','effect6','effect7','effect8','effect9','effect10']
    features = []
    labels = []
    #取得檔案列表
    def get_all_file_in_genre2(directory, genre):
        result = os.listdir(directory + "/" + genre)
        return map(lambda file : os.path.join(directory, genre, file), result)
    
    file_list = [get_all_file_in_genre2(directory, genre) for genre in genres ]
    
    file_list = list(itertools.chain(*file_list))
    #取檔案長度，類別
    def file_to_feature_label_list2(file_path): #duration, resolution=round(duration * 100)):
        print("getting feature for " + "data")
        music = MP3(file_path)
        length = music.info.length
        index = genres.index(file_path.split('\\')[1])
        return[length, index]
        #return \
        #    [ get_feature(file_path2, i * duration, duration, resolution) for i in range(math.floor(length / duration)) for file_path2 in file_list], \
        #    [ np.eye(len(genres))[index] for _ in range(math.floor(length / duration)) ]
        # features.append(get_feature(file_path, offset, duration, resolution))
        # for i in range(1, math.floor(length/duration)):
        #     timecache = time.time()
        #     offset = i * duration
        #     labelList = []
        #     features = acc_append(jnp.array(features), get_feature(file_path, offset, duration, resolution))
        #     print(features.shape)
        #     label = [1 if idx == genres.index(genre) else 0 for idx in range(len(genres))]
        #     labels = np.append(labels, [label], axis = 0)
        #     print(time.time() - timecache)
    #features = [file_to_feature_label_list(file_path, duration) for file_path in file_list]
    
    #切成0.xxx秒一份，產頻譜圖，串接資料及類別（答案）
    for file_path in file_list:
        length, index = file_to_feature_label_list2(file_path=file_path)
        if type(features) == list:
            features = np.array([get_feature(file_path, i * duration, duration, resolution) for i in range(math.floor(length / duration))])
        else:
            features = np.concatenate([features, [get_feature(file_path, i * duration, duration, resolution) for i in range(math.floor(length / duration))]], axis = 0)
        if type(labels) == list:
            labels = np.array([[index] for _ in range(math.floor(length / duration)) ])
            print(labels)
        else:
            labels = np.concatenate([labels, [[index] for _ in range(math.floor(length / duration)) ]], axis = 0)
            print(labels)
    
        print(np.array(features).shape)
        print(np.array(labels).shape)
    #features = list(itertools.chain(*features))
    print(np.array(features).shape)
    #labels = list(itertools.chain(*labels))
    print(np.array(labels).shape)
    
    # for genre in genres:
    #     print("getting feature for genre " + genre)
    #     for file in os.listdir(directory+"/"+genre):
    #         file_path = os.path.join(directory, genre, file)
    #         print("getting feature for " + file_path)
    #         music = MP3(file_path)
    #         length = music.info.length
    #         offset = 0
    #         labelList = []
            
    #         if type(features) == list:
    #             features.append(get_feature(file_path, offset, duration, resolution))
    #         else:
    #             features = acc_append(jnp.array(features), get_feature(file_path, offset, duration, resolution))
            
    #         label = [1 if idx == genres.index(genre) else 0 for idx in range(len(genres))]
            
    #         if type(labels) == list:
    #             labels.append(label)
    #         else:
    #             labels = np.append(labels, [label], axis = 0)
            
    #         for i in range(1, math.floor(length/duration)):
    #             timecache = time.time()
    #             offset = i * duration
    #             labelList = []
    #             features = acc_append(jnp.array(features), get_feature(file_path, offset, duration, resolution))
    #             print(features.shape)
    #             label = [1 if idx == genres.index(genre) else 0 for idx in range(len(genres))]
    #             labels = np.append(labels, [label], axis = 0)
    #             print(time.time() - timecache)

    time_now = datetime.now() #儲存資料
    years = time_now.year
    months = time_now.month
    days = time_now.day
    hours = time_now.hour
    minutes = time_now.minute
    secs = time_now.second
    np.save(f'feature/feature{years}{months}{days}{hours}{minutes}{secs}.npy', features)
    np.save(f'feature/label{years}{months}{days}{hours}{minutes}{secs}.npy', labels)
    print(f"used {time.time()-timecache} seconds")
    return features, labels

def TrainAI(features, labels):#訓練模型
    epochs = eval(input("Epochs times:"))#設定迭代
    splitData = eval(input("input slices of data you want to split:"))#設定分割批次
    splitratio = 1/splitData
    time_now = datetime.now()
    years = time_now.year
    months = time_now.month
    days = time_now.day
    hours = time_now.hour
    minutes = time_now.minute
    secs = time_now.second
    modelname = f'tempmodel{years}{months}{days}{hours}{minutes}{secs}'
    #def normalize(array):
    #    process = (array - np.min(array)) / (np.max(array) - np.min(array))
    #    return process
    #features = normalize(features)
    #labels = normalize(labels)
    permutations = np.random.permutation(len(labels))#打亂並分割資料為訓練集、驗證集和測試集
    features = np.array(features)[permutations]
    labels = np.array(labels)[permutations]

    features_train = features[0:round(len(labels)*0.6)]
    labels_train = labels[0:round(len(labels)*0.6)]

    features_val = features[round(len(labels)*0.6):round(len(labels)*0.8)]
    labels_val = labels[round(len(labels)*0.6):round(len(labels)*0.8)]

    features_test = features[round(len(labels)*0.8):len(labels)]
    labels_test = labels[round(len(labels)*0.8):len(labels)]
    del features
    del labels
    trainSplit = []
    trainSplitlabel = []

    for i in range(0,splitData):#分割訓練集為N份
        trainSplit.append(features_train[round(i*splitratio*len(features_train)): round((i+1)*splitratio*len(features_train))])
        trainSplitlabel.append(labels_train[round(i*splitratio*len(labels_train)): round((i+1)*splitratio*len(labels_train))])

    print(features_train.shape)
    print(labels_train.shape)
    del features_train
    del labels_train
    np.save(f'feature/testfeature{years}{months}{days}{hours}{minutes}{secs}.npy', features_test)
    np.save(f'feature/testlabel{years}{months}{days}{hours}{minutes}{secs}.npy', labels_test)
    genres = ['effect1','effect2','effect3','effect4','effect5','effect6','effect7','effect8','effect9','effect10']
    lrS = ExponentialDecay(0.1, 4000, 0.95)
    #inputs = keras.Input(shape=(166,1024,), name="feature")
    #x = keras.layers.Dense(512, activation="relu", name="layerDense1")(inputs)
    #x = keras.layers.Dense(512, activation="relu", name="layerDense2")(x)
    #x = keras.layers.Dense(512, activation="relu", name="layerDense3")(x)
    #x = keras.layers.GlobalAveragePooling1D()(x)
    #x = keras.layers.Dense(256, activation="relu", name="layerDense4")(x)
    #x = keras.layers.Dense(256, activation="relu", name="layerDense5")(x)
    #x = keras.layers.Dense(256, activation="relu", name="layerDense6")(x)
    #x = keras.layers.Dense(128, activation='relu', name="layerDense7")(x)
    #outputs = keras.layers.Dense(9, activation="softmax", name="Result")(x)
    #network = keras.Model(inputs=inputs, outputs=outputs)
    #設定模型
    network = krs.models.Sequential()
    
    network.add(Conv2D(64,(2,2), input_shape=(np.array(trainSplit[0]).shape[1],np.array(trainSplit[0]).shape[2],np.array(trainSplit[0]).shape[3],), name='cnnNet00'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D(pool_size=(2,2)))
    network.add(Dropout(0.25))
    network.add(Conv2D(128,(2,2), activation=selu , name='cnnNet0'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D(pool_size=(2,2)))
    network.add(Dropout(0.05))
    network.add(Conv2D(512,(2,2), activation=selu, name='cnnNet1'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D(pool_size=(2,2)))
    network.add(Dropout(0.05))
    network.add(Conv2D(1024,(2,2), activation=selu, name='cnnNet2'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D(pool_size=(2,2)))
    network.add(Dropout(0.05))
    network.add(Conv2D(2048,(2,2), activation=selu, name='cnnNet3'))
    network.add(BatchNormalization())
    network.add(MaxPooling2D(pool_size=(2,2)))
    #network.add(Conv2D(2048,(2,2), activation=selu, name='cnnNet4'))
    #network.add(MaxPooling2D(pool_size=(2,2)))
    #network.add(Conv2D(1024,(3,3), activation="relu", name='cnnNet5'))
    #network.add(MaxPooling2D(pool_size=(2,2)))
    network.add(GlobalAveragePooling2D())

    #network.add(Flatten())
    #network.add(Dense(512, activation="relu", name="layerDense1"))
    network.add(Dropout(0.05))
    #network.add(Dense(512, activation="relu", name="layerDense2"))
    #network.add(Dense(512, activation="relu", name="layerDense3"))
    #network.add(Dense(256, activation="relu", name="layerDense4"))
    network.add(Dense(2048, activation=selu, name="layerDense5"))
    network.add(BatchNormalization())
    network.add(Dropout(0.05))
    network.add(Dense(128, activation=selu, name="layerDense6"))
    
    #network.add(Dense(len(genres), activation="softmax", name="Result"))
    network.add(Dense(len(trainSplitlabel[0][0]), activation="softmax", name="Result"))

    network.compile(optimizer = Adam(learning_rate=0.001,beta_1=0.3,
                                     beta_2=0.5,epsilon=0.0000000001,
                                     decay=0.09), loss = 'binary_crossentropy',
	                 metrics = [AUC(),"accuracy",Precision()])
    #network.compile(optimizer = SGD(learning_rate=0.01), loss = binary_crossentropy,
	#                 metrics = [AUC(),'accuracy',Precision()])
    print(network.summary())

    #network = custom_model(len(genres))
    #network.compile(optimizer = SGD(learning_rate=0.03), loss = binary_crossentropy,
	#                 metrics = [AUC(),'accuracy',Precision()])
    #network.compile(optimizer = Adam(learning_rate=0.0003,beta_1=0.3,
    #                                 beta_2=0.5,epsilon=0.0000000001,amsgrad=True,
    #                                 decay=0.09), loss = 'binary_crossentropy',
	#                 metrics = [AUC(),"accuracy",Precision()])
    #第一批資料迭代一次
    network.fit(trainSplit[0], trainSplitlabel[0], validation_data=(features_val,labels_val),epochs = 1, verbose=40, callbacks=TqdmCallback(verbose=1))
    save_model(network, f'tempmodel/{modelname}split1+{splitData}total+epoch1+{epochs}total.h5')
    

    
    reset_keras()#重置session並創立新session
    time.sleep(5)
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(Session(config=config))
    time.sleep(5)
    traincount = 0

    #因記憶體不足，所以會依照迭代數及分批數對同個模型重複進行一次迭代的訓練，例如17epochs+5split，則需訓練17*5=85次
    for i in range(1,epochs+1):
        print(f'\nEpochs {i}/{epochs}')
        for j in range(1,splitData+1):
            print(f'\nSplit {j}/{splitData}\n')
            timerNum = time.time()
            traincount += 1
            network = Sequential()
            network = load_model(f'tempmodel/{modelname}split{j}+{splitData}total+epoch{i}+{epochs}total.h5')
            network.fit(trainSplit[j-1], trainSplitlabel[j-1], epochs = 1, verbose=0, callbacks=TqdmCallback(verbose=2))
            if j != splitData:
                save_model(network, f'tempmodel/{modelname}split{j+1}+{splitData}total+epoch{i}+{epochs}total.h5')
            reset_keras()
            if os.path.exists(f'tempmodel/{modelname}split{j}+{splitData}total+epoch{i-130}+{epochs}total.h5'):
                os.remove(f'tempmodel/{modelname}split{j}+{splitData}total+epoch{i-130}+{epochs}total.h5')
            else:
                print("The file does not exist")
            time.sleep(1)
            config = ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 1
            config.gpu_options.visible_device_list = "0"
            set_session(Session(config=config))
            time.sleep(1)
            timeuse = time.time() - timerNum
            #取得一次訓練的時間，並計算剩餘時間
            print(f'\nUsed {timeuse} seconds.\nETA --- {timeuse * (epochs * splitData - traincount)}s\n')
        print(network.evaluate(features_val, labels_val, verbose=0, callbacks=TqdmCallback(verbose=2)))
        save_model(network, f'tempmodel/{modelname}split1+{splitData}total+epoch{i+1}+{epochs}total.h5')

    
    
    
    time_now = datetime.now()
    years = time_now.year
    months = time_now.month
    days = time_now.day
    hours = time_now.hour
    minutes = time_now.minute
    secs = time_now.second
    save_model(network, f'model/model{years}{months}{days}{hours}{minutes}{secs}.h5')#儲存模型檔
    #network     .save(f'model/tempmodel{years}{months}{days}{hours}{minutes}{secs}.keras')
    #network.save_weights(f'model/tempmodelweight{years}{months}{days}{hours}{minutes}{secs}.h5')
    
    print("file saved as tempmodel.keras and tempmodel.wht")
    K.clear_session()
    #cuda.select_device(0)
    #cuda.close()

    time.sleep(20)
    #network = Sequential()
    #network = load_model(input("Re-Enter temp model file path:").split('"')[1])
    #network.load_weights(input("Re-Enter temp h5 file path:").split('"')[1])
    #time.sleep(3)
    #
    #score = network.evaluate(features_test,labels_test, verbose=1)
    #network.fit(x=np.array(features_train.tolist()), y=np.array(labels_train.tolist()), validation_data=(np.array(features_val.tolist()),np.array(labels_val.tolist())),epochs = 100)
    #score = network.evaluate(x=np.array(features_test.tolist()),y=np.array(labels_test.tolist()), verbose=1)
    #print('Accuracy:'+str(score[1]))
    #print('Loss:'+str(score[0]))

    #time_now = datetime.now()
    #years = time_now.year
    #months = time_now.month
    #days = time_now.day
    #hours = time_now.hour
    #minutes = time_now.minute
    #secs = time_now.second
    #network.save(f'model/model{years}{months}{days}{hours}{minutes}{secs}.keras')
    #network.save_weights(f'model/model{years}{months}{days}{hours}{minutes}{secs}.h5')
    #print("file saved as model.keras and model.wht")
    #time.sleep(5)
    return network

def TestAI(network):#測試模型
    file_path = input("Enter music file path:")
    if '"' in file_path:
        file_path = input("Enter music file path:").strip('"')
    features = []
    labels = []
    resolution = -1
    print("getting feature for " + file_path)
    music = MP3(file_path)#轉換指定資料為測試陣列集
    duration = 0.3
    length = music.info.length
    features = np.array([get_feature(file_path, i * duration, duration, resolution) for i in range(math.floor(length / duration))])
    for i in range(3,len(features)):#模型預測結果
        timecache = time.time()
        predictingData = np.array((features[i-3],features[i-2],features[i-1],features[i]))
        prediction = network.predict(predictingData)
        prediction = np.add(np.add(np.add(prediction[0],prediction[1]),prediction[2]),prediction[3])
        print(prediction)
        print(prediction.tolist().index(np.max(prediction)))
        print(time.time() - timecache)
def evaluate(network,features,labels):#使用一給定答案之陣列集進行測試並得出準確率
    #permutations = np.random.permutation(len(labels))
    #features = np.array(features)[permutations]
    #labels = np.array(labels)[permutations]
    #features_test = features[round(len(labels)*0.8):len(labels)]
    #labels_test = labels[round(len(labels)*0.8):len(labels)]
    score = network.evaluate(features,labels, verbose=1)
    print('Accuracy:'+str(score[1]))
    print('Loss:'+str(score[0]))
    return
def loadArray():#在訓練前載入陣列
    featurespath = input("Enter feature file path:").split('"')[1]
    labelspath = input("Enter label file path:").split('"')[1]

    features = np.load(featurespath)
    labels = np.load(labelspath)
    print(features.shape)
    print(labels.shape)
    return features, labels


def Main():
    while True:
        commend = input("1---------Generate features from data folder\n2---------Train AI\n3---------Load array\n4---------Test AI\n5---------Debugging data\n6---------Load AI model\n7---------Evaluate using model and data\nPlease enter the commend:")
        if commend == '1':
            features, labels = generateFeature()
        elif commend == '2':
            network = TrainAI(features, labels)
        elif commend == '3':
            features, labels = loadArray()
        elif commend == '5':
            featurespath = input("Enter feature file path:").split('"')[1]
            labelspath = input("Enter label file path:").split('"')[1]
            
            features2 = np.load(featurespath)
            labels2 = np.load(labelspath)
            print(features2.shape)
            print(labels2.shape)
            if (features - features2).all:
                print("features verified")
            else:
                print("features not verified")
            if (labels - labels2).all:
                print("labels verified")
            else:
                print("labels not verified")
        elif commend == '4':
            TestAI(network)
        elif commend == '6':
            network = Sequential()
            network = load_model(input("Enter model file path:").split('"')[1])
            network.load_weights(input("Enter h5 file path:").split('"')[1])
        elif commend == '7':
            evaluate(network,features,labels)
        elif commend == '8':
            features, labels = generateFeatureExpiremental()

Main()

