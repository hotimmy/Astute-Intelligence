import re
import math
import os
from keras.models import Sequential, load_model
import numpy as np
from tensorflow import keras
from keras.layers import Normalization, Conv1D, Conv2D, Dense, Dropout, Flatten, GlobalAveragePooling1D, GlobalAveragePooling2D, MaxPooling1D, MaxPooling2D
import tensorflow as tf
import keras as krs
import librosa
from matplotlib import pyplot
from datetime import datetime
import time
import cv2
from mutagen.mp3 import MP3
#from jax import jit
import jax.numpy as jnp
from numba import jit
import itertools

print(tf.config.list_physical_devices('GPU'))

def GetMFCC(path, offset, duration, resolution):
  y, sr = librosa.load(path, offset=offset, duration=duration, sr=320000)
  mfcc = np.array(librosa.feature.mfcc(y=y, sr=sr))
  #mfcc = np.array(cv2.resize(mfcc, (resolution,20)))
  #mfcc = cv2.cvtColor(mfcc, cv2.COLOR_GRAY2BGR)
  return mfcc


def GetMelspectrogram(path, offset, duration, resolution):
  y, sr = librosa.load(path, offset=offset, duration=duration, sr=320000)
  melspectrogram = np.array(librosa.feature.melspectrogram(y=y, sr=sr))
  #melspectrogram = np.array(cv2.resize(melspectrogram, (resolution,128)))
  #melspectrogram = cv2.cvtColor(melspectrogram, cv2.COLOR_GRAY2BGR)
  return melspectrogram


def GetChromaVector(path, offset, duration, resolution):
  y, sr = librosa.load(path, offset=offset, duration=duration, sr=320000)
  chroma = np.array(librosa.feature.chroma_stft(y=y, sr=sr))
  #chroma = np.array(cv2.resize(chroma, (resolution,12)))
  #chroma = cv2.cvtColor(chroma, cv2.COLOR_GRAY2BGR)
  return chroma


def NParraymap(x,arrayMin,arrayMax):
    return (x + (-(arrayMin))) / (arrayMax-arrayMin) * 255


def GetTonnetz(path, offset, duration, resolution):
  y, sr = librosa.load(path, offset=offset, duration=duration, sr=320000)
  tonnetz = np.array(librosa.feature.tonnetz(y=y, sr=sr), dtype='float32')
  #tonnetz = np.array(cv2.resize(tonnetz, (resolution,6)))
  #arrayMax = np.max(tonnetz)
  #arrayMin = np.min(tonnetz)
  #tonnetz = np.round(NParraymap(tonnetz,arrayMin,arrayMax))
  #print(tonnetz.shape)
  #print(tonnetz)
  #cv2.cvtColor(tonnetz, cv2.COLOR_BGRA2RGB)
  #tonnetz = cv2.cvtColor(tonnetz, cv2.COLOR_GRAY2BGR)
  return tonnetz


def get_feature(path, offset, duration, resolution):
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

def acc_append(array1, array2):
    return jnp.append(array1, jnp.array([array2]), axis=0)

def generateFeature():
    timecache = time.time()
    directory = 'data'
    duration = eval(input("Enter duration:"))
    resolution = round(duration * 100)
    genres = ['effect1','effect2','effect3','effect4','effect5','effect6','effect7','effect8','effect9','effect10']
    
    def get_all_file_in_genre(directory, genre):
        result = os.listdir(directory + "/" + genre)
        return map(lambda file : os.path.join(directory, genre, file), result)
    
    file_list = [get_all_file_in_genre(directory, genre) for genre in genres ]
    
    file_list = list(itertools.chain(*file_list))
    
    def file_to_feature_label_list(file_path, duration, resolution=round(duration * 100)):
        print("getting feature for " + file_path)
        music = MP3(file_path)
        length = music.info.length
        index = genres.index(file_path.split('\\')[1])
        
        return \
            [ get_feature(file_path, i * duration, duration, resolution) for i in range(math.floor(length / duration)) ], \
            [ np.eye(len(genres))[index] for _ in range(math.floor(length / duration)) ]
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
    features, labels = [file_to_feature_label_list(file_path, duration) for file_path in file_list]
    features = list(itertools.chain(*features))
    print(np.array(features).shape)
    labels = list(itertools.chain(*labels))
    
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

    time_now = datetime.now()
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
        
def TrainAI(features, labels):
    epochs = eval(input("Epochs times:"))
    #def normalize(array):
    #    process = (array - np.min(array)) / (np.max(array) - np.min(array))
    #    return process
    #features = normalize(features)
    #labels = normalize(labels)
    permutations = np.random.permutation(len(labels))
    features = np.array(features)[permutations]
    labels = np.array(labels)[permutations]

    features_train = features[0:round(len(labels)*0.6)]
    labels_train = labels[0:round(len(labels)*0.6)]

    features_val = features[round(len(labels)*0.6):round(len(labels)*0.8)]
    labels_val = labels[round(len(labels)*0.6):round(len(labels)*0.8)]

    features_test = features[round(len(labels)*0.8):len(labels)]
    labels_test = labels[round(len(labels)*0.8):len(labels)]

    print(features_train.shape)
    print(labels_train.shape)
    genres = ['effect1','effect2','effect3','effect4','effect5','effect6','effect7','effect8','effect9','effect10']

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
    network = krs.models.Sequential()
    network.add(Conv2D(64,(3,3), input_shape=(features_train.shape[1],features_train.shape[2],features_train.shape[3],), name='cnnNet00'))
    network.add(MaxPooling2D(pool_size=(2,2)))
    network.add(Conv2D(256,(3,3), name='cnnNet0'))
    network.add(MaxPooling2D(pool_size=(2,2)))
    network.add(Dropout(0.15))
    network.add(Conv2D(512,(3,3), name='cnnNet1'))
    network.add(MaxPooling2D(pool_size=(1,1)))
    network.add(Conv2D(512,(3,3), activation="relu", name='cnnNet2'))

    network.add(MaxPooling2D(pool_size=(1,1)))
    network.add(Conv2D(1024,(3,3), activation="relu", name='cnnNet3'))
    network.add(MaxPooling2D(pool_size=(2,2)))
    network.add(Conv2D(1024,(3,3), activation="relu", name='cnnNet4'))
    network.add(MaxPooling2D(pool_size=(2,2)))
    network.add(Conv2D(1024,(3,3), activation="relu", name='cnnNet5'))
    network.add(MaxPooling2D(pool_size=(2,2)))
    network.add(GlobalAveragePooling2D())

    #network.add(Flatten())
    network.add(Dense(512, activation="relu", name="layerDense1"))
    network.add(Dropout(0.15))
    network.add(Dense(256, activation="relu", name="layerDense2"))
    network.add(Dense(128, activation="relu", name="layerDense3"))
    network.add(Normalization())
    network.add(Dense(len(genres), activation="softmax", name="Result"))

    network.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',
	                 metrics = ['accuracy'])
    print(network.summary())

    network.fit(features_train, labels_train, validation_data=(features_val,labels_val),epochs = epochs)
    score = network.evaluate(features_test,labels_test, verbose=1)
    #network.fit(x=np.array(features_train.tolist()), y=np.array(labels_train.tolist()), validation_data=(np.array(features_val.tolist()),np.array(labels_val.tolist())),epochs = 100)
    #score = network.evaluate(x=np.array(features_test.tolist()),y=np.array(labels_test.tolist()), verbose=1)
    print('Accuracy:'+str(score[1]))
    print('Loss:'+str(score[0]))

    time_now = datetime.now()
    years = time_now.year
    months = time_now.month
    days = time_now.day
    hours = time_now.hour
    minutes = time_now.minute
    secs = time_now.second
    network.save(f'model/model{years}{months}{days}{hours}{minutes}{secs}.keras')
    network.save_weights(f'model/model{years}{months}{days}{hours}{minutes}{secs}.h5')
    print("file saved as model.keras and model.wht")
    time.sleep(5)
    return network

def TestAI(network):
    file_path = input("Enter music file path:")
    if '"' in file_path:
        file_path = input("Enter music file path:")
    features = []
    labels = []
    print("getting feature for " + file_path)
    music = MP3(file_path)
    duration = 0.5
    length = music.info.length
    for i in range(0, math.floor(length/duration)):
        offset = i * duration
        labelList = []
        print(get_feature(file_path, offset, duration).shape)
        features.append(get_feature(file_path, offset, duration))
        npCache = np.array(features)
        print(npCache.shape)
    for i in range(3,len(features)):
        timecache = time.time()
        predictingData = np.array((features[i-3],features[i-2],features[i-1],features[i]))
        prediction = network.predict(predictingData)
        prediction = np.add(np.add(np.add(prediction[0],prediction[1]),prediction[2]),prediction[3])
        print(prediction.tolist().index(np.max(prediction)))
        print(time.time() - timecache)

def loadArray():
    featurespath = input("Enter feature file path:").split('"')[1]
    labelspath = input("Enter label file path:").split('"')[1]

    features = np.load(featurespath)
    labels = np.load(labelspath)
    print(features.shape)
    print(labels.shape)
    return features, labels


def Main():
    while True:
        commend = input("1---------Generate features from data folder\n2---------Train AI\n3---------Load array\n4---------Test AI\n5---------Debugging data\n6---------Load AI model\nPlease enter the commend:")
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

Main()
