import random
import numpy as np
#import matplotlib.pyplot as plt
#import scipy.misc
import os
#from datetime import datetime as dt
import utils as ut
import wave
import SoundUtils as SU
import pylab
import Spectograms as spectograms
#import tensorflow as tf
import forwardPath as forward

model_weights_path = '/Users/Sergey/Thesis/Projects/neural-art-tf/models/vgg'
out_dir ="neural-art-tf/output"
device="/cpu:0"

path_to_params = "/Volumes/Storage/parameters100/"
#path_to_spectrums = "/Volumes/Storage/spectrums/1673"

#sound
path_to_flac = "/Volumes/Storage/LibriSpeech2/train-clean-100"
path_to_flac_test_folder = "/Volumes/Storage/LibriSpeech 2/train-clean-100/19/198"
path_to_flac_test_folder_track = "/Volumes/Storage/LibriSpeech 2/train-clean-100/19/198/19-198-0001.flac"

#temp
temp_folder = "/Volumes/Storage/Temp"
temp_image_to_process = "/Volumes/Storage/processing.png"

#load params ones
params = np.load(model_weights_path).item()
pathes = ut.reverseFolder(path_to_flac, ".flac")

#take one file and conevert it to wav then cut it into chunks
for path in pathes:
    print path
    wav_file = SU.Flac2Wav(path, temp_folder)
    SU.cutIntoChunks(wav_file, temp_folder)
    chunk_pathes = ut.reverseFolder(temp_folder, '.wav')
    for chunk in chunk_pathes:
        spectograms.graph_spectrogram(chunk, temp_image_to_process)
        folder, name = forward.getPath(path_to_params, os.path.basename(wav_file), os.path.basename(chunk))
        forward.spectrumToArrays(device, params, temp_image_to_process, folder, name)
    #clean temp folder
    ut.cleanFolder(temp_folder)