import random
import numpy as np
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

#model_weights_path = '/Users/Sergey/Thesis/Projects/neural-art-tf/models/vgg'
model_weights_path = '/home/ubuntu/vgg'

#out_dir ="neural-art-tf/output"
device="/gpu:0"

#STYLE
#path_to_params = "/home/ubuntu/parameters100/"

#PHONEMS
#path_to_params = "/home/ubuntu/phonemsParams/"
#cutted by chunks of 100ms
path_to_params = "/home/ubuntu/cleanedPhonemsParams20ms/"


#path_to_spectrums = "/Volumes/Storage/spectrums/1673"

#sound STYLE
#path_to_flac = "/home/ubuntu/LibriSpeech/train-clean-100"

#PHONEMS
#path_to_wav = "/home/ubuntu/Phonemes"
path_to_wav = "/home/ubuntu/PhonemesCleaned"

#path_to_flac_test_folder = "/Volumes/Storage/LibriSpeech 2/train-clean-100/19/198"
#path_to_flac_test_folder_track = "/Volumes/Storage/LibriSpeech 2/train-clean-100/19/198/19-198-0001.flac"

#temp
temp_folder = "/home/ubuntu/Temp"
temp_image_to_process = "/home/ubuntu/ThesisProj/processing.jpg"

#load params ones
params = np.load(model_weights_path).item()
#style
#pathes = ut.reverseFolder(path_to_flac, ".flac")

#PHONEMS
pathes = ut.reverseFolder(path_to_wav, ".wav")

#STYLE
#take one file and conevert it to wav then cut it into chunks
#for path in pathes:
#    print path
#    wav_file = SU.Flac2Wav(path, temp_folder)
#    SU.cutIntoChunks(wav_file, temp_folder)
#    chunk_pathes = ut.reverseFolder(temp_folder, '.wav')
#    for chunk in chunk_pathes:
#        print chunk
#	spectograms.graph_spectrogram(chunk, temp_image_to_process)
#        folder, name = forward.getPath(path_to_params, os.path.basename(wav_file), os.path.basename(chunk))
#        forward.spectrumToArrays(device, params, temp_image_to_process, folder, name)
#    #clean temp folder
#    ut.cleanFolder(temp_folder)

#PHONEMS 100 ms
#take one file and conevert it to wav then cut it into chunks
for path in pathes:
    print path
    #wav_file = SU.Flac2Wav(path, temp_folder)
    SU.cutIntoChunks(path, temp_folder)
    chunk_pathes = ut.reverseFolder(temp_folder, '.wav')
    for chunk in chunk_pathes:
        print chunk
	spectograms.graph_spectrogram(chunk, temp_image_to_process)
        folder, name = forward.getPathWithPhoneme(path_to_params, path, os.path.basename(chunk))
        #print "folder %s" %  folder
        #print "name %s" % name
        forward.spectrumToArrays(device, params, temp_image_to_process, folder, name)
    #clean temp folder
    ut.cleanFolder(temp_folder)

#PHONEMS
#for path in pathes:
#    print path
#    spectograms.graph_spectrogram(path, temp_image_to_process)
#    folder, name = forward.getPhonemsPath(path_to_params, path)
    #print folder
    #print name
    #print
#    forward.spectrumToArrays(device, params, temp_image_to_process, folder, name)
