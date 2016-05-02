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


path_to_params = "/home/ubuntu/parametersTIMIT/"


#PHONEMS
path_to_sound = "/home/ubuntu/TIMIT/timit/train"


#temp
temp_folder = "/home/ubuntu/Temp"
temp_phonemes_folder = "/home/ubuntu/TempPhonemes"
temp_20ms_phonemes_folder = "/home/ubuntu/Temp20msPhonemes"
temp_image_to_process = "/home/ubuntu/ThesisProj/processing.jpg"


#load params ones
params = np.load(model_weights_path).item()

#PHONEMS
pathes = ut.reverseFolder(path_to_sound, ".wav")


for path in pathes:
    #sentence
    print path
    phonemes = ut.parcePhonemeFile(path)
    speaker = path.split("/")[-2]

    temp_speaker_folder = temp_folder + "/" + speaker
    if not os.path.exists(temp_speaker_folder):
        os.makedirs(temp_speaker_folder)
    #converting to wave
    wav_file = SU.PCM2Wav(path, temp_speaker_folder)
    
    for phoneme in phonemes:
        #Cutting one phoneme
        phone_file = SU.cutPhonemeChunk(wav_file, temp_phonemes_folder, int(phoneme[0]), int(phoneme[1]), phoneme[2])

        ut.cleanFolder(temp_20ms_phonemes_folder)
        #cutting a phoneme into chunks of 20 ms
        SU.cutIntoChunks(phone_file, temp_20ms_phonemes_folder)
        chunk_pathes = ut.reverseFolder(temp_20ms_phonemes_folder, '.wav')
        for chunk in chunk_pathes:
            print chunk
            spectograms.graph_spectrogram(chunk, temp_image_to_process)
            folder, name = forward.getPathTIMITSPeakerAndPhoneme(path_to_params, speaker, phoneme[2] ,os.path.basename(phone_file), os.path.basename(chunk))
            forward.spectrumToArrays(device, params, temp_image_to_process, folder, name)

        ut.cleanFolder(temp_folder)
        ut.cleanFolder(temp_phonemes_folder)
