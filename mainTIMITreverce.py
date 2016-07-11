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
import scipy.misc

#model_weights_path = '/Users/Sergey/Thesis/Projects/neural-art-tf/models/vgg'
model_weights_path = '/home/ubuntu/vgg'

#out_dir ="neural-art-tf/output"
device="/gpu:0"


path_to_params = "/home/ubuntu/parametersTIMIT10POOL2/"


#PHONEMS
path_to_sound = "/home/ubuntu/TIMIT10POOL1/timit/train"


#temp
temp_folder = "/home/ubuntu/Temp"
temp_phonemes_folder = "/home/ubuntu/TempPhonemes"
temp_image_to_process = "/home/ubuntu/ThesisProj/processing.jpg"


#load params ones
params = np.load(model_weights_path).item()

#PHONEMS
pathes = ut.reverseFolder(path_to_sound, ".WAV")

#because window sizr is 11, 1 in the middle and 5 on the left and 5 on the right
phonemeOffset = spectograms.frameStep * 5

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
    
    for i in range(len(phonemes)):
        phoneme = phonemes[i]
        #Cutting one phoneme
        if i == 0 or i == len(phonemes):
            left = int(phoneme[0])
            right = int(phoneme[1])
        else:
            left = int(phoneme[0]) - phonemeOffset
            right = int(phoneme[1]) + phonemeOffset
            
        phone_file = SU.cutPhonemeChunk(wav_file, temp_phonemes_folder, left, right, phoneme[2])
        imageRGB, minEl, maxEl = spectograms.SpectogramToImage(phone_file, temp_image_to_process)
        
        chunkLength = 11
        totalFeatures = imageRGB.shape[1]
        #The stepLength is 1 therefore the number of chunks is calculated as follows
        numChunks = totalFeatures-chunkLength + 1
        
        for i in range(numChunks):
            chunk = imageRGB[:,i:i+chunkLength,:]
            chunk = scipy.misc.imresize(chunk, (224, 224))
            spectograms.SaveRGB(chunk, temp_image_to_process)
            folder, name = forward.getPathTIMITSPeakerAndPhoneme(path_to_params, speaker, phoneme[2] ,os.path.basename(phone_file), i)
            forward.spectrumToArraysTIMIT(device, params, chunk, folder, name)
        
        ut.cleanFolder(temp_folder)
        ut.cleanFolder(temp_phonemes_folder)
