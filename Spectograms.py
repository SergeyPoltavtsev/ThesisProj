import numpy as np
import wave
import pylab
import os
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
   
def get_wav_info(file_path):
    wav = wave.open(file_path, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

def graph_spectrogram(file_path, out_img_file):
    sound_info, frame_rate = get_wav_info(file_path)
    fig = plt.figure(num=None, figsize=(19, 12), edgecolor = None, frameon=False)
    #plt.subplot(111)
    Pxx, freqs, bins, im = plt.specgram(sound_info, Fs=frame_rate)#, cmap=plt.cm.gist_heat)
    im.axes.axis('off')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    #im.frameon = False
    plt.axis((0,max(bins),0,max(freqs)))
    
    plt.savefig(out_img_file, bbox_inches='tight', transparent=True, pad_inches=0, edgecolor='w')
    
    #fig.savefig(out_img_file, bbox_inches='tight', transparent=True, pad_inches=0, edgecolor='w')
    plt.clf()
    plt.close()
