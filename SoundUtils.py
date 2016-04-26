import wave
import pylab
import audiotools
import os

def Flac2Wav(flac_file, wav_path):
    wav_path = wav_path + "/"
    if not os.path.exists(wav_path):
        os.makedirs(wav_path)
        
    file_name = os.path.basename(flac_file)
    wav_name = file_name.replace(".flac", ".wav")
    wav_file= wav_path + wav_name
    audiotools.open(flac_file).convert(wav_file, audiotools.WaveAudio)
    return wav_file

def cutIntoChunks(file_path, output_folder):
    output_folder = output_folder + "/"
    chunkDuration = 2./100; # 20 ms
    chankStep =  1./100; # 10 ms
    wav = wave.open(file_path, 'r')

    frameRate = wav.getframerate()
    nChannels = wav.getnchannels()
    sampWidth = wav.getsampwidth()
    
    numSamplesPerChunk = int(chunkDuration*frameRate);
    numSamplesPerStep = int(chankStep*frameRate);
    
    totalNumSamples = wav.getnframes();

    chunkCnt = 0;
    startLoc = 0;
    
    while startLoc + numSamplesPerChunk <= totalNumSamples:
        endLoc = min(startLoc + numSamplesPerChunk - 1,totalNumSamples)
        wav.setpos(startLoc)
        chunk_frames = wav.readframes(endLoc-startLoc)
    
        test_chunk = output_folder + str(chunkCnt) + ".wav"
        chunkAudio = wave.open(test_chunk,'w')
        chunkAudio.setnchannels(nChannels)
        chunkAudio.setsampwidth(sampWidth)
        chunkAudio.setframerate(frameRate)
        chunkAudio.writeframes(chunk_frames)
        chunkAudio.close()
    
        startLoc += numSamplesPerStep
        chunkCnt += 1
    
    wav.close()
    #os.remove(file_path)
