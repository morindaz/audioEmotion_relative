import stft
import scipy.io.wavfile as wav
import os
import glob
import numpy as np
import pickle
inDir = "C:\\Users\\Mypc\\Desktop\\morindaz\\pingan\\selectedaudio"
outDir = "C:\\Users\\Mypc\\Desktop\\morindaz\\audios\\STFT"
txt = ['Acceptance', 'Admiration', 'Aggressiveness', 'Angry', 'Annoyance', 'Anticipation', 'Apprehension', 'Awe',
       'Boastfulness', 'Boredom', 'Bravery', 'Calm', 'Conflict', 'Contempt', 'Cowardice', 'Deceptiveness',
       'Defiance', 'Depression', 'Desire', 'Disapproval', 'Disgust', 'Distraction', 'Embarrassed', 'Envy',
       'Fatigue', 'Fear', 'Gratitude', 'Grievance', 'Harmony', 'Hate', 'Insincerity', 'Insult', 'Interest', 'Joy',
       'Love', 'Neglect', 'Optimism', 'Passiveness', 'Pensiveness', 'Pessimism', 'Pride', 'Puzzlement', 'Remorse',
       'Sadness', 'Serenity', 'Shame', 'Sincerity', 'Submission', 'Surprise', 'Suspicion', 'Tension', 'Trust',
       'Uneasiness', 'vitality']
count = 0
if __name__ == '__main__':
    for eachDir in txt:
    #存储第几个文件夹
        inMovDir = inDir + "\\" + eachDir
        outMovDir = outDir + "\\"
        # print formatInDir
        #转到50中情绪对应的文件夹
        os.chdir(inMovDir)
        # os.chdir(inMovDir)
        formatLines = glob.glob('*.wav')
        result = []
        label = []
        # print(formatLines)
        for line in formatLines:
            fs, audio = wav.read(line)
            specgram = stft.spectrogram(audio)
            specgram = np.transpose(specgram,(1,0,2))
            print(specgram.shape)
            result.append(specgram)
            label.append(count)
        # result.append(f)
        print(type(eachDir))
        count = count+1
        output_data = open(outMovDir+eachDir+'.pkl', 'wb')
        output_label = open(outMovDir+"label\\"+eachDir+'label.pkl', 'wb')
        pickle.dump(result, output_data, -1)
        pickle.dump(label, output_label, -1)
