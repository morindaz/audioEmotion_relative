import wave
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import NaN, Inf, arange, isscalar, asarray, array
from scipy import signal
import os



# Untested, but apart from typos this should work fine
# No attention paid to speed, just to clarify the algorithm
# Input signal and output signal are Python lists
# Listcomprehensions will be a bit faster
# Numpy will be a lot faster
def getEnvelope (inputSignal, interval):

    # Taking the absolute value

    absoluteSignal = []
    for sample in inputSignal:
        absoluteSignal.append (abs (sample))

    # Peak detection

    intervalLength = interval # Experiment with this number, it depends on your sample frequency and highest "whistle" frequency
    outputSignal = []

    for baseIndex in range (intervalLength, len (absoluteSignal)):
        maximum = 0
        for lookbackIndex in range (intervalLength):
            maximum = max (absoluteSignal [baseIndex - lookbackIndex], maximum)
        outputSignal.append (maximum)

    return outputSignal


def ZeroCR(waveData,frameSize,overLap):
    wlen = len(waveData)
    step = frameSize - overLap
    frameNum = math.ceil(wlen/step)
    zcr = np.zeros((frameNum,1))
    for i in range(frameNum):
        curFrame = waveData[np.arange(i*step,min(i*step+frameSize,wlen))]
        #To avoid DC bias, usually we need to perform mean subtraction on each frame
        #ref: http://neural.cs.nthu.edu.tw/jang/books/audiosignalprocessing/basicFeatureZeroCrossingRate.asp
        curFrame = curFrame - np.mean(curFrame) # zero-justified
        zcr[i] = sum(curFrame[0:-1]*curFrame[1::]<0)
    return zcr


def peakdet(v, delta, x=None):

    maxtab = []
    mintab = []
    if x is None:
        x = arange(len(v))
    v = asarray(v)
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    lookformax = True

    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
    start = 0
    for i in range(0,int(array(maxtab)[0,0]+1)):
        if v[i]>0.1:
            start = i
            break

    max_len = len(array(maxtab)[:,0])
    kk =0
    for i in range(int(array(maxtab)[-1,0]),len(v),10):
        if v[i]<0.05:
            mintab.append((i,v[i]))
            kk=1
            break

    if kk==0:
        mintab.append((len(v)-10,v[len(v)-10]))

    length = array(mintab)[-1,0]-start

    maxtab1 =[]
    tuoyin = 0
    for i in range(max_len):
        if i==0:
            a1 = v[start]
        else:
            a1 = mintab[i-1][0]

        a2 = maxtab[i][0]
        a3 = mintab[i][0]
        if (a3-a1)>10000:
            tuoyin +=1
        energy1 = np.sum(v[int(a1):int(a2)])
        energy2 = np.sum(v[int(a2):int(a3)])
        if (energy1+energy2)>=800:
            maxtab1.append(maxtab[i])


    return array(maxtab), array(maxtab1), array(mintab),tuoyin,length
# filepath = "./data/"  # 添加路径
# filename = os.listdir(filepath)  # 得到文件夹下的所有文件名称
# f = wave.open(filepath + filename[0], 'rb')
def readwav(path):
    f = wave.open(path)
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    strData = f.readframes(nframes)  # 读取音频，字符串格式
    waveData = np.fromstring(strData, dtype=np.int16)  # 将字符串转化为int
    waveData = waveData * 1.0 / (max(abs(waveData)))  # wave幅值归一化
    waveData = np.reshape(waveData, [nframes, nchannels])
    f.close()

    return waveData[:,0],framerate


waveData,framerate = readwav('C:\\Users\\YY\\PycharmProjects\\test\\voiceRecognition\\samples\\low_speechrate.wav')
time = np.arange(0, len(waveData)) * (1.0 / framerate)
enve1 = getEnvelope(waveData,600)
max1_,max1,min1,tuoyin1,length1 = peakdet(enve1,0.15)
plt.subplot(221)
plt.plot(time,waveData)
plt.subplot(223)
plt.plot(enve1)
plt.scatter(array(max1)[:, 0], array(max1)[:, 1], color='blue')
#plt.scatter(array(max1_)[:, 0], array(max1_)[:, 1], color='yellow')
plt.scatter(array(min1)[:, 0], array(min1)[:, 1], color='red')
timelen = length1*(len(waveData)/len(enve1) / framerate)
speechrate = len(max1)/timelen
#d = waveData[:,0]
# idx1 = np.array(np.where(abs(d) < 0.25))
# d[idx1]=0
# enve2 = getEnvelope(d,500)
# max2,min2,tuoyin2 = peakdet(enve2,0.15)
# plt.subplot(222)
# plt.plot(time,d)
# plt.subplot(224)
# plt.plot(enve2)
# plt.scatter(array(max2)[:, 0], array(max2)[:, 1], color='blue')
# plt.scatter(array(min2)[:, 0], array(min2)[:, 1], color='red')










frameSize = 256
overLap = 0
zcr = ZeroCR(waveData[:,0],frameSize,overLap)
d = waveData[:,0]
idx = np.array(np.where(zcr >= 10))
idx1 = np.array(np.where(abs(d) < 0.25))
d[idx1]=0
d1 = filter(lambda x: x!=0,d )
d1 = [x for x in d1]
sum_d1 = 0
for i in range(len(d1)):
    sum_d1 += abs(d1[i])
mean_d1 = sum_d1/len(d1)
print(sum_d1)
print(mean_d1)
plt.subplot(312)
plt.plot(time,d)
zcr1 = ZeroCR(d,frameSize,overLap)
d2 = filter(lambda x: x!=0,zcr1)
d2 = [x for x in d2]
sum_d2 = 0
for i in range(len(d2)):
    sum_d2 += abs(d2[i])
mean_d2 = sum_d2/len(d2)
print(sum_d2)
print(mean_d2)
time2 = np.arange(0, len(zcr1)) * (len(d)/len(zcr1) / framerate)
plt.subplot(313)
plt.plot(time2,zcr1)
# for i in range(len(idx[0,:])):
#     del d1[idx[0,i]*frameSize:(idx[0,i]+1)*frameSize]
# bb = sum(map(abs,d1))/len(d1)
# print(bb)
# a = getEnvelope(d1,200)
# bb2 = np.mean(a)
# print(bb2)


# plot the wave
time = np.arange(0, len(waveData[:,0])) * (1.0 / framerate)
time2 = np.arange(0, len(zcr)) * (len(waveData[:,0])/len(zcr) / framerate)
time3 = np.arange(0, len(zcr1)) * (len(d)/len(zcr1) / framerate)
plt.plot(time,d-waveData[:,0])
plt.subplot(411)
plt.plot(time, waveData[:,0])
plt.subplot(412)
plt.plot(time2,zcr)
plt.subplot(413)
plt.plot(time,d)
plt.subplot(414)
plt.plot(time3,zcr1)
plt.show()
# plot the wave
time = np.arange(0, nframes) * (1.0 / framerate)
plt.figure()
plt.subplot(3,1,1)
plt.plot(time, waveData[:,0])
#plt.subplot(3,1,2)
#plt.plot(np.sqrt(waveData[:,0]**2+hwave**2))
plt.subplot(3,1,3)
a = getEnvelope(waveData[:,0],200)
b = getEnvelope(a,400)
plt.title('envelope')
plt.plot(b)
# plt.xlabel("Time(s)")
# plt.ylabel("Amplitude")
# plt.title("Ch-1 wavedata")
# plt.grid('on')  # 标尺，on：有，off:无。
# plt.show()

from scipy.ndimage.filters import gaussian_filter

blurred = gaussian_filter(waveData[:,0], sigma=7)
plt.subplot(3,1,2)
plt.plot(blurred)
plt.title('after Gaussian filter')
plt.show()



import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array

# peak detection


if __name__ == "__main__":
    from matplotlib.pyplot import plot, scatter, show

    #series = [0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 2, 0, 0, 0, -2, 0]
    series = a  # a是envelope
    maxtab, mintab = peakdet(series, .3)
    plot(series)
    scatter(array(maxtab)[:, 0], array(maxtab)[:, 1], color='blue')
    scatter(array(mintab)[:, 0], array(mintab)[:, 1], color='red')
    show()

#画频谱
from scipy.fftpack import fft

x = np.arange(0,1,len(waveData[:,0]))
yy = abs(fft(waveData[:,0]))

xf = np.arange(len(waveData[:,0]))
plt.plot(xf,yy)

x = [[1,1],[2,3],[3,3],[4,6],[5,3],[6,1]]
y = [1,2,3,4,5]
print(array(x)[-1,0])