import numpy as np
import wfdb
import matplotlib.pyplot as plt
from scipy.signal import decimate
import numpy as np
import matplotlib.pyplot as plt
from ssqueezepy import Wavelet, cwt

def find_deceleration_anchor_points(fhr_series, T):
    dec_points = []
    for t in range(T, len(fhr_series) - T):
        if np.mean(fhr_series[t:t+T]) > np.mean(fhr_series[t-T:t]):
            dec_points.append(t)
            
    print('num: dec points:', len(dec_points))
    return dec_points

def generate_prsa_curve(fhr_series, L, T, sampling_rate):
    overall_mean = np.mean(fhr_series)
    
    dec_points = find_deceleration_anchor_points(fhr_series, T)
    prsa_curves = []
    
    for dec in dec_points:
        if dec - L >= 0 and dec + L < len(fhr_series):
            window = [fhr_series[x] for x in range(dec-L, dec+L)]
            window_relative = window - overall_mean  # 计算相对变化
            prsa_curves.append(window_relative)
    
    if prsa_curves:
        prsa_curve = np.mean(prsa_curves, axis=0)
        time_axis = np.linspace(-L/sampling_rate, L/sampling_rate, num=2*L)
        return time_axis, prsa_curve
    else:
        return [], []


def prsa_spectrum(data):
    
    # generate prsa_curve
    L = 100
    T = 1
    sample_rate = 2
    time_axis, prsa_curve = generate_prsa_curve(data, L, T, sample_rate)
    print(prsa_curve.shape)
    # compute scales 
    N = len(prsa_curve)          
    delta_t = 0.5      
    s0 = 1.0           
    J = 179         
    delta_j = 1 / J * np.log2(N * delta_t / s0)
    scales = s0 * 2 ** (np.arange(0, J + 1) * delta_j)
    time = np.linspace(0, 1, N)
    
    wavename = 'cmor'
    central_frequency = 6
    bandwidth = 1.5  

    wavelet = pywt.ContinuousWavelet(wavename + str(bandwidth) + '-' + str(central_frequency))
    coefficients, frequencies = pywt.cwt(prsa_curve, scales, wavelet, sampling_period=1/sample_rate)

    spectrogram = np.abs(coefficients)
  
    return spectrogram
    

fhr_series = np.load('../fragments2/1123_1.npy')

fhr_series_downsampled = decimate(fhr_series, q=2, n=None, ftype='iir', axis=-1, zero_phase=True)

frequencies, spectrogram = prsa_spectrum(fhr_series_downsampled,
L=100, T=1, sample_rate=2)

