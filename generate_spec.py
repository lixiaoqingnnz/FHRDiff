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
    

def prsa_spectrum(fhr_series, L, T, sample_rate):
    fhr_series_ms = bpm_to_ms(fhr_series)
    time_axis, prsa_curve = generate_prsa_curve(fhr_series_ms, L, T, smaple_rate)


    prsa_data = prsa_curve
    fs = 4 #sample rate
    t = np.linspace(0, len(prsa_data) / fs, num=len(prsa_data), endpoint=False)

    # Morse wavelet
    gamma = 3
    # based on the time-bandwidth product equal to 60 in the paper
    # beta = gamma * time-bandwidth product
    beta = 180
    wavelet = Wavelet(('gmw', {'gamma': gamma, 'beta': beta}))

    # cwt
    Wx, scales = cwt(prsa_data, wavelet)
    print("Wx:", Wx)
    print("scales:", scales)

    # wavelet power
    spectrogram = np.abs(Wx)**2


    sampling_period = 1 / fs
    #frequencies = (beta ** (1 / gamma)) / (np.sqrt(2 * np.pi) * scales * sampling_period)
    frequencies = ((2**(1/beta)) * ((gamma/beta)**(1/gamma))) / (scales * fs)

    # when k = 0, central freq spectrogram
    central_freq_index = len(prsa_data) // 2  # 中心频率索引
    PRSA_Spt = spectrogram[:, central_freq_index]

    
    return frequencies, spectrogram

fhr_series = np.load('../fragments2/1123_1.npy')

fhr_series_downsampled = decimate(fhr_series, q=2, n=None, ftype='iir', axis=-1, zero_phase=True)

frequencies, spectrogram = prsa_spectrum(fhr_series_downsampled,
L=100, T=1, sample_rate=2)

