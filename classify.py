import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import re
import struct
import csv
import os
from scipy.interpolate import PchipInterpolator

def visualization(filepath):
    datas = pd.read_csv(filepath, header=None, names=['ffhr', 'fhr', 'uc', 'acc', 'dec'])
    fhr = datas['fhr']
    ffhr = datas['ffhr']

    acc = datas['acc']
    dec = datas['dec']

    plt.subplot(411)
    # plt.scatter(x, fhr1, s=0.1)
    plt.plot(fhr)

    plt.subplot(412)
    # plt.scatter(x, fhr1, s=0.1)
    plt.plot(ffhr)

    plt.subplot(413)
    # plt.scatter(x, fhr1, s=0.1)
    plt.plot(acc)

    plt.subplot(414)
    # plt.scatter(x, fhr1, s=0.1)
    plt.plot(dec)
    plt.show()

def read_original():
# 原始数据
    data = scipy.io.loadmat('./READ_CTU-CHB/Data.mat')

    print(data.keys())
    print(len(data))
    # print(data['Data'])
    print(len(data['Data']))
    print(data['Data'][0])

def read_annotation():
    # 还是用matlab解决把（见read_ann.m)
    anno = scipy.io.loadmat('./anno_CTU-CHB/annotation_1001.mat')
    print(anno['dataloss'])
    # print(anno['ann'])
    print(anno['ann'].shape)

    a,b = anno['ann'].shape
    # print(a,b)
    for j in range(0,b):
        if anno['ann'][4][j]:
            print(anno['ann'][3][j],anno['ann'][4][j])


Pname = './READ_CTU-CHB/dataset/'
normal = []
illness = []

fid = open('./READ_CTU-CHB/RECORDS.txt', 'r')
DataName = [int(line) for line in fid.readlines()]
fid.close()

def classify():
    N_files = len(DataName)
    # print(N_files) =552
    normal = []
    illness = []

    for i in range(N_files):
        Name = str(DataName[i])
        fid = open(Pname + Name + '.hea', 'r')

        # 读pH
        # pH 值大于等于 7.15 为正常（阴性），小于 7.15 为病理（阳性）
        ####################
        for line in fid:
            if line.startswith('#pH'):
                matches = re.findall(r'\d+\.\d+', line)
                if matches:
                    pH = float(matches[0])
                else:
                    matches = re.findall(r'\d', line)
                    pH = float(matches[0])
                # print(pH)
                if pH >= 7.15:
                    normal.append(Name)
                else:
                    illness.append(Name)
                break
        ####################
        fid.close()

    print(illness)
    print(normal)
    print(len(illness)+len(normal))

def toCSV():
    data = scipy.io.loadmat('./READ_CTU-CHB/Data.mat')['Data']
    for i, entry in enumerate(data):
        FHR = entry['FHR'][0].tolist()[0]
        UC = entry['UC'][0].tolist()[0]

        csv_filename = f"{DataName[i]}.csv"
        with open('./O552/'+csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['FHR', 'UC'])
            for j in range(len(FHR)):
                writer.writerow([FHR[j], UC[j]])

        print(f"CSV file created: {csv_filename}")

original_path = './O552/'
anno_path = './anno_csv/'
merged_path = './M552/'

def merge():
    if not os.path.exists(merged_path):
        os.makedirs(merged_path)

    csv_files = [f for f in os.listdir(original_path) if f.endswith('.csv')]

    for csv_file in csv_files:
        csv_path = os.path.join(original_path, csv_file)
        data = pd.read_csv(csv_path)

        def read_ann_file(filename):
            data = []
            with open(filename, 'r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    data.append(row)
            return data

        data['acc'] = 0
        data['dec'] = 0

        ann = read_ann_file(os.path.join(anno_path, csv_file))
        length = len(ann)
        # print(len(ann),len(data))
        # print(length - length % 2)
        # input()
        for i in range(0,int(length - length % 2),2):
            if ann[i][0] == '3':
                for j in range(int(ann[i][1])-2, int(ann[i+1][1])):
                    data.at[j,'acc'] = 1
            elif ann[i][0] == '4':
                for j in range(int(ann[i][1])-2, int(ann[i+1][1])):
                    data.at[j,'dec'] = 1
            else :
                print('error')
            # print(int(ann[i][1])-1, int(ann[i+1][1]))

        if length % 2 == 1:
            if ann[length-1][0] == '3':
                for j in range(int(ann[length-1][1])-2, len(data)):
                    data.at[j, 'acc'] = 1
            elif ann[length-1][0] == '4':
                for j in range(int(ann[length-1][1])-2, len(data)):
                    data.at[j, 'dec'] = 1
            else :
                print('error')

        print(csv_file)
        # 创建完整的保存路径
        new_csv_path = os.path.join(merged_path, csv_file)
        data.to_csv(new_csv_path, index=False)

def clean(data):
    # 1. 胎心率值小于50bpm或大于200bpm时的点值为零
    data.loc[(data['FHR'] < 50) | (data['FHR'] > 200), 'FHR'] = 0

    # 2. 删除胎心率中超过15s为0值的数据
    mask = data['FHR'] == 0
    zeros_count = mask.groupby((~mask).cumsum()).cumsum()
    delete_mask = (zeros_count > 60) & mask
    delete_indices = data[delete_mask].index
    data = data.drop(delete_indices)
    data_ = data.reset_index(drop=True)

    data_['FHR'] = data_['FHR'].replace(0, np.nan)
    # 3. 跟前后1s相比变化超过25的信号,置空
    previous_values = data['FHR'].shift(1)
    next_values = data['FHR'].shift(-1)
    mask = (abs(data_['FHR'] - previous_values) > 25) | (abs(data_['FHR'] - next_values) > 25)
    data_.loc[mask, 'FHR'] = np.nan

    # 处理末端空值，重新确定信号长度
    detect = 10
    nan_rows = data_['FHR'].tail(detect).isna().sum()
    while nan_rows == detect:
        detect += 10
        nan_rows = data_['FHR'].tail(detect).isna().sum()
    data = data_.loc[:len(data_)-nan_rows]

    # 4. Hermit插值/线性插值
    data['FHR'] = data['FHR'].interpolate(method='linear')
    # x = data.index
    # y = data['FHR']
    # interp = PchipInterpolator(x, y)
    # data['FHR'] = interp(x)

    return data

cleaned_path = './C552/'
def cleanall():
    csv_files = [f for f in os.listdir(merged_path) if f.endswith('.csv')]

    for csv_file in csv_files:
        csv_path = os.path.join(merged_path, csv_file)
        data = pd.read_csv(csv_path)
        cleaned = clean(data)

        print(csv_file)
        new_csv_path = os.path.join(cleaned_path, csv_file)
        cleaned.to_csv(new_csv_path, index=False)
def movingaverage():
    pass

def slideSample():
    pass

def gerBaseline():
    pass


# toCSV()
# classify()
# np.save('illness.npy', illness)
# np.save('normal.npy', normal)

# read_annotation()
# visualization('./333/1011.csv')
# merge()

num = 1008
for num in range(2030,2046):
    data = pd.read_csv('./M552/'+str(num)+'.csv')
    plt.subplot(211)
    plt.plot(data['FHR'])
    data = pd.read_csv('./C552/'+str(num)+'.csv')

    plt.subplot(212)
    plt.plot(data['FHR'])
    plt.show()
