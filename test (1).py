# ECG PPG Feature Extraction
# Author : Venkatesh B
# version : V5.4.31K
# Comment : Corrected Notch and FFT Feature and added IQR method for outlier detection 
# Date   :  19-03-2023

'Importing Required Libraries'
##########################################################################################################
from numpy.lib.function_base import append
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import heartpy as hp
from heartpy import filter_signal
from heartpy import smooth_signal
import scipy.signal
# from scipy import signals
import math
import neurokit2 as nk
import warnings
from BaselineRemoval import BaselineRemoval
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = [10, 5]
import glob
from pathlib import Path
import math
import statistics as stat
from itertools import groupby
from operator import itemgetter
from pymongo import MongoClient
import json
##########################################################################################################

'Function to Normalize PPG from 0  to 1'
##########################################################################################################
def normalization(data):
    data_max_value = max(data)
    data_min_value = min(data)
    data_sub = []
    for j in range(len(data)):
        temp = data[j] - data_min_value
        data_sub.append(temp)
    normalized_data = []
    for i in range(len(data_sub)):
        normalized_data.append((data_sub[i])/(data_max_value-data_min_value))
    return normalized_data
##########################################################################################################

'Function to find peak'
##########################################################################################################
def findpeaks_nabian2018(signal,sampling_rate):
    window_size = int(0.4 * sampling_rate)
    peaks = np.zeros(len(signal))
    for i in range(1 + window_size, len(signal) - window_size):
        ecg_window = signal[i - window_size : i + window_size]
        rpeak = np.argmax(ecg_window)
        if i == (i - window_size - 1 + rpeak):
            peaks[i] = 1
    rpeaks = np.where(peaks == 1)[0]
    return rpeaks
##########################################################################################################

'Function for getting Linear Interpolation'
#########################################################################################################
def interpolate(xval, df, xcol, ycol):
    return np.interp([xval], df[xcol], df[ycol])
#########################################################################################################

'Function for filtering the ECG Signal for 1KHz'
######################################################################################################### 
def ECG_Filter_1K(ECG):
    fs = 1000
    ECG_1 = [524288 - i for i in ECG]
    # high_pass = hp.filter_signal(np.ravel(ECG_1), cutoff = 0.5, filtertype = 'highpass', sample_rate = fs, order = 3, return_top = False)
    # Low_pass = hp.filter_signal(np.ravel(high_pass), cutoff = 99, filtertype = 'lowpass', sample_rate = fs, order = 3, return_top = False)
    mean = sum(ECG_1)/len(ECG_1)
    DC_offset = [i-mean for i in ECG_1]
    b = [0.000190593790000433,0.0096745577472388,0.0481162825717218,0.122505021079206,0.201564236322917,0.235898616977831,0.201564236322917,0.122505021079206,0.0481162825717218,0.0096745577472388,0.000190593790000433]
    a = [1]
    p = DC_offset
    low_pass = scipy.signal.filtfilt(b, a, p, axis=- 1)
    low_pass_delay = ((len(b)-1)/(2*fs))
    c = [0.003686865,0.003948262,0.004725365,0.005999182,0.00773847,0.009900502,0.012432121,0.015271048,0.018347415,0.02158549,0.024905543,0.028225809,0.031464508,0.034541858,0.037382046,0.039915097,0.042078597,0.043819237,0.045094125,0.045871845,0.046133232,0.045871845,0.045094125,0.043819237,0.042078597,0.039915097,0.037382046,0.034541858,0.031464508,0.028225809,0.024905543,0.02158549,0.018347415,0.015271048,0.012432121,0.009900502,0.00773847,0.005999182,0.004725365,0.003948262,0.003686865]
    d = [1]
    q = low_pass
    high_pass = scipy.signal.filtfilt(c, d, q, axis=- 1)
    high_pass_delay = ((len(c)-1)/(2*fs))
    e = [0.00213153959319260000, -0.00000000000000000348, 0.00631494409001077000, 0.00000000000000003805, -0.00000000000000000394, 0.00000000000000000235, -0.03301767458473810000, 0.00000000000000002615,-0.03993543702176470000,0.00000000000000005748,0.07710361101127590000,0.00000000000000001245,0.28803171716927600000,0.00000000000000000000,0.39874259948549400000,0.00000000000000000000,0.28803171716927600000,0.00000000000000001245,0.07710361101127590000,0.00000000000000005748,-0.03993543702176470000,0.00000000000000002615,-0.03301767458473810000,0.00000000000000000235,-0.00000000000000000394,0.00000000000000003805,0.00631494409001077000,-0.00000000000000000348,0.00213153959319260000]
    f = [1]
    r = high_pass
    notch = scipy.signal.filtfilt(e, f, r, axis=- 1).tolist()
    
    

    WinSize = round((0.2*1000)+1)
    median_filter = scipy.signal.medfilt(notch,WinSize)
    winsize = round((0.6*1000)+1)
    baseline = scipy.signal.medfilt(median_filter,winsize)
    filtered_ECG_baseline_2 = list((notch - baseline))
    
    
    zeroes= [0,0,0,0,0,0,0,0]
    filtered_ECG_baseline_1 = filtered_ECG_baseline_2 + zeroes
    filtered_ECG_baseline = []
    for i in range(len(filtered_ECG_baseline_1)-8):
        filtered_ECG_baseline.append((filtered_ECG_baseline_1[i]+filtered_ECG_baseline_1[i+1]+filtered_ECG_baseline_1[i+2]+filtered_ECG_baseline_1[i+3]+filtered_ECG_baseline_1[i+4]+filtered_ECG_baseline_1[i+5]+filtered_ECG_baseline_1[i+6]+filtered_ECG_baseline_1[i+7])/8)
    
    
    x = []
    for i in range(len(filtered_ECG_baseline)):
        y = filtered_ECG_baseline[i]*(510)*(10**-6)
        x.append(y)
    avg_x = (sum(x)/len(x))
    ECG_EDF = []
    for k in range(len(x)):
        EDF = (x[k]-avg_x)/30
        ECG_EDF.append(float(EDF))

    return ECG_EDF
##############################################################################################################
'Function for Filtering the the PPG Signal for 1KHz'
##############################################################################################################
def PPG_Filter_1k(PPG,fs):
    PPG_BL_corrected = BaselineRemoval(PPG)
    Zhangfit_output=PPG_BL_corrected.ZhangFit() ### 512,20,20
    high_pass = hp.filter_signal(np.ravel(PPG), cutoff = 0.75, filtertype = 'highpass', sample_rate = fs, order = 3, return_top = False)
    Low_pass = hp.filter_signal(np.ravel(high_pass), cutoff = 6, filtertype = 'lowpass', sample_rate = fs, order = 3, return_top = False)
    # Low_pass = hp.filter_signal(np.ravel(PPG), cutoff = [0.75,5], filtertype = 'bandpass', sample_rate = fs, order = 2, return_top = False)
    norm_ppg = normalization([-1*i for i in Low_pass])
    return norm_ppg
################################################################################################################

'Function for filtering the ECG Signal for 200Hz'
######################################################################################################### 
def ECG_Filter(Raw_ECG):
    DC_offset = [524288 - i for i in Raw_ECG]
    b = [0.000806644437331855, 0.998386711125336, 0.000806644437331855]
    a = [1]
    p = DC_offset
    low_pass = scipy.signal.filtfilt(b, a, p, axis=- 1)
    c = [0.0287709643071516, 0.143105926500316, 0.328123109192532, 0.328123109192532, 0.143105926500316, 0.0287709643071516]
    d = [1]
    q = low_pass
    high_pass = scipy.signal.filtfilt(c, d, q, axis=- 1)
    e = [0.00213153959319260000, -0.00000000000000000348, 0.00631494409001077000, 0.00000000000000003805, -0.00000000000000000394, 0.00000000000000000235, -0.03301767458473810000, 0.00000000000000002615,-0.03993543702176470000,0.00000000000000005748,0.07710361101127590000,0.00000000000000001245,0.28803171716927600000,0.00000000000000000000,0.39874259948549400000,0.00000000000000000000,0.28803171716927600000,0.00000000000000001245,0.07710361101127590000,0.00000000000000005748,-0.03993543702176470000,0.00000000000000002615,-0.03301767458473810000,0.00000000000000000235,-0.00000000000000000394,0.00000000000000003805,0.00631494409001077000,-0.00000000000000000348,0.00213153959319260000]
    f = [1]
    r = high_pass
    notch = scipy.signal.filtfilt(e, f, r, axis=- 1)
    g = [0.000454327,0.000459397,0.000454453,0.000438163,0.000409183,3.66E-04,0.000307766,0.000232675,0.00013961,2.73E-05,-0.000105324,-0.000259466,-0.000436106,-0.000636159,-0.000860428,-0.001109596,-0.00138421,-0.001684672,-0.002011233,-0.002363979,-0.002742829,-0.003147525,-0.003577628,-0.004032516,-0.00451138,-0.005013224,-0.005536866,-0.006080942,-0.006643905,-0.007224036,-0.007819448,-0.008428096,-0.009047785,-0.009676184,-0.010310838,-0.01094918,-0.011588548,-0.012226202,-0.012859339,-0.013485111,-0.014100646,-0.014703065,-0.0152895,-0.015857119,-0.016403137,-0.016924844,-0.017419619,-0.01788495,-0.018318451	,-0.018717882,-0.019081162,-0.019406385,-0.019691836,-0.019936002,-0.020137581,-0.020295498,-0.020408906,-0.020477196,0.9795,-0.020477196,-0.020408906,-0.020295498,-0.020137581,-0.019936002,-0.019691836,-0.019406385,-0.019081162,-0.018717882	,-0.018318451,-0.01788495,-0.017419619,-0.016924844,-0.016403137,-0.015857119,-0.0152895,-0.014703065,-0.014100646,-0.013485111,-0.012859339,-0.012226202,-0.011588548,-0.01094918,-0.010310838,-0.009676184,-0.009047785,-0.008428096,-0.007819448,-0.007224036,-0.006643905,-0.006080942,-0.005536866,-0.005013224,-0.00451138,-0.004032516,-0.003577628,-0.003147525,-0.002742829,-0.002363979,-0.002011233,-0.001684672,-0.00138421,-0.001109596,-0.000860428,-6.36E-04,-0.000436106,-0.000259466,-0.000105324,2.73E-05,0.00013961,0.000232675,0.000307766,0.000366163,0.000409183,0.000438163,0.000454453,0.000459397,0.000454327]
    h = [1]
    s = notch
    filtered_ECG_baseline = scipy.signal.filtfilt(g, h, s, axis=- 1).tolist()
    
    
    x = []
    for i in range(len(filtered_ECG_baseline)):
        y = filtered_ECG_baseline[i]*(510)*(10**-6)
        x.append(y)
    avg_x = (sum(x)/len(x))
    ECG_EDF = []
    for k in range(len(x)):
        EDF = (x[k]-avg_x)/30
        ECG_EDF.append(float(EDF))
    return ECG_EDF
##############################################################################################################
'Function for Filtering the the PPG Signal for 200Hz'
##############################################################################################################
def PPG_Filter(PPG,fs):
    PPG_BL_corrected = BaselineRemoval(PPG)
    Zhangfit_output=PPG_BL_corrected.ZhangFit() ### 512,20,20
    high_pass = hp.filter_signal(np.ravel(PPG), cutoff = 0.5, filtertype = 'highpass', sample_rate = fs, order = 2, return_top = False)
    Low_pass = hp.filter_signal(np.ravel(high_pass), cutoff = 6, filtertype = 'lowpass', sample_rate = fs, order = 2, return_top = False)
    # Low_pass = hp.filter_signal(np.ravel(PPG), cutoff = [0.75,5], filtertype = 'bandpass', sample_rate = fs, order = 2, return_top = False)
    norm_ppg = normalization([-1*i for i in Low_pass])
    return norm_ppg
################################################################################################################

'Checking Signal Quality for Both ECG and PPG'
################################################################################################################
def signal_qulaity(filtred_signal,peaks):
    median_ppg_peaks=np.median(np.diff(peaks))
    half_window_size = math.floor(median_ppg_peaks/2)
    QRS_signal_total_1=[]
    for i in range(1,len(peaks)-1):
        first_sample = peaks[i]-half_window_size
        last_sample =  peaks[i]+half_window_size
        QRS_signal = filtred_signal[first_sample+1:last_sample]
        QRS_signal_total_1.append(QRS_signal)
    QRS_signal_total_array = np.array(QRS_signal_total_1)
    QRS_signal_total= pd.DataFrame(np.transpose(QRS_signal_total_array))
    AVG_QRS_signal_total = QRS_signal_total.mean(axis=1)
    corr = []
    A = np.array(AVG_QRS_signal_total)
    for i in range(len(QRS_signal_total.axes[1])-1):
        B=np.array(QRS_signal_total[i])
        R = np.corrcoef(A,B)
        corr.append(R[0][1])
    corr_mean = np.mean(corr)
    quality = []
    if corr_mean > 0.75:
        quality.append(1)
    else:
        quality.append(0)
    return quality[0],corr
###################################################################################################################

'Flattening the List which contian list in list'
###################################################################################################################
def flatten(l):
    return [item for sublist in l for item in sublist]
####################################################################################################################

'Baseline Correction For PPG Signal Using Envelope Method'
###################################################################################################################
def Base_line_Correction(PPG):

    Filtred_PPG = PPG
    PPG_Peaks = scipy.signal.find_peaks(Filtred_PPG, height=np.mean(Filtred_PPG)+np.std(Filtred_PPG), threshold=None, distance= 110, prominence=None, width=None, wlen=None, rel_height=None, plateau_size=None)[0].tolist()
    PPG_Foot = scipy.signal.find_peaks([-1*i for i in Filtred_PPG], height=np.mean([-1*i for i in Filtred_PPG])+0.75*np.std(Filtred_PPG), threshold=None, distance= 110, prominence=None, width=None, wlen=None, rel_height=None, plateau_size=None)[0].tolist()
   
    length_first_peaks = [len(PPG_Peaks),len(PPG_Foot)]
    
    First_peaks = [PPG_Peaks[0],PPG_Foot[0]]
    
    Upper_envelope =[]
    Lower_envelope = []
    for i in range(min(length_first_peaks)-1):
        ppg_under_test = Filtred_PPG
        onset_foot= PPG_Foot[i]
        offset_foot= PPG_Foot[i+1]
        onset_peak= PPG_Peaks[i]
        offset_peak= PPG_Peaks[i+1]
        PPG_area_under_curve_foot = ppg_under_test[onset_foot:offset_foot]
        PPG_area_under_curve_peak = ppg_under_test[onset_peak:offset_peak]
        PPG_x_foot = np.arange(onset_foot,offset_foot,1)
        PPG_y1_foot = PPG_area_under_curve_foot
        PPG_x_peak = np.arange(onset_peak,offset_peak,1)
        PPG_y1_peak = PPG_area_under_curve_peak
        PPG_footonset = [PPG_x_foot[0], PPG_y1_foot[0]]
        PPG_footoffset = [PPG_x_foot[-1], PPG_y1_foot[-1]]
        
        PPG_peakonset = [PPG_x_peak[0], PPG_y1_peak[0]]
        PPG_peakoffset = [PPG_x_peak[-1], PPG_y1_peak[-1]]
        
        PPG_df_ff_foot = pd.DataFrame(
            {
                "PPG_x": [PPG_footonset[0], PPG_footoffset[0]],
                "PPG_y": [PPG_footonset[1], PPG_footoffset[1]],
                "label": ['PPG_a', 'PPG_b'],
            }
        )
        
        PPG_df_ff_Peak = pd.DataFrame(
            {
                "PPG_x": [PPG_peakonset[0], PPG_peakoffset[0]],
                "PPG_y": [PPG_peakonset[1], PPG_peakoffset[1]],
                "label": ['PPG_a', 'PPG_b'],
            }
        )
        
        
        
        for j in PPG_x_foot :
            temp1 = np.array(interpolate(j, PPG_df_ff_foot, 'PPG_x', 'PPG_y'))
            temp2 = np.array(temp1).tolist()
            Lower_envelope.append(temp2[0])
            
        for k in PPG_x_peak :
            temp3 = np.array(interpolate(k, PPG_df_ff_Peak, 'PPG_x', 'PPG_y'))
            temp4 = np.array(temp3).tolist()
            Upper_envelope.append(temp4[0])
            
    length = [len(Upper_envelope),len(Lower_envelope)]
   
    Diff_Envelope = [Upper_envelope[i] - Lower_envelope[i] for i in range(min(length))]
    
    Reconstructed_PPG = [Filtred_PPG[i]/Diff_Envelope[i] for i in range(len(Diff_Envelope))]
   


    PPG_Peaks_1 = scipy.signal.find_peaks(Reconstructed_PPG, height=None, threshold=None, distance= 110, prominence=None, width=None, wlen=None, rel_height=None, plateau_size=None)[0].tolist()
    PPG_foot_1 = scipy.signal.find_peaks([-1*i for i in Reconstructed_PPG], height=np.mean([-1*i for i in Filtred_PPG])+0.75*np.std(Filtred_PPG), threshold=None, distance= 110, prominence=None, width=None, wlen=None, rel_height=None, plateau_size=None)[0].tolist()


    Baselined_peak_envelope = []
    
    if len(PPG_foot_1) > 5:
        for i in range(len(PPG_foot_1)-1):
            ppg_under_test_1 = Reconstructed_PPG
            onset_peak_1= PPG_foot_1[i]
            offset_peak_1= PPG_foot_1[i+1]
            PPG_area_under_curve_peak_1 = ppg_under_test_1[onset_peak_1:offset_peak_1]
            PPG_x_peak_1 = np.arange(onset_peak_1,offset_peak_1,1)
            PPG_y1_peak_1 = PPG_area_under_curve_peak_1
            
            PPG_peakonset_1 = [PPG_x_peak_1[0], PPG_y1_peak_1[0]]
            PPG_peakoffset_1 = [PPG_x_peak_1[-1], PPG_y1_peak_1[-1]]
            
            
            PPG_df_ff_Peak_1 = pd.DataFrame(
                {
                    "PPG_x": [PPG_peakonset_1[0], PPG_peakoffset_1[0]],
                    "PPG_y": [PPG_peakonset_1[1], PPG_peakoffset_1[1]],
                    "label": ['PPG_a', 'PPG_b'],
                }
            )
            
            
            
            for j in PPG_x_peak_1 :
                temp5 = np.array(interpolate(j, PPG_df_ff_Peak_1, 'PPG_x', 'PPG_y'))
                temp6 = np.array(temp5).tolist()
                Baselined_peak_envelope.append(temp6[0])
                
        
    
        Corrected_PPG_1 = [Reconstructed_PPG[i] - Baselined_peak_envelope[i] for i in range(len(Baselined_peak_envelope))]
        second_peaks = [PPG_Peaks_1[0],PPG_foot_1[0],PPG_Peaks_1[-1],PPG_foot_1[-1]]
        length_second_peaks = [len(PPG_Peaks_1),len(PPG_foot_1)]
    else:
        Corrected_PPG_1 = Reconstructed_PPG
        second_peaks = [PPG_Peaks[0],PPG_Foot[0],PPG_Peaks[-1],PPG_Foot[-1]]
        length_second_peaks = [len(PPG_Peaks),len(PPG_Foot)]
    return Corrected_PPG_1,First_peaks,length_first_peaks,second_peaks,length_second_peaks
########################################################################################################################

'Function to get max length of the list'
#######################################################################################################################
def FindMaxLength(lst):
    maxList = max(lst, key = len)
    return maxList
#########################################################################################################################
'Outlier detection using IQR Filter'
###########################################################################################################################
def outlier_detection(data):    
    data_1 = np.array(sorted((data)))
    if len(data_1) > 2:
        Z3,Z1 = np.percentile(data_1 , [75,25])
        IQR_HR = Z3-Z1
        lower_range = Z1-(1.5 * IQR_HR)
        upper_range = Z3+(1.5 * IQR_HR)
        data_2=[]
        for i in range(len(data)):
            if (data[i]<lower_range) or (data[i]>upper_range):
                data_2.append(data[i])
    return data_2
###########################################################################################################################
aws_client = MongoClient('mongodb+srv://puneetgani:puneetgani@quentdb.04y5a6m.mongodb.net/test')
aws_db = aws_client['Quent_DB']
aws_collection = aws_db['1k_madurai_23rd_march']
timeStamps = aws_collection.distinct('watchTimeStamp')
# query = {"date": "2023-03-23"}
# results = aws_collection.find(query)
ECG_1 = []
PPG_1 = []
excel_list = []
for ts in timeStamps:
# ts = '1679578117'
# query = {"watchTimeStamp": ts}
    query = {"date": {"$gte":"2023-03-23" , "$lte":"2023-03-26" }, "watchTimeStamp": ts}
    # query = {"date": "2023-03-23","watchTimeStamp": ts}
    results = aws_collection.find(query)
    print(ts)

    userinfo = []
    ECG = []
    PPG = []

    for result in results:
        # print(result['dataType'])
        try:
            if result['dataType'] == 'BP_GRN_PPG':
                a = result['data']
                dict_obj = json.loads(a)
                ppg_list = list(dict_obj['data'].values())
                PPG.append(ppg_list)
            if result['dataType'] == 'BP_RAW_ECG':
                a = result['data']
                dict_obj = json.loads(a)
                ecg_list = list(dict_obj['data'].values())
                ECG.append(ecg_list)
                userinfo.append(result['userId'])
                userinfo.append(result['userGender'])
                userinfo.append(result['userAge'])
                userinfo.append(result['userHeight'])
                userinfo.append(result['userWeight'])
                userinfo.append(result['watchTimeStamp'])
                userinfo.append(result['systolic'])
                userinfo.append(result['diastolic'])     
    #         print('ecg_list :',len(ecg_list))
    #         print()
    #         print('ppg_list :',len(ppg_list))
        except:
            print('no data')
    if len(ECG) and len(PPG) > 0:
        # print(len(ECG),len(PPG))
        # print()
        print(userinfo)
        ecg_1 = ECG[0]
        ppg_1 = PPG[0]
        ppg_2 = [i for i in ppg_1 if i > 0]
        ecg_2 = ecg_1[:len(ppg_2)]
        # plt.plot(ecg_2)
        # plt.show()
        # print(len(ppg_2))
        # plt.plot(ppg_2)
        # plt.show()
        i=0
        try:
            Raw_ECG_1 = [x for x in ecg_2 if math.isnan(x) == False]
            Raw_PPG_1 = [x for x in ppg_2 if math.isnan(x) == False]
            #############################################################################################################################
            
            'Spike Removal From the ECG and PPG'
            #############################################################################################################################
            ECG_diff = np.diff(Raw_ECG_1)
            PPG_diff = np.diff(Raw_PPG_1)
            ECG_diff = list(ECG_diff)
            
            ECG1 = []
            for idx, value in enumerate(ECG_diff):
                if abs(value) != 0:
                    ECG1.append(idx)

            ECG2 = []
            for k, g in groupby(enumerate(ECG2), lambda i_x: i_x[0] - i_x[1]):
                ECG2.append(list(map(itemgetter(1), g)))
            if len(ECG2) > 0:
                New_ECG_list1 = FindMaxLength(ECG2)
            else:
                New_ECG_list1 = list(range(0,len(ECG_diff)))
            
            
            New_ECG_list = []
            New_index_ECG = []
            for i in range(len(New_ECG_list1)-1):
                x = ECG_diff[New_ECG_list1[i]]
                if x < 400000:
                    New_index_ECG.append(New_ECG_list1[i])
                    New_ECG_list.append(x)
                    
            PPG_after_spikes_removal = []
            ECG_after_spikes_removal = []
            for i in range(len(New_index_ECG)):
                ECG_after_spikes_removal.append(Raw_ECG_1[New_index_ECG[i]])
                PPG_after_spikes_removal.append(Raw_PPG_1[New_index_ECG[i]])  
            
            Raw_ECG = ECG_after_spikes_removal
            Raw_PPG = PPG_after_spikes_removal
            # plt.plot(Raw_ECG)
            # plt.show()
            #######################################################################################################################

            'Calling the ECG and PPG Filter Function'
            ####################################################################################################
            fs = 1000
            # desired_sampling_rate = fs
            # sampling_rate = 200
            # Raw_ECG_1 = nk.signal_resample(Raw_ECG, desired_length=(desired_sampling_rate*len(Raw_ECG))/(sampling_rate), sampling_rate=sampling_rate, desired_sampling_rate=desired_sampling_rate, method='interpolation')
            # Raw_PPG_1 = nk.signal_resample(Raw_PPG, desired_length=(desired_sampling_rate*len(Raw_PPG))/(sampling_rate), sampling_rate=sampling_rate, desired_sampling_rate=desired_sampling_rate, method='interpolation')
            
            ECG_EDF_1 = ECG_Filter_1K(Raw_ECG)[1000:-1000]
            norm_ppg_1 = PPG_Filter_1k(Raw_PPG,fs)[1000:-1000]
            # plt.plot(norm_ppg_1)
            # plt.show()
            # plt.plot(ECG_EDF_1)
            # plt.show()
            ####################################################################################################
            
            'Calling the PPG Baseline Function '
            ################################################################################################################
            Modified_PPG_1,First_peaks,length_first_peaks,second_peaks,length_second_peaks = Base_line_Correction(norm_ppg_1)
            ECG_EDF = ECG_EDF_1
            # print(First_peaks,length_first_peaks,second_peaks,length_second_peaks)
            zeroes= [0,0,0,0,0]
            Modified_PPG_2 = Modified_PPG_1 + zeroes
            norm_ppg = []
            for i in range(len(Modified_PPG_2)-5):
                norm_ppg.append((Modified_PPG_2[i]+Modified_PPG_2[i+1]+Modified_PPG_2[i+2]+Modified_PPG_2[i+3]+Modified_PPG_2[i+4])/5)
            # print(len(ECG_EDF),len(norm_ppg))
            #################################################################################################################

            'Detection of PQRST and Onset and Offset of P and T wave using Single and Double order Differation'
            #################################################################################################################
            ECG_diff = np.diff(ECG_EDF) # First Order Differation of ECG
            ECG_diff_2 = list(np.diff(ECG_EDF,2)) # Second order Differation of ECG
            zero_crossing_ECG = np.where(np.diff(np.sign(ECG_EDF)))[0] # Caluclating zero_crossing_ECG for ECG First diff
            zero_crossing_ECG_1 = np.where(np.diff(np.sign(ECG_diff)))[0] # Caluclating zero_crossing_ECG for ECG Second diff
            Ampli_zero_crossing_ECG_1 = [ECG_diff_2[i] for i in zero_crossing_ECG_1] # Getting Amplitude of Peaks
            Inveted_double_diff = [-1*i for i in ECG_diff_2] # Invering the Second order Diff of ECG
            Height = np.mean(Inveted_double_diff)+(np.std(Inveted_double_diff)/0.75) #Threshold for Height for ECG Peaks
            Double_diff_ECG_Peaks = findpeaks_nabian2018(Inveted_double_diff,fs) # Detecting the peaks using findpeaks_nabian2018 Function
            Ampli_Double_diff_ECG_Peaks = [Inveted_double_diff[i] for i in Double_diff_ECG_Peaks] # Getting Amplitude of of Peaks
            Corrected_Ampli_Double_diff_ECG_Peaks = [i for i in Ampli_Double_diff_ECG_Peaks if i > Height] # Correcting Ampli using Threshold
            New_Double_diff_ECG_Peaks = [Inveted_double_diff.index(i) for i in Corrected_Ampli_Double_diff_ECG_Peaks] #Getting Corrected Peaks
            # plt.plot(ECG_EDF)
            # # plt.plot([10*i for i in ECG_diff])
            # plt.plot([40*i for i in ECG_diff_2])
            # plt.scatter(New_Double_diff_ECG_Peaks,np.array(ECG_EDF)[New_Double_diff_ECG_Peaks])
            # # plt.scatter(zero_crossing_ECG,np.array([10*i for i in ECG_diff])[zero_crossing_ECG],label='zero_crossing_ECG')
            # # plt.scatter(zero_crossing_ECG_1,np.array([10*i for i in ECG_diff])[zero_crossing_ECG_1],label='zero_crossing_ECG_1')
            # plt.legend()
            # plt.show() 
            ################################################################################################################################
            'Detecting the PQRST'
            #################################################################################################################################
            Q_Peak = []
            ECG_peaks = []
            S_Peak = []
            for i in range(len(New_Double_diff_ECG_Peaks)):
                Double_order_peak = New_Double_diff_ECG_Peaks[i]
                Q_Peak_1 = [i for i in zero_crossing_ECG_1 if i < Double_order_peak-10][-1]
                R_Peak_2 = [i for i in zero_crossing_ECG_1 if i > Double_order_peak-10 and i < Double_order_peak+10]
                if len(R_Peak_2) > 0:
                    R_Peak_1 = R_Peak_2[0]
                else:
                    continue
                S_Peak_1 = [i for i in zero_crossing_ECG_1 if i > R_Peak_1+5][0]
                Q_Peak.append(Q_Peak_1)
                ECG_peaks.append(R_Peak_1)
                S_Peak.append(S_Peak_1)

            P_onset = []
            P_offset = []
            T_onset = []
            T_offset = []
            P_Peak = []
            T_Peak = []
            for i in range(len(Q_Peak)):
                Q_peak_2 = Q_Peak[i]
                S_Peak_2 = S_Peak[i]
                P_offset_1 = [i for i in zero_crossing_ECG if i < Q_peak_2][-1]
                P_onset_1 = [i for i in zero_crossing_ECG if i < P_offset_1-4][-1]
                T_onset_1 = [i for i in zero_crossing_ECG if i > S_Peak_2+10][0]
                T_offset_1 = [i for i in zero_crossing_ECG if i > T_onset_1+10][0]
                P_Peak_1 = [i for i in zero_crossing_ECG_1 if i > P_onset_1 and i < P_offset_1]
                T_Peak_1 = [i for i in zero_crossing_ECG_1 if i > T_onset_1 and i < T_offset_1]
                P_onset.append(P_onset_1)
                P_offset.append(P_offset_1)
                if len(P_Peak_1) == 1:
                    P_Peak.append(P_Peak_1[0])
                else:
                    P_Peak.append(P_Peak_1[1])
                T_onset.append(T_onset_1)
                T_offset.append(T_offset_1)
                if len(T_Peak_1) == 1:
                    T_Peak.append(T_Peak_1[0])
                else:
                    T_Peak.append(T_Peak_1[1])
            # plt.plot(ECG_EDF)
            # plt.plot(ECG_diff)
            # plt.scatter(zero_crossing_ECG,np.array(ECG_diff)[zero_crossing_ECG],label = 'P_Peak')
            # plt.scatter(P_Peak,np.array(ECG_EDF)[P_Peak],label = 'P_Peak')
            # plt.scatter(Q_Peak,np.array(ECG_EDF)[Q_Peak],label = 'Q_Peak')
            # plt.scatter(ECG_peaks,np.array(ECG_EDF)[ECG_peaks],label = 'R_Peak')
            # plt.scatter(S_Peak,np.array(ECG_EDF)[S_Peak],label = 'S_Peak')
            # plt.scatter(T_Peak,np.array(ECG_EDF)[T_Peak],label = 'T_Peak')
            # plt.scatter(P_onset,np.array(ECG_EDF)[P_onset],label = 'P_onset')
            # plt.scatter(P_offset,np.array(ECG_EDF)[P_offset],label = 'P_offset')
            # plt.scatter(T_onset,np.array(ECG_EDF)[T_onset],label = 'T_onset')
            # plt.scatter(T_offset,np.array(ECG_EDF)[T_offset],label = 'T_offset')
            # plt.legend()
            # plt.show()
            
            #################################################################################################################
            
            'Correcting the ECG Peaks according to PPG'
            #################################################################################################################
            New_ECG_Peak = []
            # print(second_peaks)
            for i in range(len(ECG_peaks)):
                if (ECG_peaks[i] > First_peaks[1]) and (ECG_peaks[i] < second_peaks[3]): 
                    New_ECG_Peak.append(ECG_peaks[i])
            ################################################################################################################       
                        
            'Normalizing the Modified PPG Signal'
            #################################################################################################################
            Modified_PPG_3 = normalization(norm_ppg)
            Modified_PPG = [1+i for i in Modified_PPG_3]
            
            # plt.plot(ECG_EDF)
            # plt.plot(Modified_PPG)
            # plt.scatter(New_ECG_Peak,np.array(ECG_EDF)[New_ECG_Peak])
            # plt.show()
            ##################################################################################################################

            'Finding the Quality of ECG and PPG Signal'
            ##################################################################################################################
            Systolic_Peak = findpeaks_nabian2018(Modified_PPG,fs)
            Inverted_PPG = [-1* i for i in Modified_PPG]
            PPG_Foot_5 = findpeaks_nabian2018(Inverted_PPG,fs)
            Height = np.mean(Inverted_PPG)+(np.std(Inverted_PPG))
            Ampli_PPG_Foot_5 = [Inverted_PPG[i] for i in PPG_Foot_5] 
            Corrected_Ampli_PPG = [i for i in Ampli_PPG_Foot_5 if i > Height] 
            PPG_Foot_1 = [Inverted_PPG.index(i) for i in Corrected_Ampli_PPG]
            # plt.plot(Modified_PPG)
            # plt.scatter(PPG_Foot_1,np.array(Modified_PPG)[PPG_Foot_1])
            # plt.show()
            
            # Systolic_Foot = scipy.signal.find_peaks([-1*i] for i in Modified_PPG], height=None, threshold=None, distance= 110, prominence=None, width=None, wlen=None, rel_height=None, plateau_size=None)[0].tolist()
            ECG_quality,ECG_Corr = signal_qulaity(ECG_EDF,New_ECG_Peak)
            PPG_quality,PPG_Corr = signal_qulaity(Modified_PPG,Systolic_Peak)
            # print(ECG_quality,PPG_quality)
            ##################################################################################################################
            
            if ECG_quality == PPG_quality:
                #######################################################################################################################
                
                'Getting the FFT of PPG and Caluclating the First,Second,Third Frequency from the Spectrum (Frequency Domain Feature)'
                ########################################################################################################################
                ts = 1.0/fs
                N = len(Modified_PPG[1000:10480])
                n = np.arange(N)
                T = N/fs
                freq = n/T
                t = np.arange(0,T,ts)
                FFT = scipy.fft.fft(Modified_PPG[1000:10480])
                
            
            
                # plt.figure(figsize = (12, 6))
                # plt.stem(freq, np.abs(FFT), 'b', \
                #         markerfmt=" ", basefmt="-b",label='FFT')
                # plt.suptitle('FFT_filtred_PPG')
                # plt.xlabel('Freq (Hz)')
                # plt.ylabel('FFT Amplitude |X(freq)|')   
                # plt.xlim(0,200)
                # plt.ylim(0,1000)
                # plt.legend()
                # plt.show()

                First_abs_magintude = max(abs(FFT[5:40]))
                First_max_index = np.where(abs(FFT) == First_abs_magintude)[0][0]
                First_max_1 = FFT[First_max_index]
                First_magintude = First_max_1.real
                First_Phase = First_max_1.imag
                First_abs_magintude_Left = abs(FFT[First_max_index-1])
                First_abs_magintude_Right = abs(FFT[First_max_index+1])
                First_Freq_1 = freq[First_max_index]
                FFT_HR = First_Freq_1*60
                
                Second_abs_magnitude = max(abs(FFT[np.where(abs(FFT) == First_abs_magintude)[0][0]+10:200]))
                Second_max_index = np.where(abs(FFT) == Second_abs_magnitude)[0][0]
                Second_max_1 = FFT[Second_max_index]
                Second_magintude = Second_max_1.real
                Second_Phase = Second_max_1.imag
                Second_abs_magintude_Left = abs(FFT[Second_max_index-1])
                Second_abs_magintude_Right = abs(FFT[Second_max_index+1])
                Second_Freq_1 = freq[Second_max_index]
                
                Third_abs_magintude = max(abs(FFT[np.where(abs(FFT) == Second_abs_magnitude)[0][0]+10:200]))
                Third_max_index = np.where(abs(FFT) == Third_abs_magintude)[0][0]
                Third_max_1 = FFT[Third_max_index]
                Third_magintude = Third_max_1.real
                Third_Phase = Third_max_1.imag
                Third_abs_magintude_Left = abs(FFT[Third_max_index-1])
                Third_abs_magintude_Right = abs(FFT[Third_max_index+1])
                Third_Freq_1 = freq[Third_max_index]
                # print(First_max_1,FFT_HR,First_Freq_1)
                # print(First_abs_magintude,First_magintude,First_Phase,First_Freq_1,First_abs_magintude_Left,First_abs_magintude_Right)
                # print(Second_magintude,Second_Phase,Second_Freq_1)
                # print(Third_magintude,Third_Phase,Third_Freq_1)
                ##########################################################################################################################
            
                'Getting First order and Second order differation for calculation of Slope and Dervative Features of PPG'   
                ##################################################################################################################################
                First_diff = np.diff(Modified_PPG[:-100])
                Second_diff = list(np.diff(Modified_PPG,2)[:-100])
                zero_crossing_second_diff = np.where(np.diff(np.sign(Second_diff)))[0] 
                Ampli_zero_crossing_second_diff = [Second_diff[i] for i in zero_crossing_second_diff]
                # plt.plot(Modified_PPG)
                # # plt.plot([20*i for i in First_diff])
                # plt.plot([900*i for i in Second_diff])
                # plt.scatter(zero_crossing_second_diff,np.array([900*i for i in Second_diff])[zero_crossing_second_diff])
                # plt.show()
                
                
                Slope = []
                first_diff_minimum = []
                for i in range(len(PPG_Foot_1)-1):
                    sliced_second_diff = list(Second_diff[PPG_Foot_1[i]:PPG_Foot_1[i+1]])
                    common_element_sliced_second_diff = list(set(sliced_second_diff).intersection(Ampli_zero_crossing_second_diff))
                    common_element_sliced_second_diff_index = [sliced_second_diff.index(i) for i in common_element_sliced_second_diff]
                    common_element_sliced_second_diff_index.sort()
                    Slope.append(common_element_sliced_second_diff_index[0]+PPG_Foot_1[i])
                    first_diff_minimum_1 = [i for i in common_element_sliced_second_diff_index if i > common_element_sliced_second_diff_index[0]+20]
                    first_diff_minimum.append(first_diff_minimum_1[0]+PPG_Foot_1[i])
                # plt.plot([20*i for i in np.diff(Modified_PPG)[:-100]])
                # plt.scatter(Slope,np.array([20*i for i in np.diff(Modified_PPG)[:-100]])[Slope])
                # plt.scatter(first_diff_minimum,np.array([20*i for i in np.diff(Modified_PPG)[:-100]])[first_diff_minimum])
                # plt.show()
                
                
                First_diff_max = [First_diff[i] for i in Slope]
                First_diff_min = [First_diff[i] for i in first_diff_minimum]
                First_diff_width = [first_diff_minimum[i] - Slope[i] for i in range(min([len(Slope),len(first_diff_minimum)]))]
                Total_height_first_diff = [First_diff_max[i]+First_diff_min[i] for i in range(min([len(First_diff_min),len(First_diff_max)]))]
                ###################################################################################################################################################################################################
            
                'Getting Third order Differation for Detecting the Foot,Systolic,Notch and Diastolic Peaks and Second Dervative Features of PPG'
                ###################################################################################################################################################################################################
                Derived_PPG_Foot_2 = []
                Common_element_Third_order_index_1 = []
                Third_diff = np.diff(Modified_PPG,3)[:-100]
                zero_crossing = np.where(np.diff(np.sign(Third_diff)))[0] 
                Ampli_zero_crossing = [Third_diff[i] for i in zero_crossing]
                
                for i in range(len(PPG_Foot_1)-1):
                    Sliced_Third_diff = list(Third_diff[PPG_Foot_1[i]:PPG_Foot_1[i+1]])
                    Common_element_Third_order = list(set(Sliced_Third_diff).intersection(Ampli_zero_crossing))
                    Common_element_Third_order_index = [Sliced_Third_diff.index(i) for i in Common_element_Third_order]
                    Common_element_Third_order_index.sort()
                    # Derived_foot_1 = Common_element_Third_order_index[0]
                    # second_diff_min_1 = [i for i in Common_element_Third_order_index if i > Derived_foot_1+15][0]
                    # Derived_PPG_Foot_2.append(Derived_foot_1+PPG_Foot_1[i]-10)
                    # second_diff_min.append(second_diff_min_1+PPG_Foot_1[i]-10)
                    Common_element_Third_order_index_1.append(Common_element_Third_order_index)
                
                # print(second_diff_min,len(second_diff_min))
                # print(Common_element_Third_order_index_1,len(Common_element_Third_order_index_1))
                # print()
                Derived_PPG_Peak = []
                second_diff_min_2 =[]
                PPG_Notch = []
                PPG_Diastolic = []
                Derived_PPG_Foot = []
                for i in range(len(PPG_Foot_1)-1):
                    # print(PPG_Foot_1[i+1],PPG_Foot_1[i])
                    Diff_foot_peak = (round((PPG_Foot_1[i+1] - PPG_Foot_1[i])*0.26))+80
                    Common_element_Third_order_index = Common_element_Third_order_index_1[i]
                    # print(Diff_foot_peak)
                    # print(Common_element_Third_order_index)
                    sliced_index =[i for i in Common_element_Third_order_index if i >= Diff_foot_peak-40 and i<= Diff_foot_peak+40]
                    if len(sliced_index) > 0:
                        Difference = [abs(i-Diff_foot_peak) for i in sliced_index]
                        minimum_value_index = Difference.index(min(Difference))
                        Derived_sys_peak_1 = sliced_index[minimum_value_index]
                    else:
                        sliced_index_1 =[i for i in Common_element_Third_order_index if i >= Diff_foot_peak-80 and i<= Diff_foot_peak+80]
                        Difference = [abs(i-Diff_foot_peak) for i in sliced_index_1]
                        try:
                            minimum_value_index = Difference.index(min(Difference))
                            Derived_sys_peak_1 = sliced_index_1[minimum_value_index]
                        except:
                            continue
                    
                    Common_element_Third_order_index_2 = Common_element_Third_order_index_1[i]
                    PPG_Foot_2 = PPG_Foot_1[i]
                    Notch_1 = [i for i in Common_element_Third_order_index_2 if i > Derived_sys_peak_1+20][0]
                    Diastolic_peak_1 = [i for i in Common_element_Third_order_index_2 if i > Notch_1][0]
                    Notch_2 = Notch_1+PPG_Foot_2
                    second_diff_min_1 = Second_diff.index(min(Second_diff[PPG_Foot_2:Notch_2-5]))
                    foot = Second_diff.index(max(Second_diff[PPG_Foot_2:second_diff_min_1-5]))
                    Notch = Second_diff.index(max(Second_diff[second_diff_min_1+5:Notch_2+5]))
                    Diastolic_peak_2 = [i for i in Common_element_Third_order_index_2 if i > Notch-PPG_Foot_2][0]
                    # print(Common_element_Third_order_index_2,Notch,PPG_Foot_2,second_diff_min_1-PPG_Foot_2)``
                    # print(second_diff_min_1-PPG_Foot_1[i],Common_element_Third_order_index_2,foot,Notch)
                    # # Deried_PPG_Peak.append(Derived_sys_peak_1+Derived_PPG_Foot[i])
                    Derived_PPG_Peak.append(second_diff_min_1)
                    PPG_Notch.append(Notch)
                    PPG_Diastolic.append(Diastolic_peak_2+PPG_Foot_2) 
                    Derived_PPG_Foot.append(foot)
                    second_diff_min_2.append(second_diff_min_1)
                    
                Second_diff_max = [Second_diff[i] for i in Derived_PPG_Foot]
                Second_diff_min = [Second_diff[i] for i in second_diff_min_2]
                Total_height_second_diff = [Second_diff_min[i]+Second_diff_max[i] for i in range(min(len(Second_diff_max),len(Second_diff_min)))]
                Second_diff_width = [Derived_PPG_Foot[i] - second_diff_min_2[i] for i in range(min(len(Derived_PPG_Foot),len(Second_diff_min)))]
                
                
                # # # plt.plot([10*i for i in First_diff])
                # plt.plot(Modified_PPG)
                # plt.plot([200*i for i in Second_diff])
                # # plt.scatter(Derived_PPG_Peak,np.array(Modified_PPG)[Derived_PPG_Peak],label='Derived_sys_peak')
                # plt.plot([200*i for i in Third_diff])
                # # plt.scatter(zero_crossing,np.array([200*i for i in Third_diff])[zero_crossing],label='Zero_Crossing_1')
                # plt.scatter(Derived_PPG_Foot,np.array([200*i for i in Second_diff])[Derived_PPG_Foot],label='Derived_foot')
                # # plt.scatter(second_diff_min_2,np.array([200*i for i in Second_diff])[second_diff_min_2],label='second_diff_min')
                # plt.scatter(PPG_Notch,np.array([200*i for i in Second_diff])[PPG_Notch],label='Notch')
                # plt.scatter(PPG_Diastolic,np.array([200*i for i in Second_diff])[PPG_Diastolic],label='Diastolic_peak')
                # plt.scatter(Derived_PPG_Peak,np.array([200*i for i in Second_diff])[Derived_PPG_Peak],label='Derived_sys_peak')
                # # plt.scatter(Derived_PPG_Foot,np.array([500*i for i in Third_diff])[Derived_PPG_Foot],label='Derived_foot')
                # plt.scatter(PPG_Notch,np.array(Modified_PPG)[PPG_Notch],label='PPG_Notch')
                # plt.scatter(PPG_Diastolic,np.array(Modified_PPG)[PPG_Diastolic],label='PPG_Diastolic')
                # plt.scatter(Derived_PPG_Peak,np.array(Modified_PPG)[Derived_PPG_Peak],label='Derived_PPG_Peak')
                # plt.scatter(Derived_PPG_Foot,np.array(Modified_PPG)[Derived_PPG_Foot],label='Derived_PPG_Foot')
                # plt.legend()
                # plt.show()
                
                End_of_systolic = [round((Derived_PPG_Foot[i+1] - Derived_PPG_Foot[i])*0.33)+Derived_PPG_Foot[i] for i in range(len(Derived_PPG_Foot)-1)]
                
                # plt.plot(Modified_PPG)
                # plt.scatter(Derived_PPG_Foot,np.array(Modified_PPG)[Derived_PPG_Foot],label='Derived_foot')
                # plt.scatter(Derived_PPG_Peak,np.array(Modified_PPG)[Derived_PPG_Peak],label='Derived_sys_peak')
                # plt.scatter(PPG_Notch,np.array(Modified_PPG)[PPG_Notch],label='Notch')
                # plt.scatter(PPG_Diastolic,np.array(Modified_PPG)[PPG_Diastolic],label='Diastolic_peak')
                # plt.scatter(Slope,np.array(Modified_PPG)[Slope],label='Slope')
                # plt.scatter(End_of_systolic,np.array(Modified_PPG)[End_of_systolic],label='End_of_systolic')
                # plt.legend()
                # plt.show()
                ##########################################################################################################################################
                
            
                #########################################################################################################################################################
                
                'Correcting Peaks of ECG and PPG According to PPG_Foot'
                ###########################################################################################################################################################
                ECG_peak_new = [] 
                for i in range(len(New_ECG_Peak)):
                    if (New_ECG_Peak[i] > Derived_PPG_Foot[0]) and (New_ECG_Peak[i] < Derived_PPG_Foot[-1]): 
                        ECG_peak_new.append(New_ECG_Peak[i])
                        
                PPG_Foot = []
                for i in range(len(Derived_PPG_Foot)):
                    if (Derived_PPG_Foot[i] > ECG_peak_new[0]) and (Derived_PPG_Foot[i] < ECG_peak_new[-1]): 
                        PPG_Foot.append(Derived_PPG_Foot[i])
                
                
                PPG_Systolic_peak_New = []
                for i in range(len(Derived_PPG_Peak)):
                    if (Derived_PPG_Peak[i] > PPG_Foot[0]) and (Derived_PPG_Peak[i] < PPG_Foot[-1]) and (Derived_PPG_Peak[i] > ECG_peak_new[0]):
                        PPG_Systolic_peak_New.append(Derived_PPG_Peak[i])
                        
                PPG_Notch_New = []
                for i in range(len(PPG_Notch)):
                    if (PPG_Notch[i] > PPG_Foot[0]) and (PPG_Notch[i] < PPG_Foot[-1]) and (PPG_Notch[i] > ECG_peak_new[0]):
                        PPG_Notch_New.append(PPG_Notch[i])
                        
                PPG_Diastolic_Peak_New = []
                for i in range(len(PPG_Diastolic)):
                    if (PPG_Diastolic[i] > PPG_Foot[0]) and (PPG_Diastolic[i] < PPG_Foot[-1]) and (PPG_Diastolic[i] > ECG_peak_new[0]):
                        PPG_Diastolic_Peak_New.append(PPG_Diastolic[i])
                        
                PPG_Slope_New = []
                for i in range(len(Slope)):
                    if (Slope[i] > PPG_Foot[0]) and (Slope[i] < PPG_Foot[-1]) and (Slope[i] > ECG_peak_new[0]) :
                        PPG_Slope_New.append(Slope[i]) 
                        
                        
                P_onset_New =[]
                for i in range(len(P_onset)):
                    if (P_onset[i] > Derived_PPG_Foot[0]-40) and (P_onset[i] < Derived_PPG_Foot[-1]):
                        P_onset_New.append(P_onset[i]) 
                
                P_offset_New =[]
                for i in range(len(P_offset)):
                    if (P_offset[i] > Derived_PPG_Foot[0]-40) and (P_offset[i] < Derived_PPG_Foot[-1]):
                        P_offset_New.append(P_offset[i]) 
                
                
                P_Peak_New =[]
                for i in range(len(P_Peak)):
                    if (P_Peak[i] > Derived_PPG_Foot[0]-40) and (P_Peak[i] < Derived_PPG_Foot[-1]):
                        P_Peak_New.append(P_Peak[i])     
                
                Q_Peak_New =[]
                for i in range(len(Q_Peak)):
                    if (Q_Peak[i] > Derived_PPG_Foot[0]-40) and (Q_Peak[i] < Derived_PPG_Foot[-1]):
                        Q_Peak_New.append(Q_Peak[i])   
                        
                S_Peak_New =[]
                for i in range(len(S_Peak)):
                    if (S_Peak[i] > Derived_PPG_Foot[0]) and (S_Peak[i] < Derived_PPG_Foot[-1]) and (S_Peak[i] > ECG_peak_new[0]):
                        S_Peak_New.append(S_Peak[i])
                        
                T_onset_New =[]
                for i in range(len(T_onset)):
                    if (T_onset[i] > Derived_PPG_Foot[0]) and (T_onset[i] < Derived_PPG_Foot[-1]) and (T_onset[i] > ECG_peak_new[0]):
                        T_onset_New.append(T_onset[i])
                        
                T_offset_New =[]
                for i in range(len(T_offset)):
                    if (T_offset[i] > Derived_PPG_Foot[0]) and (T_offset[i] < Derived_PPG_Foot[-1]) and (T_offset[i] > ECG_peak_new[0]):
                        T_offset_New.append(T_offset[i])
                        
                T_Peak_New =[]
                for i in range(len(T_Peak)):
                    if (T_Peak[i] > Derived_PPG_Foot[0]) and (T_Peak[i] < Derived_PPG_Foot[-1]) and (T_Peak[i] > ECG_peak_new[0]):
                        T_Peak_New.append(T_Peak[i])
                
                PPG_Foot = []
                for i in range(len(Derived_PPG_Foot)):
                    if (Derived_PPG_Foot[i] > ECG_peak_new[0]) and (Derived_PPG_Foot[i] < ECG_peak_new[-1]): 
                        PPG_Foot.append(Derived_PPG_Foot[i])
            

                # plt.plot(Modified_PPG)
                # plt.plot(ECG_EDF)
                # plt.scatter(P_Peak_New,np.array(ECG_EDF)[P_Peak_New],label = 'P_Peak')
                # plt.scatter(Q_Peak_New,np.array(ECG_EDF)[Q_Peak_New],label = 'Q_Peak')
                # plt.scatter(ECG_peak_new,np.array(ECG_EDF)[ECG_peak_new],label = 'R_Peak')
                # plt.scatter(S_Peak_New,np.array(ECG_EDF)[S_Peak_New],label = 'S_Peak')
                # plt.scatter(T_Peak_New,np.array(ECG_EDF)[T_Peak_New],label = 'T_Peak')
                # plt.scatter(P_onset_New,np.array(ECG_EDF)[P_onset_New],label = 'P_onset')
                # plt.scatter(P_offset_New,np.array(ECG_EDF)[P_offset_New],label = 'P_offset')
                # plt.scatter(T_onset_New,np.array(ECG_EDF)[T_onset_New],label = 'T_onset')
                # plt.scatter(T_offset_New,np.array(ECG_EDF)[T_offset_New],label = 'T_offset')
                # plt.scatter(PPG_Foot,np.array(Modified_PPG)[PPG_Foot],label='FOOT')
                # plt.scatter(PPG_Slope_New,np.array(Modified_PPG)[PPG_Slope_New],label='Slope')
                # plt.scatter(PPG_Systolic_peak_New,np.array(Modified_PPG)[PPG_Systolic_peak_New],label='Systolic_peak')
                # # plt.scatter(End_of_systolic,np.array(Modified_PPG)[End_of_systolic],label='End_of_systolic')
                # plt.scatter(PPG_Notch_New,np.array(Modified_PPG)[PPG_Notch_New],label='Notch')
                # plt.scatter(PPG_Diastolic_Peak_New,np.array(Modified_PPG)[PPG_Diastolic_Peak_New],label='Diastolic')
                # plt.scatter(ECG_peak_new,np.array(ECG_EDF)[ECG_peak_new],label='Diastolic')
                # plt.legend()
                # plt.show()
                ######################################################################################################################################################################
                'Caluclating ECG Features'
                ######################################################################################################################################################################
                ECG_length = [len(P_Peak_New),len(Q_Peak_New),len(ECG_peak_new),len(S_Peak_New),len(T_Peak_New),len(P_onset_New),len(P_offset_New),len(T_onset_New),len(T_offset_New)]
                P_R_interval = []
                QRS_interval = []
                Q_T_interval = []
                T_P_interval = []
                R_R_interval = []
                ECG_HR = []
                ECG_PTT = []
                Corrected_QT_Bazett_formula = []
                Corrected_QT_from_Hodges_formula = []
                P_Duration = []
                T_Duration = []
                T_onset_to_Peak = []
                T_peak_to_offset = []
                for i in range(min(ECG_length)-1):
                    P_R_interval.append((ECG_peak_new[i] - P_onset_New[i])/fs)
                    QRS_interval.append((S_Peak_New[i] - Q_Peak_New[i])/fs)
                    Q_T_interval_1 = ((T_offset_New[i] - Q_Peak_New[i])/fs)
                    Q_T_interval.append(Q_T_interval_1)
                    T_P_interval.append((P_onset_New[i+1] - T_onset_New[i])/fs)
                    R_R_interval_1 = ((ECG_peak_new[i+1] - ECG_peak_new[i])/fs)
                    R_R_interval.append(R_R_interval_1)
                    ECG_PTT.append((T_offset_New[i] - ECG_peak_new[i])/fs)
                    ECG_HR_1 = (round(60000/((ECG_peak_new[i+1] - ECG_peak_new[i])*1),5))
                    ECG_HR.append(ECG_HR_1)
                    Corrected_QT_Bazett_formula.append(Q_T_interval_1/(math.sqrt(R_R_interval_1)))
                    Corrected_QT_from_Hodges_formula.append((Q_T_interval_1*1000)+(1.75*(ECG_HR_1-60))/1000)
                    P_Duration.append((P_offset_New[i] - P_onset_New[i])/fs)
                    T_Duration.append((T_offset_New[i] - T_onset_New[i])/fs)
                    T_onset_to_Peak.append((T_Peak_New[i] - T_onset_New[i])/fs)
                    T_peak_to_offset.append((T_offset_New[i] - T_Peak_New[i])/fs)
                Ratio_of_onset_peak_to_offset = [T_onset_to_Peak[i]/T_peak_to_offset[i] for i in range(min([len(T_onset_to_Peak),len(T_peak_to_offset)]))]
                ######################################################################################################################################################################
                'Calculating PTT from ECG and PPG'
                ######################################################################################################################################################################
                PTT_F = []
                PTT_S = []
                PTT_P = []
                PTT_Notch = []
                PTT_Dia = []
                
                # print(PPG_Foot,ECG_peak_new)
                # print(PPG_Slope_New,ECG_peak_new)
                # print(PPG_Systolic_peak_New,ECG_peak_new)
                # print(PPG_Notch_New,ECG_peak_new)
                # print(PPG_Diastolic_Peak_New,ECG_peak_new)
                length = [len(ECG_peak_new),len(PPG_Foot),len(PPG_Slope_New),len(PPG_Systolic_peak_New),len(PPG_Notch_New),len(PPG_Diastolic_Peak_New)]
                
                for i in range(len(ECG_peak_new)):
                    PTT_P_New = []
                    for j in range(len(PPG_Systolic_peak_New)):
                        if PPG_Systolic_peak_New[j] - ECG_peak_new[i] > 0 and  PPG_Systolic_peak_New[j] - ECG_peak_new[i] < 800:
                            PTT_P_New.append(PPG_Systolic_peak_New[j] - ECG_peak_new[i])
                    if len(PTT_P_New) == 1:
                        PTT_P.append(PTT_P_New[0])
                        
                for i in range(len(ECG_peak_new)):
                    PTT_F_New = []
                    for j in range(len(PPG_Foot)):
                        if PPG_Foot[j] - ECG_peak_new[i] > 0 and  PPG_Foot[j] - ECG_peak_new[i] < 600:
                            PTT_F_New.append(PPG_Foot[j] - ECG_peak_new[i])
                    if len(PTT_F_New) == 1:
                        PTT_F.append(PTT_F_New[0])
                        
                for i in range(len(ECG_peak_new)):
                    PTT_S_New = []
                    for j in range(len(PPG_Slope_New)):
                        if PPG_Slope_New[j] - ECG_peak_new[i] > 0 and  PPG_Slope_New[j] - ECG_peak_new[i] < 700:
                            PTT_S_New.append(PPG_Slope_New[j] - ECG_peak_new[i])
                    if len(PTT_S_New) == 1:
                        PTT_S.append(PTT_S_New[0])
                        
            
                for i in range(len(ECG_peak_new)):
                    PTT_N_New = []
                    for j in range(len(PPG_Notch_New)):
                        if PPG_Notch_New[j] - ECG_peak_new[i] > 0 and  PPG_Notch_New[j] - ECG_peak_new[i] < 800:
                            PTT_N_New.append(PPG_Notch_New[j] - ECG_peak_new[i])
                    if len(PTT_N_New) == 1:
                        PTT_Notch.append(PTT_N_New[0])
                    elif len(PTT_N_New) > 1:
                        PTT_Notch.append(PTT_N_New[-1])
                        
                
                for i in range(len(ECG_peak_new)):
                    PTT_D_New = []
                    for j in range(len(PPG_Diastolic_Peak_New)):
                        if PPG_Diastolic_Peak_New[j] - ECG_peak_new[i] > 0 and  PPG_Diastolic_Peak_New[j] - ECG_peak_new[i] < 1000:
                            PTT_D_New.append(PPG_Diastolic_Peak_New[j] - ECG_peak_new[i])
                    if len(PTT_D_New) == 1:
                        PTT_Dia.append(PTT_D_New[0])
                    elif len(PTT_D_New) > 1:
                        PTT_Dia.append(PTT_D_New[-1])
                        
                # print(PTT_F)
                # print(PTT_S)
                # print(PTT_P)
                # print(PTT_Notch)
                # print(PTT_Dia)
                #############################################################################################################################################################################
                'Discarding the peaks which as high difference'
                ##############################################################################################################################################################################
                Ampli_systolic = [Modified_PPG[i] for i in PPG_Systolic_peak_New]
                Ampli_Foot = [Modified_PPG[i] for i in PPG_Foot]
                PIR = [Ampli_systolic[i]/Ampli_Foot[i] for i in range(min(length))]
                
                
                Corrected_PPG_Slope = PPG_Slope_New
                Corrected_PPG_Notch = PPG_Notch_New
                Corrected_PPG_Diastolic_Peak = PPG_Diastolic_Peak_New
                
                # plt.plot(Modified_PPG)
                # plt.plot(ECG_EDF)
                # plt.scatter(PPG_Foot,np.array(Modified_PPG)[PPG_Foot],label='FOOT')
                # plt.scatter(Corrected_PPG_Slope,np.array(Modified_PPG)[Corrected_PPG_Slope],label='Slope')
                # plt.scatter(PPG_Systolic_peak_New,np.array(Modified_PPG)[PPG_Systolic_peak_New],label='Systolic_peak')
                # # plt.scatter(Corrected_End_of_systolic,np.array(Modified_PPG)[Corrected_End_of_systolic],label='End_of_systolic')
                # plt.scatter(Corrected_PPG_Notch,np.array(Modified_PPG)[Corrected_PPG_Notch],label='Notch')
                # plt.scatter(Corrected_PPG_Diastolic_Peak,np.array(Modified_PPG)[Corrected_PPG_Diastolic_Peak],label='Diastolic')
                # plt.scatter(ECG_peak_new,np.array(ECG_EDF)[ECG_peak_new],label='ECG_Peak')
                # plt.legend()
                # plt.show()
                #######################################################################################################################################################################
                'Taking Peaks Which are between FOOt to Foot of PPG'
                #######################################################################################################################################################################
                # length_1 = [len(ECG_peak_new),len(PPG_Foot),len(ECG_HR)]
                corrected_peaks_1 = []
                for i in range(min(length)-1):
                    data = []
                    sliced_data = norm_ppg[PPG_Foot[i]:PPG_Foot[i+1]+1]
                    y_slope = [norm_ppg[j] for j in Corrected_PPG_Slope]
                    y_peak = [norm_ppg[j] for j in PPG_Systolic_peak_New]
                    y_notch = [norm_ppg[j] for j in Corrected_PPG_Notch]
                    y_diastolic = [norm_ppg[j] for j in Corrected_PPG_Diastolic_Peak]
                    y_end_of_systolic = [norm_ppg[j] for j in End_of_systolic]
                    y_foot = [norm_ppg[j] for j in PPG_Foot]
                    data_2 = [i for i in y_foot if i in sliced_data]
                    if len(data_2) == 2:
                        data.append(norm_ppg.index(data_2[0]))
                    data_3 = [i for i in y_slope if i in sliced_data]
                    if len(data_3) == 1:
                        data.append(norm_ppg.index(data_3[0]))
                    data_4 = [i for i in y_peak if i in sliced_data]
                    if len(data_4) == 1:
                        data.append(norm_ppg.index(data_4[0]))
                    data_5 = [i for i in y_notch if i in sliced_data]
                    if len(data_5) == 1:
                        data.append(norm_ppg.index(data_5[0]))
                    data_6 = [i for i in y_diastolic if i in sliced_data]
                    if len(data_6) == 1:
                        data.append(norm_ppg.index(data_6[0]))
                    if len(data_2) == 2:
                        data.append(norm_ppg.index(data_2[1]))
                    data_7 = [i for i in y_end_of_systolic if i in sliced_data]
                    if len(data_7) == 1:
                        data.append(norm_ppg.index(data_7[0]))
                    corrected_peaks_1.append(data)
                # print(corrected_peaks_1)
                ##############################################################################################################################################################################
                'Extrcting PPG Features'
                ###############################################################################################################################################################################
                Feature_len = [len(ECG_peak_new),len(PPG_Foot),len(PPG_Slope_New),len(PPG_Systolic_peak_New),len(PPG_Notch_New),len(PPG_Diastolic_Peak_New),len(PTT_F),len(PTT_P),len(PTT_S),len(PTT_Notch),len(PTT_Dia),len(ECG_HR),len(PIR),len(ECG_Corr),len(PPG_Corr),len(P_R_interval),len(QRS_interval),len(Q_T_interval),len(T_P_interval),len(R_R_interval),len(ECG_PTT),len(Corrected_QT_Bazett_formula),len(Corrected_QT_from_Hodges_formula),len(P_Duration),len(T_Duration),len(T_onset_to_Peak),len(T_peak_to_offset),len(Ratio_of_onset_peak_to_offset)]
                # print(Feature_len)
                ST = []
                DT = []
                S1_time = []
                New_Ampli_Systolic = []
                New_Ampli_Foot= []
                S2_time = []
                S3_time = []
                S4_time = []
                S5_time = []
                PTT_PPG_Peak = []
                PTT_PPG_notch = []
                PTT_PPG_diastolic = []
                PTT_PPG_foot = []
                New_PTT_F = []
                New_PTT_P = []
                New_PTT_S = []
                New_PTT_DN = []
                New_PTT_DP = []
                S1_1 = []
                S2_1 = []
                S3_1 = []
                S4_1 = []
                LASI = []
                AI = []
                New_ECG_HR = []
                PPG_HR = []
                New_P_R_interval = []
                New_QRS_interval = []
                New_Q_T_interval = []
                New_T_P_interval = []
                New_R_R_interval = []
                New_ECG_PTT = []
                New_PIR = []
                Delta_T = []
                quent_data = []
                Mean = []
                Median = []
                Mode = []
                Variance = []
                Standard_Devation = []
                Minimum = []
                Maximum = []
                Skewness = []
                Kurtosis = []
                Twenty_Fifth_Percentile = []
                Fifty_Percentile = []
                Seventy_Fifth_Percentile = []
                Shannon_Entropy = []
                New_first_diff_max = []
                New_first_diff_min = []
                New_first_diff_width = []
                New_Total_height_first_diff = []
                New_Second_diff_max = []
                New_Second_diff_min = []
                New_Second_diff_width = []
                New_Total_height_second_diff = []
                ECG_Corr_1 = []
                PPG_Corr_1 = []
                PTT_PPG_end_of_sys = []
                sys_area = []
                New_Corrected_QT_Bazett_formula = []
                New_Corrected_QT_from_Hodges_formula = []
                New_P_Duration = []
                New_T_Duration = []
                New_T_onset_to_Peak = []
                New_T_peak_to_offset = []
                New_Ratio_of_onset_peak_to_offset = []
                for k in range(min(Feature_len)-1):
                    corrected_peaks = corrected_peaks_1[k]
                    if (len(corrected_peaks) == 7) and (corrected_peaks[0]<corrected_peaks[1]<corrected_peaks[2]<corrected_peaks[3]<corrected_peaks[4]<corrected_peaks[5]): 
                        ST.append((corrected_peaks[2] - corrected_peaks[0])/fs)
                        DT.append((corrected_peaks[5] - corrected_peaks[2])/fs)
                        S1_time.append((corrected_peaks[1] - corrected_peaks[0])/fs)
                        New_Ampli_Systolic.append(Ampli_systolic[k])
                        New_Ampli_Foot.append(Ampli_Foot[k])
                        S2_time.append((corrected_peaks[2] - corrected_peaks[1])/fs)
                        S3_time.append(((corrected_peaks[3] - corrected_peaks[2])/fs))
                        S4_time.append(((corrected_peaks[4] - corrected_peaks[3])/fs))
                        S5_time.append((corrected_peaks[5] - corrected_peaks[4])/fs)
                        PTT_PPG_Peak.append((corrected_peaks[2] - corrected_peaks[0])/fs)
                        PTT_PPG_notch.append((corrected_peaks[3] - corrected_peaks[0])/fs)
                        PTT_PPG_diastolic.append((corrected_peaks[4] - corrected_peaks[0])/fs)
                        PTT_PPG_end_of_sys.append((corrected_peaks[6] - corrected_peaks[0])/fs)
                        PTT_PPG_foot.append((corrected_peaks[5] - corrected_peaks[0])/fs)
                        Sliced_ppg = Modified_PPG
                        Foot_Slope = Sliced_ppg[corrected_peaks[0] : corrected_peaks[1]]
                        # Foot_Peak = Sliced_ppg[corrected_peaks[0] : corrected_peaks[2]]
                        # peak_next_foot = Sliced_ppg[corrected_peaks[2] : corrected_peaks[5]]
                        Slope_Peak = Sliced_ppg[corrected_peaks[1] : corrected_peaks[2]]
                        Peak_Notch = Sliced_ppg[corrected_peaks[2] : corrected_peaks[3]]
                        # Notch_Diastolic = Sliced_ppg[corrected_peaks[3] : corrected_peaks[4]]
                        Notch_next_foot = Sliced_ppg[corrected_peaks[3] : corrected_peaks[5]]
                        Foot_end_of_sys = Sliced_ppg[corrected_peaks[0] : corrected_peaks[6]]
                        Sub_min_in_foot_slope = [j-min(Sliced_ppg[corrected_peaks[0]  :  corrected_peaks[5]]) for j in Foot_Slope]
                        # Sub_min_in_foot_Peak = [j-min(Sliced_ppg[corrected_peaks[0]  :  corrected_peaks[5]]) for j in Foot_Peak]
                        # Sub_min_in_peak_next_foot = [j-min(Sliced_ppg[corrected_peaks[0]  :  corrected_peaks[5]]) for j in peak_next_foot]
                        Sub_min_in_slope_peak = [j-min(Sliced_ppg[corrected_peaks[0]  :  corrected_peaks[5]]) for j in Slope_Peak]
                        Sub_min_in_peak_notch = [j-min(Sliced_ppg[corrected_peaks[0]  :  corrected_peaks[5]]) for j in Peak_Notch]
                        Sub_min_in_notch_foot = [j-min(Sliced_ppg[corrected_peaks[0]  :  corrected_peaks[5]]) for j in Notch_next_foot]
                        # Sub_min_in_notch_dia = [j-min(Sliced_ppg[corrected_peaks[0]  :  corrected_peaks[5]]) for j in Notch_Diastolic]
                        Sub_min_foot_end_of_sys = [j-min(Sliced_ppg[corrected_peaks[0]  :  corrected_peaks[5]]) for j in Foot_end_of_sys]
                        S1_1.append(round(np.trapz(Sub_min_in_foot_slope)/fs,5))
                        # ST_area.append(round(np.trapz(Sub_min_in_foot_Peak)/fs,5))
                        # DT_area.append(round(np.trapz(Sub_min_in_peak_next_foot)/fs,5))
                        S2_1.append(round(np.trapz(Sub_min_in_slope_peak)/fs,5))
                        S3_1.append(round(np.trapz(Sub_min_in_peak_notch)/fs,5))
                        S4_1.append(round(np.trapz(Sub_min_in_notch_foot)/fs,5))
                        # S4_1.append(round(np.trapz(Sub_min_in_notch_dia)/fs,5))
                        sys_area.append((np.trapz(Sub_min_foot_end_of_sys)/fs))
                        Sliced_ppg_1 = Sliced_ppg[corrected_peaks[0]:corrected_peaks[5]]
                        MEAN = stat.mean(Sliced_ppg_1)
                        MEDIAN = stat.median(Sliced_ppg_1)
                        MODE = stat.mode(Sliced_ppg_1)
                        VARIANCE = stat.variance(Sliced_ppg_1)
                        STANDARD_DEVATION = stat.stdev(Sliced_ppg_1)
                        MIN = min(Sliced_ppg_1)
                        MAX = max(Sliced_ppg_1)
                        SKEW = 3*(MEAN-MEDIAN)/STANDARD_DEVATION
                        KURTOSIS = scipy.stats.skew(Sliced_ppg_1)
                        Z1,Z2,Z3 = np.percentile(Sliced_ppg_1 , [25,50,75])
                        ENTROPY = nk.entropy_shannon(Sliced_ppg_1)
                        Mean.append(MEAN)
                        Median.append(MEDIAN)
                        Mode.append(MODE)
                        Variance.append(VARIANCE)
                        Standard_Devation.append(STANDARD_DEVATION)
                        Minimum.append(MIN)
                        Maximum.append(MAX)
                        Skewness.append(SKEW)
                        Kurtosis.append(KURTOSIS)
                        Twenty_Fifth_Percentile.append(Z1)
                        Fifty_Percentile.append(Z2)
                        Seventy_Fifth_Percentile.append(Z3)
                        Shannon_Entropy.append(ENTROPY[0])
                        New_first_diff_max.append(abs(First_diff_max[k]))
                        New_first_diff_min.append(abs(First_diff_min[k]))
                        New_first_diff_width.append(abs(First_diff_width[k]))
                        New_Total_height_first_diff.append(abs(Total_height_first_diff[k]))
                        New_Second_diff_max.append(abs(Second_diff_max[k]))
                        New_Second_diff_min.append(abs(Second_diff_min[k]))
                        New_Second_diff_width.append(abs(Second_diff_width[k]))
                        New_Total_height_second_diff.append(abs(Total_height_second_diff[k]))
                        LASI.append(round(float(fs/(corrected_peaks[3]- corrected_peaks[2])),5))
                        Delta_T.append(corrected_peaks[4]- corrected_peaks[2])
                        AI.append(round(Sliced_ppg[corrected_peaks[3]]/Sliced_ppg[corrected_peaks[2]],5))
                        New_P_R_interval.append(P_R_interval[k])
                        New_QRS_interval.append(QRS_interval[k])
                        New_Q_T_interval.append(Q_T_interval[k])
                        New_T_P_interval.append(T_P_interval[k])
                        New_R_R_interval.append(R_R_interval[k])
                        New_ECG_PTT.append(ECG_PTT[k])
                        New_Corrected_QT_Bazett_formula.append(Corrected_QT_Bazett_formula[k])
                        New_Corrected_QT_from_Hodges_formula.append(Corrected_QT_from_Hodges_formula[k])
                        New_P_Duration.append(P_Duration[k])
                        New_T_Duration.append(T_Duration[k])
                        New_T_onset_to_Peak.append(T_onset_to_Peak[k])
                        New_T_peak_to_offset.append(T_peak_to_offset[k])
                        New_Ratio_of_onset_peak_to_offset.append(Ratio_of_onset_peak_to_offset[k])
                        New_PTT_F.append(PTT_F[k])
                        New_PTT_P.append(PTT_P[k])
                        New_PTT_S.append(PTT_S[k])
                        New_PTT_DN.append(PTT_Notch[k])
                        New_PTT_DP.append(PTT_Dia[k])
                        New_ECG_HR.append(ECG_HR[k])
                        New_PIR.append(PIR[k])
                        PPG_HR.append(round(60000/((corrected_peaks[5] - corrected_peaks[0])*1),5))
                        ECG_Corr_1.append(ECG_Corr[k])
                        PPG_Corr_1.append(PPG_Corr[k])
                        quent_data.append("quent"+ "_" + str(k))
                        

                IPA = [(S4_1[i]/(S1_1[i]+S2_1[i]+S3_1[i])) for i in range(len(S4_1))]
                Ratio_of_sys_time_Ampli = ([ST[i]/New_Ampli_Systolic[i] for i in range(len(ST))])
                Ratio_of_sys_pulse_interval = ([ST[i]/PTT_PPG_foot[i] for i in range(len(ST))])
                
                
                # print('s1_time :',S1_time)
                # print()
                # print('s2_time :',S2_time)
                # print()
                # print('s3_time :',S3_time)
                # print()
                # print('s4_time :',S4_time)
                # print()
                # print('s5_time :',S5_time)
                # print()
                # print('ST :',ST)
                # print()
                # print('DT :',DT)
                # print()
                # print('PTT_PPG_Peak :',PTT_PPG_Peak)
                # print()
                # print('PTT_PPG_notch :',PTT_PPG_notch)
                # print()
                # print('PTT_PPG_diastolic :',PTT_PPG_diastolic)
                # print()
                # print('PTT_PPG_foot :',PTT_PPG_foot)
                # print()
                # print('PTT_F :',New_PTT_F)
                # print()
                # print('PTT_P :',New_PTT_P)
                # print()
                # print('PTT_S :',New_PTT_S)
                # print()
                # print('PTT_DN :',New_PTT_DN)
                # print()
                # print('PTT_DP :',New_PTT_DP)
                # print()
                # print('s1_1 :',S1_1)
                # print()
                # print('s2_1 :',S2_1)
                # print()
                # print('s3_1 :',S3_1)
                # print()
                # print('s4_1 :',S4_1)
                # print()
                # print('LASI :',LASI)
                # print()
                # print('AI :',AI)
                # print()
                # print('ECG_HR :',New_ECG_HR)
                # print()
                # print('PPG_HR :',PPG_HR)
                # print()
                # print('New_P_R_interval :',New_P_R_interval)
                # print()
                # print('New_QRS_interval :',New_QRS_interval)
                # print()
                # print('New_Q_T_interval :',New_Q_T_interval)
                # print()
                # print('New_T_P_interval :',New_T_P_interval)
                # print()
                # print('New_R_R_interval :',New_R_R_interval)
                # print()
                # print('ECG_PTT :',New_ECG_PTT)
                # print()
                # print('New_PIR :',New_PIR)
                # print()
                # print('Delta_T :',Delta_T)
                # print()
                # print('quent_data :',quent_data)
                
                
                First_max = [First_magintude]*len(S1_time)
                First_abs_max = [First_abs_magintude]*len(S1_time)
                First_abs_max_Left = [First_abs_magintude_Left]*len(S1_time)
                First_abs_max_Right = [First_abs_magintude_Right]*len(S1_time)
                Second_max = [Second_magintude]*len(S1_time)
                Second_abs_max = [Second_abs_magnitude]*len(S1_time)
                Second_abs_max_Left = [Second_abs_magintude_Left]*len(S1_time)
                Second_abs_max_Right = [Second_abs_magintude_Right]*len(S1_time)
                Third_max = [Third_magintude]*len(S1_time)
                Third_abs_max = [Third_abs_magintude]*len(S1_time)
                Third_abs_max_Left = [Third_abs_magintude_Left]*len(S1_time)
                Third_abs_max_Right = [Third_abs_magintude_Right]*len(S1_time)
                First_Phase_1 = [First_Phase]*len(S1_time)
                Second_Phase_1 = [Second_Phase]*len(S1_time)
                Third_Phase_1 = [Third_Phase]*len(S1_time)
                First_Freq = [First_Freq_1]*len(S1_time)
                Second_Freq = [Second_Freq_1]*len(S1_time)
                Third_Freq = [Third_Freq_1]*len(S1_time)
                FFT_HR_1 = [FFT_HR]*len(S1_time)
                #######################################################################################################################################################################################
                'Adding Extracted Features to the DataFrame'
                #######################################################################################################################################################################################
                df_1 = pd.DataFrame(New_P_R_interval,columns=['P_R_interval'])
                df_2 = pd.DataFrame(New_QRS_interval,columns=['QRS_interval'])
                df_3 = pd.DataFrame(New_Q_T_interval,columns=['Q_T_interval'])
                df_4 = pd.DataFrame(New_T_P_interval,columns=['T_P_interval'])
                df_5 = pd.DataFrame(New_R_R_interval,columns=['R_R_interval'])
                df_6 = pd.DataFrame(New_ECG_PTT,columns=['ECG_PTT'])
                df_7 = pd.DataFrame(ST,columns=['Systolic_Time'])
                df_8 = pd.DataFrame(DT,columns=['Dystolic_Time'])
                df_9 = pd.DataFrame(S1_time,columns=['S1_time'])
                df_10 = pd.DataFrame(New_Ampli_Systolic,columns=['Ampli_Systolic'])
                df_11 = pd.DataFrame(New_Ampli_Foot,columns=['Ampli_Foot'])
                df_12 = pd.DataFrame(New_first_diff_max,columns=['dppg_max_Height'])
                df_13 = pd.DataFrame(New_first_diff_min,columns=['dppg_min_Height'])
                df_14 = pd.DataFrame(New_first_diff_width,columns=['dppgW'])
                df_15 = pd.DataFrame(New_Total_height_first_diff,columns=['dppgPH'])
                df_16 = pd.DataFrame(New_Second_diff_max,columns=['ddppg_max_Height'])
                df_16 = pd.DataFrame(New_Second_diff_min,columns=['ddppg_min_Height'])
                df_17 = pd.DataFrame(New_Second_diff_width,columns=['ddppgW'])
                df_18 = pd.DataFrame(New_Total_height_second_diff,columns=['ddppgPH'])
                df_19 = pd.DataFrame(PTT_PPG_Peak,columns=['PTT_PPG_Peak'])
                df_20 = pd.DataFrame(PTT_PPG_notch,columns=['PTT_PPG_notch'])
                df_21 = pd.DataFrame(PTT_PPG_diastolic,columns=['PTT_PPG_diastolic'])
                df_22 = pd.DataFrame(PTT_PPG_foot,columns=['PTT_PPG_foot'])
                df_23 = pd.DataFrame(S1_1,columns=['S1'])
                df_26 = pd.DataFrame(IPA,columns=['IPA'])
                df_27 = pd.DataFrame(Ratio_of_sys_time_Ampli,columns=['Ratio_of_sys_time_Ampli'])
                df_28 = pd.DataFrame(Ratio_of_sys_pulse_interval,columns=['Ratio_of_sys_pulse_interval'])
                df_29 = pd.DataFrame(New_PTT_F,columns=['PTT_F'])
                df_30 = pd.DataFrame(New_PTT_S,columns=['PTT_S'])
                df_31 = pd.DataFrame(New_PTT_P,columns=['PTT_P'])
                df_32 = pd.DataFrame(New_PTT_DN,columns=['PTT_DN'])
                df_33 = pd.DataFrame(New_PTT_DP,columns=['PTT_DP'])
                df_34 = pd.DataFrame(LASI,columns=['LASI'])
                df_35 = pd.DataFrame(AI,columns=['AI'])
                df_36 = pd.DataFrame(New_ECG_HR,columns=['ECG_HR'])
                df_37 = pd.DataFrame(PPG_HR,columns=['PPG_HR'])
                df_38 = pd.DataFrame(New_PIR,columns=['PIR'])
                df_39 = pd.DataFrame(Delta_T,columns=['Time_Dia_sys'])
                df_40 = pd.DataFrame(First_max,columns=['First_Magnitude'])
                df_41 = pd.DataFrame(Second_max,columns=['Second_magnitude'])
                df_42 = pd.DataFrame(Third_max,columns=['Third_magnitude'])
                df_43 = pd.DataFrame(First_Freq,columns=['First_Freq'])
                df_44 = pd.DataFrame(Second_Freq,columns=['Second_Freq'])
                df_45 = pd.DataFrame(Third_Freq,columns=['Third_Freq'])
                df_46 = pd.DataFrame(Mean,columns=['Mean'])
                df_47 = pd.DataFrame(Median,columns=['Median'])
                df_48= pd.DataFrame(Mode,columns=['Mode'])
                df_49 = pd.DataFrame(Variance,columns=['Variance'])
                df_50 = pd.DataFrame(Standard_Devation,columns=['Standard_Devation'])
                df_51 = pd.DataFrame(Minimum,columns=['Minimum_value'])
                df_52 = pd.DataFrame(Maximum,columns=['Maximum_Value'])
                df_53 = pd.DataFrame(Skewness,columns=['Skewness'])
                df_54 = pd.DataFrame(Kurtosis,columns=['Kurtosis'])
                df_55 = pd.DataFrame(Twenty_Fifth_Percentile,columns=['Twenty_Fifth_Percentile'])
                df_56 = pd.DataFrame(Fifty_Percentile,columns=['Fifty_Percentile'])
                df_57 = pd.DataFrame(Seventy_Fifth_Percentile,columns=['Seventy_Fifth_Percentile'])
                df_58 = pd.DataFrame(Shannon_Entropy,columns=['Shannon_Entropy'])
                df_59 = pd.DataFrame(quent_data,columns=['Cycle'])
                df_60 = pd.DataFrame(First_Phase_1,columns=['First_Phase'])
                df_61 = pd.DataFrame(Second_Phase_1,columns=['Second_Phase'])
                df_62 = pd.DataFrame(Third_Phase_1,columns=['Third_Phase'])
                df_63 = pd.DataFrame(S2_time,columns=['s2_time'])
                df_64 = pd.DataFrame(S3_time,columns=['s3_time'])
                df_65 = pd.DataFrame(S4_time,columns=['s4_time'])
                df_66 = pd.DataFrame(S5_time,columns=['s5_time'])
                df_67 = pd.DataFrame(S2_1,columns=['s2'])
                df_68 = pd.DataFrame(S3_1,columns=['s3'])
                df_69 = pd.DataFrame(S4_1,columns=['s4'])
                df_70 = pd.DataFrame(First_abs_max,columns=['First_abs_max'])
                df_71 = pd.DataFrame(First_abs_max_Left,columns=['First_abs_max_Left'])
                df_72 = pd.DataFrame(First_abs_max_Right,columns=['First_abs_max_Right'])
                df_73 = pd.DataFrame(Second_abs_max,columns=['Second_abs_max'])
                df_74 = pd.DataFrame(Second_abs_max_Left,columns=['Second_abs_max_Left'])
                df_75 = pd.DataFrame(Second_abs_max_Right,columns=['Second_abs_max_Right'])
                df_76 = pd.DataFrame(Third_abs_max,columns=['Third_abs_max'])
                df_77 = pd.DataFrame(Third_abs_max_Left,columns=['Third_abs_max_Left'])
                df_78 = pd.DataFrame(Third_abs_max_Right,columns=['Third_abs_max_Right'])
                df_79 = pd.DataFrame(FFT_HR_1,columns=['FFT_HR'])
                df_80 = pd.DataFrame(ECG_Corr_1,columns=['ECG_Corr'])
                df_81 = pd.DataFrame(PPG_Corr_1,columns=['PPG_Corr'])
                df_82 = pd.DataFrame(PTT_PPG_end_of_sys,columns=['PTT_PPG_end_of_sys'])
                df_83 = pd.DataFrame(sys_area,columns=['Systolic_area'])
                df_84 = pd.DataFrame(New_Corrected_QT_Bazett_formula,columns=['Corrected_QT_Bazett_formula'])
                df_85 = pd.DataFrame(New_Corrected_QT_from_Hodges_formula,columns=['Corrected_QT_from_Hodges_formula'])
                df_86 = pd.DataFrame(New_P_Duration,columns=['P_Duration'])
                df_87 = pd.DataFrame(New_T_Duration,columns=['T_Duration'])
                df_88 = pd.DataFrame(New_T_onset_to_Peak,columns=['T_onset_to_Peak'])
                df_89 = pd.DataFrame(New_T_peak_to_offset,columns=['New_T_peak_to_offset'])
                df_90 = pd.DataFrame(New_Ratio_of_onset_peak_to_offset,columns=['Ratio_of_onset_peak_to_offset'])
                df_1 = pd.concat([df_59,df_1,df_2,df_3,df_4,df_5,df_6,df_84,df_85,df_86,df_87,df_88,df_89,df_90,df_7,df_8,df_9,df_63,df_64,df_65,df_66,df_10,df_11,df_12,df_13,df_14,df_15,df_16,df_17,df_18,df_82,df_19,df_20,df_21,df_22,df_83,df_23,df_67,df_68,df_69,df_26,df_27,df_28,df_29,df_30,df_31,df_32,df_33,df_34,df_35,df_36,df_37,df_79,df_38,df_39,df_40,df_70,df_71,df_72,df_41,df_73,df_74,df_75,df_42,df_76,df_77,df_78,df_43,df_44,df_45,df_60,df_61,df_62,df_46,df_47,df_48,df_49,df_50,df_51,df_52,df_53,df_54,df_55,df_56,df_57,df_58,df_80,df_81],axis=1)
                df = df_1
                df = df[(df['P_R_interval']>0) & (df['QRS_interval']>0) & (df['Q_T_interval']>0) & (df['T_P_interval']>0) & (df['ECG_PTT']>0) & (df['PTT_F']>0) & (df['PTT_S']>0) & (df['PTT_P']>0) & (df['PTT_DN']>0) & (df['PTT_DP']>0)]
                df_4 = df
                data_1 = df_4.copy()
                data_2 = data_1.drop(['Cycle','ECG_Corr','PPG_Corr','LASI','dppgW','P_Duration','T_Duration','Ratio_of_onset_peak_to_offset','T_P_interval','FFT_HR','PTT_PPG_end_of_sys','Shannon_Entropy','Time_Dia_sys'],axis=1)
                columns = data_2.columns.tolist()
                index = []
                for i in columns:
                    Feature = data_1[i].tolist()
                    outlier = outlier_detection(Feature)
                    index.append([Feature.index(i) for i in outlier])
                index_1 = flatten(index)
                New_index = []
                for index in index_1:
                    if index not in New_index:
                        New_index.append(index)
                df4 = df_4.drop(df_4.index[i] for i in New_index)
                
                df3 = pd.DataFrame()
                if len(df4) >= 0 and len(df4) < 3:
                    df3 = df3.append(df_4,ignore_index=True)
                else:
                    df3 = df3.append(df4,ignore_index=True)
                print(df3)
                ##########################################################################################################################################################################################################################
                'Adding User info to the DataFrame'
                ##########################################################################################################################################################################################################################
                df_1 = df3
                # df_1['MSR_ID'] = userinfo[0]+'ASM_000'+str(i)
                df_1['USR_ID'] = userinfo[0]          
                df_1['AGE'] = userinfo[2] 
                df_1['GENDER'] = userinfo[1] 
                df_1['HEIGHT'] = userinfo[3] 
                df_1['WEIGHT'] = userinfo[4] 
                # df_1['BMI'] = BMI
                df_1['REF_SBP'] = userinfo[6] 
                df_1['REF_DBP'] = userinfo[7] 
                # excel_list.append(df_1)
                # HR_1 = df_1['ECG_HR'].tolist()
                # SBP_1 = df_1['REF_SBP'].tolist()
                # DBP_1 = df_1['REF_DBP'].tolist()
                # New_SBP = []
                # for  y in range(len(SBP_1)):
                #     a = HR_1[y]/SBP_1[y]
                #     sbp = a+SBP_1[y]
                #     New_SBP.append(sbp)
                # New_DBP = []
                # for  z in range(len(DBP_1)):
                #     b=(HR_1[z]/DBP_1[z])
                #     dbp=b+DBP_1[z]
                #     New_DBP.append(dbp)
                # df_1['New_SBP'] = New_SBP
                # df_1['New_DBP'] = New_DBP
                unique_msr_id = df_1.MSR_ID.unique()
                for a in range(len(unique_msr_id)):
                    df2 = df_1[df_1["MSR_ID"] == unique_msr_id[a]]
                    df2["average"] = df2["ECG_HR"].mean()
                    df2["average_plus_10"] = df2["average"].add(10)
                    df2["average_minus_10"] = df2["average"].sub(10)
                    df2["condition"] = np.where((df2["ECG_HR"] >= df2["average_minus_10"]) & (df2["ECG_HR"] <= df2["average_plus_10"]), df2["ECG_HR"], np.nan)
                    df_1["new_condition"] = np.where(((abs(df_1["ECG_HR"] - df_1["PPG_HR"])) <= 2), df_1["ECG_HR"], np.nan )
                    final_df = df_1.dropna()
                    final_df.drop(["new_condition"], axis=1, inplace=True)
                    excel_list.append(final_df)
                i+=1
            else:
                print('Bad_Quality')
        except:
            print('Error_data')
# #########################################################################################################################################################################################################################################
# 'Mergging All dataFrame to single Dataframe'
# #########################################################################################################################################################################################################################################
excel_merged_2 = pd.DataFrame()
for excel_file in excel_list:
    excel_merged_2 = excel_merged_2.append(excel_file, ignore_index=True)
excel_merged_2.sort_values(by=['Cycle'],ascending=True)
print(excel_merged_2)
excel_merged_2.to_excel(r'D:\working_directory\test_madhrai1k.xlsx')
# excel_merged_2.to_excel(r'D:\User_Trail_data\data_user_info\BP_Features_V5.4.3_21_03_2023_with_Zhang_fit_Upsampled_to_1KHZ.xlsx')
########################################################### END ##############################################################################################
        
        
            
        
        
        
        
        
        
        


    