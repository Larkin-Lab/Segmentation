#%% IMPORT DATA TABLE
import os 
# import math
import pandas as pd
# from skimage.filters import  median
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import copy
import datetime 
# import cv2
import seaborn as sns
import cmapy
from skimage.feature import peak_local_max
from scipy.signal import find_peaks
from scipy import stats
import math
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed
from scipy.fft import fft, fftfreq

# bert
bert_trap_pix = 124

trap_um = 200

bert_um2_per_pix = (trap_um / bert_trap_pix) **2

perseph_trap_pix = 60

perseph_um2_per_pix = (trap_um / perseph_trap_pix) **2
# %% IMPORT DATA F0230
# file number
file_num = '010'
# experiment number
exp = 'F0230'
# chamber
ch = "ch1"
strain= 'wt'
kcon = '1x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0230_ch1_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0230_ch1_ID_count = max(F0230_ch1_tracking_table["Track ID"].tolist()) +1
F0230_ch1_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0230_ch1_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0230_ch1_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0230_ch1_tracking_table.insert(3, 'kcon', '{}'.format(kcon))
## %% IMPORT DATA
# file number
file_num = '011'
# experiment number
exp = 'F0230'
# chamber
ch = "ch2"
strain= 'wt'
kcon = '1x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
# os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0230_ch2_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0230_ch2_ID_count = max(F0230_ch2_tracking_table["Track ID"].tolist()) +1
F0230_ch2_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0230_ch2_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0230_ch2_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0230_ch2_tracking_table.insert(3, 'kcon', '{}'.format(kcon))
## %% IMPORT DATA
# file number
file_num = '012'
# experiment number
exp = 'F0230'
# chamber
ch = "ch3"
strain= 'wt'
kcon = '5x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
# os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0230_ch3_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0230_ch3_ID_count = max(F0230_ch3_tracking_table["Track ID"].tolist()) +1
F0230_ch3_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0230_ch3_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0230_ch3_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0230_ch3_tracking_table.insert(3, 'kcon', '{}'.format(kcon))
## %% IMPORT DATA
# file number
file_num = '013'
# experiment number
exp = 'F0230'
# chamber
ch = "ch4"
strain= 'wt'
kcon = '5x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
# os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0230_ch4_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0230_ch4_ID_count = max(F0230_ch4_tracking_table["Track ID"].tolist()) +1
F0230_ch4_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0230_ch4_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0230_ch4_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0230_ch4_tracking_table.insert(3, 'kcon', '{}'.format(kcon))
# %%
tables = [F0230_ch1_tracking_table, F0230_ch2_tracking_table, F0230_ch3_tracking_table, F0230_ch4_tracking_table]
F0230_df1 = pd.concat(tables)
F0230_df = F0230_df1.loc[(F0230_df1['Time pt'] >= 9) & (F0230_df1['Time pt'] <= 120+15 + 9)] 
F0230_df['Time pt'] = F0230_df['Time pt'] - 9
F0230_df['area'] = (F0230_df['area'] / 4)*perseph_um2_per_pix
F0230_df['strip_area'] = F0230_df['strip_area'] / 4

# %% IMPORT DATA F0261
# file number
file_num = '001'
# experiment number
exp = 'F0261'
# chamber
ch = "ch1"
strain= 'wt'
kcon = '1x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0261_ch1_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0261_ch1_ID_count = max(F0261_ch1_tracking_table["Track ID"].tolist()) +1
F0261_ch1_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0261_ch1_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0261_ch1_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0261_ch1_tracking_table.insert(3, 'kcon', '{}'.format(kcon))
## %% IMPORT DATA
# file number
file_num = '002'
# experiment number
exp = 'F0261'
# chamber
ch = "ch2"
strain= 'wt'
kcon = '5x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
# os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0261_ch2_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0261_ch2_ID_count = max(F0261_ch2_tracking_table["Track ID"].tolist()) +1
F0261_ch2_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0261_ch2_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0261_ch2_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0261_ch2_tracking_table.insert(3, 'kcon', '{}'.format(kcon))
## %% IMPORT DATA
# file number
file_num = '003'
# experiment number
exp = 'F0261'
# chamber
ch = "ch3"
strain= 'ktrD'
kcon = '1x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
# os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0261_ch3_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0261_ch3_ID_count = max(F0261_ch3_tracking_table["Track ID"].tolist()) +1
F0261_ch3_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0261_ch3_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0261_ch3_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0261_ch3_tracking_table.insert(3, 'kcon', '{}'.format(kcon))
## %% IMPORT DATA
# file number
file_num = '004'
# experiment number
exp = 'F0261'
# chamber
ch = "ch4"
strain= 'ktrD'
kcon = '5x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
# os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0261_ch4_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0261_ch4_ID_count = max(F0261_ch4_tracking_table["Track ID"].tolist()) +1
F0261_ch4_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0261_ch4_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0261_ch4_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0261_ch4_tracking_table.insert(3, 'kcon', '{}'.format(kcon))
# %%
tables = [F0261_ch1_tracking_table, F0261_ch2_tracking_table, F0261_ch3_tracking_table, F0261_ch4_tracking_table]
F0261_df1 = pd.concat(tables)
F0261_df = F0261_df1.loc[(F0261_df1['Time pt'] <= 120+15)]# & (F0261_df1['Time pt'] >= 9)] 
F0261_df['area'] = F0261_df['area'] *perseph_um2_per_pix
# %% IMPORT DATA F0269
# file number
file_num = '000'
# experiment number
exp = 'F0269'
# chamber
ch = "ch1"
strain= 'wt'
kcon = '1x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0269_ch1_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0269_ch1_ID_count = max(F0269_ch1_tracking_table["Track ID"].tolist()) +1
F0269_ch1_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0269_ch1_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0269_ch1_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0269_ch1_tracking_table.insert(3, 'kcon', '{}'.format(kcon))
## %% IMPORT DATA
# file number
file_num = '001'
# experiment number
exp = 'F0269'
# chamber
ch = "ch2"
strain= 'wt'
kcon = '5x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
# os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0269_ch2_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0269_ch2_ID_count = max(F0269_ch2_tracking_table["Track ID"].tolist()) +1
F0269_ch2_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0269_ch2_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0269_ch2_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0269_ch2_tracking_table.insert(3, 'kcon', '{}'.format(kcon))
## %% IMPORT DATA
# file number
file_num = '002'
# experiment number
exp = 'F0269'
# chamber
ch = "ch3"
strain= 'ktrB'
kcon = '1x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
# os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0269_ch3_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0269_ch3_ID_count = max(F0269_ch3_tracking_table["Track ID"].tolist()) +1
F0269_ch3_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0269_ch3_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0269_ch3_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0269_ch3_tracking_table.insert(3, 'kcon', '{}'.format(kcon))
## %% IMPORT DATA
# file number
file_num = '003'
# experiment number
exp = 'F0269'
# chamber
ch = "ch4"
strain= 'ktrB'
kcon = '5x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
# os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0269_ch4_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0269_ch4_ID_count = max(F0269_ch4_tracking_table["Track ID"].tolist()) +1
F0269_ch4_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0269_ch4_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0269_ch4_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0269_ch4_tracking_table.insert(3, 'kcon', '{}'.format(kcon))
# %%
tables = [F0269_ch1_tracking_table, F0269_ch2_tracking_table, F0269_ch3_tracking_table, F0269_ch4_tracking_table]
F0269_df1 = pd.concat(tables)
F0269_df = F0269_df1.loc[(F0269_df1['Time pt'] <= 120+15) & (F0269_df1['Time pt'] >= 9)] 
F0269_df['area'] = F0269_df['area'] *perseph_um2_per_pix

# %% IMPORT DATA F0265
# file number
file_num = '000'
# experiment number
exp = 'F0265'
# chamber
ch = "ch1.1"
strain= 'wt'
kcon = '1x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0265_ch1_1_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0265_ch1_1_ID_count = max(F0265_ch1_1_tracking_table["Track ID"].tolist()) +1
F0265_ch1_1_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0265_ch1_1_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0265_ch1_1_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0265_ch1_1_tracking_table.insert(3, 'kcon', '{}'.format(kcon))
# #%% IMPORT DATA
# file number
file_num = '000'
# experiment number
exp = 'F0265'
# chamber
ch = "ch1.3"
strain= 'wt'
kcon = '1x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
# os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0265_ch1_3_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0265_ch1_3_ID_count = max(F0265_ch1_3_tracking_table["Track ID"].tolist()) +1
F0265_ch1_3_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0265_ch1_3_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0265_ch1_3_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0265_ch1_3_tracking_table.insert(3, 'kcon', '{}'.format(kcon))

## %% IMPORT DATA
# file number
file_num = '001'
# experiment number
exp = 'F0265'
# chamber
ch = "ch2.1"
strain= 'wt'
kcon = '5x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
# os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0265_ch2_1_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0265_ch2_1_ID_count = max(F0265_ch2_1_tracking_table["Track ID"].tolist()) +1
F0265_ch2_1_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0265_ch2_1_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0265_ch2_1_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0265_ch2_1_tracking_table.insert(3, 'kcon', '{}'.format(kcon))

## %% IMPORT DATA
# file number
file_num = '001'
# experiment number
exp = 'F0265'
# chamber
ch = "ch2.2"
strain= 'wt'
kcon = '5x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
# os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0265_ch2_2_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0265_ch2_2_ID_count = max(F0265_ch2_2_tracking_table["Track ID"].tolist()) +1
F0265_ch2_2_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0265_ch2_2_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0265_ch2_2_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0265_ch2_2_tracking_table.insert(3, 'kcon', '{}'.format(kcon))

## %% IMPORT DATA
# file number
file_num = '002'
# experiment number
exp = 'F0265'
# chamber
ch = "ch3.1"
strain= 'ktrB'
kcon = '1x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
# os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0265_ch3_1_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0265_ch3_1_ID_count = max(F0265_ch3_1_tracking_table["Track ID"].tolist()) +1
F0265_ch3_1_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0265_ch3_1_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0265_ch3_1_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0265_ch3_1_tracking_table.insert(3, 'kcon', '{}'.format(kcon))

## %% IMPORT DATA
# file number
file_num = '002'
# experiment number
exp = 'F0265'
# chamber
ch = "ch3.2"
strain= 'ktrB'
kcon = '1x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
# os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0265_ch3_2_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0265_ch3_2_ID_count = max(F0265_ch3_2_tracking_table["Track ID"].tolist()) +1
F0265_ch3_2_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0265_ch3_2_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0265_ch3_2_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0265_ch3_2_tracking_table.insert(3, 'kcon', '{}'.format(kcon))

## %% IMPORT DATA
# file number
file_num = '003'
# experiment number
exp = 'F0265'
# chamber
ch = "ch4.1"
strain= 'ktrB'
kcon = '5x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
# os.chdir('/Volumes/MPK_sandisk\Experiments\{}\tracking tables'.format(exp))
F0265_ch4_1_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0265_ch4_1_ID_count = max(F0265_ch4_1_tracking_table["Track ID"].tolist()) +1
F0265_ch4_1_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0265_ch4_1_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0265_ch4_1_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0265_ch4_1_tracking_table.insert(3, 'kcon', '{}'.format(kcon))

## %% IMPORT DATA
# file number
file_num = '003'
# experiment number
exp = 'F0265'
# chamber
ch = "ch4.2"
strain= 'ktrB'
kcon = '5x'
# os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
# os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))
F0265_ch4_2_tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch), usecols = ['Time pt', 'Centroid', 'Track ID', 'strip_radii',
       'strip_pixels', 'strip_area', 'mean_CFP', 'mean_edge_CFP', 'area'])
F0265_ch4_2_ID_count = max(F0265_ch4_2_tracking_table["Track ID"].tolist()) +1
F0265_ch4_2_tracking_table.insert(0, 'Exp', '{}'.format(exp))
F0265_ch4_2_tracking_table.insert(1, 'ch', '{}'.format(ch))
F0265_ch4_2_tracking_table.insert(2, 'strain', '{}'.format(strain))
F0265_ch4_2_tracking_table.insert(3, 'kcon', '{}'.format(kcon))
# %%
tables = [F0265_ch1_1_tracking_table, F0265_ch1_3_tracking_table, F0265_ch2_1_tracking_table, F0265_ch2_2_tracking_table, F0265_ch3_1_tracking_table, 
          F0265_ch3_2_tracking_table, F0265_ch4_1_tracking_table, F0265_ch4_2_tracking_table]
F0265_df1 = pd.concat(tables)
F0265_df = F0265_df1.loc[(F0265_df1['Time pt'] <= 120+15) & (F0265_df1['Time pt'] >= 9)] 
F0265_df['mean_edge_CFP'] = F0265_df['mean_edge_CFP'] / 8
F0265_df['area'] = F0265_df['area'] *bert_um2_per_pix


# %%
tables_all = [F0230_df, F0261_df, F0269_df, F0265_df]
all_df = pd.concat(tables_all)
# %% extract dif groups

wt_1xk = all_df[(all_df['strain'] == 'wt') & (all_df['kcon'] == '1x')]
wt_5xk = all_df[(all_df['strain'] == 'wt') & (all_df['kcon'] == '5x')]
ktrD_1xk = all_df[(all_df['strain'] == 'ktrD') & (all_df['kcon'] == '1x')]
ktrD_5xk = all_df[(all_df['strain'] == 'ktrD') & (all_df['kcon'] == '5x')]
ktrB_1xk = all_df[(all_df['strain'] == 'ktrB') & (all_df['kcon'] == '1x')]
ktrB_5xk = all_df[(all_df['strain'] == 'ktrB') & (all_df['kcon'] == '5x')]

F0230_t = all_df[(all_df['Exp'] == 'F0230') & (all_df['ch'] == 'ch1') & (all_df['Track ID'] == 0)]['Time pt']
F0261_t = all_df[(all_df['Exp'] == 'F0261') & (all_df['ch'] == 'ch1') & (all_df['Track ID'] == 0)]['Time pt']
F0269_t = all_df[(all_df['Exp'] == 'F0269') & (all_df['ch'] == 'ch1') & (all_df['Track ID'] == 0)]['Time pt']
F0265_t = all_df[(all_df['Exp'] == 'F0265') & (all_df['ch'] == 'ch1.1') & (all_df['Track ID'] == 0)]['Time pt']

# %% assign groups
# os.chdir(r'F:\GRC_2022\graphs')
os.chdir(r'/Users/danielkostman/Documents/maya testos/Larkin Lab/GRC Conference 2022/graphs')

img_freq = 5
line_width = 3
labels = ['colony 1', 'colony 2']

# # wt_1xk
# group = 'WT 1xK'
# group_save = 'WT 1xK all'
# # group_a = all_df.loc[(all_df['Exp'] == 'F0230') & (all_df['ch'] == 'ch1') & (all_df['Track ID'] == 0)]
# # group_b = all_df.loc[(all_df['Exp'] == 'F0230') & (all_df['ch'] == 'ch1') & (all_df['Track ID'] == 1)]
# # group_c = all_df.loc[(all_df['Exp'] == 'F0230') & (all_df['ch'] == 'ch2') & (all_df['Track ID'] == 0)]
# # group_d = all_df.loc[(all_df['Exp'] == 'F0230') & (all_df['ch'] == 'ch2') & (all_df['Track ID'] == 1)]
# group_e = all_df.loc[(all_df['Exp'] == 'F0261') & (all_df['ch'] == 'ch1') & (all_df['Track ID'] == 0)]
# group_f = all_df.loc[(all_df['Exp'] == 'F0261') & (all_df['ch'] == 'ch1') & (all_df['Track ID'] == 1)]
# # group_g = all_df.loc[(all_df['Exp'] == 'F0269') & (all_df['ch'] == 'ch1') & (all_df['Track ID'] == 0)]
# # group_h = all_df.loc[(all_df['Exp'] == 'F0269') & (all_df['ch'] == 'ch1') & (all_df['Track ID'] == 1)]
# group_i = all_df.loc[(all_df['Exp'] == 'F0265') & (all_df['ch'] == 'ch1.1') & (all_df['Track ID'] == 0)]
# group_j = all_df.loc[(all_df['Exp'] == 'F0265') & (all_df['ch'] == 'ch1.3') & (all_df['Track ID'] == 1)]
# # all_groups = [group_a, group_b, group_c, group_d, group_e, group_f, group_g, group_h, group_i, group_j]
# all_groups = [group_e, group_f, group_i, group_j]
# # labels = ['F0230-1.0', 'F0230-1.1', 'F0230-2.0', 'F0230-2.1', 'F0261-1', 'F0261-2', 'F0269-1', 'F0269-2', 'F0265-1', 'F0265-2']
# labels = ['F0261-1', 'F0261-2', 'F0265-1', 'F0265-2']
# times = [F0230_t]
# colors = ['grey', 'grey', 'black', 'black']

# # wt_5xk
# group = 'WT 5xK'
# group_save = 'WT 5xK all'
# # group_a = all_df.loc[(all_df['Exp'] == 'F0230') & (all_df['ch'] == 'ch3') & (all_df['Track ID'] == 0)]
# # group_b = all_df.loc[(all_df['Exp'] == 'F0230') & (all_df['ch'] == 'ch3') & (all_df['Track ID'] == 1)]
# # group_c = all_df.loc[(all_df['Exp'] == 'F0230') & (all_df['ch'] == 'ch4') & (all_df['Track ID'] == 0)]
# # group_d = all_df.loc[(all_df['Exp'] == 'F0230') & (all_df['ch'] == 'ch4') & (all_df['Track ID'] == 1)]
# group_e = all_df.loc[(all_df['Exp'] == 'F0261') & (all_df['ch'] == 'ch2') & (all_df['Track ID'] == 0)]
# group_f = all_df.loc[(all_df['Exp'] == 'F0261') & (all_df['ch'] == 'ch2') & (all_df['Track ID'] == 1)]
# # group_g = all_df.loc[(all_df['Exp'] == 'F0269') & (all_df['ch'] == 'ch2') & (all_df['Track ID'] == 0)]
# # group_h = all_df.loc[(all_df['Exp'] == 'F0269') & (all_df['ch'] == 'ch2') & (all_df['Track ID'] == 1)]
# group_i = all_df.loc[(all_df['Exp'] == 'F0265') & (all_df['ch'] == 'ch2.1') & (all_df['Track ID'] == 0)]
# group_j = all_df.loc[(all_df['Exp'] == 'F0265') & (all_df['ch'] == 'ch2.2') & (all_df['Track ID'] == 0)]
# # all_groups = [group_a, group_b, group_c, group_d, group_e, group_f, group_g, group_h, group_i, group_j]
# all_groups = [group_e, group_f, group_i, group_j]
# # labels = ['F0230-1.0', 'F0230-1.1', 'F0230-2.0', 'F0230-2.1', 'F0261-1', 'F0261-2', 'F0269-1', 'F0269-2', 'F0265-1', 'F0265-2']
# # labels = ['F0261-1', 'F0261-2', 'F0265-1', 'F0265-2']
# times = [F0230_t]
# colors = ['grey', 'grey', 'black', 'black']


# wt_1xk
group = 'WT 1xK'
group_save = 'WT 1xK 261'
group_e = all_df.loc[(all_df['Exp'] == 'F0261') & (all_df['ch'] == 'ch1') & (all_df['Track ID'] == 0)]
group_f = all_df.loc[(all_df['Exp'] == 'F0261') & (all_df['ch'] == 'ch1') & (all_df['Track ID'] == 1)]
all_groups = [group_e, group_f]
# labels = ['F0230-1.0', 'F0230-1.1', 'F0230-2.0', 'F0230-2.1', 'F0261-1', 'F0261-2', 'F0269-1', 'F0269-2', 'F0265-1', 'F0265-2']
times = [F0261_t]
colors = ['darkgrey', 'dimgrey']

# # wt_1xk
# group = 'WT 1xK'
# group_save = 'WT 1xK 265'
# group_i = all_df.loc[(all_df['Exp'] == 'F0265') & (all_df['ch'] == 'ch1.1') & (all_df['Track ID'] == 0)]
# group_j = all_df.loc[(all_df['Exp'] == 'F0265') & (all_df['ch'] == 'ch1.3') & (all_df['Track ID'] == 1)]
# all_groups = [group_i, group_j]
# # labels = ['F0230-1.0', 'F0230-1.1', 'F0230-2.0', 'F0230-2.1', 'F0261-1', 'F0261-2', 'F0269-1', 'F0269-2', 'F0265-1', 'F0265-2']
# times = [F0265_t]
# colors = ['darkgrey', 'dimgrey']


# # wt_5xk
# group = 'WT 5xK'
# group_save = 'WT 5xK 261'
# group_e = all_df.loc[(all_df['Exp'] == 'F0261') & (all_df['ch'] == 'ch2') & (all_df['Track ID'] == 0)]
# group_f = all_df.loc[(all_df['Exp'] == 'F0261') & (all_df['ch'] == 'ch2') & (all_df['Track ID'] == 1)]
# all_groups = [group_e, group_f]
# times = [F0261_t]
# colors = ['darkgrey', 'dimgrey']

# # wt_5xk
# group = 'WT 5xK'
# group_save = 'WT 5xK 265'
# group_i = all_df.loc[(all_df['Exp'] == 'F0265') & (all_df['ch'] == 'ch2.1') & (all_df['Track ID'] == 0)]
# group_j = all_df.loc[(all_df['Exp'] == 'F0265') & (all_df['ch'] == 'ch2.2') & (all_df['Track ID'] == 0)]
# all_groups = [group_i, group_j]
# times = [F0265_t]
# colors = ['darkgrey', 'dimgrey']

# # ktrD 1xk
# group = 'ktrD 1xK'
# group_save = group+'261'
# group_a = all_df.loc[(all_df['Exp'] == 'F0261') & (all_df['ch'] == 'ch3') & (all_df['Track ID'] == 0)]
# group_b = all_df.loc[(all_df['Exp'] == 'F0261') & (all_df['ch'] == 'ch3') & (all_df['Track ID'] == 1)]
# all_groups = [group_a, group_b]
# # labels = ['biofilm 1', 'biofilm 2']
# times = [F0261_t]
# colors = ['mediumturquoise', 'paleturquoise']


# # ktrD 5x
# group = 'ktrD 5xK'
# group_save = group+'261'
# group_a = all_df.loc[(all_df['Exp'] == 'F0261') & (all_df['ch'] == 'ch4') & (all_df['Track ID'] == 0)]
# group_b = all_df.loc[(all_df['Exp'] == 'F0261') & (all_df['ch'] == 'ch4') & (all_df['Track ID'] == 1)]
# all_groups = [group_a, group_b]
# # labels = ['F0261-1', 'F0261-2']
# times = [F0261_t]
# colors = ['mediumturquoise', 'paleturquoise']


# # ktrAB 1xk
# group = 'ktrAB 1xK'
# group_save = group+'265'
# # group_a = all_df.loc[(all_df['Exp'] == 'F0269') & (all_df['ch'] == 'ch3') & (all_df['Track ID'] == 0)]
# # group_b = all_df.loc[(all_df['Exp'] == 'F0269') & (all_df['ch'] == 'ch3') & (all_df['Track ID'] == 1)]
# group_c = all_df.loc[(all_df['Exp'] == 'F0265') & (all_df['ch'] == 'ch3.1') & (all_df['Track ID'] == 0)]
# group_d = all_df.loc[(all_df['Exp'] == 'F0265') & (all_df['ch'] == 'ch3.2') & (all_df['Track ID'] == 0)]
# # all_groups = [group_a, group_b, 
# all_groups = [group_c, group_d]
# # labels = ['F0269-1', 'F0269-2', 'F0265-1', 'F0265-2']
# # labels = ['F0265-1', 'F0265-2']
# times = [F0265_t]
# colors = ['mediumseagreen', 'lightgreen']


# # ktrAB 5x
# group = 'ktrAB 5xK'
# group_save = group+'265'
# # group_a = all_df.loc[(all_df['Exp'] == 'F0269') & (all_df['ch'] == 'ch4') & (all_df['Track ID'] == 0)]
# # group_b = all_df.loc[(all_df['Exp'] == 'F0269') & (all_df['ch'] == 'ch4') & (all_df['Track ID'] == 1)]
# group_c = all_df.loc[(all_df['Exp'] == 'F0265') & (all_df['ch'] == 'ch4.1') & (all_df['Track ID'] == 0)]
# group_d = all_df.loc[(all_df['Exp'] == 'F0265') & (all_df['ch'] == 'ch4.2') & (all_df['Track ID'] == 0)]
# # all_groups = [group_a, group_b, group_c, group_d]
# all_groups = [group_c, group_d]
# # labels = ['F0269-1', 'F0269-2', 'F0265-1', 'F0265-2']
# # labels = ['F0265-1', 'F0265-2']
# times = [F0265_t]
# colors = ['mediumseagreen', 'lightgreen']


## %% THT ROLLING AVE
if '261' in group_save:
    ylimfour = [0, 55]
    ylimtht = [-150, 160]
    yscale = np.arange(0, 55, step=10)
    window0 = 20
    window1 = 2
if '265' in group_save:
    ylimfour = [0, 28]
    ylimtht = [-150, 220]
    yscale = np.arange(0, 28, step=5)
    window0 = 15
    window1 = 2
    
value = 'mean_edge_CFP'
mean_edge_CFP_roll_ave_w1 = []
mean_edge_CFP_roll_ave = []
mean_edge_CFP_mean_sub = []
mean_edge_CFP_mean_sub_w0w1 = []

for k in range(0, len(all_groups)):
    group_mean_w1 = all_groups[k]['{}'.format(value)].rolling(window1, center=True).mean()
    mean_edge_CFP_roll_ave_w1.append(group_mean_w1)
    
    group_mean_w0 = all_groups[k]['{}'.format(value)].rolling(window0, center=True).mean()
    mean_edge_CFP_roll_ave.append(group_mean_w0)
    
    group_mean_sub = all_groups[k]['{}'.format(value)].subtract(group_mean_w0)
    mean_edge_CFP_mean_sub.append(group_mean_sub)
    
    group_mean_sub_w0_w1 = group_mean_w1.subtract(group_mean_w0)
    mean_edge_CFP_mean_sub_w0w1.append(group_mean_sub_w0_w1)    
    
plt.figure(figsize=(7, 4))
for i in range(0, len(all_groups)):
    plt.plot(times[0]*img_freq/60, mean_edge_CFP_mean_sub_w0w1[i], label = labels[i], color = colors[i], linewidth=line_width)

plt.xlabel('Time (hours)')
plt.ylabel('Intensity')
plt.title('{} Detrended ThT'.format(group))
plt.xlim([0, 10])
plt.ylim(ylimtht)
plt.legend(loc='best')
# plt.show()
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = "Arial"
plt.savefig('{} mean_edge_CFP Mean-Sub.pdf'.format(group_save), bbox_inches='tight')
plt.close()


## %% MEAN EDGE CFP FOURIER

def fourier_fun(column):
    data = column.dropna()
    # Number of sample points
    N = len(data)
    # sample spacing
    T = img_freq / 60.0
    x = np.linspace(0.0, N*T, N)
    y = data.tolist()
    yf = fft(y)
    xf = fftfreq(N, T)[:N//2]
    return xf, yf, N

ft_x = []
ft_y = []
N_all = []

for i in range(0, len(all_groups)):
    xf, yf, N = fourier_fun(mean_edge_CFP_mean_sub_w0w1[i])
    ft_x.append(xf)
    ft_y.append(yf)
    N_all.append(N)

plt.figure(figsize=(7.1, 4))
for i in range(0, len(all_groups)):
    plt.plot(ft_x[i], 2.0/N_all[i] * np.abs(ft_y[i][0:N_all[i]//2]),linewidth=line_width, label = labels[i], color = colors[i])
plt.xlabel('Freq. (1/hours)')
plt.ylabel('Fourier Magnitude')
plt.title('{} Fourier ThT'.format(group))
plt.axvline(x = 1/2, linewidth = 2, linestyle ="--", color ='black', label = '2hr period')
plt.xlim([0, 3])
plt.ylim(ylimfour)
plt.yticks(yscale)
plt.legend(loc='best')
# plt.figure(figsize=(5, 5))

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = "Arial"
plt.savefig('{} ThT Fourier.pdf'.format(group_save), bbox_inches='tight')

## %%
# value = 'area'

# for i in range(0, len(all_groups)):
#     plt.plot(times[0]*img_freq/60, all_groups[i]['{}'.format(value)], label = labels[i], color = colors[i], linewidth=line_width)
# plt.xlabel('Time (hours)')
# plt.ylabel('BF Area (um2)')
# plt.title('{} Area raw'.format(group))
# plt.xlim([0, 10])
# # plt.ylim([0, 70000])
# # plt.ylim(-200, 20000)
# # plt.ylim(0, 70000)
# plt.legend(loc='best')

# plt.show()
# plt.savefig('{} Area raw.jpg'.format(group_save), bbox_inches='tight')
# plt.close()

# # %%
# value = 'strip_area'

# for i in range(0, len(all_groups)):
#     plt.plot(times[0]*img_freq/60, all_groups[i]['{}'.format(value)], label = labels[i])
# plt.xlabel('Time (hours)')
# plt.ylabel('Size')
# plt.title('strip_area raw'.format(exp,ch))
# plt.legend(loc='best')
# # plt.show()
# plt.savefig('{} strip_area raw.jpg'.format(group), bbox_inches='tight')

# # %%# edge CFP graph all together
# value = 'mean_edge_CFP'

# for i in range(0, len(all_groups)):
#     plt.plot(times[0]*img_freq/60, all_groups[i]['{}'.format(value)], label = labels[i], color = colors[i], linewidth=line_width)
# plt.xlabel('Time (hours)')
# plt.ylabel('Intesity')
# plt.title('{} ThT raw'.format(group))
# plt.legend(loc='best')
# plt.xlim([0, 10])
# # plt.ylim([0, 1000])
# # plt.show()
# # plt.savefig('{} mean_edge_CFP raw.jpg'.format(group_save), bbox_inches='tight')
# # plt.close()

# # %%  AREA ROLLIN AVE

# window0 = 20
# window1 = 2

# value = 'area'
# area_roll_ave_w1 = []
# area_roll_ave = []
# area_mean_sub = []
# area_mean_sub_w0w1 = []

# for k in range(0, len(all_groups)):
#     group_mean_w1 = all_groups[k]['{}'.format(value)].rolling(window1, center=True).mean()
#     area_roll_ave_w1.append(group_mean_w1)
    
#     group_mean_w0 = all_groups[k]['{}'.format(value)].rolling(window0, center=True).mean()
#     area_roll_ave.append(group_mean_w0)
    
#     group_mean_sub = all_groups[k]['{}'.format(value)].subtract(group_mean_w0)
#     area_mean_sub.append(group_mean_sub)
    
#     group_mean_sub_w0_w1 = group_mean_w1.subtract(group_mean_w0)
#     area_mean_sub_w0w1.append(group_mean_sub_w0_w1)    
    

# for i in range(0, len(all_groups)):
#     plt.plot(times[0]*img_freq/60, area_mean_sub_w0w1[i], label = labels[i], color = colors[i], linewidth=line_width)
    
# plt.xlabel('Time (hours)')
# plt.ylabel('Mean-Sub Area (um2)')
# plt.title('{} Mean-Sub Area'.format(group))
# plt.legend(loc='best')
# plt.xlim([0, 10])
# # plt.show()
# # plt.savefig('{} Area Mean-Sub.jpg'.format(group_save), bbox_inches='tight')
# # plt.close()

## %% THT VS AREA

# bf = 0
# bf_table = all_groups[bf]

# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True

# ax1 = plt.subplot()
# l1, = ax1.plot(bf_table['Time pt']*img_freq/60, mean_edge_CFP_mean_sub_w0w1[bf], color=colors[bf], linewidth=line_width)
# ax2 = ax1.twinx()
# l2, = ax2.plot(bf_table['Time pt']*img_freq/60, area_mean_sub_w0w1[bf], color='darkgoldenrod', linewidth=line_width)
# plt.title('{} Mean-Sub ThT and Area'.format(group))
# plt.legend([l1, l2], ["ThT", "area"])
# ax1.set_ylabel('Intensity', color='cyan')
# ax2.set_ylabel('Area (um2)', color='darkgoldenrod')
# ax1.set_xlabel('Time (hours)')
# plt.xlim([0, 10])
# # plt.savefig('{} Mean-Sub ThT and Area.jpg'.format(group_save), bbox_inches='tight')
# # # plt.show()
# # plt.close()

# # %% AREA FOURIER


# def fourier_fun(column):
#     data = column.dropna()
#     # Number of sample points
#     N = len(data)
#     # sample spacing
#     T = img_freq / 60.0
#     x = np.linspace(0.0, N*T, N)
#     y = data.tolist()
#     yf = fft(y)
#     xf = fftfreq(N, T)[:N//2]
#     return xf, yf, N

# ft_x = []
# ft_y = []
# N_all = []
# for i in range(0, len(all_groups)):
#     xf, yf, N = fourier_fun(area_mean_sub_w0w1[i])
#     ft_x.append(xf)
#     ft_y.append(yf)
#     N_all.append(N)
    
# for i in range(0, len(all_groups)):
#     plt.plot(ft_x[i], 2.0/N_all[i] * np.abs(ft_y[i][0:N_all[i]//2]),linewidth=line_width, label = labels[i], color = colors[i])
# plt.xlabel('Freq. (1/hours)')
# plt.ylabel('Fourier Magnitude')
# plt.title('{} Fourier Area'.format(group))
# plt.axvline(x = 1/2, linewidth = 2, linestyle ="--", color ='black', label = '2hr period')
# plt.xlim([0, 5])
# plt.legend(loc='best')
# # plt.savefig('{} Area Fourier.jpg'.format(group_save), bbox_inches='tight')
# # plt.close()
