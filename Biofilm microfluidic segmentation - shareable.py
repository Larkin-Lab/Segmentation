#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 13:07:09 2022

@author: danielkostman
"""
import os 
# import math
import pandas as pd
from skimage.filters import  median
from matplotlib import pyplot as plt
import numpy as np
import copy
import datetime 
# import cv2
import seaborn as sns
import cmapy
# from skimage.feature import canny
# from skimage import data, img_as_float
from skimage import io, measure
from skimage import exposure
# from skimage.morphology import disk
from skimage.filters import sobel
from skimage import segmentation
# from skimage.filters import threshold_otsu
# from skimage.segmentation import clear_border
# from skimage.measure import label, regionprops
from skimage.morphology import  binary_erosion, remove_small_holes, remove_small_objects, disk
# from skimage.color import label2rgb
from skimage.feature import peak_local_max
from scipy.signal import find_peaks
from scipy import stats
import math
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed
# from sklearn.linear_model import LinearRegression

# from scipy import ndimage as ndi
# from skimage.morphology import disk
# from skimage.filters import rank
# from skimage.util import img_as_ubyte

# from scipy.spatial.distance import cdist
# from scipy import ndimage as ndi
# os.chdir('/Users/danielkostman/Documents/maya testos/Sgro Lab/scripts/Microfluidic segmentation')
os.chdir('F:\scripts\Microfluidic segmentation')

from centroid_tracker import CentroidTracker, feature_dist
# os.chdir('/Users/danielkostman/')
# os.chdir('C:/')
# %% IMPORT DATA
# file number
file_num = '010'
# experiment number
exp = 'F0230'
# chamber
ch = "ch1"
# time btw imaging
img_freq = 5

# graph_directory = ('/Users/danielkostman/Documents/maya testos/Larkin Lab/experiments/F0230/binned/graphs')
graph_directory = ('F:\Experiments\{}\graphs'.format(exp))
# graph_directory = os.chdir('/Volumes/MPK_sandisk/Experiments/{}/graphs'.format(exp))
# /Volumes/MPK_sandisk/Experiments/F0230

# # os.chdir('/Users/danielkostman/Documents/maya testos/Larkin Lab/experiments/F0230/binned')
# os.chdir('/Volumes/MPK_sandisk/Experiments/{}/processed tifs'.format(exp))
os.chdir(r'F:\Experiments\{}\processed tifs'.format(exp))
# os.chdir('/Users/danielkostman/Dropbox/F0194_03312022/cropped tif')

phase = io.imread('{}_{}_C0001.tif'.format(exp,file_num))
CFP = io.imread('{}_{}_C0002.tif'.format(exp,file_num))


# # # os.chdir('/Users/danielkostman/Documents/maya testos/Larkin Lab/experiments/F0230/binned')
# os.chdir(r'F:\Experiments\{}\processed tifs'.format(exp))
# phase = io.imread('{}-1.tif'.format(file_num))
# CFP = io.imread('{}-2.tif'.format(file_num))

phase_correct = []
for i in range(0,len(phase)):
    # slicing
    crop_phase_img = phase[i][0:len(CFP[i]), 0:len(CFP[i][0])]
    phase_correct.append(crop_phase_img)
    
CFP_correct = []
for i in range(0,len(phase)):
    # slicing
    crop_CFP_img = CFP[i][0:len(phase[i]), 0:len(phase[i][0])]
    CFP_correct.append(crop_CFP_img)
    
CFP_crop = CFP_correct
phase_crop = phase_correct

#%% BLUE MASKING PARALLELIZED


def mask_fun(mask=None,i=None,phase_crop=None,CFP_crop=None):
    crop_img = np.where(phase_crop[i] == 0, np.nan, phase_crop[i])
    med = median(crop_img, disk(5))
    # logarithmic_corrected = exposure.adjust_log(med, 3)
    # ave_background = gaussian(logarithmic_corrected, sigma = 10)
    # backsub = np.subtract(logarithmic_corrected, ave_background)
    # edge detection on phase
    sobel_edge = sobel(med)
    
    CFP_img = np.where(CFP_crop[i] == 0, 10000, CFP_crop[i])
    CFP_med = median(CFP_img, disk(5))
    CFP_logarithmic_corrected = exposure.adjust_log(CFP_med, 3)
    # ave_background = gaussian(logarithmic_corrected, sigma = 10)|
    # backsub = np.subtract(logarithmic_corrected, ave_background)

    # assign markers: <min value inside biofilm, >max outside biofilm
    markers = np.zeros_like(CFP_logarithmic_corrected)
    
    # markers[CFP_logarithmic_corrected < 3500] = 1
    # markers[CFP_logarithmic_corrected > 4000] = 2
    markers[CFP_logarithmic_corrected < 2.2*np.median(CFP_logarithmic_corrected)] = 1
    markers[CFP_logarithmic_corrected > 2.2*np.median(CFP_logarithmic_corrected)] = 2

    segmentation_img = segmentation.watershed(sobel_edge, markers)
    bool_seg = np.where(segmentation_img == 2, 1, 0)
    mask.append(bool_seg)
    return mask

mask=[]
mask=Parallel(n_jobs=mp.cpu_count())(delayed(mask_fun)(mask=mask,phase_crop=phase_crop,CFP_crop=CFP_crop,i=i) 
                                          for i in tqdm(range(0,len(phase_crop)), desc="Masking"))



areaLow = 6000
hole_max = 300000

colony_label_filt = []
for t in range(0, len(mask)):
    a = measure.label(mask[t][0]).astype(int)
    filtered = remove_small_objects(a, min_size=areaLow)
    closed =  remove_small_holes(filtered, hole_max)
    b = measure.label(closed).astype(int)
    colony_label_filt.append(b)

#  filtered labels and no traps
colony_label_filt_nt = []
for t in range(0, len(mask)):
    b = colony_label_filt[t]
    c = np.where(phase_crop[t] == 0, 0, b)
    colony_label_filt_nt.append(c)
    
props = []
for t in range(0, len(colony_label_filt)):
    img_props = measure.regionprops(colony_label_filt[t])
    props.append(img_props)

centroids = []
for t in range(0, len(colony_label_filt)):
    img_centroids = []
    for i in range(0, len(props[t])):
        single_centroid = props[t][i]['centroid']
        img_centroids.append(single_centroid)
    centroids.append(img_centroids)
    
all_labels = []
for t in range(0, len(colony_label_filt)):
    labels = np.unique(colony_label_filt[t])
    labels = np.delete(labels, np.where(labels == 0))
    all_labels.append(labels)

area = []
for t in range(0, len(centroids)):
    frame_areas = []
    for k in range(0, len(centroids[t])):
        single_area = props[t][k]['area']
        frame_areas.append(single_area)
    area.append(frame_areas)

#%% Check mask!

fps = 10
nSeconds = 27

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure( figsize=(8,8) )

a = colony_label_filt_nt[0]
im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)

def animate_func(i):
    if i % fps == 0:
        print( '.', end ='' )

    im.set_array(colony_label_filt_nt[i])
    plt.title('Tp{}'.format(i))
    return [im]

anim = FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )
print('Behold mortals')

# %% EXTRACT DATA - MAKE TABLE
# track centroids
ct = CentroidTracker()

# key is an array of ordered dictionaries. each time point has a dictionary of centroids and IDs
key = []
for t in range(0, len(centroids)):
    objects = ct.update(centroids[t])
    key.append(copy.copy(objects))

# #%%    
# extract data from mask with traps excluded (colony_label_filt_nt)
# organized by (t1[biofilm1, biofilm2,...])

indiv_masks = []
indiv_masks_CFP = []
mean_CFP = []
indiv_masks_zeros = []

indiv_masks_edge = []
mean_edge_CFP = []

# indiv_masks_edge_top = []
# mean_edge_CFP_top = []
# indiv_masks_edge_bot = []
# mean_edge_CFP_bot = []


for t in tqdm(range(0, len(colony_label_filt)), desc="Signal processing"):
    tp_single_colony = []
    tp_single_colony_CFP = []
    tp_single_colony_mean_CFP = []
    tp_single_colony_zeros = []
    
    tp_single_colony_edge_mask = []
    tp_single_colony_edge_mean_CFP = []
    
    # tp_single_edge_CFP_top = []
    # tp_single_edge_mean_top = []
    # tp_single_edge_CFP_bot = []
    # tp_single_edge_mean_bot = []
    
    for k in range(1, len(all_labels[t])+1):
        raw_img = CFP_crop[t]
        single_mask = np.where(colony_label_filt_nt[t] == int(k), 1, np.nan)
        single_mask_CFP = np.multiply(raw_img, single_mask)
        single_mean = np.nanmean(single_mask_CFP)
        # get full colony data
        single_mask_wtrap = np.where(colony_label_filt[t] == int(k), 1, 0)
        single_mask_eroded = binary_erosion(single_mask_wtrap, selem=np.ones((50, 50))).astype(int)
        single_mask_zeros = np.where(colony_label_filt_nt[t] == int(k), 1, 0)
        single_edge_mask_zeros = np.where(single_mask_eroded == 1, 0, single_mask_zeros)
        single_edge_mask = np.where(single_edge_mask_zeros == 0, np.nan, 1)
        single_edge_CFP = np.multiply(raw_img, single_edge_mask)
        single_edge_mean = np.nanmean(single_edge_CFP)
            # append lists
        tp_single_colony.append(single_mask)
        tp_single_colony_CFP.append(single_mask_CFP)
        tp_single_colony_mean_CFP.append(single_mean)
        tp_single_colony_zeros.append(single_mask_zeros)
        
        tp_single_colony_edge_mask.append(single_edge_mask)
        tp_single_colony_edge_mean_CFP.append(single_edge_mean)
    indiv_masks.append(tp_single_colony)
    indiv_masks_CFP.append(tp_single_colony_CFP)
    mean_CFP.append(tp_single_colony_mean_CFP)
    indiv_masks_zeros.append(tp_single_colony_zeros)
    
    indiv_masks_edge.append(tp_single_colony_edge_mask)
    mean_edge_CFP.append(tp_single_colony_edge_mean_CFP)
    
##%% get pixel radius and intensity [radius, intensity]

def pix_radii_fun(coords, t, i):
    radii_output = []
    pix_output = []
    cs_output = []
    for j in range(0, len(coords)):
            single_radius = (math.dist(coords[j], centroids[t][i]))/(math.sqrt(area[t][i]/math.pi))
            pix_intensity = CFP_crop[t][coords[j][0]][coords[j][1]]
            if pix_intensity > 0:
                radius, pix = single_radius, pix_intensity
                radii_output.append(radius)
                pix_output.append(pix)
                if .5 <single_radius < .75:
                    cs_output.append(pix_intensity)
    return radii_output, pix_output, cs_output

def strip(mask, centroid, width):
    strip_mask = np.zeros_like(mask)
    strip_mask[int(centroid[0]-width):int(centroid[0]+width):,int(centroid[1]):len(CFP_crop[0][0])] = 1
    return strip_mask

width = 10 #1/2 of the actual width
strip_radii = []
strip_pixels = []
strip_area = []
for t in  tqdm(range(0, len(CFP_crop)), desc='STRIP pixel by radius'):
    img_radii = []
    img_pix = []
    img_area = []
    for k in range(1, len(all_labels[t])+1):
        single_mask = np.where(colony_label_filt_nt[t] == int(k), 1, 0)
        strip_shape = strip(single_mask, centroids[t][k-1], width)
        strip_mask = np.multiply(single_mask, strip_shape)
        strip_props = measure.regionprops(strip_mask)
        colony_area = strip_props[0]['area']
        colony_pixels = strip_props[0]['coords']
        colony_radii, colony_pix, colony_cs_pixels = pix_radii_fun(colony_pixels, t, k-1)
        img_radii.append(colony_radii)
        img_pix.append(colony_pix)
        img_area.append(colony_area)
    strip_radii.append(img_radii)
    strip_pixels.append(img_pix)
    strip_area.append(img_area)

## %%
def build_column(info):
    ttable = pd.DataFrame(info).T 
    return pd.melt(ttable)

# make table of colony frame labels
l_table = build_column(all_labels)
c_table =  build_column(centroids)
c_table.columns =['Time pt', 'Centroid']
centroid_column = c_table['Centroid']
label_table = l_table.join(centroid_column)
label_table.columns =['Time pt', 'Label', 'Centroid']

strip_radii_col = build_column(strip_radii)
strip_pixels_col = build_column(strip_pixels)
strip_area_col = build_column(strip_area)


indiv_masks_CFP_col = build_column(indiv_masks_CFP)
mean_CFP_col = build_column(mean_CFP)
indiv_masks_zeros_col = build_column(indiv_masks_zeros)
indiv_masks_edge_col = build_column(indiv_masks_edge)
mean_edge_CFP_col = build_column(mean_edge_CFP)
area_col = build_column(area)

label_table['strip_radii'] = strip_radii_col[strip_radii_col.columns[1]]
label_table['strip_pixels'] = strip_pixels_col[strip_pixels_col.columns[1]]
label_table['strip_area'] = strip_area_col[strip_area_col.columns[1]]

label_table['indiv_masks_CFP'] = indiv_masks_CFP_col[indiv_masks_CFP_col.columns[1]]
label_table['mean_CFP'] = mean_CFP_col[mean_CFP_col.columns[1]]
label_table['indiv_masks_zeros'] = indiv_masks_zeros_col[indiv_masks_zeros_col.columns[1]]
label_table['indiv_masks_edge'] = indiv_masks_edge_col[indiv_masks_edge_col.columns[1]]
label_table['mean_edge_CFP'] = mean_edge_CFP_col[mean_edge_CFP_col.columns[1]]
label_table['area'] = area_col[area_col.columns[1]]
label_table = label_table.dropna()

# # make colony ID table
key_table1 = pd.DataFrame(key).T
key_table = pd.melt(key_table1)
key_table.columns =['Time pt', 'Centroid']

all_IDs = np.arange(0, key_table1.shape[0])
ID_column = []
for i in range(0, len(key)):
    for element in all_IDs:
        ID_column.append(element)
key_table['Track ID'] = ID_column

# join tables
tracking_table = pd.merge(key_table, label_table,
    how="left", on=['Time pt','Centroid'], #maybe try left join?
    copy=True, validate=None,)

ID_count = max(tracking_table["Track ID"].tolist()) +1
print('Table built!')

# %% SAVE TABLE

os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
tracking_table.to_csv('tracking table {}-{}.csv'.format(exp, ch), index=False)
#%% IMPORT DATA TABLE
os.chdir(r'F:\Experiments\{}\tracking tables'.format(exp))
# os.chdir('/Volumes/MPK_sandisk/Experiments/{}/tracking tables'.format(exp))

tracking_table = pd.read_csv(r'tracking table {}-{}.csv'.format(exp, ch))
ID_count = max(tracking_table["Track ID"].tolist()) +1

#%% CREATE HEX GRAPH TIFF SERIES FOR BF STRIPS
os.chdir(graph_directory+'/strip radial intensity tifs/')

# df_all = []
# for i in range(0, ID_count):
#     group = tracking_table[tracking_table['Track ID'] == i]
#     df_all.append(group)
colonyID = 0

bf = tracking_table[tracking_table['Track ID'] == colonyID]
# bf = df_all[colonyID]

import ast

x_raw = bf['strip_radii'].tolist()
y_raw = bf['strip_pixels'].tolist()
x_all = []
y_all = []
for t in range(0, len(CFP_crop)):
    x1 = x_raw[t]
    y1 = y_raw[t]
    x2 = ast.literal_eval(x1)
    y2 = ast.literal_eval(y1)
    x_all.append(x2)
    y_all.append(y2)

sns.set_theme(style="ticks")
hex_graphs = []

for t in tqdm(range(0, len(CFP_crop)), desc="Hex graphing"):
    time = t*img_freq
    p = sns.jointplot(x=x_all[t], y=y_all[t], kind="hex", color="#4CB391")
    p.set_axis_labels('Radius from centroid (pixels)', 'Intensity', fontsize=16)
    p.fig.suptitle("Radial Intensity {}min-{}-{}-bf{}".format(time,exp,ch,str(colonyID)))
    p.ax_marg_x.set_xlim(0, 1.2) #np.max(x_all[t]))
    p.ax_marg_y.set_ylim(0, 3000) #np.max(np.max(y_all)))
    # plt.close('all')
    hex_graphs.append(p)
    plt.savefig('Strip Radial Intensity {}min-{}-{}-bf{}.tiff'.format(time,exp,ch,str(colonyID)), bbox_inches='tight')
    
plt.close('all')

# %% TESTING REGRESSION CURVE FOR ALL FRAMES
os.chdir(graph_directory+'/radial intensity regression tifs/')

from scipy.optimize import curve_fit
from scipy import signal


# colonyID = 0

bf = tracking_table[tracking_table['Track ID'] == colonyID]

x_all = bf['radii'].tolist()
y_all = bf['pixels'].tolist()

# define the true objective function
def objective(x, a, b, c, d):
	return a * np.sin(b - x) + c * x**2 + d

y_lines = []
x_lines = []
y_detrended = []
for t in tqdm(range(0, len(CFP_crop)), desc="regression processing"):
    x, y = x_all[t], y_all[t]
    # curve fit
    popt, _ = curve_fit(objective, x, y)
    # summarize the parameter values
    a, b, c, d = popt
    # plot input vs output
    # plt.scatter(x, y)
    # define a sequence of inputs between the smallest and largest known inputs
    x_line = np.arange(min(x), max(x), .1)
    # calculate the output for the range
    y_line = objective(x_line, a, b, c, d)
    y_detrend = signal.detrend(y_line)
    y_lines.append(y_line)
    x_lines.append(x_line)
    y_detrended.append(y_detrend)

for t in tqdm(range(0, len(CFP_crop)), desc="graphing regression"):
    time = t*5
    p = plt.plot(x_lines[t], y_detrended[t], '--', color='red')
    p = plt.scatter(x_all[t], y_all[t])
    # p.set_axis_labels('Radius from centroid (pixels)', 'Intensity', fontsize=16)
    # p.fig.suptitle("Radial Intensity tp{}-{}-{}-bf{}".format(t,exp,ch,str(colonyID)))
    plt.xlim(0, 1.2) #np.max(x_all[t]))
    plt.ylim(-200, 2000)
    plt.title("Radial Intensity {}min-{}-{}-bf{}".format(time,exp,ch,str(colonyID)))
    # plt.savefig('Radial Intensity Regression {}min-{}-{}-bf{}.tiff'.format(time,exp,ch,str(colonyID)), bbox_inches='tight')
    # plt.pause(.3)
    # plt.close()
    


# %%
# area graph all together
# df_all breaks down table into individual tables for each ID

# os.chdir(graph_directory)
detrend_data = True

df_all = []
for i in range(0, ID_count):
    # group = tracking_table[tracking_table['Track ID'] == i]
    # df_all.append(group)

    group = tracking_table[tracking_table['Track ID'] == i]
    df_all.append(group)

# detrend
def detrend(x, y):
    # find linear regression line, subtract off data to detrend
    not_nan_ind = ~np.isnan(y)
    m, b, r_val, p_val, std_err = stats.linregress(x[not_nan_ind],y[not_nan_ind])
    detrended = y - (m*x + b)
    return detrended

detrended_area = []
for i in range(0, ID_count):
    a = detrend(df_all[i]['Time pt'], df_all[i]['area']).tolist()
    detrended_area.append(a)
    

# name different biofilms
# bf_positions = ['top trap', 'bottom trap', 'outlet', 'other', 'anticor','other','other', 'other', 'other']
bf_positions = ['0', '1', '2', '3', '4', '5','6','7', '8', '9']

peaks_all = []
for i in range(0, ID_count):
    peak, _ = find_peaks(detrended_area[i])
    peaks_all.append(peak)
    
for i in range(0, ID_count):
    if detrend_data == True:
        plt.plot((df_all[i]['Time pt']*img_freq), detrended_area[i],'-x',markevery=peaks_all[i], label = bf_positions[i])
    if detrend_data == False:
        plt.plot(df_all[i]['Time pt']*img_freq, df_all[i]['area'],'-x',markevery=peaks_all[i], label = bf_positions[i])

plt.xlabel('Time (min)')
plt.ylabel('Size')
if detrend_data == True:
    plt.title('{}-{} Area detrended'.format(exp,ch))
if detrend_data == False:
    plt.title('{}-{} Area raw'.format(exp,ch))
plt.legend(loc='best')
# plt.show()

# if detrend_data == True:
#     plt.savefig('Area detrended {}-{}.tiff'.format(exp,ch), bbox_inches='tight')
# if detrend_data == False:
#     plt.savefig('Area raw {}-{}.tiff'.format(exp,ch), bbox_inches='tight')

# plt.close()
# %%
detrended_area_strip = []
for i in range(0, ID_count):
    a = detrend(df_all[i]['Time pt'], df_all[i]['strip_area']).tolist()
    detrended_area_strip.append(a)

for i in range(0, ID_count):
    if detrend_data == True:
        plt.plot((df_all[i]['Time pt']*img_freq), detrended_area_strip[i],'-x',markevery=peaks_all[i], label = bf_positions[i])
    if detrend_data == False:
        plt.plot(df_all[i]['Time pt']*img_freq, df_all[i]['strip_area'],'-x',markevery=peaks_all[i], label = bf_positions[i])

plt.xlabel('Time (min)')
plt.ylabel('Size')
if detrend_data == True:
    plt.title('{}-{} Area Strip detrended'.format(exp,ch))
if detrend_data == False:
    plt.title('{}-{} Area Strip raw'.format(exp,ch))
plt.legend(loc='best')
# plt.show()

if detrend_data == True:
    plt.savefig('Area detrended {}-{}.tiff'.format(exp,ch), bbox_inches='tight')
if detrend_data == False:
    plt.savefig('Area raw {}-{}.tiff'.format(exp,ch), bbox_inches='tight')

# plt.close()


# %%
## %%
# edge CFP graph all together
os.chdir(graph_directory)
# detrend
# detrend_data = True
detrended_edge_CFP = []
for i in range(0, ID_count):
    a = detrend(df_all[i]['Time pt'], df_all[i]['mean_edge_CFP']).tolist()
    detrended_edge_CFP.append(a)

# name different biofilms
# bf_positions = ['top trap', 'bottom trap', 'outlet', 'other', 'anticor','other','other', 'other', 'other']
bf_positions = ['0', '1', '2', '3', '4', '5','6','7', '8', '9']

peaks_all = []
for i in range(0, ID_count):
    peak, _ = find_peaks(detrended_edge_CFP[i], distance=10, prominence=30)
    peaks_all.append(peak)
    
for i in range(0, ID_count):
    if detrend_data == True:
        plt.plot(df_all[i]['Time pt']*img_freq, detrended_edge_CFP[i],'-x',markevery=peaks_all[i], label = bf_positions[i])
    if detrend_data == False:
        plt.plot(df_all[i]['Time pt']*img_freq, df_all[i]['mean_edge_CFP'],'-x',markevery=peaks_all[i], label = bf_positions[i])

plt.xlabel('Time (min)')
plt.ylabel('Intensity')
if detrend_data == True:
    plt.title('{}-{} mean_edge_CFP detrended'.format(exp,ch))
if detrend_data == False:
    plt.title('{}-{} mean_edge_CFP raw'.format(exp,ch))
plt.legend(loc='best')
# plt.show()

if detrend_data == True:
    plt.savefig('mean_edge_CFP detrended {}-{}.tiff'.format(exp,ch), bbox_inches='tight')
if detrend_data == False:
    plt.savefig('mean_edge_CFP raw {}-{}.tiff'.format(exp,ch), bbox_inches='tight')



# %%

# pearson correlation
# measure synchrony https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9
os.chdir(graph_directory)
wave_end = len(CFP_crop)
max_IDs = 2

mean_edge_CFP_table = pd.DataFrame(detrended_edge_CFP).T
pearsoncorr = mean_edge_CFP_table.iloc[:, 0:max_IDs].iloc[0:wave_end].corr(method='pearson')

sns.heatmap(pearsoncorr, 
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5).set(title='{}-{} Pearson of Edge ThT 0-{}0min'.format(exp,ch,str(wave_end)))
plt.savefig(('{}-{} Pearson of Edge ThT 0-{}0min'.format(exp,ch,str(wave_end))), bbox_inches='tight')
# plt.close()
## %%
# Pearson rolling window 
# Set window size to compute moving window synchrony.
os.chdir(graph_directory)
r_window_size = 30
ID1 = 0
ID2 = 1
# Compute rolling window synchrony
rolling_r = mean_edge_CFP_table[ID1].rolling(window=r_window_size, center=True).corr(mean_edge_CFP_table[ID2])
f,ax=plt.subplots(figsize=(14,4))
rolling_r.plot(ax=ax)
ax.set(xlabel='Frame',ylabel='Pearson r')
ax.set_ylim([-1, 1])
plt.suptitle("{}-{} Pearson mean_edge_CFP and rolling window correlation ID:{}vs{} (window={})".format(exp,ch,str(ID1), str(ID2), str(r_window_size)))
# plt.savefig(("{}-{} Pearson mean_edge_CFP and rolling window correlation ID:{}vs{} (window={})".format(exp,ch,str(ID1), str(ID2), str(r_window_size))), bbox_inches='tight')

plt.savefig(("{}-{} Pearson mean_edge_CFP rolling window corr BF{} vs BF{} - window {}".format(exp,ch,str(ID1), str(ID2), str(r_window_size))),bbox_inches='tight')
# plt.close()
## %%
# Crosscorrelation
def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))

d1 = mean_edge_CFP_table[0].iloc[0:wave_end]
d2 = mean_edge_CFP_table[1].iloc[0:wave_end]
lags = 10
rs = [crosscorr(d1,d2, lag) for lag in range(-lags,lags+1)]
offset = np.floor(len(rs)/2)-np.argmax(rs)
f,ax=plt.subplots(figsize=(14,3))
ax.plot(rs)
ax.axvline(np.ceil((len(rs)-1)/2),color='k',linestyle='--',label='Center')
ax.axvline(np.argmax(rs),color='r',linestyle='--',label='Peak synchrony')
ax.set(title="Time Lagged Cross Correlation")
# ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads',ylim=[.1,.31],xlim=[0,301], xlabel='Offset',ylabel='Pearson r')
# ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
# ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150]);
plt.legend()

plt.savefig(("{}-{} Pearson mean_edge_CFP cross corr BF{} vs BF{} - window {}".format(exp,ch,str(ID1), str(ID2), str(r_window_size))),bbox_inches='tight')



# %%
# derivative of area
df_all = []
for i in range(0, ID_count):
    # group = tracking_table[tracking_table['Track ID'] == i]
    # df_all.append(group)

    group = tracking_table[tracking_table['Track ID'] == i]
    df_all.append(group)

# name different biofilms
# bf_positions = ['top trap', 'bottom trap', 'outlet', 'other', 'anticor','other','other', 'other', 'other']
bf_positions = ['0', '1', '2', '3', '4', '5','6','7', '8', '9']

peaks_all = []
for i in range(0, ID_count):
    peak, _ = find_peaks(detrended_area[i])
    peaks_all.append(peak)

y_area_deriv = []
x_area_deriv = []
for i in range(0, ID_count):
    y_list = df_all[i]['area'].tolist()
    x_list = df_all[i]['Time pt'].tolist()
    y = np.diff(y_list) / np.diff(x_list)
    x = (np.array(x_list)[:-1] + np.array(x_list)[1:]) / 2
    y_area_deriv.append(y)
    x_area_deriv.append(x)

for i in range(0, ID_count):
    plt.plot(x_area_deriv[i]*img_freq, y_area_deriv[i],'-x',markevery=peaks_all[i], label = bf_positions[i])

plt.xlabel('Time (min)')
plt.ylabel('size derivative')
plt.title('{}-{} Area derivative'.format(exp,ch))
plt.ylim(-100, 600)
plt.show()


# %%
from scipy import interpolate
os.chdir(graph_directory)

# detrend
detrend_data = True
detrended_mean_cs_CFP = []


for i in range(0, ID_count):
    a = detrend(df_all[i]['Time pt'], df_all[i]['mean_cs']).tolist()
    f2 = interpolate.splrep(df_all[i]['Time pt'], a, s=50)
    xfit = np.arange(0, len(df_all[i]['Time pt']), 3)
    yfit = interpolate.splev(xfit, f2, der=0)
    b = interpolate.splev(xfit, f2, der=0)
    detrended_mean_cs_CFP.append(b)
    
bf_positions = ['0', '1', '2', '3', '4', '5','6','7', '8', '9']

peaks_all = []
for i in range(0, ID_count):
    peak, _ = find_peaks(detrended_mean_cs_CFP[i], distance=10, prominence=30)
    peaks_all.append(peak)
    
for i in range(0, ID_count):
    if detrend_data == True:
        plt.plot(df_all[i]['Time pt'], detrended_mean_cs_CFP[i],'-x',markevery=peaks_all[i], label = bf_positions[i])
    if detrend_data == False:
        plt.plot(df_all[i]['Time pt'], df_all[i]['mean_cs'],'-x',markevery=peaks_all[i], label = bf_positions[i])

plt.xlabel('Time (min)')
plt.ylabel('Intensity')
if detrend_data == True:
    plt.title('{}-{} mean_cs_CF detrended'.format(exp,ch))
if detrend_data == False:
    plt.title('{}-{} mean_cs_CF raw'.format(exp,ch))
plt.legend(loc='best')
plt.show()

# if detrend_data == True:
#     plt.savefig('meas_cs_CFP detrended {}-{}-bf{}.tiff'.format(exp,ch,str(colonyID)), bbox_inches='tight')
# if detrend_data == False:
#     plt.savefig('meas_cs_CFP raw {}-{}-bf{}.tiff'.format(exp,ch,str(colonyID)), bbox_inches='tight')




    


#%%
# plot centroids to check tracking

df = tracking_table[tracking_table['Track ID'] == 0]
df = df['Centroid'].tolist()
# plt.figure()
# plt.imshow(colony_label_filt[0])
# plt.plot(df[0][1], df[0][0],marker = 'o',ms = 10)
fps = 5
nSeconds = 5

fig = plt.figure( figsize=(8,8) )

a = colony_label_filt[0]
im = plt.imshow(a, interpolation='none', aspect='auto', vmin=0, vmax=1)


def animate_func(t):
    if i % fps == 0:
        print( '.', end ='' )

    im.set_array(colony_label_filt[t])
    plt.plot(df[t][1], df[t][0],marker = 'o',ms = 10)
    return [im]

anim = FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )
print('Behold mortals')
    
    
# %%
# make images flash to test tracking

df2 = tracking_table['indiv_masks_CFP']
df1 = tracking_table[tracking_table['Track ID'] == 1]
# # df = df['indiv_masks_zeros']
listt = df1['indiv_masks_CFP'].to_list()
cent = df1['Centroid'].to_list()


fig, ax = plt.subplots()

for i in range(len(listt)):
    ax.cla()
    ax.imshow(listt[i])
    ax.set_title("frame {}".format(i))
    plt.plot(cent[i][1], cent[i][0],marker = 'o',ms = 10)
    # Note that using time.sleep does *not* work here!
    plt.pause(0.0001)
    
# %% mask video
import cv2
os.chdir(r'F:\Experiments\{}\videos'.format(exp))

vidSize = colony_label_filt_nt[0].shape

# name new video file, assign it to be a jpeg movie, set frame rate, vid dimentions, set to color or no
vid = cv2.VideoWriter('{}_{}_mask.avi'.format(exp,ch), cv2.VideoWriter_fourcc(*'MJPG'),15,(vidSize[1],vidSize[0]),1)

for i in range(0,len(colony_label_filt_nt)):
    
    im = copy.copy(colony_label_filt_nt[i])
    im8 = np.multiply(im,255).astype('uint8')
    # # im = indiv_masks_test[i]
    # imzeros = np.where(indiv_masks_test[i] == np.nan, 1, 0)
    # im8 = np.multiply(imzeros,255).astype('uint8')
    
    frame = cv2.applyColorMap(im8,cmapy.cmap('gray'))
    
    # timestamp
    # t = datetime.time(i*img_freq // 60,i*img_freq % 60)
    # cv2.putText(frame,t.strftime('%H:%M'),(25,100),cv2.FONT_HERSHEY_SIMPLEX,3,[255,255,255],5,cv2.LINE_AA)

    # write frame 
    vid.write(frame)

vid.release()
cv2.destroyAllWindows()

#%% fancy side by side video

mask_vidSize = mask[0].shape
phase_vidSize = phase_crop[0].shape


# name new video file, assign it to be a jpeg movie, set frame rate, vid dimentions, set to color or no
vid = cv2.VideoWriter('{}_{}_{}_mask.avi'.format(exp,day,ch), cv2.VideoWriter_fourcc(*'MJPG'),15,(mask_vidSize[1]*2,mask_vidSize[0]),1)

for i in range(0,len(mask)):
    
    phase_im = phase_crop[i]
    phase_im8 = np.multiply(phase_im,255).astype('uint8')
    phase_frame = cv2.applyColorMap(phase_im8,cmapy.cmap('gray'))
    
    mask_im = mask[i]
    mask_im8 = np.multiply(mask_im-1,255).astype('uint8')
    mask_frame = cv2.applyColorMap(im8,cmapy.cmap('gray'))
    
    side_by_side = np.zeros((2*vidSize[1], vidSize[0]),dtype='uint8')
    side_by_side[:mask_vidSize[1], mask_vidSize[0]] = phase_frame
    side_by_side[mask_vidSize[1]:,mask_vidSize[0]] = mask_frame
    
    vid.write(frame)
    

vid.release()
cv2.destroyAllWindows()

# %%



# %% AREA ROLLIN AVE

df_all = []
for i in range(0, ID_count):
    group = tracking_table[tracking_table['Track ID'] == i]
    df_all.append(group)

area_df = []
for i in range(0, ID_count):
    group = tracking_table[tracking_table['Track ID'] == i]
    group_area = group['area']
    area_df.append(group_area)

window0 = 20
window1 = 5

area_roll_ave_w1 = []
area_roll_ave = []
area_mean_sub = []
area_mean_sub_w0w1 = []

for k in range(0, ID_count):
    group_mean_w1 = area_df[k].rolling(window1, center=True).mean()
    area_roll_ave_w1.append(group_mean_w1)
    
    group_mean_w0 = area_df[k].rolling(window0, center=True).mean()
    area_roll_ave.append(group_mean_w0)
    
    group_mean_sub = area_df[k].subtract(group_mean_w0)
    area_mean_sub.append(group_mean_sub)
    
    group_mean_sub_w0_w1 = group_mean_w1.subtract(group_mean_w0)
    area_mean_sub_w0w1.append(group_mean_sub_w0_w1)    
    
bf_positions = ['0', '1', '2', '3', '4', '5','6','7', '8', '9']

for i in range(0, ID_count):
    plt.plot(df_all[i]['Time pt']*img_freq, area_mean_sub_w0w1[i], label = bf_positions[i])
    
plt.xlabel('Time (min)')
plt.ylabel('Size')
plt.title('{}-{} Area Mean-Sub'.format(exp,ch))
plt.legend(loc='best')
# plt.show()

# %% THT ROLLING AVE

df_all = []
for i in range(0, ID_count):
    group = tracking_table[tracking_table['Track ID'] == i]
    df_all.append(group)

mean_edge_CFP_df = []
for i in range(0, ID_count):
    group = tracking_table[tracking_table['Track ID'] == i]
    group_CFP = group['mean_edge_CFP']
    mean_edge_CFP_df.append(group_CFP)

window0 = 20
window1 = 5

mean_edge_CFP_roll_ave_w1 = []
mean_edge_CFP_roll_ave = []
mean_edge_CFP_mean_sub = []
mean_edge_CFP_mean_sub_w0w1 = []

for k in range(0, ID_count):
    group_mean_w1 = mean_edge_CFP_df[k].rolling(window1, center=True).mean()
    mean_edge_CFP_roll_ave_w1.append(group_mean_w1)
    
    group_mean_w0 = mean_edge_CFP_df[k].rolling(window0, center=True).mean()
    mean_edge_CFP_roll_ave.append(group_mean_w0)
    
    group_mean_sub = mean_edge_CFP_df[k].subtract(group_mean_w0)
    mean_edge_CFP_mean_sub.append(group_mean_sub)
    
    group_mean_sub_w0_w1 = group_mean_w1.subtract(group_mean_w0)
    mean_edge_CFP_mean_sub_w0w1.append(group_mean_sub_w0_w1)    
    

# name different biofilms
bf_positions = ['0', '1', '2', '3', '4', '5','6','7', '8', '9']

for i in range(0, ID_count):
    plt.plot(df_all[i]['Time pt']*img_freq, mean_edge_CFP_mean_sub_w0w1[i], label = bf_positions[i])

plt.xlabel('Time (min)')
plt.ylabel('Intensity')
plt.title('{}-{} mean_edge_CFP Mean-Sub'.format(exp,ch))
plt.legend(loc='best')
# plt.show()

#%% THT VS AREA

bf = 0

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

ax1 = plt.subplot()
l1, = ax1.plot(df_all[bf]['Time pt']*img_freq/60, mean_edge_CFP_mean_sub_w0w1[bf], color='cyan')
ax2 = ax1.twinx()
l2, = ax2.plot(df_all[bf]['Time pt']*img_freq/60, area_mean_sub_w0w1[bf], color='grey')
plt.title('{}-{}-bf{} mean_edge_CFP Mean-Sub'.format(exp,ch,bf))
plt.legend([l1, l2], ["ThT", "area"])
ax1.set_ylabel('Intensity', color='cyan')
ax2.set_ylabel('Size (pixels)', color='grey')
ax1.set_xlabel('Time (hours)')
plt.show()

# %%

from scipy.fft import fft, fftfreq
# bf = 0

# mean_edge_CFP_mean_sub_w0w1[bf].dropna()

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
for i in range(0, ID_count):
    xf, yf, N = fourier_fun(mean_edge_CFP_mean_sub_w0w1[i])
    ft_x.append(xf)
    ft_y.append(yf)
    N_all.append(N)
    
for i in range(0, ID_count):
    plt.plot(ft_x[i], 2.0/N_all[i] * np.abs(ft_y[i][0:N_all[i]//2]),linewidth=2.0)
plt.xlabel('Freq. (1/hours)')
plt.ylabel('Fourier Magnitude')

plt.show()



