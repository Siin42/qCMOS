# -*- coding: utf-8 -*-
# %%
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import re
import pickle
# from tifffile import imread
# from scipy.sparse import csr_matrix
# from concurrent.futures import ThreadPoolExecutor

from utils import get_def_name, get_tags_from_first_tiff, get_tiff_list
from multi_threading import Threading_read_images


# %%
def read_all_images(tiff_path, **kwargs):
    debugging = kwargs.get('debugging', True)
    pickle_usage = kwargs.get('pickle_usage', True)

    image_arrays = []
    start_time = time.time()
    if debugging==True and pickle_usage==False:
        print('NOT USING PICKLE')
    
    if os.path.exists(f'{tiff_path}\\image_arrays.pkl') and pickle_usage==True:
        with open(f'{tiff_path}\\image_arrays.pkl', 'rb') as f:
            image_arrays = pickle.load(f)
        print(f'{tiff_path}\nLoaded image_arrays.pkl')
    else:
        image_arrays = Threading_read_images(tiff_path)

        if pickle_usage==True:
            with open(f'{tiff_path}\\image_arrays.pkl', 'wb') as f:
                pickle.dump(image_arrays, f)

    end_time = time.time()
    if debugging==True:
        print(f'File reading time: {end_time - start_time:.1f} seconds. With {get_def_name()}')

    
    return image_arrays

# %% [markdown]
# ##### Testing block for cProfile

# %%
import cProfile

def profile_read_image(file_path):
    profiler = cProfile.Profile()
    profiler.enable()

    # 调用你想要分析的函数
    # result = read_image(file_path)
    result = to_call_with_cProfile(file_path=file_path)

    profiler.disable()
    profiler.print_stats(sort='time')

    return result

def to_call_with_cProfile(**args):
    file_path = args['file_path']
    
    result = read_image(file_path)

    return result

# 使用你的文件路径调用函数
# example_tiff =  'C:\\3.12 qCMOS\\full frame\\Background001.tif'
# profile_read_image(example_tiff)

# %% [markdown]
# ##### Keep going

# %%
def sum_all_images(tiff_path, **kwargs):
    debugging = kwargs.get('debugging', False)
    pickle_usage = kwargs.get('pickle_usage', True)

    image_arrays = read_all_images(tiff_path, **kwargs)
    start_time = time.time()
    if debugging==True and pickle_usage==False:
        print('NOT USING PICKLE')
    
    if os.path.exists(f'{tiff_path}\\sum_image_arrays.pkl') and pickle_usage==True:
        with open(f'{tiff_path}\\sum_image_arrays.pkl', 'rb') as f:
            sum_image = pickle.load(f)
        print('Loaded sum_image_arrays.pkl')

    else:
        sum_image = np.zeros_like(image_arrays[0], dtype=np.float64)
        for img in image_arrays:
            sum_image += img

        if pickle_usage==True:
            with open(f'{tiff_path}\\sum_image_arrays.pkl', 'wb') as f:
                pickle.dump(sum_image, f)
        
    return sum_image

# %%

def total_pixel_values(sum_image_arrays, tiff_path, **kwargs):
    plot_type = kwargs.get('plot_type', 'bar')
    debug = kwargs.get('debug', True)
    bin_amount = kwargs.get('bin_amount', 100)
    save = kwargs.get('save', False)

    tiff_filenames = get_tiff_list(tiff_path)

    # we assume to have a csr sparse matrix as input
    sum_image_arrays = sum_image_arrays.toarray()

    # start_time = time.time()

    # region get tags from img_first
    # image_first = imread(f'{tiff_path}\\{tiff_filenames[0]}')
    image_first = Image.open(f'{tiff_path}\\{tiff_filenames[0]}')
    # image_arrays = np.zeros_like(image_first, dtype=np.float64)

    # tag 270 Image description
    tag_content = image_first.tag_v2.get(270)

    # Extract exposure time using regular expression
    match = re.search(r'Prop_ExposureTime2=([\d.]+)', tag_content)
    if match:
        exposure_time_str = match.group(1)
        exposure_time_ms = float(exposure_time_str) * 1000
        # if debug==True:
        #     print(f'exposure time match {match}')
        #     # print(f"Exposure time: {exposure_time_ms:.0f} ms")
    else:
        print("Exposure time not found in tag 270")

    # tag 258 bit rate (16,) 
    bit_rate = image_first.tag_v2.get(258)
    # if debug==True:
    #     print(f"Bit rate: {bit_rate[0]} bits")

    # tag 256 image width tag 257 image length
    image_width = image_first.tag_v2.get(256)
    image_length = image_first.tag_v2.get(257)
    # image_dimension = str(image_first.tag_v2.get(256)) + "x" + str(image_first.tag_v2.get(257))
    image_dimension = str(image_width) + "x" + str(image_length)
    # if debug==True:
    #     print(f"Dimension: {image_dimension}")
    #     print(f'{np.sum(sum_image_arrays==0)} pixels had 0 count {np.sum(sum_image_arrays==0) / (image_width * image_length) * 100:.1f}%')
    #     print(f'{np.sum(sum_image_arrays==1)} had 1 count {np.sum(sum_image_arrays==1) / (image_width * image_length) * 100:.1f}%')
    #     print(f'Total amount of pixels: {image_width * image_length}')
    # endregion
        
    # if reader=='PIL':
    #     # File reading time: 658.416844367981 seconds
    #     for file_name in tiff_filenames:
    #         # 只处理tif文件
    #         if file_name.endswith('.tif'):
    #             img = Image.open(f'{tiff_path}\\{file_name}')
    #             img_array = np.array(img)
    #             image_arrays += img_array
    #     print('With PIL')
    # elif reader=='tifffile':
    #     # File reading time: 766.2443685531616 seconds
    #     for file_name in tiff_filenames:
    #         if file_name.endswith('.tif'):
    #             img_array = imread(f'{tiff_path}\\{file_name}')
    #             image_arrays += img_array
    #     print('With tifffile')
    # end_time = time.time()
    # if debug==True:
    #     print(f"File reading time: {end_time - start_time} seconds")

    # if debug==True:
    #     frames = len(tiff_filenames)
    #     print('exposures:', frames)
    # if debug==True:
    #     print('Maximum value of image_arrays:', np.max(sum_image_arrays))


    if plot_type == 'heat':
        fig = plt.figure(figsize=(10, 6))
        plt.imshow(sum_image_arrays, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Pixel Value (Sum)')
        fig.text(0.5, 0.01, f'Exposure time: {exposure_time_ms:.0f} ms, Bit rate: {bit_rate[0]} bits, Dimension: {image_dimension}, frames: {len(tiff_filenames)}', ha='center') # Add caption to the figure
        if save==True:
            plt.savefig('total_pixel_cmap.png')
        plt.show()

    elif plot_type == 'bar':
        # bins = max(int(0.1*frames), 20) # Set bins amount. if bins is less than 20, set it to there
        counts, bin_edges = np.histogram(sum_image_arrays.flatten(), bins=bin_amount)
        if debug==True:
            # print('bin amount:', bin_amount)
            print(f'bin amount: {bin_amount}')
            max_fraction_bin = np.max(counts) / np.sum(counts)
            print(f'{100*max_fraction_bin:.1f}% of pixels counted between {np.argmax(counts)} and {np.argmax(counts) + np.diff(bin_edges)[0]} in all {len(tiff_filenames)} frames')
            # print the amount of pixels in image_arrays that has 0

        counts_normalized = counts / counts.max()
        # has to be printed on every branches of if-else
        # because bins height normalization is done in different ways
        # if debug==True:
        #     # min_value = np.min(counts_normalized[counts_normalized > 0])
        #     print('Minimum in counts_normalized: %.e' % np.min(counts_normalized[counts_normalized > 0]))

        fig = plt.figure(figsize=(10, 6))
        plt.bar(bin_edges[:-1], counts_normalized, width=np.diff(bin_edges), color='gray', log=True, align='edge')

        if debug==True:
            # print('bins:', bin_edges[:-1])
            # print('freq_array:', counts_normalized)
            print('bins width:', np.average(np.diff(bin_edges)))

        # bars = plt.bar(bins, freq_arrays_normalized, color='gray')

        # xticks = np.arange(0, int(value_max)) # 生成一个从x轴的最小值到最大值的整数序列
        # plt.xticks(xticks) # 设置x轴的刻度

        # # 在每个柱子上方添加一个文本标签
        # for bar in bars:
        #     height = bar.get_height()
        #     plt.text(bar.get_x() + bar.get_width() / 2, height, 
        #             f'{height:.1f}', ha='center', va='bottom')

        # plt.title(f'Histogram of Total Counts v.s. Amount of Pixels after {len(freq_arrays)} exposures')
        plt.xlabel('Counts')
        plt.ylabel('Frequency (normalized)')
        
        fig.text(0.15, 0.01, f'Exposure time: {exposure_time_ms:.0f} ms, Bit rate: {bit_rate[0]} bits, Dimension: {image_dimension}, frames: {len(tiff_filenames)}\n' \
            f'{np.sum(sum_image_arrays==0)} pixels had 0 count in total, {np.sum(sum_image_arrays==0) / (image_width * image_length) * 100:.1f}%; ' \
            f'{np.sum(sum_image_arrays==1)} pixels had 1 count, {np.sum(sum_image_arrays==1) / (image_width * image_length) * 100:.1f}%\n' \
            f'10% of the pixels had less than {np.percentile(sum_image_arrays, 10):.0f} counts, 10% of the pixels had more than {np.percentile(sum_image_arrays, 90):.0f} counts' \
            , ha='left')
        plt.subplots_adjust(bottom=0.18)

        if save==True:
            plt.savefig('total_pixel_values.png')

        # 显示图表
        plt.show()

    elif plot_type=="hist":
        if debug==True:
            print('Minimum non-zero value of image_arrays: %.e' % np.min(sum_image_arrays[sum_image_arrays > 0]))

        fig = plt.figure(figsize=(10, 6))
        bin_amount = max(int(0.1*frames), 20) # Set bins amount. if bins is less than 20, set it to there
        if debug==True:
            print('amount of bins:', bin_amount)
        hist = plt.hist(sum_image_arrays.flatten(), bins=bin_amount, color='gray', log=True)
        if debug==True:
            print('hist:', hist)

        plt.xlabel('Counts')
        plt.ylabel('Frequency')
        fig.text(0.5, 0.01, f'Exposure time: {exposure_time_ms} ms, Bit rate: {bit_rate[0]} bits, Dimension: {image_dimension}, frames: {len(tiff_filenames)}', ha='center') # Add caption to the figure
        if save==True:
            plt.savefig('total_pixel_values.png')
        # 显示图表
        plt.show()
