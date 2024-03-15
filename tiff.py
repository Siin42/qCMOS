import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import re
# from multiprocessing import Pool, cpu_count
import pickle
from tifffile import imread

# TODO
# 1. use imread to read tiff files
# 2. try pickle

def grouping_intensity_values(image_array, bin_width=1):
    # 1. image_array is a list of zero or positive integers. 
    # 2. returns will be a list from 0 to the max of image_array, and another list of how many elements having the value in the first list respectively. 
    # 3. returns should be ready for a bar plot. 
    max_value = np.max(image_array)

    bins = np.arange(0, max_value + bin_width, bin_width)
    pixels, _ = np.histogram(image_array, bins=bins)
    bins = bins[:-1]

    if debug==True:
        print('bins:', bins)
        print('pixels:', pixels)

    return bins, pixels

def total_pixel_values(tiff_path, plot_type,  save=False):

    # 定义pickle文件的名字
    # pickle_file = 'img_data.pkl'

    tiff_filenames = get_tiff_list(tiff_path)

    start_time = time.time()
    # # 检查pickle文件是否存在
    # if os.path.exists(f'{tiff_path}/{pickle_file}'):
    #     # 如果pickle文件存在，从pickle文件中加载数据
    #     with open(f'{tiff_path}/{pickle_file}', 'rb') as f:
    #         img_arrays = pickle.load(f)
    #     print('Data loaded')
    # else:
    #     # 如果pickle文件不存在，读取所有的图像数据
    #     img_arrays = [imread(tiff_path + '\\' + file_name) for file_name in tiff_filenames]

    #     # 将数据保存到pickle文件
    #     with open(f'{tiff_path}/{pickle_file}', 'wb') as f:
    #         pickle.dump(img_arrays, f)
    #     print('Data saved to pickle file')

    # region get tags from img_first
    img_first = Image.open(f'{tiff_path}\\{tiff_filenames[0]}')
    image_arrays = np.zeros_like(img_first, dtype=np.float64)

    # tag 270 Image description
    tag_content = img_first.tag_v2.get(270)

    # Extract exposure time using regular expression
    match = re.search(r'Prop_ExposureTime2=([\d.]+)', tag_content)
    if match:
        exposure_time_str = match.group(1)
        exposure_time_ms = round(float(exposure_time_str) * 1000)
        if debug==True:
            print(f"Exposure time: {exposure_time_ms} ms")
    else:
        if debug==True:
            print("Exposure time not found in tag 270")

    # tag 258 bit rate (16,) 
    bit_rate = img_first.tag_v2.get(258)
    if debug==True:
        print(f"Bit rate: {bit_rate[0]} bits")

    # tag 256 image width tag 257 image length
    image_dimension = str(img_first.tag_v2.get(256)) + "x" + str(img_first.tag_v2.get(257))
    if debug==True:
        print(f"Dimension: {image_dimension}")
    # endregion
        
    if reader=='PIL':
        # File reading time: 893.8105647563934 seconds
        for file_name in tiff_filenames:
            # 只处理tif文件
            if file_name.endswith('.tif'):
                img = Image.open(f'{tiff_path}\\{file_name}')
                img_array = np.array(img)
                image_arrays += img_array
        print('With PIL')
    elif reader=='tifffile':
        # File reading time: 194.37059926986694 seconds
        for file_name in tiff_filenames:
            if file_name.endswith('.tif'):
                img_array = imread(f'{tiff_path}\\{file_name}')
                image_arrays += img_array
        print('With tifffile')

    if debug==True:
        frames = len(tiff_filenames)
        print('exposures:', frames)
    if debug==True:
        print('Maximum value of image_arrays:', np.max(image_arrays))

    end_time = time.time()
    if debug==True:
        print(f"File reading time: {end_time - start_time} seconds")


    if plot_type == 'heat':
        fig = plt.figure(figsize=(10, 6))
        plt.imshow(image_arrays, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Pixel Value (Sum)')
        fig.text(0.5, 0.01, f'Exposure time: {exposure_time_ms} ms, Bit rate: {bit_rate[0]} bits, Dimension: {image_dimension}, frames: {len(tiff_filenames)}', ha='center') # Add caption to the figure
        if save==True:
            plt.savefig('total_pixel_cmap.png')
        plt.show()

    elif plot_type == 'bar':
        bins = max(int(0.1*frames), 20) # Set bins amount. if bins is less than 20, set it to there
        counts, bin_edges = np.histogram(image_arrays.flatten(), bins=bins)

        counts_normalized = counts / counts.max()

        fig = plt.figure(figsize=(10, 6))
        plt.bar(bin_edges[:-1], counts_normalized, width=np.diff(bin_edges), color='gray', log=True, align='edge')

        if debug==True:
            print('bins:', bin_edges[:-1])
            print('freq_array:', counts_normalized)

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
        fig.text(0.5, 0.01, f'Exposure time: {exposure_time_ms} ms, Bit rate: {bit_rate[0]} bits, Dimension: {image_dimension}, frames: {len(tiff_filenames)}', ha='center') # Add caption to the figure

        if save==True:
            plt.savefig('total_pixel_values.png')

        # 显示图表
        plt.show()

    elif plot_type=="hist":
        if debug==True:
            print('Minimum non-zero value of image_arrays:', np.min(image_arrays[image_arrays > 0]))# 使用plt.hist()函数绘制直方图

        fig = plt.figure(figsize=(10, 6))
        bins = max(int(0.1*frames), 20) # Set bins amount. if bins is less than 20, set it to there
        if debug==True:
            print('amount of bins:', bins)
        hist = plt.hist(image_arrays.flatten(), bins=bins, color='gray', log=True)
        if debug==True:
            print('hist:', hist)

        plt.xlabel('Counts')
        plt.ylabel('Frequency')
        fig.text(0.5, 0.01, f'Exposure time: {exposure_time_ms} ms, Bit rate: {bit_rate[0]} bits, Dimension: {image_dimension}, frames: {len(tiff_filenames)}', ha='center') # Add caption to the figure
        if save==True:
            plt.savefig('total_pixel_values.png')
        # 显示图表
        plt.show()

def read_image(file_name):
    img = Image.open(f'{tiff_path}\\{file_name}')
    img_array = np.array(img)
    return img_array

def get_tiff_list(tiff_path):
    file_names = os.listdir(tiff_path)
    tiff_files = [f for f in file_names if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
    return tiff_files

debug = True
tiff_path = 'W:\\Quant_Opt_Group\\Group\\Zhengyin_public\\LABDATA\\3.12 qCMOS\\data' + '\\3056x124'
# tiff_path = 'W:\\Quant_Opt_Group\\Group\\Zhengyin_public\\LABDATA\\3.12 qCMOS\\data' + '\\1024x1024_quick'
reader = 'tifffile'
# reader = 'PIL'

def main():
    total_pixel_values(tiff_path, 'bar', save=False)

if __name__ == '__main__':
    main()