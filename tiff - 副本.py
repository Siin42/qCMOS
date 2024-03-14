import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import re
from multiprocessing import Pool, cpu_count
from tifffile import imread

def average_pixel_values(file_names, value_max, save=False):
    freq_arrays = []

    for file_name in file_names:
        # 只处理tif文件
        if file_name.endswith('.tif'):
            img = Image.open(f'{tiff_path}\\{file_name}')
            img_array = np.array(img)
            flattened_img_array = img_array.flatten()

            # 计算每个强度值的频率
            freq_array, _ = np.histogram(flattened_img_array, bins=value_max, range=(0, value_max))
            freq_arrays.append(freq_array)

    # 计算频率的平均值
    average_freq_array = np.mean(freq_arrays, axis=0)

    # 设置图表的大小
    plt.figure(figsize=(10, 6))

    # 创建直方图
    bars = plt.bar(range(value_max), average_freq_array, color='gray', alpha=0.7)

    # 将y轴标签改为对数值
    plt.yscale('log')

    # 在每个柱子上方添加一个文本标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, 
                f'{height:.1f}', ha='center', va='bottom')

    plt.title(f'Histogram of Average Pixel Values from {len(freq_arrays)} tiff files')
    plt.xlabel('Pixel Value')
    plt.ylabel('Log Frequency')

    if save==True:
        plt.savefig('average_pixel_values.png')

    # 显示图表
    plt.show()

def total_pixel_values(file_names, value_max, save=False):
    freq_arrays = []

    start_time = time.time()
    for file_name in file_names:
        # 只处理tif文件
        if file_name.endswith('.tif'):
            img = Image.open(f'{tiff_path}\\{file_name}')
            img_array = np.array(img)
            flattened_img_array = img_array.flatten()

            # 计算每个强度值的频率
            freq_array, _ = np.histogram(flattened_img_array, bins=value_max, range=(0, value_max))
            freq_arrays.append(freq_array)
    end_time = time.time()
    if debug==True:
        print(f"File reading time: {end_time - start_time} seconds")

    # 计算频率的总和
    total_freq_array = np.sum(freq_arrays, axis=0)

    # 设置图表的大小
    plt.figure(figsize=(10, 6))

    # 创建直方图
    bars = plt.bar(range(value_max), total_freq_array, color='gray', alpha=0.7)

    # 将y轴标签改为对数值
    plt.yscale('log')

    # 在每个柱子上方添加一个文本标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, 
                f'{height:.1f}', ha='center', va='bottom')

    plt.title(f'Histogram of Total Pixel Values from {len(freq_arrays)} tiff files')
    plt.xlabel('Pixel Value')
    plt.ylabel('Log Frequency')

    if save==True:
        plt.savefig('total_pixel_values.png')

    # 显示图表
    plt.show()

def pixel_fluctuation(file_names, value_range, save=False):
    img_arrays = []

    # With PIL
    # File reading time: 893.8105647563934 seconds
    start_time = time.time()
    # for file_name in file_names:
    #     if file_name.endswith('.tif'):
    #         img = Image.open(f'{tiff_path}\\{file_name}')
    #         img_array = np.array(img)
    #         img_arrays.append(img_array)
    # print('With PIL')

    # With tifffile
    # File reading time: 194.37059926986694 seconds
    for file_name in file_names:
        if file_name.endswith('.tif'):
            img_array = imread(f'{tiff_path}\\{file_name}')
            img_arrays.append(img_array)
    print('With tifffile')

    # Create a process pool with the number of processes equal to the number of CPU cores
    # made things 20% slower. Probably IO not multiprocessable
    # with Pool(cpu_count()) as p:
    #     img_arrays = p.map(read_image, tiff_files)
    # # print amount of pools being created
    # if debug==True:
    #     print(f"Amount of pools: {cpu_count()}")

    end_time = time.time()
    if debug==True:
        print(f"File reading time: {end_time - start_time} seconds")
    
    # read exposure time from the first tiff file
    img_first = Image.open(f'{tiff_path}\\{file_names[0]}')

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

    # tag 256 image width
    # tag 257 image length
    image_dimension = str(img_first.tag_v2.get(256)) + "x" + str(img_first.tag_v2.get(257))
    if debug==True:
        print(f"Dimension: {image_dimension}")

    # exposure_time = img_first.tag_v2[33434][0] / img_first.tag_v2[33434][1] doesn't work
    # print(list(img_first.tag_v2.keys()))

    img_arrays = np.array(img_arrays)
    pixel_readout_noise_RMS = []

    start_time = time.time()
    pixel_readout_noise_RMS = np.sqrt(np.mean(np.square(img_arrays), axis=0))
    end_time = time.time()
    if debug==True:
        print(f"Loop execution time: {end_time - start_time} seconds")

    # stds = np.array(stds).reshape(img_arrays.shape[1], img_arrays.shape[2])

    counts, bins = np.histogram(pixel_readout_noise_RMS, bins=500, range=value_range)
    bins = bins[:-1]

    plt.figure(figsize=(15, 9)) # Set the size of the plot
    plt.plot(bins, counts)
    plt.xlabel('Readout Noise RMS (ADU)')
    plt.ylabel('Pixel Count')
    plt.title(f'Pixel Value Fluctuation RMS\nExposure time: {exposure_time_ms} ms, Bit rate: {bit_rate[0]} bits, Dimension: {image_dimension}')
    # plt.xlim([0, 2]) # Set the x-axis limits
    plt.yscale('log') # Set the y-axis to a logarithmic scale

    if save==True:
        plt.savefig('pixel_fluctuation_rms.png')

    plt.show()

def read_image(file_name):
    img = Image.open(f'{tiff_path}\\{file_name}')
    img_array = np.array(img)
    return img_array

debug = True
tiff_path = 'W:\\Quant_Opt_Group\\Group\\Zhengyin_public\\LABDATA\\3.12 qCMOS\\data' + '\\3056x124'
file_names = os.listdir(tiff_path)
tiff_files = [f for f in file_names if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]

def main():
    # average_pixel_values(tiff_files, 18, save=False)
    pixel_fluctuation(tiff_files, (0, 6), save=False)
    # total_pixel_values(tiff_files, 20, save=True)

if __name__ == '__main__':
    main()