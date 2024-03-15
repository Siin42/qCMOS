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