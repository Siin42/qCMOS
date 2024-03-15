import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import re
# from multiprocessing import Pool, cpu_count
import pickle
from tifffile import imread

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

    for file_name in tiff_filenames:
        # 只处理tif文件
        if file_name.endswith('.tif'):
            img = Image.open(f'{tiff_path}\\{file_name}')
            img_array = np.array(img)
            image_arrays += img_array

    if debug==True:
        frames = len(tiff_filenames)
        print('exposures:', frames)
    if debug==True:
        print('Dimension of image_arrays:', image_arrays.shape)
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
    elif plot_type == 'hist':
        bins, freq_array = grouping_intensity_values(image_arrays, bin_width=int(0.1*frames))

        if debug==True:
            print('bins:', bins)
            print('freq_array:', freq_array)

        # 确保bins和频率数组的长度相同
        assert len(bins) == len(freq_array), 'The lengths of bins and freq_array are not the same.'

        # 确保频率数组中的值不全为0
        assert np.max(freq_array) > 0, 'The values in freq_array are all zero.'

        # Normalizing the frequency array
        freq_arrays_normalized = freq_array / np.max(freq_array)

        # 设置图表的大小
        fig = plt.figure(figsize=(10, 6))
        bars = plt.bar(bins, freq_arrays_normalized, color='gray')

        # xticks = np.arange(0, int(value_max)) # 生成一个从x轴的最小值到最大值的整数序列
        # plt.xticks(xticks) # 设置x轴的刻度
        plt.yscale('log') # 将y轴标签改为对数值

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

def read_image(file_name):
    img = Image.open(f'{tiff_path}\\{file_name}')
    img_array = np.array(img)
    return img_array

def get_tiff_list(tiff_path):
    file_names = os.listdir(tiff_path)
    tiff_files = [f for f in file_names if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
    return tiff_files

debug = True
# tiff_path = 'W:\\Quant_Opt_Group\\Group\\Zhengyin_public\\LABDATA\\3.12 qCMOS\\data' + '\\3056x124'
tiff_path = 'W:\\Quant_Opt_Group\\Group\\Zhengyin_public\\LABDATA\\3.12 qCMOS\\data' + '\\1024x1024_quick'

def main():
    total_pixel_values(tiff_path, 'hist', save=False)

if __name__ == '__main__':
    main()