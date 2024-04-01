import inspect
import os
from PIL import Image
import re
import numpy as np
import time
# import pickle
# from multi_threading import Threading_read_images



def timer_decorator(func):

    def wrapper(*args, **kwargs):

        debugging = kwargs.get('debugging', True)

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if debugging==True:
            print(f"Function   {func.__name__}()   took {end_time - start_time:.1f} seconds to run.")
        
        # if debugging==True:
            # print(f'File reading time: {end_time - start_time:.1f} seconds. With {get_def_name()}')
        return result
    return wrapper


def get_def_name():
    # print(f"This message is printed from {inspect.currentframe().f_back.f_code.co_name}")
    def_name = 'DEF ' + inspect.currentframe().f_back.f_code.co_name + '()'
    return def_name


def get_tags_from_first_tiff(tiff_path):

    tiff_filenames = get_tiff_list(tiff_path)
    image_first = Image.open(f'{tiff_path}\\{tiff_filenames[0]}')

    # tag 270 Image description
    tag_content = image_first.tag_v2.get(270)

    # Extract exposure time using regular expression
    match = re.search(r'Prop_ExposureTime2=([\d.]+)', tag_content)
    if match:
        exposure_time_str = match.group(1)
        exposure_time_ms = np.round(float(exposure_time_str) * 1000, 0).astype(int)
    else:
        print("Exposure time not found in tag 270")

    # tag 258 bit rate (16,) 
    bit_rate = image_first.tag_v2.get(258)

    # tag 256 image width tag 257 image length
    image_width = image_first.tag_v2.get(256)
    image_length = image_first.tag_v2.get(257)
    image_dimension = str(image_width) + "x" + str(image_length)

    output_str = f'Exposure time: {exposure_time_ms} ms, Bit rate: {bit_rate[0]} bits, Dimension: {image_dimension}, frames: {len(tiff_filenames)}\n'

    return [output_str, image_width, image_length, exposure_time_ms]


def get_tiff_list(tiff_path):
    file_names = os.listdir(tiff_path)
    tiff_files = [f for f in file_names if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
    return tiff_files


# @timer_decorator
# def read_all_images(tiff_path, **kwargs):
#     debugging = kwargs.get('debugging', True)
#     pickle_usage = kwargs.get('pickle_usage', True)
#     tiff_amount_cutoff = kwargs.get('tiff_amount_cutoff', None)

#     image_arrays = []
#     if debugging==True and pickle_usage==False:
#         print('NOT USING PICKLE')
    
#     if os.path.exists(f'{tiff_path}\\image_arrays.pkl') and pickle_usage==True:
#         with open(f'{tiff_path}\\image_arrays.pkl', 'rb') as f:
#             image_arrays = pickle.load(f)
#         print(f'{tiff_path}\nLoaded image_arrays.pkl')
#     else:
#         image_arrays = Threading_read_images(tiff_path, tiff_amount_cutoff)

#         if pickle_usage==True:
#             with open(f'{tiff_path}\\image_arrays.pkl', 'wb') as f:
#                 pickle.dump(image_arrays, f)

#     return image_arrays