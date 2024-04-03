import inspect
import os
from PIL import Image
import re
import numpy as np
import time
from functools import wraps
from typing import TypedDict, Union, Optional


class configDict(TypedDict):
    """
    A class to hold configuration values.
    
    Attributes:
        debugging (bool): Whether debugging is enabled.
        pickle_usage (bool): Whether to use pickle.
        tiff_amount_cutoff(int): The amount of tiff files to process. If None, all files are processed.
    """
    debugging: bool
    pickle_usage: bool
    tiff_amount_cutoff: int

class plotDict(TypedDict):
    """
    A class to hold configuration values.
    
    Attributes:
        save(bool): Whether to save the plot.
        array_type(str): 'SUM', 'RMS', or 'individual'
        heatmap_max(int): The maximum value for the heatmap. None for automatic scaling.
    """
    array_type: str
    bin_amount: int
    heatmap_max: Optional[int]  # int or None
    dpi: int
    save: bool

def timer_decorator(configs:configDict):
    debugging = configs['debugging']
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            if debugging==True:
                print(f"Function   {func.__name__}() took {end_time - start_time:.1f} seconds to run.")
            
            # if debugging==True:
                # print(f'File reading time: {end_time - start_time:.1f} seconds. With {get_def_name()}')
            return result
        return wrapper
    return decorator


# deprecated
# def get_def_name():
#     # print(f"This message is printed from {inspect.currentframe().f_back.f_code.co_name}")
#     def_name = 'DEF ' + inspect.currentframe().f_back.f_code.co_name + '()'
#     return def_name


def get_tags_from_first_tiff(tiff_path:str) -> list[Union[str, int]]:

    tiff_filenames = get_tiff_list(tiff_path)
    image_first = Image.open(f'{tiff_path}\\{tiff_filenames[0]}')

    # tag 270 Image description
    tag_content = image_first.tag_v2.get(270)

    # Extract exposure time using regular expression
    match = re.search(r'Prop_ExposureTime2=([\d.]+)', tag_content)
    if match:
        exposure_time_str = match.group(1)
        exposure_time_ms:int = np.round(float(exposure_time_str) * 1000, 0).astype(int)
    else:
        exposure_time_ms:int = -1 # default value
        # print("Exposure time not found in tag 270")
        raise ValueError("Exposure time not found in tag 270")

    # tag 258 bit rate (16,) 
    bit_rate = image_first.tag_v2.get(258)

    # tag 256 image width tag 257 image length
    image_width:int = image_first.tag_v2.get(256)
    if image_width is None:
        raise ValueError("Image width not found in tag 256")
    image_length:int = image_first.tag_v2.get(257)
    if image_length is None:
        raise ValueError("Image length not found in tag 257")
    image_dimension:str = str(image_width) + "x" + str(image_length)

    output_str = f'Exposure time: {exposure_time_ms} ms, Bit rate: {bit_rate[0]} bits, Dimension: {image_dimension}, frames: {len(tiff_filenames)}\n'

    return [output_str, image_width, image_length, exposure_time_ms]


def get_tiff_list(tiff_path:str) -> list[str]:
    file_names = os.listdir(tiff_path)
    tiff_files = [f for f in file_names if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
    return tiff_files