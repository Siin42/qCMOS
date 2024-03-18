import inspect
import os
from PIL import Image
import re
import numpy as np



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

# def read_image(file_path):
#     img = Image.open(file_path)
#     img_array = np.array(img)
#     return csr_matrix(img_array)

# def worker(file_path):
#     # 你的函数
#     return read_image(file_path)

# def Threading_read_images(file_path):
#     tiff_filenames = get_tiff_list(file_path)
#     tiff_addresses = [os.path.join(file_path, fn) for fn in tiff_filenames]

#     # max_workers should be a bit bigger than the number of threads for I/O intensive tasks
#     with ThreadPoolExecutor(max_workers=60) as executor:
#         futures = [executor.submit(worker, addr) for addr in tiff_addresses]
#     image_arrays = [future.result() for future in futures]

#     return image_arrays