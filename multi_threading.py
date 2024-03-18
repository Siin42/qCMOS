from scipy.sparse import csr_matrix
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
import os

from utils import get_tiff_list


def read_image(file_path):
    img = Image.open(file_path)
    img_array = np.array(img)
    return csr_matrix(img_array)

def worker(file_path):
    # Your function
    return read_image(file_path)

def Threading_read_images(file_path):
    tiff_filenames = get_tiff_list(file_path)
    tiff_addresses = [os.path.join(file_path, fn) for fn in tiff_filenames]

    # max_workers should be a bit bigger than the number of threads for I/O intensive tasks
    with ThreadPoolExecutor(max_workers=60) as executor:
        futures = [executor.submit(worker, addr) for addr in tiff_addresses]
    image_arrays = [future.result() for future in futures]

    return image_arrays