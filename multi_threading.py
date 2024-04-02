from scipy.sparse import csr_matrix
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from PIL import Image
import numpy as np
import os
from scipy.sparse import csr_matrix
from functools import reduce
import numpy as np
from operator import add
import pickle
from utils import get_tiff_list


def read_image(file_path):
    img = Image.open(file_path)
    img_array = np.array(img)
    return csr_matrix(img_array)

def worker(file_path):
    # Your function
    return read_image(file_path)

# def Threading_read_images(file_path):
# def Threading_read_images(file_path, tiff_amount_cutoff=None):
def Threading_read_images(file_path, configs):
    tiff_amount_cutoff = configs['tiff_amount_cutoff']

    tiff_filenames = get_tiff_list(file_path)
    if tiff_amount_cutoff is not None:
        if len(tiff_filenames) < tiff_amount_cutoff:
            raise ValueError(f"Insufficient tiff files. Expected at least {tiff_amount_cutoff}, but got {len(tiff_filenames)}")
        tiff_filenames = tiff_filenames[:tiff_amount_cutoff]
    tiff_addresses = [os.path.join(file_path, fn) for fn in tiff_filenames]

    # max_workers should be a bit bigger than the number of threads for I/O intensive tasks
    with ThreadPoolExecutor(max_workers=60) as executor:
        futures = [executor.submit(worker, addr) for addr in tiff_addresses]
    image_arrays = [future.result() for future in futures]

    return image_arrays

# @timer_decorator(debugging)
def read_all_images(tiff_path, configs):
    # pickle_usage = kwargs.get('pickle_usage', True)
    # tiff_amount_cutoff = kwargs.get('tiff_amount_cutoff', None)
    debugging = configs['debugging']
    pickle_usage = configs['pickle_usage']
    tiff_amount_cutoff = configs['tiff_amount_cutoff']

    image_arrays = []
    if debugging==True and pickle_usage==False:
        print('NOT USING PICKLE')
    
    if os.path.exists(f'{tiff_path}\\image_arrays.pkl') and pickle_usage==True:
        with open(f'{tiff_path}\\image_arrays.pkl', 'rb') as f:
            image_arrays = pickle.load(f)
        if debugging==True:
            print(f'{tiff_path}\nLoaded image_arrays.pkl')

    else:
        image_arrays = Threading_read_images(tiff_path, tiff_amount_cutoff)
        if pickle_usage==True:
            with open(f'{tiff_path}\\image_arrays.pkl', 'wb') as f:
                pickle.dump(image_arrays, f)

    return image_arrays



def exponentiate_RMS(img, exponent):
    return img.power(exponent)

def parallel_exponentiate_RMS(image_arrays, exponent):
    with ProcessPoolExecutor() as executor:
        exponentiation = list(executor.map(exponentiate_RMS, image_arrays, [exponent]*len(image_arrays)))

    return exponentiation





def sum_sublist(sublist):
    return reduce(add, sublist)

def parallel_sum(image_arrays):
    num_splits = os.cpu_count()
    sublists = np.array_split(image_arrays, num_splits)
    with ProcessPoolExecutor() as executor:
        sublist_sums = list(executor.map(sum_sublist, sublists))

    total_sum = reduce(add, sublist_sums)

    return total_sum

def frame_sum_sublist(sublist):
    return [np.sum(arr) for arr in sublist]
    
def parallel_frame_sum(image_arrays):
    
    num_splits = os.cpu_count()
    sublists = np.array_split(image_arrays, num_splits)
    with ProcessPoolExecutor() as executor:
        sublist_frame_sums = list(executor.map(frame_sum_sublist, sublists))

    whole_frame_sum = []
    for sublist_sum in sublist_frame_sums:
        whole_frame_sum.extend(sublist_sum)

    return whole_frame_sum