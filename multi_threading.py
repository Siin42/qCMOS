from scipy.sparse import csr_matrix
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from PIL import Image
import numpy as np
import os
from functools import reduce
import numpy as np
from operator import add
import pickle
from utils import get_tiff_list, configDict


def read_image(file_path:str) -> np.ndarray:
    """
    worker of `Threading_read_images()`
    """
    img = Image.open(file_path)
    img_array = np.array(img)
    return img_array

def worker(file_path:str) -> np.ndarray:
    # Your function
    return read_image(file_path)

def Threading_read_images(file_path:str, configs:configDict) -> list[np.ndarray]:
    """
    read images in parallel
    """
    # tiff_amount_cutoff = configs['tiff_amount_cutoff']

    tiff_filenames = get_tiff_list(file_path, configs)
    # if tiff_amount_cutoff is not None:
    #     if len(tiff_filenames) < tiff_amount_cutoff:
    #         raise ValueError(f"Insufficient tiff files. Expected at least {tiff_amount_cutoff}, but got {len(tiff_filenames)}")
    #     tiff_filenames = tiff_filenames[:tiff_amount_cutoff]
    tiff_addresses = [os.path.join(file_path, fn) for fn in tiff_filenames]

    # max_workers should be a bit bigger than the number of threads for I/O intensive tasks
    with ThreadPoolExecutor(max_workers=60) as executor:
        futures = [executor.submit(worker, addr) for addr in tiff_addresses]
    image_arrays = [future.result() for future in futures]

    return image_arrays

# @timer_decorator(debugging)
def read_all_images(tiff_path:str, configs:configDict) -> list[np.ndarray]:
    """
    Parameters:
        - `tiff_path`(str):
    
    Optionals packed in _configs:
        - `debugging`(bool):
        - `pickle_usage`(bool):
        - `tiff_amount_cutoff`(int):
    """
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
            image_arrays:list[np.ndarray] = pickle.load(f)
        if debugging==True:
            print(f'{tiff_path}\nLoaded image_arrays.pkl')
        
        image_arrays = image_arrays[:tiff_amount_cutoff]

    else:
        image_arrays = Threading_read_images(tiff_path, configs)
        if pickle_usage==True:
            with open(f'{tiff_path}\\image_arrays.pkl', 'wb') as f:
                pickle.dump(image_arrays, f)

    return image_arrays



def exponentiate_RMS(img:np.ndarray, exponent:int) ->np.ndarray:
    """
    Square on every elements of the matrix
    """
    return np.power(img, exponent)

def parallel_exponentiate_RMS(image_arrays_arr:list[np.ndarray], exponent:int) -> list[np.ndarray]:
    """
    Square on every elements of the matrix, but in parallel
    """
    with ProcessPoolExecutor() as executor:
        exponentiation = list(executor.map(exponentiate_RMS, image_arrays_arr, [exponent]*len(image_arrays_arr)))

    return exponentiation





def sum_sublist(sublist:list[np.ndarray]) -> np.ndarray:
    """
    Add intensity values for each pixel across the frames. 
    """
    if isinstance(sublist[0], csr_matrix):
        sublist = [item.toarray() for item in sublist]
    result = np.sum(sublist, axis=0)
    return result

def parallel_sum(image_arrays_arr:list[np.ndarray]) -> np.ndarray:
    """
    divide the list into sublists and sum them in parallel. 

    - Generate a total sum for each pixel across the frames. 
    """
    num_splits = os.cpu_count()
    sublists:list[np.ndarray] = np.array_split(image_arrays_arr, num_splits)
    with ProcessPoolExecutor() as executor:
        sublist_sums:list[np.ndarray] = list(executor.map(sum_sublist, sublists))
    
    print(f'sublist_sums:{sublist_sums}')
    total_sum = np.sum(sublist_sums, axis=0)

    return total_sum

def frame_sum_sublist(sublist:list[np.ndarray]) -> list[int]:
    """
    add intensity values 
    
    - on the whole frame
    """
    return [np.sum(arr) for arr in sublist]
    
def parallel_frame_sum(image_arrays_arr:list[np.ndarray]) -> list[int]:
    """
    get the sum of intensity values in parallel
    
    - on the whole frame
    """
    num_splits = os.cpu_count()
    sublists:list[np.ndarray] = np.array_split(image_arrays_arr, num_splits)
    with ProcessPoolExecutor() as executor:
        sublist_frame_sums:list[list[int]] = list(executor.map(frame_sum_sublist, sublists))

    whole_frame_sum:list[int] = []
    for sublist_sum in sublist_frame_sums:
        whole_frame_sum.extend(sublist_sum)

    return whole_frame_sum