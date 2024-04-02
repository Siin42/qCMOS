import inspect
import os
from PIL import Image
import re
import numpy as np
import time
from functools import wraps


def timer_decorator(configs):
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



# # @timer_decorator(debugging)
# def plot_SUM_or_RMS(array_to_plot, tiff_path, configs, **kwargs):
#     # # if debugging is not yet defined in global, set it to be True
#     # if 'debugging' not in globals():
#     #     debugging = True
#     debugging = configs['debugging']
    
#     plot_type = kwargs.get('plot_type', 'bar')
#     bin_amount = kwargs.get('bin_amount', 100)
#     heatmap_max = kwargs.get('heatmap_max')
#     save = kwargs.get('save', False)

#     array_type = kwargs.get('array_type')
#     # raise an error if array_type is not defined
#     if array_type is None:
#         raise ValueError('array_type is not defined. Please provide either "SUM" or "RMS"')

#     tiff_filenames = get_tiff_list(tiff_path)


#     caption_tags, image_width, image_length, exposure_time_ms = get_tags_from_first_tiff(tiff_path)[:4]
#     total_pixel_amount = image_width * image_length

#     caption_statistics_SUM = f'{np.sum(array_to_plot==0)} pixels had 0 count in the whole set of data, {np.sum(array_to_plot==0) / total_pixel_amount * 100:.1f}%; \n' + \
#             f'SUM(10% of the pixels) <= {np.percentile(array_to_plot, 10)}; \n' + \
#             f'SUM(90% of the pixels) <= {np.percentile(array_to_plot, 90)}'
#     caption_statistics_RMS = f'{np.sum(array_to_plot==0)} pixels had 0 count in the whole set of data, {np.sum(array_to_plot==0) / total_pixel_amount * 100:.1f}%; \n' + \
#             f'RMS(10% of the pixels) <= {np.percentile(array_to_plot, 10):.3f}; \n' + \
#             f'RMS(90% of the pixels) <= {np.percentile(array_to_plot, 90):.3f}; \n' + \
#             f'pixel max-RMS: {np.max(array_to_plot):.3f} e- \n' + \
#             f'Camera RMS: {np.sqrt(np.mean(array_to_plot**2)):.3f} e-'

#     figname_optional = ''


#     if plot_type == 'bar':
#         # It was cutting right tail of the histogram. Name a max for it
#         # counts, bin_edges = np.histogram(sum_image_arrays.flatten(), bins=bin_amount)
#         min_value = 0
#         max_value = np.max(array_to_plot)
#         counts, bin_edges = np.histogram(array_to_plot.flatten(), bins=bin_amount, range=(min_value, max_value))

#         if debugging==True:
#             # print(f'bin amount: {bin_amount}')
#             print(f'bins width: {np.average(np.diff(bin_edges))}')
#             max_fraction_bin = np.max(counts) / np.sum(counts)
#             if array_type=='SUM':
#                 print(f'{100*max_fraction_bin:.1f}% of pixel Sum counts fall between '
#                     f'{np.argmax(counts)} and {np.argmax(counts) + np.diff(bin_edges)[0]} counts in all {len(tiff_filenames)} frames')
#             elif array_type=='RMS':
#                 print(f'{100*max_fraction_bin:.1f}% of pixel RMS errors fall between '
#                     f'{np.argmax(counts)} and {np.argmax(counts) + np.diff(bin_edges)[0]} counts in all {len(tiff_filenames)} frames')

#         # counts_normalized = counts / counts.max()

#         fig = plt.figure(figsize=(10, 6))
#         # plt.bar(bin_edges[:-1], counts_normalized, width=np.diff(bin_edges), color='gray', log=True, align='edge')
#         plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), color='gray', log=True, align='edge')

#         if array_type=='SUM':
#             plt.xlabel('Total Counts')
#         elif array_type=='RMS':
#             plt.xlabel('RMS')
#         plt.ylabel('Frequency')

#         figname_optional += f'_{len(tiff_filenames)}frames'

#         if array_type=='SUM':
#             fig.text(0.15, 0.05, caption_tags + caption_statistics_SUM, ha='left')
#         elif array_type=='RMS':
#             fig.text(0.15, 0.05, caption_tags + caption_statistics_RMS, ha='left')
#         # plt.subplots_adjust(bottom=0.3)

#     elif plot_type == 'heat':
#         heat_plot_width = np.round(image_width/image_length * 6, 1) + 2
#         fig = plt.figure(figsize=(heat_plot_width, 6), dpi=400)

#         plt.imshow(array_to_plot, cmap='hot', interpolation='nearest', vmax=heatmap_max)
#         if array_type=='SUM':
#             plt.colorbar(label='Counts (Sum)')
#         elif array_type=='RMS':
#             plt.colorbar(label='RMS')
#         if heatmap_max is not None:
#             caption_tags += f'Clipped at {heatmap_max} counts\n'
#             figname_optional += f'_clip{heatmap_max}'

#         figname_optional += f'_{len(tiff_filenames)}frames'
        
#         if array_type=='SUM':
#             fig.text(0.15, 0.05, caption_tags + caption_statistics_SUM, ha='left')
#         elif array_type=='RMS':
#             fig.text(0.15, 0.05, caption_tags + caption_statistics_RMS, ha='left')
    


#     plt.subplots_adjust(bottom=0.3)

#     if save==True:
#         if array_type=='SUM':
#             plt.savefig(f'SUM_{plot_type}{figname_optional}_{exposure_time_ms}ms_{image_width}x{image_length}.png')
#         elif array_type=='RMS':
#             plt.savefig(f'RMS_{plot_type}{figname_optional}_{exposure_time_ms}ms_{image_width}x{image_length}.png')

#     plt.show()