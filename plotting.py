from utils import *
import matplotlib.pyplot as plt

# @timer_decorator(debugging)
def plot_SUM_or_RMS(array_to_plot:np.ndarray, tiff_path:str, array_configs:configDict, plot_configs:plotDict, plot_type:str) -> None:
    """
    Parameters:
        - array_to_plot():
        - tiff_path(str):
        - array_type(str): SUM, RMS, invididual
    
    Optionals packed in _array_configs:
        - debugging(bool):
        - pickle_usage(bool):
        - tiff_amount_cutoff(int):
    
    Optionals packed in _plot_configs:
        - array_type(str): 'SUM', 'RMS', or 'individual'
        - bin_amount(int): amount of bins for the histogram
        - heatmap_max(int): maximum value for the heatmap
        - dpi(int): dpi for the heatmap
        - save(bool): whether to save the plot

    - plot_type(str): 'bar' or 'heat'
    """
    debugging = array_configs['debugging']
    
    # plot_type:str = kwargs.get('plot_type', 'bar')
    if plot_type not in ['heat', 'bar']:
        raise ValueError('plot_type is wrong. Please provide "heat" or "bar"')
    
    # bin_amount:int = kwargs.get('bin_amount', 100)
    # heatmap_max:int = kwargs.get('heatmap_max')
    # dpi:int = kwargs.get('dpi', 400)
    # save:bool = kwargs.get('save', False)
    array_type = plot_configs['array_type']
    bin_amount = plot_configs.get('bin_amount', 200)    # two ways of phrasing
    heatmap_max = plot_configs.get('heatmap_max', None)
    dpi = plot_configs['dpi']
    save = plot_configs.get('save', False)


    # array_type:str = kwargs.get('array_type')
    # if array_type is None:
    #     raise ValueError('array_type is not defined. Please provide "SUM", "RMS" or "individual"')

    tiff_filenames = get_tiff_list(tiff_path, array_configs)



    caption_tags, image_width, image_length, exposure_time_ms = get_tags_from_first_tiff(tiff_path)[:4]
    total_pixel_amount = image_width * image_length

    figname_optional = ''
    if array_type=='SUM':
        caption_statistics_SUM = f'{np.sum(array_to_plot==0)} pixels had 0 count in the whole set of data, {np.sum(array_to_plot==0) / total_pixel_amount * 100:.1f}%; \n' + \
                f'SUM(10% of the pixels) <= {np.percentile(array_to_plot, 10)}; \n' + \
                f'SUM(90% of the pixels) <= {np.percentile(array_to_plot, 90)}'
    elif array_type=='RMS':
        caption_statistics_RMS = f'{np.sum(array_to_plot==0)} pixels had 0 count in the whole set of data, {np.sum(array_to_plot==0) / total_pixel_amount * 100:.1f}%; \n' + \
                f'RMS(10% of the pixels) <= {np.percentile(array_to_plot, 10):.3f}; \n' + \
                f'RMS(90% of the pixels) <= {np.percentile(array_to_plot, 90):.3f}; \n' + \
                f'pixel max-RMS: {np.max(array_to_plot):.3f} e- \n' + \
                f'Camera RMS: {np.sqrt(np.mean(array_to_plot**2)):.3f} e-'


    if plot_type == 'bar':
        # It was cutting right tail of the histogram. Name a max for it
        # counts, bin_edges = np.histogram(sum_image_arrays.flatten(), bins=bin_amount)
        min_value = 0
        max_value = np.max(array_to_plot)
        counts, bin_edges = np.histogram(array_to_plot.flatten(), bins=bin_amount, range=(min_value, max_value))

        if debugging==True:
            # print(f'bin amount: {bin_amount}')
            print(f'bins width: {np.average(np.diff(bin_edges))}')
            max_fraction_bin = np.max(counts) / np.sum(counts)
            if array_type=='SUM':
                print(f'{100*max_fraction_bin:.1f}% of pixel Sum counts fall between '
                    f'{np.argmax(counts)} and {np.argmax(counts) + np.diff(bin_edges)[0]} counts in all {len(tiff_filenames)} frames')
            elif array_type=='RMS':
                print(f'{100*max_fraction_bin:.1f}% of pixel RMS errors fall between '
                    f'{np.argmax(counts)} and {np.argmax(counts) + np.diff(bin_edges)[0]} counts in all {len(tiff_filenames)} frames')

        # counts_normalized = counts / counts.max()

        fig = plt.figure(figsize=(10, 6))
        # plt.bar(bin_edges[:-1], counts_normalized, width=np.diff(bin_edges), color='gray', log=True, align='edge')
        plt.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), color='gray', log=True, align='edge')

        if array_type=='SUM':
            plt.xlabel('Total Counts')
        elif array_type=='RMS':
            plt.xlabel('RMS')
        plt.ylabel('Frequency')

        figname_optional += f'_{len(tiff_filenames)}frames'

        if array_type=='SUM':
            fig.text(0.15, 0.05, caption_tags + caption_statistics_SUM, ha='left')
        elif array_type=='RMS':
            fig.text(0.15, 0.05, caption_tags + caption_statistics_RMS, ha='left')
        else:
            fig.text(0.15, 0.05, caption_tags, ha='left')
        # plt.subplots_adjust(bottom=0.3)

    elif plot_type == 'heat':
        heat_plot_width = np.round(image_width/image_length * 6, 1) + 2
        fig = plt.figure(figsize=(heat_plot_width, 6), dpi=dpi)

        plt.imshow(array_to_plot, cmap='hot', interpolation='nearest', vmax=heatmap_max)
        if array_type=='SUM':
            plt.colorbar(label='Counts (Sum)')
        elif array_type=='RMS':
            plt.colorbar(label='RMS')
        else:
            plt.colorbar(label='None')
        if heatmap_max is not None:
            caption_tags += f'Clipped at {heatmap_max} counts\n'
            figname_optional += f'_clip{heatmap_max}'

        figname_optional += f'_{len(tiff_filenames)}frames'
        
        if array_type=='SUM':
            fig.text(0.15, 0.05, caption_tags + caption_statistics_SUM, ha='left')
        elif array_type=='RMS':
            fig.text(0.15, 0.05, caption_tags + caption_statistics_RMS, ha='left')
        else:
            fig.text(0.15, 0.05, caption_tags, ha='left')
            
    


    plt.subplots_adjust(bottom=0.3)

    if save==True:
        if array_type=='SUM':
            plt.savefig(f'SUM_{plot_type}{figname_optional}_{exposure_time_ms}ms_{image_width}x{image_length}.png')
        elif array_type=='RMS':
            plt.savefig(f'RMS_{plot_type}{figname_optional}_{exposure_time_ms}ms_{image_width}x{image_length}.png')
        elif array_type=='individual':
            pass
        else:
            raise ValueError('array_type is not defined. Please provide "SUM", "RMS" or "individual"')

    plt.show()