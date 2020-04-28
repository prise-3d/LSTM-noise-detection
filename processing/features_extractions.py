# main imports
import numpy as np
from math import log10
import gzip
import sys


# librairies imports
from ipfml.utils import get_entropy, normalize_arr, normalize_arr_with_range
from ipfml.processing import transform, compression, segmentation


def _extract_svd(image, params):
    begin, end = tuple(map(int, params.split(',')))

    sigma = transform.get_LAB_L_SVD_s(image)
    return list(sigma[begin:end])


def _extract_svd_log10(image, params):
    begin, end = tuple(map(int, params.split(',')))

    sigma = transform.get_LAB_L_SVD_s(image)
    sigma_interval = list(sigma[begin:end])
    return [ log10(y) for y in sigma_interval ]


def _extract_stats_luminance(image, params):
    L = transform.get_LAB_L(image)
    sigma = compression.get_SVD_s(L)
    return list([np.mean(L), np.std(L), get_entropy(sigma)])


def _extract_svd_norm(image, params):
    begin, end = tuple(map(int, params.split(',')))

    sigma = transform.get_LAB_L_SVD_s(image)
    return list(normalize_arr_with_range(sigma[begin:end]))


def _extract_svd_norm_log10(image, params):
    begin, end = tuple(map(int, params.split(',')))

    sigma = transform.get_LAB_L_SVD_s(image)
    sigma_interval = list(normalize_arr_with_range(sigma[begin:end]))
    return [ log10(y) for y in sigma_interval ]


def _extract_mu_sigma(image, params):
    image = np.array(image)
    return list([np.mean(image), np.std(image)])


def _extract_svd_entropy(image, params):
    
    begin, end = tuple(map(int, params.split(',')))

    sigma = transform.get_LAB_L_SVD_s(image)
    return get_entropy(sigma[begin:end])


def _extract_svd_entropy_split(image, params):
    
    begin, end, split = tuple(map(int, params.split(',')))

    sigma = transform.get_LAB_L_SVD_s(image)

    # split in equals parts
    sigma_parts = np.array_split(sigma[begin:end], split)

    return list([ get_entropy(part) for part in sigma_parts ])


def _extract_svd_entropy_norm_split(image, params):

    begin, end, split = tuple(map(int, params.split(',')))

    sigma = transform.get_LAB_L_SVD_s(image)
    sigma_interval = list(normalize_arr_with_range(sigma[begin:end]))

    # split in equals parts
    sigma_parts = np.array_split(sigma_interval, split)

    return list([ get_entropy(part) for part in sigma_parts ])


def _extract_svd_entropy_log10_split(image, params):

    begin, end, split = tuple(map(int, params.split(',')))

    sigma = transform.get_LAB_L_SVD_s(image)
    sigma_interval = list(sigma[begin:end])
    sigma_log_based = [ log10(y) for y in sigma_interval ]

    # split in equals parts
    sigma_parts = np.array_split(sigma_log_based, split)

    return list([ get_entropy(part) for part in sigma_parts ])


def _extract_svd_entropy_norm_log10_split(image, params):

    begin, end, split = tuple(map(int, params.split(',')))

    sigma = transform.get_LAB_L_SVD_s(image)
    sigma_interval = list(normalize_arr_with_range(sigma[begin:end]))
    sigma_log_based = [ log10(y) for y in sigma_interval ]

    # split in equals parts
    sigma_parts = np.array_split(sigma_log_based, split)

    return list([ get_entropy(part) for part in sigma_parts ])


def _extract_svd_entropy_norm(image, params):
    
    begin, end = tuple(map(int, params.split(',')))

    sigma = transform.get_LAB_L_SVD_s(image)
    return get_entropy(normalize_arr_with_range(sigma[begin:end]))


def _extract_svd_entropy_blocks(image, params):
    
    w_block, h_block = tuple(map(int, params.split(',')))

    # get L channel
    L_channel = transform.get_LAB_L(image)

    # split in n block
    blocks = segmentation.divide_in_blocks(L_channel, (w_block, h_block))

    entropy_list = []

    for block in blocks:
        reduced_sigma = compression.get_SVD_s(block)
        reduced_entropy = get_entropy(reduced_sigma)
        entropy_list.append(reduced_entropy)

    return entropy_list


def _extract_svd_entropy_blocks_norm(image, params):

    return normalize_arr_with_range(_extract_svd_entropy_blocks(image, params))


def _extract_svd_entropy_blocks_permutation(image, params):
    
    w_block, h_block = tuple(map(int, params.split(',')))

    # get L channel
    L_channel = transform.get_LAB_L(image)

    # split in n block
    blocks = segmentation.divide_in_blocks(L_channel, (w_block, h_block))

    entropy_list = []

    for block in blocks:
        reduced_sigma = compression.get_SVD_s(block)
        reduced_entropy = get_entropy(reduced_sigma)
        entropy_list.append(reduced_entropy)

    return list(np.argsort(entropy_list))


def _extract_svd_entropy_blocks_permutation_norm(image, params):
    
    return normalize_arr_with_range(_extract_svd_entropy_blocks_permutation(image, params))


def _extract_entropy_blocks(image, params):
    
    w_block, h_block = tuple(map(int, params.split(',')))

    # get L channel
    L_channel = transform.get_LAB_L(image)

    # split in n block
    blocks = segmentation.divide_in_blocks(L_channel, (w_block, h_block))

    entropy_list = []

    for block in blocks:
        reduced_entropy = get_entropy(np.array(block).flatten())
        entropy_list.append(reduced_entropy)

    return entropy_list


def _extract_entropy_blocks_norm(image, params):

    return normalize_arr_with_range(_extract_entropy_blocks(image, params))


def _extract_entropy_blocks_permutation(image, params):
    
    w_block, h_block = tuple(map(int, params.split(',')))

    # get L channel
    L_channel = transform.get_LAB_L(image)

    # split in n block
    blocks = segmentation.divide_in_blocks(L_channel, (w_block, h_block))

    entropy_list = []

    for block in blocks:
        reduced_entropy = get_entropy(np.array(block).flatten())
        entropy_list.append(reduced_entropy)

    return list(np.argsort(entropy_list))


def _extract_entropy_blocks_permutation_norm(image, params):
    
    return normalize_arr_with_range(_extract_entropy_blocks_permutation(image, params))


def _extract_kolmogorov_complexity(image, params):

    bytes_data = np.array(image).tobytes()
    compress_data = gzip.compress(bytes_data)

    return sys.getsizeof(compress_data)


def extract_data(image, method, params = None):

    if method == 'svd_entropy':
        return _extract_svd_entropy(image, params)

    if method == 'svd_entropy_norm':
        return _extract_svd_entropy_norm(image, params)
        
    if method == 'mu_sigma':
        return _extract_mu_sigma(image, params)

    if method == 'svd':
        return _extract_svd(image, params)

    if method == 'svd_norm':
        return _extract_svd_norm(image, params)

    if method == 'svd_log10':
        return _extract_svd_log10(image, params)

    if method == 'svd_norm_log10':
        return _extract_svd_norm_log10(image, params)

    if method == 'stats_luminance':
        return _extract_stats_luminance(image, params)

    if method == 'svd_entropy_split':
        return _extract_svd_entropy_split(image, params)
    
    if method == 'svd_entropy_log10_split':
        return _extract_svd_entropy_log10_split(image, params)

    if method == 'svd_entropy_norm_split':
        return _extract_svd_entropy_norm_split(image, params)
    
    if method == 'svd_entropy_norm_log10_split':
        return _extract_svd_entropy_norm_log10_split(image, params)

    if method == 'kolmogorov_complexity':
        return _extract_kolmogorov_complexity(image, params)

    if method == 'svd_entropy_blocks':
        return _extract_svd_entropy_blocks(image, params)
    
    if method == 'svd_entropy_blocks_norm':
        return _extract_svd_entropy_blocks_norm(image, params)

    if method == 'svd_entropy_blocks_permutation':
        return _extract_svd_entropy_blocks_permutation(image, params)

    if method == 'svd_entropy_blocks_permutation_norm':
        return _extract_svd_entropy_blocks_permutation_norm(image, params)

    if method == 'entropy_blocks':
        return _extract_entropy_blocks(image, params)
    
    if method == 'entropy_blocks_norm':
        return _extract_entropy_blocks_norm(image, params)

    if method == 'entropy_blocks_permutation':
        return _extract_entropy_blocks_permutation(image, params)
    
    if method == 'entropy_blocks_permutation_norm':
        return _extract_entropy_blocks_permutation_norm(image, params)
    
    # no method found
    return None