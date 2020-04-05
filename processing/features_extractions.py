# main imports
import numpy as np
from math import log10

# librairies imports
from ipfml.utils import get_entropy, normalize_arr
from ipfml.processing import transform, compression

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
    return list(normalize_arr(sigma[begin:end]))

def _extract_svd_norm_log10(image, params):
    begin, end = tuple(map(int, params.split(',')))

    sigma = transform.get_LAB_L_SVD_s(image)
    sigma_interval = list(normalize_arr(sigma[begin:end]))
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
    sigma_interval = list(normalize_arr(sigma[begin:end]))

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
    sigma_interval = list(normalize_arr(sigma[begin:end]))
    sigma_log_based = [ log10(y) for y in sigma_interval ]

    # split in equals parts
    sigma_parts = np.array_split(sigma_log_based, split)

    return list([ get_entropy(part) for part in sigma_parts ])

def _extract_svd_entropy_norm(image, params):
    
    begin, end = tuple(map(int, params.split(',')))

    sigma = transform.get_LAB_L_SVD_s(image)
    return get_entropy(normalize_arr(sigma[begin:end]))

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
    
    # no method found
    return None