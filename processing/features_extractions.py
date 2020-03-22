# main imports
import numpy as np

# librairies imports
from ipfml.utils import get_entropy, normalize_arr
from ipfml.processing import transform, compression

def _extract_svd(image, params):
    begin, end = tuple(map(int, params.split(',')))

    sigma = transform.get_LAB_L_SVD_s(image)
    return sigma[begin:end]

def _extract_stats_luminance(image, params):
    L = transform.get_LAB_L(image)
    sigma = compression.get_SVD_s(L)
    return list([np.mean(L), np.std(L), get_entropy(sigma)])

def _extract_svd_norm(image, params):
    begin, end = tuple(map(int, params.split(',')))

    sigma = transform.get_LAB_L_SVD_s(image)
    return normalize_arr(sigma[begin:end])

def _extract_mu_sigma(image, params):
    image = np.array(image)
    return list([np.mean(image), np.std(image)])

def _extract_svd_entropy(image, params):
    
    begin, end = tuple(map(int, params.split(',')))

    sigma = transform.get_LAB_L_SVD_s(image)
    return get_entropy(sigma[begin:end])

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

    if method == 'stats_luminance':
        return _extract_stats_luminance(image, params)
    
    # no method found
    return None