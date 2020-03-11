# main imports
import numpy as np

# librairies imports
from ipfml.utils import get_entropy, normalize_arr
from ipfml.processing.transform import get_LAB_L_SVD_s

def _extract_svd(image, params):
    begin, end = tuple(map(int, params.split(',')))

    sigma = get_LAB_L_SVD_s(image)
    return sigma[begin:end]

def _extract_svd_norm(image, params):
    begin, end = tuple(map(int, params.split(',')))

    sigma = get_LAB_L_SVD_s(image)
    return normalize_arr(sigma[begin:end])

def _extract_mu_sigma(image, params):
    image = np.array(image)
    return list([np.mean(image), np.std(image)])

def _extract_svd_entropy(image, params):
    
    begin, end = tuple(map(int, params.split(',')))

    sigma = get_LAB_L_SVD_s(image)
    return get_entropy(sigma[begin:end])

def extract_data(image, method, params = None):

    if method == 'svd_entropy':
        return _extract_svd_entropy(image, params)

    if method == 'mu_sigma':
        return _extract_mu_sigma(image, params)

    if method == 'svd':
        return _extract_svd(image, params)

    if method == 'svd_norm':
        return _extract_svd_norm(image, params)
    
    # no method found
    return None