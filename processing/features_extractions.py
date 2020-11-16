# main imports
import numpy as np
import math
from math import log10
import gzip
import sys
import cv2


# librairies imports
from ipfml.utils import get_entropy, normalize_arr, normalize_arr_with_range
from ipfml.processing import transform, compression, segmentation

from skimage import color, restoration
from numpy.linalg import svd as lin_svd
from scipy.signal import medfilt2d, wiener, cwt
import pywt
import cv2

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
    
    # params : w_b, w_h, begin_sv, end_sv
    w_b, w_h, begin, end = tuple(map(int, params.split(',')))

    l_image = transform.get_LAB_L(image)
    blocks = segmentation.divide_in_blocks(l_image, (w_b, w_h))

    entropies = []

    for b in blocks:
        sigma = compression.get_SVD_s(b)
        entropies.append(get_entropy(sigma[begin:end]))

    return entropies

def _extract_svd_entropy_blocks_disordered(image, params):
    
    # params : w_b, w_h, begin_sv, end_sv
    w_b, w_h, begin, end = tuple(map(int, params.split(',')))

    l_image = transform.get_LAB_L(image)
    blocks = segmentation.divide_in_blocks(l_image, (w_b, w_h))

    entropies = []

    for b in blocks:
        sigma = compression.get_SVD_s(b)
        entropies.append(get_entropy(sigma[begin:end]))

    # change order of entropies
    indices = [] 
    index = 0 

    m_size = int(math.sqrt(len(entropies)))

    # compute b matrix in order to reorder elements using dot product
    b = np.arange(m_size * m_size).reshape(m_size, m_size)
    entropies = np.array(entropies).reshape(m_size, m_size)
    step_val = 3 # by default

    # set automatically indices of b matrix
    for i in range(m_size): 
        indices.append(index % m_size) 
        index += step_val

    for i in range(m_size): 
        b[i][indices[i]] = 1 

    # change order of entropies matrix
    entropies = np.linalg.multi_dot([entropies, b])

    for i in range(m_size):
        if i % index == 1:
            entropies[i] = np.concatenate((entropies[i][1:], entropies[i][:1]))

    return list(entropies.flatten())


def _extract_svd_entropy_blocks_divided(image, params):
    
    # params : w_b, w_h, begin_sv, end_sv
    w_b, w_h, begin, end = tuple(map(int, params.split(',')))

    l_image = transform.get_LAB_L(image)
    blocks = segmentation.divide_in_blocks(l_image, (w_b, w_h))

    entropies = []

    for b in blocks:
        sigma = compression.get_SVD_s(b)
        sigma_size = len(sigma)
        entropies.append(get_entropy(sigma[0:int(sigma_size/4)]))
        entropies.append(get_entropy(sigma[int(sigma_size/4):]))

    return entropies


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

def _extracts_complexity_stats(image, params):

    stats_attributes = []

    image = np.array(image)

    # get lightness
    lab_img = transform.get_LAB_L(image)

    # 1. extract sobol complexity with kernel 3
    sobelx = cv2.Sobel(lab_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(lab_img, cv2.CV_64F, 0, 1,ksize=3)

    sobel_mag = np.array(np.hypot(sobelx, sobely), 'uint8')  # magnitude

    stats_attributes.append(np.std(sobel_mag))

    # 2. extract sobol complexity with kernel 5
    sobelx = cv2.Sobel(lab_img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(lab_img, cv2.CV_64F, 0, 1,ksize=5)

    sobel_mag = np.array(np.hypot(sobelx, sobely), 'uint8')  # magnitude

    stats_attributes.append(np.std(sobel_mag))

    # 3. extract kolmogorov
    bytes_data = image.tobytes()
    compress_data = gzip.compress(bytes_data)

    mo_size = sys.getsizeof(compress_data) / 1024.
    go_size = mo_size / 1024.

    stats_attributes.append(go_size)

    # 4. extract svd_entropy
    begin, end = tuple(map(int, params.split(',')))

    sigma = compression.get_SVD_s(lab_img)
    stats_attributes.append(get_entropy(sigma[begin:end]))

    # 5. extract lightness mean
    stats_attributes.append(np.std(lab_img))

    # 6. extract lightness std
    stats_attributes.append(np.mean(lab_img))

    return list(stats_attributes)

def _filters_statistics(image, params):

    img_width, img_height = 200, 200

    lab_img = transform.get_LAB_L(image)
    arr = np.array(lab_img)

    # compute all filters statistics
    def get_stats(arr, I_filter):

        e1       = np.abs(arr - I_filter)
        L        = np.array(e1)
        mu0      = np.mean(L)
        A        = L - mu0
        H        = A * A
        E        = np.sum(H) / (img_width * img_height)
        P        = np.sqrt(E)

        return mu0, P
        # return np.mean(I_filter), np.std(I_filter)

    stats = []

    kernel = np.ones((3,3),np.float32)/9
    stats.append(get_stats(arr, cv2.filter2D(arr,-1,kernel)))

    kernel = np.ones((5,5),np.float32)/25
    stats.append(get_stats(arr, cv2.filter2D(arr,-1,kernel)))

    stats.append(get_stats(arr, cv2.GaussianBlur(arr, (3, 3), 0.5)))

    stats.append(get_stats(arr, cv2.GaussianBlur(arr, (3, 3), 1)))

    stats.append(get_stats(arr, cv2.GaussianBlur(arr, (3, 3), 1.5)))

    stats.append(get_stats(arr, cv2.GaussianBlur(arr, (5, 5), 0.5)))

    stats.append(get_stats(arr, cv2.GaussianBlur(arr, (5, 5), 1)))

    stats.append(get_stats(arr, cv2.GaussianBlur(arr, (5, 5), 1.5)))

    stats.append(get_stats(arr, medfilt2d(arr, [3, 3])))

    stats.append(get_stats(arr, medfilt2d(arr, [5, 5])))

    stats.append(get_stats(arr, wiener(arr, [3, 3])))

    stats.append(get_stats(arr, wiener(arr, [5, 5])))

    wave = w2d(arr, 'db1')
    stats.append(get_stats(arr, np.array(wave, 'float64')))

    data = []

    for stat in stats:
        data.append(stat[0])

    for stat in stats:
        data.append(stat[1])
    
    data = np.array(data)

    return list(data)

def _filters_statistics_norm(image, params):
    return normalize_arr_with_range(_filters_statistics(image, params))


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
    
    if method == 'svd_entropy_blocks_disordered':
        return _extract_svd_entropy_blocks_disordered(image, params)

    if method == 'svd_entropy_blocks_divided':
        return _extract_svd_entropy_blocks_divided(image, params)
    
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
    
    # TODO : add stats method
    # lightness complexity, mean, std, variance, what more ? kolmogorov ?
    if method == 'complexity_stats':
        return _extracts_complexity_stats(image, params)

    if method == 'filters_statistics':
        return _filters_statistics(image, params)
    
    if method == 'filters_statistics_norm':
        return _filters_statistics_norm(image, params)

    # no method found
    return None


def w2d(arr, mode):
    # convert to float   
    imArray = arr
    # np.divide(imArray, 100) # because of lightness channel, use of 100

    # compute coefficients 
    # same to: LL (LH, HL, HH)
    # cA, (cH, cV, cD) = pywt.dwt2(imArray, mode)
    # cA *= 0 # remove low-low sub-bands data

    # reduce noise from the others cofficients
    # LH, HL and HH
    # ----
    # cannot use specific method to predict thresholds...
    # use of np.percentile(XX, 5) => remove data under 5 first percentile
    # cH = pywt.threshold(cH, np.percentile(cH, 5), mode='soft')
    # cV = pywt.threshold(cV, np.percentile(cV, 5), mode='soft')
    # cD = pywt.threshold(cD, np.percentile(cD, 5), mode='soft')

    # reconstruction
    # imArray_H = pywt.idwt2((cA, (cH, cV, cD)), mode)
    # print(np.min(imArray_H), np.max(imArray_H), np.mean(imArray_H))
    # imArray_H *= 100 # because of lightness channel, use of 100
    # imArray_H = np.array(imArray_H)

    # coeffs = pywt.wavedec2(imArray, mode, level=2)

    # #Process Coefficients
    # coeffs_H=list(coeffs)  
    # coeffs_H[0] *= 0;  

    # # reconstruction
    # imArray_H=pywt.waverec2(coeffs_H, mode)
    # print(np.min(imArray_H), np.max(imArray_H), np.mean(imArray_H))

    # using skimage
    sigma = restoration.estimate_sigma(imArray, average_sigmas=True, multichannel=False)
    imArray_H = restoration.denoise_wavelet(imArray, sigma=sigma, wavelet='db1', mode='soft', 
        wavelet_levels=2, 
        multichannel=False, 
        convert2ycbcr=False, 
        method='VisuShrink', 
        rescale_sigma=True)

    # imArray_H *= 100

    return imArray_H
