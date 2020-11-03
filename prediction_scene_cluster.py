# main imports
import numpy as np
import pandas as pd
import sys, os, argparse
import joblib

# image processing
from PIL import Image
from ipfml import utils
from ipfml.processing import transform, segmentation, compression
import matplotlib.pyplot as plt

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt

from keras.models import load_model

from processing.features_extractions import extract_data
from complexity.run.estimators import estimate, estimators_list

zones_indices  = cfg.zones_indices
block_size     = (200, 200)


'''
Display progress information as progress bar
'''
def write_progress(progress):
    barWidth = 180

    output_str = "["
    pos = barWidth * progress
    for i in range(barWidth):
        if i < pos:
           output_str = output_str + "="
        elif i == pos:
           output_str = output_str + ">"
        else:
            output_str = output_str + " "

    output_str = output_str + "] " + str(int(progress * 100.0)) + " %\r"
    print(output_str)
    sys.stdout.write("\033[F")


def predict_ensemble(models, data, cluster_id, weight):
    """Predict label from list of models (return probability of label)

    Args:
        models ([Sequential]): models list
        data ([float]): input data for model
        cluster_id (int): cluster id of current zone (in order to target specific model)
        weight (int): weight of targeted cluster model
    """
    prob_sum = 0.
    clusters_weight = len(models) - 1 + weight

    for i in range(len(models)):
        
        prob = models[i].predict(data, batch_size=1)[0][0]

        # if cluster id model is targeted
        if i == cluster_id:
            prob = prob * weight

        #print('[clusterId: {0}] Model {1} predicts => {2}'.format(cluster_id, i, prob))

        prob_sum += prob
    
    return prob_sum / clusters_weight
        
def main():

    parser = argparse.ArgumentParser(description="Extract data from image dataset")

    parser.add_argument('--scene', type=str, help='folder path with all scenes', required=True)
    parser.add_argument('--cluster', type=str, help='clustering model to use', required=True)
    parser.add_argument('--nclusters', type=int, help='number of clusters', required=True)
    parser.add_argument('--estimators', type=str, help='list of estimators', default='l_mean,l_variance')
    parser.add_argument('--thresholds', type=str, help='file which contains all thresholds', required=True)
    parser.add_argument('--method', type=str, help='method name to used', choices=cfg.features_choices_labels, default=cfg.features_choices_labels[0])
    parser.add_argument('--params', type=str, help='param of the method used', default="", required=True)
    parser.add_argument('--sequence', type=int, help='sequence length expected', required=True)
    parser.add_argument('--seq_norm', type=int, help='normalization sequence by features', choices=[0, 1])
    parser.add_argument('--n_stop', type=int, help='n elements for stopping criteria', default=1)
    parser.add_argument('--models', type=str, help='folder with all models', required=True)
    parser.add_argument('--imnorm', type=int, help="specify if image is normalized before computing something", default=0, choices=[0, 1])
    parser.add_argument('--label', type=str, help='output label for each simulation', default='clusters')
    parser.add_argument('--output', type=str, help='output folder name with predictions', required=True)

    args = parser.parse_args()

    p_scene      = args.scene
    p_thresholds = args.thresholds
    p_cluster    = args.cluster
    p_nclusters  = args.nclusters
    p_estimators = [ i.strip() for i in args.estimators.split(',') ]
    p_output     = args.output
    p_method     = args.method
    p_params     = args.params
    p_sequence   = args.sequence
    p_seq_norm   = args.seq_norm
    p_n_stop     = args.n_stop
    p_models     = args.models
    p_label      = args.label
    p_imnorm     = args.imnorm

    # 1. Load cluster model
    cluster_model = joblib.load(p_cluster)

    # 2. Load each classification model
    models_path = sorted([ os.path.join(p_models, m) for m in os.listdir(p_models)])

    models_list = []

    for model_p in models_path:

        model = load_model(model_p)
        model.compile(loss='binary_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

        models_list.append(model)

    if len(models_list) != p_nclusters:
        raise('Error, number of loaded models is not correct, expecter {0}'.format(p_nclusters))

    # 3. extract all thresholds from threshold file
    human_thresholds = {}

    with open(p_thresholds) as f:
        thresholds_line = f.readlines()

        for line in thresholds_line:
            data = line.split(';')
            del data[-1] # remove unused last element `\n`
            scene = data[0]
            thresholds_scene = data[1:]

            human_thresholds[scene] = thresholds_scene

    # 4. Extract all images of scene
    images_path = []
    number_of_images = 0

    images_path = sorted([os.path.join(p_scene, img) for img in os.listdir(p_scene) if cfg.scene_image_extension in img])
    number_of_images = len(images_path)

    # 5. Construct here dictionnary of associated cluster for each block
    # Use of estimators passed as attribute
    # Use of the first image only
    clusters_block = {}

    first_image = images_path[0]

    blocks = segmentation.divide_in_blocks(Image.open(first_image), block_size)

    clusters_block[scene] = {}

    for id_b, block in enumerate(blocks):

        # extract data and write into file
        x = []
        
        for estimator in p_estimators:
            estimated = estimate(estimator, block)
            
            if not isinstance(estimated, np.float64):
                for v in estimated:
                    x.append(v)
            else:
                x.append(estimated)

        # call cluster model
        predicted_label = cluster_model.predict([x])[0]

        # add label for this specific zone
        clusters_block[id_b] = predicted_label

    # 6. get estimated thresholds using model and specific method
    image_indices = [ dt.get_scene_image_quality(img_path) for img_path in images_path ]

    blocks_sequence = []
    estimated_thresholds = []
    n_estimated_thresholds = []
    image_counter = 0

    # append empty list
    for _ in np.arange(16):
        blocks_sequence.append([])
        estimated_thresholds.append(None)
        n_estimated_thresholds.append(0)

    for img_i, img_path in enumerate(images_path):

        blocks = segmentation.divide_in_blocks(Image.open(img_path), (200, 200))

        for index, block in enumerate(blocks):
            
            if estimated_thresholds[index] is None:
                # normalize if necessary
                if p_imnorm:
                    block = np.array(block) / 255.

                blocks_sequence[index].append(np.array(extract_data(block, p_method, p_params)))

                # check if prediction is possible
                if len(blocks_sequence[index]) >= p_sequence:
                    data = np.array(blocks_sequence[index])
                    
                    if data.ndim == 1:
                        data = data.reshape(len(blocks_sequence[index]), 1)
                    else:
                        # check if sequence normalization is used
                        if p_seq_norm:

                            # check snorm process
                            #for _, seq in enumerate(data):
                                
                            s, f = data.shape
                            for i in range(f):
                                #final_arr[index][]
                                data[:, i] = utils.normalize_arr_with_range(data[:, i])
                                    
                    data = np.expand_dims(data, axis=0)
                    
                    prob = predict_ensemble(models_list, data, clusters_block[index], 2.5)
                    #print(index, ':', image_indices[img_i], '=>', prob)

                    if prob < 0.5:
                        n_estimated_thresholds[index] += 1

                        # if same number of detection is attempted
                        if n_estimated_thresholds[index] >= p_n_stop:
                            estimated_thresholds[index] = image_indices[img_i]
                    else:
                        n_estimated_thresholds[index] = 0

                    #print('Block @', index, ':', len(blocks_sequence[index]))
                    # delete first element (just like sliding window)
                    del blocks_sequence[index][0]

        # write progress bar
        write_progress((image_counter + 1) / number_of_images)
        
        image_counter = image_counter + 1
    
    # default label
    for i in np.arange(16):
        if estimated_thresholds[i] == None:
            estimated_thresholds[i] = image_indices[-1]

    # 7. check if learned zones
    if p_output is not None:
        with open(p_output, 'a') as f:
            f.write(p_label + ';')

            for t in estimated_thresholds:
                f.write(str(t) + ';')
            f.write('\n')

if __name__== "__main__":
    main()
