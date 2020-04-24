# main imports
import numpy as np
import pandas as pd
import sys, os, argparse

# image processing
from PIL import Image
from ipfml import utils
from ipfml.processing import transform, segmentation

import matplotlib.pyplot as plt

# model imports
import joblib

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg
from modules.utils import data as dt

from processing.features_extractions import extract_data

dataset_folder = cfg.dataset_path
scenes_list    = cfg.scenes_names
zones_indices  = cfg.zones_indices

output_figures = cfg.output_figures

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

def display_estimated_thresholds(scene, estimated, humans, max_index, zones_learned=None, p_save=False):
    
    colors = ['C0', 'C1', 'C2', 'C3']
    
    plt.figure(figsize=(25, 20))
    plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
    
    plt.plot(estimated, 
             color=colors[0], 
             label='Estimated thresholds')

    
    plt.plot(humans, 
             color=colors[1], 
             label='Human thresholds')
        

    if zones_learned is not None:
        del zones_learned[-1]
        zones_learned = [ int(i) for i in zones_learned ]

        for x, estimation in enumerate(estimated):

            if x in zones_learned:
                plt.scatter(x, estimation, s=30, marker='H', color='red')
    

    plt.ylim(0, max_index) 
    plt.xticks(zones_indices)
    plt.title('Comparisons of estimated vs human thresholds for ' + scene, fontsize=22)
    plt.legend(fontsize=20)

    if p_save:

        if not os.path.exists(output_figures):
            os.makedirs(output_figures)
        plt.savefig(os.path.join(output_figures, 'thresholds_' + scene + '.png'))
    else:
        plt.show()

def main():

    parser = argparse.ArgumentParser(description="Read and compute entropy data file")

    parser.add_argument('--model', type=str, help='entropy file data with estimated threshold to read and compute')
    parser.add_argument('--method', type=str, help='method name to used', choices=cfg.features_choices_labels, default=cfg.features_choices_labels[0])
    parser.add_argument('--params', type=str, help='param of the method used', default="")
    parser.add_argument('--sequence', type=int, help='sequence length expected')
    parser.add_argument('--n_stop', type=int, help='n elements for stopping criteria', default=1)
    parser.add_argument('--imnorm', type=int, help="specify if image is normalized before computing something", default=0, choices=[0, 1])
    parser.add_argument('--learned_zones', type=str, help="Filename which specifies if zones are learned or not and which zones", default="")
    parser.add_argument('--scene', type=str, help='Scene index to use', choices=cfg.scenes_indices)
    parser.add_argument('--save', type=int, help='save or not figure', choices=[0, 1])

    args = parser.parse_args()

    p_model    = args.model
    p_method   = args.method
    p_params   = args.params
    p_sequence = args.sequence
    p_n_stop   = args.n_stop
    p_imnorm   = args.imnorm
    p_zones    = args.learned_zones
    p_scene    = args.scene
    p_save     = bool(args.save)

    # 1. get scene name
    scenes_list = cfg.scenes_names
    scenes_indices = cfg.scenes_indices

    scene_index = scenes_indices.index(p_scene.strip())
    scene = scenes_list[scene_index]

    scene_path = os.path.join(cfg.dataset_path, scene)

    # 2. load model and compile it
    model = joblib.load(p_model)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


    estimated_thresholds = []
    n_estimated_thresholds = []
    human_thresholds = []

    # 3. retrieve human_thresholds
    # construct zones folder
    zones_list = []

    for index in zones_indices:

        index_str = str(index)

        while len(index_str) < 2:
            index_str = "0" + index_str
        
        zones_list.append(cfg.zone_folder + index_str)

    for zone in zones_list:
            zone_path = os.path.join(scene_path, zone)

            with open(os.path.join(zone_path, cfg.seuil_expe_filename), 'r') as f:
                human_thresholds.append(int(f.readline()))

    # 4. get estimated thresholds using model and specific method
    images_path = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])
    number_of_images = len(images_path)
    image_indices = [ dt.get_scene_image_quality(img_path) for img_path in images_path ]

    blocks_sequence = []
    image_counter = 0

    print(human_thresholds)

    # append empty list
    for zone in zones_list:
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

                    data = np.expand_dims(data, axis=0)
                    
                    prob = model.predict(data, batch_size=1)[0][0]
                    #print(index, ':', image_indices[img_i], '=>', prob)

                    if prob < 0.5:
                        n_estimated_thresholds[index] += 1

                        # if same number of detection is attempted
                        if n_estimated_thresholds[index] >= p_n_stop:
                            estimated_thresholds[index] = image_indices[img_i]
                    else:
                        n_estimated_thresholds[index] = 0

                    # delete first element (just like sliding window)
                    del blocks_sequence[index][0]

        # write progress bar
        write_progress((image_counter + 1) / number_of_images)
        
        image_counter = image_counter + 1
    
    # default label
    for i, _ in enumerate(zones_list):
        if estimated_thresholds[i] == None:
            estimated_thresholds[i] = image_indices[-1]

    # 5. check if learned zones
    zones_learned = None

    if len(p_zones) > 0:
        with open(p_zones, 'r') as f:
            lines = f.readlines()

            for line in lines:
                data = line.split(';')

                if data[0] == scene:
                    zones_learned = data[1:]

    # 6. display results
    display_estimated_thresholds(scene, estimated_thresholds, human_thresholds, image_indices[-1], zones_learned, p_save)

if __name__== "__main__":
    main()
