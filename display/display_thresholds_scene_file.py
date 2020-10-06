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
from keras.models import load_model

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
    plt.rc('xtick', labelsize=22)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=22)    # fontsize of the tick labels
    
    plt.plot(estimated, 
             color=colors[0], 
             label='Estimated thresholds',
             lw=3)

    
    plt.plot(humans, 
             color=colors[1], 
             label='Human thresholds', 
             lw=3)
        

    plt.xticks(zones_indices)

    if zones_learned:

        for i in cfg.zones_indices:
            if i in zones_learned:
                
                plt.plot([i, i], [0, max_index], '--', color='black', alpha=0.5)
                plt.gca().get_xticklabels()[i].set_color('black')
            else:
                plt.plot([i, i], [0, max_index], '-.', color='red', alpha=0.5)
                plt.gca().get_xticklabels()[i].set_color('red')
    
    plt.xlabel('Image zone indices', fontsize=28)
    plt.ylabel('Number of samples', fontsize=28)

    plt.ylim(0, max_index) 
    plt.title('Comparisons of estimated vs human thresholds for ' + scene, fontsize=22)
    plt.legend(fontsize=26)

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
    parser.add_argument('--selected_zones', type=str, help="Filename which specifies if zones are learned or not and which zones", default="")
    parser.add_argument('--scene', type=str, help='Scene folder to use')
    parser.add_argument('--save', type=int, help='save or not figure', choices=[0, 1])
    parser.add_argument('--thresholds', type=str, help='file which cantains all thresholds')
    parser.add_argument('--save_thresholds', type=str, help='save or not thresholds')
    parser.add_argument('--label_thresholds', type=str, help='thresholds method label')
    parser.add_argument('--seq_norm', type=int, help='normalization sequence by features', choices=[0, 1])

    args = parser.parse_args()

    p_model      = args.model
    p_method     = args.method
    p_params     = args.params
    p_sequence   = args.sequence
    p_n_stop     = args.n_stop
    p_imnorm     = args.imnorm
    p_zones      = args.selected_zones
    p_scene      = args.scene
    p_save       = bool(args.save)
    p_thresholds = args.thresholds
    p_save_thresholds = args.save_thresholds
    p_label_thresholds = args.label_thresholds
    p_seq_norm   = bool(args.seq_norm)

    # 1. get scene name
    scene_path = os.path.join(cfg.dataset_path, p_scene)

    # 2. load model and compile it
    model = load_model(p_model)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


    estimated_thresholds = []
    n_estimated_thresholds = []
    human_thresholds = []

    _, scene_name = os.path.split(p_scene)

    # 3. retrieve human_thresholds
    # construct zones folder
    with open(p_thresholds) as f:
        thresholds_line = f.readlines()

        for line in thresholds_line:
            data = line.split(';')
            del data[-1] # remove unused last element `\n`
            current_scene = data[0]
            thresholds_scene = data[1:]

            if scene_name == current_scene:
                human_thresholds = [ int(threshold) for threshold in  thresholds_scene ]


    # 4. get estimated thresholds using model and specific method
    images_path = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])
    number_of_images = len(images_path)
    image_indices = [ dt.get_scene_image_quality(img_path) for img_path in images_path ]

    blocks_sequence = []
    image_counter = 0

    print(human_thresholds)

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
                    
                    prob = model.predict(data, batch_size=1)[0][0]
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

    # 5. check if learned zones
    zones_learned = None

    if len(p_zones) > 0:
        with open(p_zones, 'r') as f:
            lines = f.readlines()

            for line in lines:
                data = line.split(';')

                if data[0] == current_scene:
                    zones_learned = data[1:]

    if p_save_thresholds is not None:
        with open(p_save_thresholds, 'a') as f:
            f.write(p_label_thresholds + ';')

            for t in estimated_thresholds:
                f.write(str(t) + ';')
            f.write('\n')

    # 6. display results
    display_estimated_thresholds(scene_name, estimated_thresholds, human_thresholds, image_indices[-1], zones_learned, p_save)

if __name__== "__main__":
    main()
