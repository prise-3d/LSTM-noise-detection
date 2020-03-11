# main imports
import numpy as np
import pandas as pd
import sys, os, argparse

# image processing
from PIL import Image
from ipfml import utils
from ipfml.processing import transform, segmentation

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg

dataset_folder = cfg.dataset_path
scenes_list    = cfg.scenes_names
zones_indices  = cfg.zones_indices


def main():

    parser = argparse.ArgumentParser(description="Read and compute entropy data file (using diff)")

    parser.add_argument('--data', type=str, help='entropy file data to read and compute')
    parser.add_argument('--output', type=str, help='output dataset prefix file used (saved into .train and .test extension)')
    parser.add_argument('--sequence', type=int, help='sequence length expected')
    parser.add_argument('--train_scenes', type=str, help='list of train scenes used', default='')

    args = parser.parse_args()

    p_data         = args.data
    p_output       = args.output
    p_sequence     = args.sequence
    p_train_scenes = args.train_scenes.split(',')

     # list all possibles choices of renderer
    scenes_list = cfg.scenes_names
    scenes_indices = cfg.scenes_indices

    # getting scenes from indexes user selection
    scenes_selected = []

    # if training set is empty then use all scenes
    if p_train_scenes[0] == '':
        scenes_selected = scenes_list
    else:
        for scene_id in p_train_scenes:
            index = scenes_indices.index(scene_id.strip())
            scenes_selected.append(scenes_list[index])

    print("Scenes used in train:", scenes_selected)

    # create output path if not exists
    p_output_path = os.path.join(cfg.output_datasets, p_output)
    p_output_path_train = os.path.join(p_output_path, p_output + '.train')
    p_output_path_test = os.path.join(p_output_path, p_output + '.test')

    if not os.path.exists(p_output_path):
        os.makedirs(p_output_path)

    # read line by line file to estimate threshold entropy stopping criteria
    f_train = open(p_output_path_train, 'w')
    f_test = open(p_output_path_test, 'w')

    with open(p_data, 'r') as f:
        lines = f.readlines()

        for line in lines:

            data = line.split(';')

            # only if scene is used for training part
            scene_name = data[0]
            threshold = int(data[3])
            image_indices = data[4].split(',')
            values_list = data[5].split(',')

            sequence_data = []
            # one element is removed using this function (first element of list for computing first difference)
            # TODO : remove previous and add new
            
            for i, index in enumerate(image_indices):
                
                values = values_list[i].split(' ')
                nb_elements = len(values)

                # append new sequence
                sequence_data.append(values)

                if i + 1 >= p_sequence:

                    label = int(threshold > int(index))

                    line = str(label) + ';'

                    for index_v, values in enumerate(sequence_data):

                        for index_x, x in enumerate(values):
                            line += str(x)

                            if index_x + 1 < len(values):
                                line += ' '

                        if index_v + 1 < len(sequence_data):
                            line += ';'

                    line += '\n'

                    if scene_name in scenes_selected:
                        f_train.write(line)
                    else:
                        f_test.write(line)

                    # del previous element
                    del sequence_data[0]

    f_test.close()
    f_train.close()    
    


if __name__== "__main__":
    main()
