# main imports
import numpy as np
import pandas as pd
import sys, os, argparse
import random

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


def save_learned_zones(output_name, scene, zones):

    if not os.path.exists(cfg.output_zones_learned):
        os.makedirs(cfg.output_zones_learned)

    with open(os.path.join(cfg.output_zones_learned, output_name), 'a') as f:
        f.write(scene + ';')

        for zone in zones:
            f.write(str(zone) + ';')

        f.write('\n')

def main():

    parser = argparse.ArgumentParser(description="Read and compute entropy data file (using diff)")

    parser.add_argument('--data', type=str, help='entropy file data to read and compute')
    parser.add_argument('--output', type=str, help='output dataset prefix file used (saved into .train and .test extension)')
    parser.add_argument('--sequence', type=int, help='sequence length expected')
    parser.add_argument('--n_zones', type=int, help='number of zones used in train', default='')

    args = parser.parse_args()

    p_data         = args.data
    p_output       = args.output
    p_sequence     = args.sequence
    p_n_zones      = args.n_zones

    print("Number of zones used in train:", p_n_zones)

    # create output path if not exists
    p_output_path = os.path.join(cfg.output_datasets, p_output)
    p_output_path_train = os.path.join(p_output_path, p_output + '.train')
    p_output_path_test = os.path.join(p_output_path, p_output + '.test')

    if not os.path.exists(p_output_path):
        os.makedirs(p_output_path)

    # read line by line file to estimate threshold entropy stopping criteria
    f_train = open(p_output_path_train, 'w')
    f_test = open(p_output_path_test, 'w')

    zones = np.arange(16)

    with open(p_data, 'r') as f:
        lines = f.readlines()

        new_scene = False
        current_scene = None
        selected_zones = None

        for line in lines:

            data = line.split(';')

            # only if scene is used for training part
            scene_name = data[0]
            zones_index = int(data[1])
            threshold = int(data[3])
            image_indices = data[4].split(',')
            values_list = data[5].split(',')

            sequence_data = []
            # one element is removed using this function (first element of list for computing first difference)
            # TODO : remove previous and add new

            if current_scene == None:
                new_scene == True
                current_scene = scene_name
                random.shuffle(zones)
                selected_zones = zones[0:p_n_zones]
                save_learned_zones(p_output, scene_name, selected_zones)

            if scene_name != current_scene:
                new_scene = True
                random.shuffle(zones)
                selected_zones = zones[0:p_n_zones]
                save_learned_zones(p_output, scene_name, selected_zones)
            else:
                new_scene = False

            current_scene = scene_name

            for i, index in enumerate(image_indices):
                
                values = values_list[i].split(' ')

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

                    if zones_index in selected_zones:
                        f_train.write(line)
                    else:
                        f_test.write(line)

                    # del previous element
                    del sequence_data[0]

    f_test.close()
    f_train.close()    
    


if __name__== "__main__":
    main()
