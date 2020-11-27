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
import tensorflow as tf

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

def main():

    parser = argparse.ArgumentParser(description="Read and compute entropy data file")

    parser.add_argument('--model', type=str, help='entropy file data with estimated threshold to read and compute')
    parser.add_argument('--dataset', type=str, help='datasets file to load and predict from')
    parser.add_argument('--output', type=str, help="output folder")

    args = parser.parse_args()

    p_model      = args.model
    p_dataset    = args.dataset
    p_output     = args.output

    # 2. load model and compile it
    model = load_model(p_model)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    if not os.path.exists(p_output):
        os.makedirs(p_output)

    scene_predictions = {}
    data_lines = []

    dataset_files = os.listdir(p_dataset)

    for filename in dataset_files:
        filename_path = os.path.join(p_dataset, filename)

        with open(filename_path, 'r') as f:
            for line in f.readlines():
                data_lines.append(line)

    nlines = len(data_lines)
    ncounter = 0

    for line in data_lines:
        data = line.split(';')

        scene_name = data[0]
        zone_index = int(data[1])

        if scene_name not in scene_predictions:
            scene_predictions[scene_name] = []

            for _ in range(16):
                scene_predictions[scene_name].append([])

        # prepare input data
        sequence_data = np.array([ l.split(' ') for l in data[4:] ], 'float32')
        sequence_data = np.expand_dims(sequence_data, axis=0)
                
        prob = model.predict(sequence_data, batch_size=1)[0][0]

        scene_predictions[scene_name][zone_index].append(prob)

        ncounter += 1
        write_progress(float(ncounter / nlines))


    # 6. save predictions results
    for key, blocks_predictions in scene_predictions.items():

        output_file = os.path.join(p_output, key + '.csv')

        f = open(output_file, 'w')
        for i, data in enumerate(blocks_predictions):
            f.write(scene_name + ';')
            f.write(str(i) + ';')

            for v in data:
                f.write(str(v) + ';')
            
            f.write('\n')
        f.close()


if __name__== "__main__":
    main()
