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

    parser = argparse.ArgumentParser(description="Read and process prediction file")

    parser.add_argument('--predictions', type=str, help='prediction file of scene')
    parser.add_argument('--sequence', type=int, help='sequence length expected')
    parser.add_argument('--thresholds', type=str, help='file which cantains all thresholds')
    parser.add_argument('--every', type=int, help="every images only", default=1)
    parser.add_argument('--nstop', type=int, help='number of predictions before estimating label', default=1)
    parser.add_argument('--prob', type=float, help='max probability for not noisy label', default=0.5)
    parser.add_argument('--scene', type=str, help='Scene folder to use')
    parser.add_argument('--output', type=str, help='output filename')

    args = parser.parse_args()

    p_prediction = args.predictions
    p_sequence = args.sequence
    p_thresholds = args.thresholds
    p_every    = args.every
    p_nstop    = args.nstop
    p_prob     = args.prob
    p_scene    = args.scene
    p_output   = args.output

    # 1. get scene name
    scene_path = p_scene

    if scene_path[-1] == '/':
        scene_path = scene_path[:-1]

    _, scene_name = os.path.split(scene_path)


    human_thresholds = []

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

    print('Human')
    print(human_thresholds)
    
    images_path = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])
    number_of_images = len(images_path)
    image_indices = [ dt.get_scene_image_quality(img_path) for img_path in images_path ]

    zones_predictions = []
    # Get predictions from model (let by default to 1 the first sequence values)
    with open(p_prediction, 'r') as f:
        lines = f.readlines()

        for line in lines:
            data = line.split(';')

            predictions = []
            
            for _ in range(p_sequence - 1):
                predictions.append(1)

            for v in data[2:-1]:
                predictions.append(float(v))

            zones_predictions.append(predictions)

    zones_thresholds = []
    # 6. compute expected thresholds
    for predictions in zones_predictions:

        n_estimated = 0
        found = False
        for prob_index, prob in enumerate(predictions):

            if prob < p_prob:
                n_estimated += 1

                # if same number of detection is attempted
                if n_estimated >= p_nstop:
                    zones_thresholds.append(image_indices[prob_index * p_every])
                    found = True
                    break
            else:
                n_estimated = 0
        
        if not found:
            zones_thresholds.append(image_indices[-1])
    
    print('Predicted')
    print(zones_thresholds)

    with open(p_output, 'a') as f:

        f.write(scene_name + ';')

        for v in zones_thresholds:
            f.write(str(v) + ';')
        
        f.write('\n')
        
    

if __name__== "__main__":
    main()
