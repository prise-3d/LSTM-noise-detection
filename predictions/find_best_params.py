# main imports
import numpy as np
import pandas as pd
import sys, os, argparse
import math

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

step_indices = 20
last_index = 10000

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

def predict_on_scene(zones_predictions, limit_prob, nstop, image_indices, every):

    zones_thresholds = []
    # 6. compute expected thresholds
    for predictions in zones_predictions:

        n_estimated = 0
        found = False
        for prob_index, prob in enumerate(predictions):

            if prob < limit_prob:
                n_estimated += 1

                # if same number of detection is attempted
                if n_estimated >= nstop:
                    zones_thresholds.append(image_indices[prob_index * every])
                    found = True
                    break
            else:
                n_estimated = 0
        
        if not found:
            zones_thresholds.append(image_indices[-1])

    return zones_thresholds

def main():

    parser = argparse.ArgumentParser(description="Read and process predictions file in order to find best params")

    parser.add_argument('--predictions', type=str, help='predictions files folder of each scene')
    parser.add_argument('--sequence', type=int, help='sequence length expected')
    parser.add_argument('--thresholds', type=str, help='file which cantains all thresholds')
    parser.add_argument('--learned_zones', type=str, help="Filename which specifies if zones are learned or not and which zones", default="")
    parser.add_argument('--every', type=int, help="every images only", default=1)

    args = parser.parse_args()

    p_predictions = args.predictions
    p_sequence    = args.sequence
    p_thresholds  = args.thresholds
    p_zones       = args.learned_zones
    p_every       = args.every

    human_thresholds = {}

    # 1. retrieve human_thresholds
    # construct zones folder
    with open(p_thresholds) as f:
        thresholds_line = f.readlines()

        for line in thresholds_line:
            data = line.split(';')
            del data[-1] # remove unused last element `\n`
            current_scene = data[0]
            thresholds_scene = data[1:]

            human_thresholds[current_scene] = [ int(threshold) for threshold in  thresholds_scene ]

    scenes_predictions = {}

    for key, _ in human_thresholds.items():

        zones_predictions = []

        prediction_file = os.path.join(p_predictions, key + '.csv')

        # Get predictions from model (let by default to 1 the first sequence values)
        with open(prediction_file, 'r') as f:
            lines = f.readlines()

            for line in lines:
                data = line.split(';')

                predictions = []
                
                for _ in range(p_sequence - 1):
                    predictions.append(1)

                for v in data[2:-1]:
                    predictions.append(float(v))

                zones_predictions.append(predictions)

        scenes_predictions[key] = zones_predictions

    # 2. check if learned zones
    zones_learned = {}

    if len(p_zones) > 0:
        with open(p_zones, 'r') as f:
            lines = f.readlines()

            for line in lines:
                data = line.split(';')

                zones_selected = data[1:]
                del zones_selected[-1]
                zones_learned[data[0]] = [ int(zone) for zone in zones_selected ]

    # 3. Compute optimisation process
    images_indices = np.arange(step_indices, last_index + step_indices, step_indices)

    best_params = None
    min_error = sys.maxsize

    nstop_values = np.arange(20) + 1
    prob_values = np.arange(0, 0.5, 0.01) + 0.01

    ncombinations = len(nstop_values) * len(prob_values)
    ncounter = 0
    
    for c_nstop in nstop_values:
        for c_prob in prob_values:
    
            sum_error = 0
            for key, scene_thresholds in human_thresholds.items():

                zones_predictions = scenes_predictions[key]

                predicted_thresholds = predict_on_scene(zones_predictions, c_prob, c_nstop, images_indices, p_every)

                if key in zones_learned:
                    measured_error = sum([ abs(predicted_thresholds[i] - scene_thresholds[i]) for i in zones_learned[key] ])
                else:
                    measured_error = sum([ abs(p - scene_thresholds[i]) for i, p in enumerate(predicted_thresholds) ])

                # using custom error
                # measured_error = 0

                # for i, p in enumerate(predicted_thresholds):
                #     if p - scene_thresholds[i] < 0:
                #         measured_error += abs(p - scene_thresholds[i]) * 3
                #     else:
                #         measured_error += abs(p - scene_thresholds[i])
                
                sum_error += measured_error

            if sum_error < min_error:
                min_error = sum_error
                best_params = {'p': c_prob, 'nstop': c_nstop }

            ncounter += 1
            write_progress(float(ncounter / ncombinations))

    print()
    print('Best found is', best_params, 'with error of', min_error)
    
if __name__== "__main__":
    main()
