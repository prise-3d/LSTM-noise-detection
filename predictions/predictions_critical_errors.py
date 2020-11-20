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

    parser.add_argument('--folders', type=str, help='folder with data predictions')
    parser.add_argument('--dataset', type=str, help='scenes dataset folder')
    parser.add_argument('--sequences', type=str, help='sequence length expected of each model')
    parser.add_argument('--thresholds', type=str, help='file which cantains all thresholds')
    parser.add_argument('--learned_zones', type=str, help="Filename which specifies if zones are learned or not and which zones", default="")
    parser.add_argument('--every', type=str, help="every list for images only", default=1)
    parser.add_argument('--margins', type=str, help="\"accepted\" margins percent error list")
    parser.add_argument('--output', type=str, help='output image file name')

    args = parser.parse_args()

    p_folders  = [ f.strip() for f in args.folders.split(',') ]
    p_dataset  = args.dataset
    p_sequences = [ int(s.strip()) for s in args.sequences.split(',') ]
    p_thresholds = args.thresholds
    p_zones    = args.learned_zones
    p_every    = [ int(s.strip()) for s in args.every.split(',') ]
    p_margins   = [ float(m.strip()) for m in args.margins.split(',') ]
    p_output   = args.output

    if not os.path.exists(p_output):
        os.makedirs(p_output)

    predictions_folders = {}

    print(p_folders)

    for index_f, folder in enumerate(p_folders):

        print(os.path.abspath(folder))
        predictions_folders[folder] = {}

        every = p_every[index_f]
        sequence = p_sequences[index_f]

        for margin in p_margins:

            print('Predictions over `{0}` folder, with margin: {1}'.format(folder, margin))

            # 1. get all thresholds
            human_thresholds = {}

            # 2. retrieve human_thresholds
            # construct zones folder
            with open(p_thresholds) as f:
                thresholds_line = f.readlines()

                for line in thresholds_line:
                    data = line.split(';')
                    del data[-1] # remove unused last element `\n`
                    current_scene = data[0]
                    thresholds_scene = data[1:]

                    human_thresholds[current_scene] = [ int(threshold) for threshold in  thresholds_scene ]

            # 3. check if learned zones
            zones_learned = {}

            if len(p_zones) > 0:
                with open(p_zones, 'r') as f:
                    lines = f.readlines()

                    for line in lines:
                        data = line.split(';')

                        zones_selected = data[1:]
                        del zones_selected[-1]
                        zones_learned[data[0]] = [ int(zone) for zone in zones_selected ]

            # 4. get all predictions
            predictions_files = sorted(os.listdir(folder))

            scene_critical_predictions = {}
            global_percent = []

            for predictions in predictions_files:

                prediction_filepath = os.path.join(folder, predictions)
                zones_predictions = []

                # csv file need to have scene name as prefix
                scene_name = predictions.replace('.csv', '')

                if scene_name in human_thresholds:

                    # Get predictions from model (let by default to 1 the first sequence values)
                    with open(prediction_filepath, 'r') as f:
                        lines = f.readlines()

                        for line in lines:
                            data = line.split(';')

                            predictions = []
                            
                            for _ in range(sequence - 1):
                                predictions.append(1)

                            for v in data[2:-1]:
                                predictions.append(v)

                            zones_predictions.append(predictions)

                    scene_path = os.path.join(p_dataset, scene_name)

                    images_path = sorted([os.path.join(scene_path, img) for img in os.listdir(scene_path) if cfg.scene_image_extension in img])
                    number_of_images = len(images_path)
                    image_indices = [ dt.get_scene_image_quality(img_path) for img_path in images_path ]
                    step_index = image_indices[1] - image_indices[0]

                    # print('Scene used is', scene_name)
                    # print('nimages =>', number_of_images)
                    # print('zones_predictions =>', len(zones_predictions[0]))

                    scene_critical_predictions[scene_name] = []

                    for i in range(len(zones_predictions)):

                        critical_predictions = []

                        min_margin_threshold = human_thresholds[scene_name][i] - int(len(image_indices) * margin) * step_index
                        max_margin_threshold = human_thresholds[scene_name][i] + int(len(image_indices) * margin) * step_index

                        # if human thresholds is 10000, escape as all images is noisy
                        if human_thresholds[scene_name][i] == 10000:
                            min_margin_threshold = 10000
                            max_margin_threshold = 10000

                        # print(i, '=>', min_margin_threshold, human_thresholds[scene_name][i], max_margin_threshold)

                        pred_index = 0
                        
                        for j in range(len(image_indices)):

                            if j % every == 0:
                            
                                pred_label = 0 if float(zones_predictions[i][pred_index]) < 0.5 else 1
                                
                                critical_error = None

                                if image_indices[j] <= min_margin_threshold and pred_label == 1:
                                    critical_error = 0 # no error

                                if image_indices[j] >= max_margin_threshold and pred_label == 0:
                                    critical_error = 0

                                if image_indices[j] > min_margin_threshold and image_indices[j] < max_margin_threshold:
                                    critical_error = 0
                                
                                if critical_error is None:
                                    critical_error = 1

                                critical_predictions.append(critical_error)

                                pred_index += 1

                        scene_critical_predictions[scene_name].append(critical_predictions)

                    arr = np.array(scene_critical_predictions[scene_name]).flatten()
                    percent = np.count_nonzero(arr == 1, axis=0) / len(arr)
                    global_percent.append(percent)
                    print('[margin: {0}] Critical percent error on {1} is {2:.3f}%'.format(margin, scene_name, percent * 100))
            
            predictions_folders[folder][margin] = scene_critical_predictions

            print('Global percent is {0}%'.format(sum(global_percent) / len(global_percent)))

    scene_predictions = {}

    global_predictions = []

    for key, folder in predictions_folders.items():
        
        for key_pred, scenes in folder.items():
            
            model_predictions = []
            for key_scene, predictions  in scenes.items():
                print(key_scene)

                if key_scene not in scene_predictions:
                    scene_predictions[key_scene] = []

                arr = np.array(predictions).flatten()
                percent = np.count_nonzero(arr == 1, axis=0) / len(arr)
                scene_predictions[key_scene].append(percent)
                model_predictions.append(percent)
            global_predictions.append(sum(model_predictions) / len(model_predictions))

    print(global_predictions)

    
    line = " & "
    line_margin = " & "
    for index_f, folder in enumerate(p_folders):

        _, model = os.path.split(folder)

        for margin in p_margins:
            line += model + ' & '
            line_margin += '{0:.0f} & '.format(margin * 100)

    print(line[:-2] + ' \\\\')
    print(line_margin[:-2] + ' \\\\ \n\hline')

    for key_scene, errors in scene_predictions.items():

        line = key_scene.replace('p3d_', '').replace('_part6', '').replace('-', ' ').replace('view', 'view ').replace('_', ' ').capitalize()
        for error in errors:

            percent_error = error * 100

            cell_str = ' & {0:.2f}'

            if percent_error < 3:
                cell_str = ' & \\cellcolor{{green!55}}{0:.2f}'
            
            if percent_error < 5:
                cell_str = ' & \\cellcolor{{green!25}}{0:.2f}'

            if percent_error < 10:
                cell_str = ' & \\cellcolor{{green!10}}{0:.2f}'
            
            if percent_error > 15:
                cell_str = ' & \\cellcolor{{red!25}}{0:.2f}'

            if percent_error > 20:
                cell_str = ' & \\cellcolor{{red!55}}{0:.2f}'

            line += cell_str.format(percent_error)

        print(line + ' \\\\')
    print('\\hline')
    
    global_line = "\\textbf{Global critical error (in \%)} "

    for pred in global_predictions:
        
        percent_error = pred * 100

        cell_str = ' & {0:.2f}'

        if percent_error < 3:
            cell_str = ' & \\cellcolor{{green!55}}{0:.2f}'
        
        if percent_error < 5:
            cell_str = ' & \\cellcolor{{green!25}}{0:.2f}'
        
        if percent_error < 10:
            cell_str = ' & \\cellcolor{{green!10}}{0:.2f}'

        if percent_error > 15:
            cell_str = ' & \\cellcolor{{red!25}}{0:.2f}'

        if percent_error > 20:
            cell_str = ' & \\cellcolor{{red!55}}{0:.2f}'

        global_line += cell_str.format(percent_error)

    print(global_line + '\\\\ \n\hline')

                
            
            

if __name__== "__main__":
    main()
