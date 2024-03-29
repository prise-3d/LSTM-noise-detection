import os
import argparse
import sys
import subprocess

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg

def main():


    parser = argparse.ArgumentParser(description="Read and process prediction file")

    parser.add_argument('--dataset', type=str, help='dataset with all scenes')
    parser.add_argument('--predictions', type=str, help='all predictions scene files (need to be the scene name + .csv)')
    parser.add_argument('--npredictions', type=int, help='expected number of predictions before gettint threshold')
    parser.add_argument('--sequence', type=int, help='sequence length expected')
    parser.add_argument('--thresholds', type=str, help='file which contains all thresholds')
    parser.add_argument('--learned_zones', type=str, help="Filename which specifies if zones are learned or not and which zones", default="")
    parser.add_argument('--every', type=int, help="every images only", default=1)
    parser.add_argument('--nsamples', type=int, help="expected max number of samples", default=10000)
    parser.add_argument('--output', type=str, help="output predictions folder")

    args = parser.parse_args()

    p_prediction = args.predictions
    p_npredictions = args.npredictions
    p_sequence = args.sequence
    p_thresholds = args.thresholds
    p_zones    = args.learned_zones
    p_every    = args.every
    p_dataset    = args.dataset
    p_output   = args.output
    p_nsamples = args.nsamples

    if not os.path.exists(p_output):
        os.makedirs(p_output)

    scenes = sorted(os.listdir(p_dataset))

    for scene in scenes:
        """
        For each scene run `predictions/display_prediction_scene.py`

        Params:

        parser.add_argument('--predictions', type=str, help='prediction file of scene')
        parser.add_argument('--sequence', type=int, help='sequence length expected')
        parser.add_argument('--thresholds', type=str, help='file which cantains all thresholds')
        parser.add_argument('--learned_zones', type=str, help="Filename which specifies if zones are learned or not and which zones", default="")
        parser.add_argument('--npredictions', type=int, help='expected number of predictions before gettint threshold')
        parser.add_argument('--every', type=int, help="every images only", default=1)
        parser.add_argument('--scene', type=str, help='Scene folder to use')
        parser.add_argument('--output', type=str, help='output image file name')
        """

        scene_path = os.path.join(p_dataset, scene)
        output_name = os.path.join(p_output, scene)
        prediction_scene = os.path.join(p_prediction, scene + '.csv')

        print('Prediction for', scene)
        command_str = "python predictions/display_prediction_scene.py --predictions {0} --sequence {1} --thresholds {2} --learned_zones {3} --every {4} --scene {5} --npredictions {6} --nsamples {7} --output {8}" \
            .format(prediction_scene, p_sequence, p_thresholds, p_zones, p_every, scene_path, p_npredictions, p_nsamples, output_name)

        subprocess.call(command_str, shell=True)


if __name__ == "__main__":
    main()