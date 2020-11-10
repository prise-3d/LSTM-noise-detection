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
    parser.add_argument('--predictions', type=str, help='prediction folder for each scene (need to have the same name as scene + .csv)')
    parser.add_argument('--sequence', type=int, help='sequence length expected')
    parser.add_argument('--thresholds', type=str, help='file which cantains all thresholds')
    parser.add_argument('--every', type=int, help="every images only", default=1)
    parser.add_argument('--nstop', type=int, help='number of predictions before estimating label', default=1)
    parser.add_argument('--prob', type=float, help='max probability for not noisy label', default=0.5)
    parser.add_argument('--output', type=str, help="output predictions file")

    args = parser.parse_args()

    p_prediction = args.predictions
    p_sequence = args.sequence
    p_thresholds = args.thresholds
    p_every    = args.every
    p_nstop    = args.nstop
    p_prob     = args.prob
    p_dataset  = args.dataset
    p_output   = args.output

    folder, filename = os.path.split(p_output)

    if len(folder) > 0 and not os.path.exists(folder):
        os.makedirs(folder)

    scenes = sorted(os.listdir(p_dataset))

    for scene in scenes:
        """
        For each scene run `predictions/estimate_prediction_scene.py`

        Params:

        parser.add_argument('--predictions', type=str, help='prediction file of scene')
        parser.add_argument('--sequence', type=int, help='sequence length expected')
        parser.add_argument('--thresholds', type=str, help='file which cantains all thresholds')
        parser.add_argument('--every', type=int, help="every images only", default=1)
        parser.add_argument('--scene', type=str, help='Scene folder to use')
        parser.add_argument('--nstop', type=int, help='number of predictions before estimating label', default=1)
        parser.add_argument('--prob', type=float, help='max probability for not noisy label', default=0.5)
        parser.add_argument('--output', type=str, help='output image file name')
        """

        scene_path = os.path.join(p_dataset, scene)

        print('Prediction for', scene)
        command_str = "python predictions/estimate_prediction_scene.py --predictions {0} --sequence {1} --thresholds {2} --every {3} --scene {4} --nstop {5} --prob {6} --output {7}" \
            .format(os.path.join(p_prediction, scene + '.csv'), p_sequence, p_thresholds, p_every, scene_path, p_nstop, p_prob, filename)

        subprocess.call(command_str, shell=True)


if __name__ == "__main__":
    main()