import os
import argparse
import sys
import subprocess

# modules and config imports
sys.path.insert(0, '') # trick to enable import of main folder module

import custom_config as cfg

def main():


    parser = argparse.ArgumentParser(description="Read and compute entropy data file")

    parser.add_argument('--model', type=str, help='entropy file data with estimated threshold to read and compute')
    parser.add_argument('--method', type=str, help='method name to used', choices=cfg.features_choices_labels, default=cfg.features_choices_labels[0])
    parser.add_argument('--params', type=str, help='param of the method used', default="")
    parser.add_argument('--sequence', type=int, help='sequence length expected')
    parser.add_argument('--imnorm', type=int, help="specify if image is normalized before computing something", default=0, choices=[0, 1])
    parser.add_argument('--dataset', type=str, help='dataset with all scenes')
    parser.add_argument('--seq_norm', type=int, help='normalization sequence by features', choices=[0, 1])
    parser.add_argument('--every', type=int, help="every images only", default=1)
    parser.add_argument('--output', type=str, help="output predictions folder")

    args = parser.parse_args()

    p_model      = args.model
    p_method     = args.method
    p_params     = args.params
    p_sequence   = args.sequence
    p_imnorm     = args.imnorm
    p_dataset    = args.dataset
    p_seq_norm   = args.seq_norm
    p_every      = args.every
    p_output     = args.output

    if not os.path.exists(p_output):
        os.makedirs(p_output)

    scenes = sorted(os.listdir(p_dataset))

    for scene in scenes:
        """
        For each scene run `display/prediction_on_scene.py`

        Params:

        parser.add_argument('--model', type=str, help='entropy file data with estimated threshold to read and compute')
        parser.add_argument('--method', type=str, help='method name to used', choices=cfg.features_choices_labels, default=cfg.features_choices_labels[0])
        parser.add_argument('--params', type=str, help='param of the method used', default="")
        parser.add_argument('--sequence', type=int, help='sequence length expected')
        parser.add_argument('--imnorm', type=int, help="specify if image is normalized before computing something", default=0, choices=[0, 1])
        parser.add_argument('--scene', type=str, help='Scene folder to use')
        parser.add_argument('--seq_norm', type=int, help='normalization sequence by features', choices=[0, 1])
        parser.add_argument('--every', type=int, help="every images only", default=1)
        parser.add_argument('--output', type=str, help="output prediction file")
        """

        scene_path = os.path.join(p_dataset, scene)
        output_name = os.path.join(p_output, scene + ".csv")

        print('Prediction for', scene)
        command_str = "python predictions/prediction_on_scene.py --model {0} --method {1} --params {2} --sequence {3} --scene {4} --seq_norm {5} --every {6} --output {7} --imnorm {8}" \
            .format(p_model, p_method, p_params, p_sequence, scene_path, p_seq_norm, p_every, output_name, p_imnorm)

        subprocess.call(command_str, shell=True)


if __name__ == "__main__":
    main()