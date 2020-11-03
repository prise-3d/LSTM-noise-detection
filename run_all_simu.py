import os
import argparse


command = "python display/display_thresholds_scene_file.py --params \"{0}\" --method {1} --model {2} --selected_zones {3} --scene {4} --thresholds {5} --seq_norm {6} --sequence {7} --save 1 --save_thresholds {8} --label_thresholds {9} --every {10}"


parser = argparse.ArgumentParser(description="Compute simulation for each scenes")

parser.add_argument('--folder', type=str, help="data folder with scenes files", required=True)
parser.add_argument('--method', type=str, help="method name", required=True)
parser.add_argument('--model', type=str, help="model path for simulation", required=True)
parser.add_argument('--params', type=str, help="expected params for model", required=True)
parser.add_argument('--thresholds', type=str, help="thresholds file", required=True)
parser.add_argument('--selected_zones', type=str, help="selected zone file", required=True)
parser.add_argument('--sequence', type=str, help="sequence size of RNN model", required=True)
parser.add_argument('--seqnorm', type=str, help="normalization or not of sequence", required=True)
parser.add_argument('--output', type=str, help="output prediction filename", required=True)
parser.add_argument('--every', type=int, help='specify if all step images are used or not', default=1)

args = parser.parse_args()

p_folder = args.folder
p_method = args.method
p_model  = args.model
p_params = args.params
p_thresholds = args.thresholds
p_selected_zones = args.selected_zones
p_sequence = args.sequence
p_seqnorm = args.seqnorm
p_output = args.output
p_every  = args.every


for scene in sorted(os.listdir(p_folder)):    
    scene_path = os.path.join(p_folder, scene)
    str_command = command.format(p_params, p_method, p_model, p_selected_zones, scene_path, p_thresholds, p_seqnorm, p_sequence, p_output, scene, p_every)

    print("Run simulation for {0}".format(scene))
    os.system(str_command)
