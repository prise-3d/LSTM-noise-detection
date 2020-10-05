import os
import subprocess
import argparse


n_epochs = 30
generated_datasets_folder = "data/datasets/{0}/{1}"
results_filename = "data/results/results.csv"

already_runned = []
with open(results_filename, 'r') as f:
    
    lines = f.readlines()
    del lines[0] # remove header

    for line in lines:
        name = line.split(';')[0]
        already_runned.append(name)


def main():
    
    parser = argparse.ArgumentParser(description="For each data files found, datasets are created and RNN model trained")

    parser.add_argument('--folder', type=str, help="data folder with data files", required=True)
    parser.add_argument('--selected_zones', type=str, help='list of train scenes used', required=True)

    args = parser.parse_args()

    p_folder            = args.folder
    selected_zones_file = args.selected_zones

    # get all data files
    data_files = sorted(os.listdir(p_folder))

    for data in data_files:
        
        data_filename = os.path.join(p_folder, data)

        for seq in range(3, 8):

            output_dataset_name = data + "_seq" + str(seq)

            if not os.path.exists(os.path.join('data', 'datasets', output_dataset_name)):

                command_dataset = "python processing/prepare_dataset_zones.py --data {0} --output {1} --sequence {2} --selected_zones {3}" \
                    .format(data_filename, output_dataset_name, seq, selected_zones_file)

                
                print("Prepare dataset with name {0}".format(output_dataset_name))
                # print(command_dataset)
                subprocess.call(command_dataset, shell=True)
            else:
                print("Dataset {0} already generated".format(output_dataset_name))

            for norm in [0, 1]:

                # run pytorch RNN model with generated dataset file
                output_dataset_name_path = generated_datasets_folder.format(output_dataset_name, output_dataset_name)

                for b_size in [64, 128]:
                    
                     # specific output model name
                    output_model_name = output_dataset_name

                    if norm:
                        output_model_name += "_seqnorm"

                    output_model_name += "_bsize" + str(b_size)

                    if output_dataset_name not in already_runned:
                        
                        command_run_model = "python train_lstm_weighted.py --train {0}.train --test {1}.test --output {2} --batch_size {3} --epochs {4} --seq_norm {5}" \
                            .format(output_dataset_name_path, output_dataset_name_path, output_model_name, b_size, n_epochs, norm)

                        print("Start running model {0}".format(output_model_name))
                        # print(command_run_model)
                        subprocess.call(command_run_model, shell=True)
                    else:
                        print('{0} model already trained'.format(output_dataset_name))

if __name__ == "__main__":
    main()