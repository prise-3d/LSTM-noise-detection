import os
import argparse
import subprocess

data_filename = "cluster_data_{0}.{1}"
output_model_cluster = "cluster_{0}"

def main():

    parser = argparse.ArgumentParser(description="Read and compute training of LSTM model")

    parser.add_argument('--data', type=str, help='data will all cluster data files')
    parser.add_argument('--nclusters', type=int, help='number of expected clusters')
    parser.add_argument('--output', type=str, help='output folder name')
    parser.add_argument('--epochs', type=int, help='number of expected epochs', default=30)
    parser.add_argument('--batch_size', type=int, help='expected batch size for training model', default=64)
    parser.add_argument('--seq_norm', type=int, help='normalization sequence by features', choices=[0, 1])

    args = parser.parse_args()

    p_data         = args.data
    p_nclusters    = args.nclusters
    p_output       = args.output
    p_epochs       = args.epochs
    p_batch_size   = args.batch_size
    p_seq_norm     = args.seq_norm

    saved_models_folder = os.path.join('data/saved_models', p_output)
    
    if not os.path.exists(saved_models_folder):
        os.makedirs(saved_models_folder)

    for i in range(p_nclusters):

        train_file = os.path.join(p_data, data_filename.format(i, 'train'))
        test_file = os.path.join(p_data, data_filename.format(i, 'test'))

        output_model_name = os.path.join(p_output, output_model_cluster.format(i))

        command_run_model = "python train_lstm_weighted.py --train {0} --test {1} --output {2} --batch_size {3} --epochs {4} --seq_norm {5}" \
                    .format(train_file, test_file, output_model_name, p_batch_size, p_epochs, p_seq_norm)

        print("Start running model {0}".format(output_model_name))
        # print(command_run_model)
        subprocess.call(command_run_model, shell=True)

if __name__ == "__main__":
    main()