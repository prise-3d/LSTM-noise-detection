from modules.config.global_config import *
import os

# store all variables from cnn config
context_vars = vars()

# Custom config used for redefined config variables if necessary

# folders

output_data_folder              = 'data'
output_data_generated           = os.path.join(output_data_folder, 'generated')
output_datasets                 = os.path.join(output_data_folder, 'datasets')
## noisy_folder                    = 'noisy'
## not_noisy_folder                = 'notNoisy'

# file or extensions

features_choices_labels         = ['svd_entropy', 'mu_sigma']

# parameters
