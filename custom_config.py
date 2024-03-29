from modules.config.global_config import *
import os

# store all variables from cnn config
context_vars = vars()

# Custom config used for redefined config variables if necessary

# folders

output_data_folder              = 'data'
output_data_generated           = os.path.join(output_data_folder, 'generated')
output_datasets                 = os.path.join(output_data_folder, 'datasets')
output_results_folder           = os.path.join(output_data_folder, 'results')
output_zones_learned            = os.path.join(output_data_folder, 'learned_zones')
output_models                   = os.path.join(output_data_folder, 'saved_models')
output_figures                  = os.path.join(output_data_folder, 'figures')
output_kpi                      = os.path.join(output_data_folder, 'kpi')
output_data_scaler              = os.path.join(output_data_folder, 'scaler')
## noisy_folder                    = 'noisy'
## not_noisy_folder                = 'notNoisy'

# file or extensions

features_choices_labels         = [
                                    'svd_entropy', 
                                    'svd_entropy_norm', 
                                    'mu_sigma', 
                                    'svd', 
                                    'svd_norm', 
                                    'svd_log10', 
                                    'svd_norm_log10', 
                                    'stats_luminance',
                                    'svd_entropy_split',
                                    'svd_entropy_norm_split',
                                    'svd_entropy_log10_split',
                                    'svd_entropy_norm_log10_split',
                                    'kolmogorov_complexity',
                                    'svd_entropy_blocks',
                                    'svd_entropy_blocks_disordered',
                                    'svd_entropy_blocks_divided',
                                    'svd_entropy_blocks_norm',
                                    'svd_entropy_blocks_permutation',
                                    'svd_entropy_blocks_permutation_norm',
                                    'entropy_blocks',
                                    'entropy_blocks_norm',
                                    'entropy_blocks_permutation',
                                    'entropy_blocks_permutation_norm',
                                    'complexity_stats',
                                    'filters_statistics',
                                    'filters_statistics_norm',
                                    'extended_statistics'
                                ]

results_filename                = 'results.csv'

# parameters
