import yaml
from guess_n_check import run_guess_and_check_generalzation_exp
from gd import run_GD_generalzation_exp
from utilities import save_result_to_csv
import pandas as pd

# Load experiment configurations from a YAML file
with open('configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

# Get the configurations for each type of experiment
configs_dist = configs['experiment_dist']
configs_gd = configs['experiment_GD']

dist_vec = ["Normal", "Normal_uv", "Laplace"]
# Run the experiments for each type of experiment
for config in configs_dist:
    if config['config_status'] == 'completed':
            continue
    config['config_status'] = 'running'
    config_for_runs = {k: v for k, v in config.items() if k != 'config_status'}
    # print(pd.DataFrame(config_for_runs, index=[0]))
    for dist in dist_vec:
        test_acc_vec, status_vec = run_guess_and_check_generalzation_exp(dist, **config_for_runs)
        save_result_to_csv(test_acc_vec, status_vec, dist, config_for_runs, file_name="results.csv")
    config['config_status'] = 'completed'
for config in configs_gd:
    if config['config_status'] == 'completed':
        continue
    config['config_status'] = 'running'
    config_for_runs = {k: v for k, v in config.items() if k != 'config_status'}
    # print(pd.DataFrame(config_for_runs, index=[0]))
    test_acc_vec, status_vec = run_GD_generalzation_exp(**config_for_runs)
    dist="GD"
    save_result_to_csv(test_acc_vec, status_vec, dist, config_for_runs, file_name="results.csv")
    config['config_status'] = 'completed'

# Save the updated configurations back to the YAML file
with open('configs.yaml', 'w') as f:
    yaml.safe_dump(configs, f, sort_keys=False)