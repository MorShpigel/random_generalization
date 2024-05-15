import yaml
from guess_n_check import run_guess_and_check_generalzation_prob_exp
from utilities import save_result_to_csv

config_file_name = 'configs_for_prob_new_0102.yaml'
config_file_name = 'configs_for_prob_new_10_05.yaml'

# Load experiment configurations from a YAML file
with open(config_file_name, 'r') as f:
    configs = yaml.safe_load(f)

# Get the configurations for each type of experiment
configs_dist = configs['experiment_dist']

dist_vec = ["Normal", "Normal_uv", "Laplace"]
dist_vec = ["Normal_uv"]

# Run the experiments for each type of experiment
for config in configs_dist:
    if config['config_status'] == 'completed':
            continue
    config['config_status'] = 'running'
    config_for_runs = {k: v for k, v in config.items() if k != 'config_status'}
    # print(pd.DataFrame(config_for_runs, index=[0]))
    for dist in dist_vec:
        if dist!="Normal_uv" and config_for_runs['depth']>2:
            continue
        test_acc_vec, status_vec, iter_vec = run_guess_and_check_generalzation_prob_exp(dist, **config_for_runs)
        save_result_to_csv(test_acc_vec, status_vec, iter_vec, dist, config_for_runs, file_name=f"results_new_1005_prob_N_train_{config_for_runs['N_train']}_d_{config_for_runs['d']}_r_{config_for_runs['r']}.csv")
    config['config_status'] = 'completed'

# Save the updated configurations back to the YAML file
with open(config_file_name, 'w') as f:
    yaml.safe_dump(configs, f, sort_keys=False)