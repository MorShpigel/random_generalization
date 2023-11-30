import yaml
import os

# Define the base experiment_dist configuration
base_dist_config = {
    'N_train': 16,
    'N_test': 2000,
    'dataset': 'random',
    'd': 30,
    'mu': 0.0,
    'r': 1,
    'data_seed': 0,
    'depth': 30,
    'criterion_type': 'BCEWithLogitsLoss',
    'loss_max_thr': 0.3,
    'loss_min_thr': 0.2,
    'T_find_interpolating_sol': 100000,
    'number_of_GNC_models_to_try': 100,
    'config_status': 'completed'
}

base_gd_config = {
    'N_train': 16,
    'N_test': 2000,
    'dataset': 'random',
    'd': 30,
    'mu': 0.0,
    'r': 1,
    'data_seed': 0,
    'depth': 30,
    'criterion_type': 'BCEWithLogitsLoss',
    'loss_max_thr': 0.3,
    'loss_min_thr': 0.2,
    'number_of_GD_models_to_try': 100,
    'epochs': 20000,
    'optimizer_type': 'SGD',
    'lr': 0.1,
    'weight_decay': 0.0,
    'config_status': 'completed'
}

# Define the values for N_train and d
data_seed_values = [0, 300, 600, 900]
depth_vec = [2, 30]
loss_thr_vec = [(0.2, 0.3), (0, 1)]

if not os.path.exists('configs.yaml'):
    data = {'experiment_dist':[], 'experiment_GD': []}
    with open('configs.yaml', 'w') as f:
    # Write the data to the file
        yaml.dump(data, f)

# Load experiment configurations from a YAML file
with open('configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

# Get the configurations for each type of experiment
configs_dist = configs['experiment_dist']
configs_gd = configs['experiment_GD']

# Generate the configurations
configs = []
for data_seed in data_seed_values:
    for depth in depth_vec:
        for loss_thr in loss_thr_vec:
            config = base_dist_config.copy()
            config['data_seed'] = data_seed
            config['depth'] = depth
            config['loss_min_thr'] = loss_thr[0]
            config['loss_max_thr'] = loss_thr[1]
            if not config in configs_dist:
                config['config_status'] = 'waiting'
                configs.append(config)
configs_dist.extend(configs)

weight_decay_vec = [0.0, 0.1, 0.3]
configs = []
for data_seed in data_seed_values:
    for depth in depth_vec:
        for loss_thr in loss_thr_vec:
            for weight_decay in weight_decay_vec:
                config = base_gd_config.copy()
                config['data_seed'] = data_seed
                config['depth'] = depth
                config['loss_min_thr'] = loss_thr[0]
                config['loss_max_thr'] = loss_thr[1]
                config['weight_decay'] = weight_decay
                if not config in configs_gd:
                    config['config_status'] = 'waiting'
                    configs.append(config)
configs_gd.extend(configs)

# Write the configurations to a YAML file
with open('configs.yaml', 'w') as f:
    yaml.dump({'experiment_dist': configs_dist, 'experiment_GD': configs_gd}, f, sort_keys=False)