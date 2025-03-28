from run.DataGen.run_data_generator import run_data_generator

# Automatically create new path to save everything in

import yaml, os

def get_record_path():
    experiment_index = 1
    while True:
        record_path = "Experiment-" + str(experiment_index)
        if os.path.exists(config_SI['paths']['PATH_TO_EXPERIMENT_FOLDERS'] + record_path):
            experiment_index += 1
        else:
            record_path += "/Recordings"
            break

    record_path = config_SI['paths']['PATH_TO_EXPERIMENT_FOLDERS'] + record_path
    return record_path

if __name__ == '__main__':
    config_SI = yaml.load(open(os.path.join('SI_Toolkit_ASF', 'config_training.yml')), Loader=yaml.FullLoader)
    config_f1t = yaml.load(open('DataGen/config_data_gen.yml'), Loader=yaml.FullLoader)
    
    record_path = get_record_path()

    # Save copy of configs in experiment folder
    if not os.path.exists(record_path):
        os.makedirs(record_path)
    
    yaml.dump(config_SI, open(record_path + "/SI_Toolkit_config_savefile.yml", "w"), default_flow_style=False) 
    yaml.dump(config_f1t, open(record_path + "/F1t_config_savefile.yml", "w"), default_flow_style=False)

    # Run data generator
    run_data_generator(run_for_ML_Pipeline=True, record_path=record_path)
