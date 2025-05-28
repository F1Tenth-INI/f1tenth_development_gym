from SI_Toolkit.data_preprocessing import transform_dataset
import os

experiment_path = 'SI_Toolkit_ASF/Experiments/03_26_RCA1/Recordings/'
save_path = 'SI_Toolkit_ASF/Experiments/03_26_RCA1/Recordings/'

for folder in ['Validate', 'Test']:
    get_files_from = os.path.join(experiment_path, folder)
    save_files_to = os.path.join(save_path, folder)
    variables_to_shift = ['translational_control_calculated', 'angular_control_calculated']
    indices_by_which_to_shift = [-1]
    print(f'Adding shifted columns to {folder} dataset')
    transform_dataset(get_files_from, save_files_to, transformation='add_shifted_columns',
                        variables_to_shift=variables_to_shift, indices_by_which_to_shift=indices_by_which_to_shift)
 
