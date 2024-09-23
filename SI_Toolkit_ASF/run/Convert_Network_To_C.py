# Convert_Network_To_C.py
""" Only dense networks with tanh activation on all but last layer are supported! """
from SI_Toolkit.C_implementation.TF2C import tf2C

path_to_models = 'SI_Toolkit_ASF/Experiments/flo-mpc-cs-4/Models/'
net_name = 'Dense-48IN-32H1-64H2-32H3-3OUT-0'
batch_size = 1

if __name__ == '__main__':
    tf2C(path_to_models=path_to_models, net_name=net_name, batch_size=batch_size)