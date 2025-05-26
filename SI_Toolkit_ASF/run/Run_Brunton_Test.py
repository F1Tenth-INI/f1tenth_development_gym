from SI_Toolkit.Testing.run_brunton_test import run_brunton_test
import sys
import os

# Detect the absolute path of the f1tenth_development_gym folder
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
run_brunton_test()