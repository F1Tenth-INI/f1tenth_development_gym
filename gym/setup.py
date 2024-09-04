from setuptools import setup

setup(name='f110_gym',
      version='0.2.1',
      author='Hongrui Zheng',
      author_email='billyzheng.bz@gmail.com',
      url='https://f1tenth.org',
      install_requires=['gym==0.23.1',
                        'numpy==1.24.3',
                        'Pillow',
                        'scipy',
                        'numba',
                        'pyyaml',
                        'pyglet==1.5.26',
                        'pyopengl',
                        'matplotlib',
                        'tqdm',
                        'scikit-image',
                        'scipy',
                        'ruamel.yaml',
                        'pyserial',
                        'matplotlib',
                        ]
      )