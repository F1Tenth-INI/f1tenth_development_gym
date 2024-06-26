from setuptools import setup

setup(name='f110_gym',
      version='0.2.1',
      author='Hongrui Zheng',
      author_email='billyzheng.bz@gmail.com',
      url='https://f1tenth.org',
      install_requires=['gym==0.19.0',
                        'numpy==1.23.5',
                        'Pillow',
                        'scipy',
                        'numba',
                        'pyyaml',
                        'pyglet==1.5.26',
                        'pyopengl',
                        'matplotlib',
                        'tqdm',
                        'scikit-image==0.21.0',
                        'scipy==1.8.1',
                        'ruamel.yaml',
                        'pyserial',
                        'matplotlib',
                        ]
      )