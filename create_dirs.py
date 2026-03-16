#!/usr/bin/env python3
import os

dirs = [
    'include/neuralcore/nn',
    'include/neuralcore/optim',
    'include/neuralcore/data',
    'src/nn',
    'src/optim',
    'src/data',
    'tests',
    'examples'
]

for dir_path in dirs:
    os.makedirs(dir_path, exist_ok=True)
    print(f'Created: {dir_path}')

print('\nDirectory structure created successfully!')
