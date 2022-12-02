import os
import sys
import subprocess

print('Procesando dataset...')
if not os.path.exists(os.path.join('.', 'data')):
    os.mkdir(os.path.join('.', 'data'))
    try:
        subprocess.check_call([sys.executable, 'prepare_data.py'])
    except subprocess.CalledProcessError:
        print('Error al procesar el dataset. Intente de nuevo')
        exit(1)

subprocess.check_call([sys.executable, 'NN.py'])

