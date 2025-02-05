import sys
import os
import re

"""
Script para leer un archivo con texto y remplazar un conjunto de palabras por otras, segun una lista pasada
como input

Command line arguments:
1. Nombre de la carpeta con los archivos a modificar

2. Lista en .txt de palabras y sus reemplazos, seprados por ' --- '. Ej:
pepe --- pancho
xk --- hk
(ju + --- (ku +

3. Nombres de archivos a excluir (no modificar)
"""

folder_name = sys.argv[1] + r'\\'

with open(folder_name + sys.argv[2]) as f1:
    words = f1.readlines()
    words = [w.split(' --- ') for w in words]
    words = [[w[0], w[1].strip('\n')] for w in words]

files = os.listdir(folder_name)  # lista de nombres de los archivos en la carpeta, invluye el de la
# lista de palabras a reemplazar, hay que esquivarlo

output_folder = folder_name + r'mod\\'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for x in files:
    # chequear que no sea la lista de palabras o un directorio o un archivo excluido
    if x != sys.argv[2] and x != sys.argv[3] and not os.path.isdir(folder_name + x):
        with open(folder_name + x, 'r') as f, open(folder_name + r'mod\\' + x, 'w') as f_r:
            a = f.read()
            for w in words:
                a = a.replace(w[0], w[1])
            f_r.write(a)

