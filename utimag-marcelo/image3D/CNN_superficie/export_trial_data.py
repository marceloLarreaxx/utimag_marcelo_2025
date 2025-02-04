import json
import os
import json
import os
import csv
import numpy as np
def parse_trial_data(folder_path):
    trial_data = []
    for folder_name in os.listdir(folder_path):
        folder_name_path = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_name_path):
            trial_id = folder_name  # Nombre de la carpeta es el trial_id
            json_file_path = os.path.join(folder_path, folder_name, 'trial.json')
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
                hyperparameters = data['hyperparameters']['values']
                parameters = {param: hyperparameters[param] for param in hyperparameters if param != 'tuner/epochs' and param!= 'tuner/initial_epoch' and param!= 'tuner/bracket' and param!= 'tuner/round'}
                status = data['status']
                epochs = hyperparameters['tuner/epochs']
                initial_epochs = hyperparameters['tuner/initial_epoch']
                bracket = hyperparameters['tuner/bracket']
                round_ = hyperparameters['tuner/round']
                trial_data.append({
                    'trial_id': trial_id,
                    'parameters': list(parameters.values()),
                    'status': status,
                    'epochs': epochs,
                    'initial_epochs': initial_epochs,
                    'bracket': bracket,
                    'round': round_
                })
    return trial_data
def export_to_csv(trial_data, csv_file_path):


    with open(csv_file_path, 'w', newline='', ) as csvfile:
        fieldnames = ['trial_id', 'parameters','status', 'epochs', 'initial_epochs', 'bracket', 'round']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trial_data:
            writer.writerow(trial)


def export_trial_results(folder_path):
    all_trials=[]
    for folder_name in os.listdir(folder_path):
        folder_name_path = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_name_path):
            trial_id = folder_name  # Nombre de la carpeta es el trial_id
            json_file_path = os.path.join(folder_path, folder_name, 'trial.json')
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
                completed_data={'trial_id': trial_id,
                                'data': data}
                all_trials.append(completed_data)

    result_path=folder_path+ 'all_info_tuner11.csv'
    with open(result_path, 'w', newline='') as csvfile:
        fieldnames = ['trial_id', 'hyperparameters', 'epoch_best_performance','idx_error', 'n_out_1', 'n_out_2', 'val_idx_error', 'val_n_out_1', 'val_n_out_2']  # Define los nombres de las columnas
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')

        # Escribe las métricas para cada prueba
        writer.writeheader()
        for trial_it in all_trials:
            trial=trial_it['data']
            id=trial_it['trial_id']
            if trial['status']=='COMPLETED' and not int(trial['metrics']['metrics']['val_idx_error']['observations'][0]['value'][0])==10000:
                writer.writerow({'trial_id': id,
                                'hyperparameters': trial['hyperparameters']['values'],
                                'epoch_best_performance': trial['best_step'],
                                'idx_error': round(trial['metrics']['metrics']['idx_error']['observations'][0]['value'][0],2),
                                'n_out_1': round(trial['metrics']['metrics']['n_out_1']['observations'][0]['value'][0],2) ,
                                'n_out_2': round(trial['metrics']['metrics']['n_out_2']['observations'][0]['value'][0],2),
                                'val_idx_error': round(trial['metrics']['metrics']['val_idx_error']['observations'][0]['value'][0],2) ,
                                'val_n_out_1': round(trial['metrics']['metrics']['val_n_out_1']['observations'][0]['value'][0],2),
                                'val_n_out_2': round(trial['metrics']['metrics']['val_n_out_2']['observations'][0]['value'][0],2)

                                })
    return result_path

import pandas as pd
import matplotlib.pyplot as plt
def show_scatter_graph(csv_path):
    # Cargar el CSV con Pandas
    df = pd.read_csv(csv_path,delimiter=';')

    # Extraer los valores de las columnas
    val_idx_error = df['val_idx_error']
    n_out_1 = df['val_n_out_1']
    n_out_2 = df['val_n_out_2']

    # Calcular la suma de n_out_1 y n_out_2
    n_out_total = n_out_1 + n_out_2
    plt.ion()
    # Crear el gráfico de puntos
    plt.figure(figsize=(8, 6))
    plt.scatter(val_idx_error, n_out_total, alpha=0.5)
    plt.title('Trial results')
    plt.xlabel('val_idx_error')
    plt.ylabel('val_n_out_1 + val_n_out_2')
    plt.grid(True)
    plt.show()

def show_scatter_graph_best_models(csv_path):
    # Cargar el CSV con Pandas
    df = pd.read_csv(csv_path, delimiter=';')



    # Crear una figura y un eje
    plt.figure(figsize=(10, 8))
    plt.ion()

    # Colores para diferenciar cada modelo
    colors = plt.cm.get_cmap('Set1', len(df['config_name'].unique()))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'H', 'X', 'd']
    # Convertir comas a puntos y luego a float
    for col in ['val_idx_error', 'val_n_out_1', 'val_n_out_2']:
        df[col] = df[col].str.replace(',', '.').astype(float)
    # Pintar cada punto con un color diferente basado en 'config_name'
    for i, (config_name, group) in enumerate(df.groupby('config_name')):
        val_idx_error= np.asarray(group['val_idx_error'].to_list(), dtype='float32')
        val_n_out_1= np.asarray(group['val_n_out_1'].to_list(), dtype='float32')
        val_n_out_2 = np.asarray(group['val_n_out_2'].to_list(), dtype='float32')

        plt.scatter(val_idx_error, val_n_out_1 + val_n_out_2,
                    color=colors(i),marker=markers[i], label=config_name, alpha=0.8)

    # Añadir leyenda y etiquetas
    plt.title('Best model results')
    plt.xlabel('val_idx_error')
    plt.ylabel('val_n_out_1 + val_n_out_2')
    plt.grid(True)
    plt.legend(title='Model')
    plt.show()


trial_data_path= r'C:\Users\admin\Desktop\Mario\DEVELOPMENT\utimag\imag3D\CNN_superficie\trained_models\9emisores\rf\tuner11\cnn_surf'

csv_file_path= trial_data_path +'\data_trials_test.csv'

trial_data= parse_trial_data(trial_data_path)

export_to_csv(trial_data,csv_file_path)
results_csv_path=export_trial_results(trial_data_path)
results_csv_path= r'C:\Users\admin\Desktop\Mario\DEVELOPMENT\utimag\imag3D\CNN_superficie\trained_models\9emisores\rf\tuner11\HP_FULL.csv'
show_scatter_graph(results_csv_path)
path_best=r'C:\Users\admin\Desktop\Mario\DEVELOPMENT\utimag\imag3D\CNN_superficie\trained_models\9emisores\rf\best5\best_model_training_results.csv'
show_scatter_graph_best_models(path_best)
