import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import ipdb
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--model-names', '--list', help="delimited list input", type=str)
#parser.add_argument('--checkpoints-dir', type=str)

#checkpoints_dir = 'checkpoints-eval'
checkpoints_dir = 'checkpoints'

def load_errors(model_name):
    error_path = os.path.join(checkpoints_dir, model_name, 'errors_val.pkl')
    with open(error_path, 'rb') as f:
        errors = pickle.load(f)
    #errors = np.array(np.load(error_path))
    return errors


def plot_errors(ax, errors, key, model_name, min_len):
    error_i = errors[key][:min_len]
    ax.plot(error_i, label=model_name)
    return ax

def plot_barplots(ax, errors, key, model_name, width, i, color):
    error_i = errors[key]
    ax.bar(width*i, error_i, width, color=color, label=model_name)
    return ax


if __name__ == '__main__':

    error_titles = ['mu_loss', 'mu_hitrate', 'mu_traj_10', 'mu_traj_20']

    error_index_to_title = {i : error_titles[i] for i in range(len(error_titles))}
    error_title_to_index = {j : i for i,j in error_index_to_title.items()}

    eval_args = parser.parse_args()
    model_names = [str(item) for item in eval_args.model_names.split(',')]

    figure_root_path = 'evaluation'
    if not os.path.exists(figure_root_path):
        os.mkdir(figure_root_path)
    figure_save_path = f'{figure_root_path}/evaluation_figs_{model_names}'
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)

    all_errors = dict()
    min_len = 100
    for model_name in model_names:
        errors_m = load_errors(model_name)
        num_errors = len(errors_m['mu_loss'])
        if num_errors < min_len:
            min_len = num_errors
        all_errors[model_name] = load_errors(model_name)

    keys = ['mu_loss','mu_hitrate', 'mu_traj_10', 'mu_traj_20']
    for key in keys:
        avg_hitrate = np.array([all_errors[model_name][key] for model_name in model_names])
        mu_hitrate = np.mean(avg_hitrate)
        std_hitrate = np.std(avg_hitrate)
        print(f"Key {key} {mu_hitrate} {std_hitrate}")

    if True:
        for key in error_titles:
            fig, ax = plt.subplots()
            fig_bar, ax_bar = plt.subplots()
            colors = ['red', 'green', 'blue', 'purple', 'black']
            width = 0.2
            for ix, model_name in enumerate(model_names):
                errors_m = all_errors[model_name]
                plot_errors(ax, errors_m, key, model_name, min_len)
                plot_barplots(ax_bar, errors_m, key, model_name, width, ix, colors[ix])

            ax.legend()
            ax_bar.legend()
            ax.set_xlabel('Epochs')

            fig.suptitle(key)
            fig.savefig(os.path.join(figure_save_path, f'{key}.png'))

            fig_bar.suptitle(key)
            print("Saving fig to", figure_save_path)
            fig_bar.savefig(os.path.join(figure_save_path, f'{key}_bar.png'))


