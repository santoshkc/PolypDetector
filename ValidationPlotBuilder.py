import json
import matplotlib.pyplot as plt

experiment_folder = './output'

def build_training_validation_plot(experiment_folder: str, output_image_name: str):
    def load_json_arr(json_path):
        lines = []
        with open(json_path, 'r') as f:
            for line in f:
                lines.append(json.loads(line))
        return lines

    experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')

    plt.plot(
        [x['iteration'] for x in experiment_metrics], 
        [x['total_loss'] for x in experiment_metrics])
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
        [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
    plt.legend(['total_loss', 'validation_loss'], loc='upper left')
    plt.imsave(f"{experiment_folder}/{output_image_name}")
    #plt.show()

build_training_validation_plot(experiment_folder,"val_plot.jpg")
