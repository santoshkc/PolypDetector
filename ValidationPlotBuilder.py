import json
import matplotlib.pyplot as plt

#experiment_folder = './output'
experiment_folder = r"C:\Users\Dev2\Desktop\Jupyter Notebook\detectron2_facebook\polyp-detector\output_colab_validation_4"


def build_training_validation_plot(experiment_folder: str, output_image_name: str):
    def load_json_arr(json_path):
        lines = []
        with open(json_path, 'r') as f:
            for line in f:
                lines.append(json.loads(line))
        return lines

    experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')

    #print("Losses: ", [ x['iteration'] for x in experiment_metrics if 'total_loss' not in x ] )
    #print("Losses: ", [ x['iteration'] for x in experiment_metrics if 'validation_loss' not in x ] )

    plt.plot(
        [ x['iteration'] for x in experiment_metrics if 'total_loss' in x ] , 
        [ x['total_loss'] for x in experiment_metrics if 'total_loss' in x ] )
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
        [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
    plt.legend(['total_loss', 'validation_loss'], loc='upper left')
    plt.xlabel("epochs")
    plt.savefig(f"{experiment_folder}/{output_image_name}")
    #plt.show()

build_training_validation_plot(experiment_folder,"val_plot_colab.jpg")
