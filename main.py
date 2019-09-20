from NeuralNetwork import *
from Plotter import *
import sys, getopt


def main(argv):
    # program arguments management:
    # ------------------------------------------------------------------------------------------------------------------
    input_path = ''
    output_path = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print
        'main.py -i <path_to_images_dir_from_dataset2-master_db> -o <output_dir_path>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print
            'main.py -i <path_to_images_dir_from_dataset2-master_db> -o <output_dir_path>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_path = arg
        elif opt in ("-o", "--ofile"):
            output_path = arg

    # tmp hardcoded paths for lazy people:
    # ------------------------------------------------------------------------------------------------------------------
    if input_path == '':
        input_path = '/home/ilona/Desktop/Literatura/Bachelors_Thesis/blood_cells/blood-cells_db/dataset2-master'
    if output_path == '':
        output_path = '/home/ilona/Desktop/Literatura/Bachelors_Thesis/blood_cells/blood_cells_deep_network/' \
                      'displayforecastio'

    # how to use the script:
    # ------------------------------------------------------------------------------------------------------------------
    # if you don't have any saved model and want to create one:
    # use functions form NeuralNetwork to create and train a model:
    # train_and_save_new_model / train_and_save_pretrained_network_model
    """
    history = NeuralNetwork.train_and_save_new_model(model_name='blood_cells_2',
                                                     path_to_db=input_path,
                                                     size=(160, 120),
                                                     rescale=1. / 255,
                                                     batch_size=20,
                                                     validation_set_percentage=80,
                                                     epochs=30)
    NeuralNetwork.save_training_history(history=history, your_history_path=os.path.join(output_path, 'training_1'))
    """
    # if you have a saved training history and want to visualise it:
    # ------------------------------------------------------------------------------------------------------------------
    # load and plot training history:
    history = NeuralNetwork.load_training_history(os.path.join(output_path,'training_1'))
    Plotter.plot_history(history)

    # if you have a saved model and want to visualise its results:
    # -----------------------------------------------------------------------------------------------------------------
    # load model and use Plotter functions to visualise chosen parts:
    # model = models.load_model('blood_cells_pretrained_extended.h5')

    # NeuralNetwork.train_and_save_pretrained_network_model(size=(160, 120), extended_version=True)


if __name__ == "__main__":
    main(sys.argv[1:])

