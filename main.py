from NeuralNetwork import *
from Plotter import *
import sys, getopt
# PLAN FOR TOMORROW:
# 1) make a heatmaps visualisation
# 2) try to use it on each saved model
# 3) come back to channels visualisation and check which function is responsible for it
# 4) check it on each model
# 5) go on to filters visualisation and correct it
# 6) evaluate its correctness on each model


def main(argv):
    # ------------------------------------------------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------------------------------------------------
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
    # ------------------------------------------------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------------------------------------------------
    # if you have a saved training history and want to visualise it:
    # ------------------------------------------------------------------------------------------------------------------
    # load and plot training history:
    """
    history = NeuralNetwork.load_training_history(os.path.join(output_path,'training_1'))
    Plotter.plot_history(history)
    """

    # ------------------------------------------------------------------------------------------------------------------
    # if you have a saved model and want to visualise its results:
    # -----------------------------------------------------------------------------------------------------------------
    # load model and use Plotter functions to visualise chosen parts:
    model = models.load_model('blood_cells_1.h5')
    # model.summary() # the last conv layer is 'conv2d_4'
    frames = Dataset.get_class_sets(frames_size=(80, 60))

    # check how the model handles new frames:
    # NeuralNetwork.decode_predictions_for_classes(model=model, frames_classes=frames, print_detailed_predictions=False)

    # NeuralNetwork.decode_predictions_for_one_class(model=model, frames=frames[0], print_detailed_predictions=True)
    heatmap = Maths.get_heatmap(model=model, frame=frames[0][0], layer_name='conv2d_4', class_number=0)
    Plotter.visualise_heatmap(frames[0][0][0], heatmap=heatmap)

if __name__ == "__main__":
    main(sys.argv[1:])

