from keras import optimizers
from Plotter import *
import sys, getopt


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
    # use functions form old_NeuralNetwork to create a model:
    frame_size = (120, 160)
    activation = 'relu'
    output_activation = 'softmax'
    model_name = 'book_model_1'
    """
    ideal_model = NeuralNetwork.create_article_1_model((frame_size[0], frame_size[1], 3), kernel_size=(3,3))
    NeuralNetwork.save_model(ideal_model, model_name, output_path)

   
    # model from keras kernel does not work for me - terrible results:
    ideal_model = NeuralNetwork.create_keras_kernel_model((frame_size[0], frame_size[1], 3))
    NeuralNetwork.save_model(ideal_model, model_name, output_path)
    simple_model = NeuralNetwork.create_model_from_scratch((frame_size[0], frame_size[1], 3), activation,
                                                           output_activation, kernel_size=(3, 3), dense_units=512,
                                                           features_num_sequence=(32, 64, 128, 128))
    NeuralNetwork.save_model(simple_model, model_name, output_path)
    simple_model = NeuralNetwork.create_model_from_scratch((frame_size[0], frame_size[1], 3), activation,
    output_activation, kernel_size=(3, 3), dense_units=512, features_num_sequence=(32, 64, 128, 128))    
    NeuralNetwork.save_model(simple_model, model_name, output_path)
    MobileNet_extended = NeuralNetwork.create_MobileNet(frame_size, basic=False, freeze_layers=range(0, 70))
    NeuralNetwork.save_model(MobileNet_extended, model_name, output_path)
    VGG16_extended = NeuralNetwork.create_VGG16(frame_size, basic=False, freeze_layers=range(0, 10))
    NeuralNetwork.save_model(VGG16_extended, model_name, output_path)
    """
    # ------------------------------------------------------------------------------------------------------------------
    # if you have a saved model and want to train it:
    # ------------------------------------------------------------------------------------------------------------------

    optimizer = optimizers.RMSprop(lr=1e-4)
    loss = 'categorical_crossentropy'
    metrics = ['acc']
    model = NeuralNetwork.load_model(output_path, model_name)
    NeuralNetwork.save_summary(model, model_name, output_path)
    """
    history = NeuralNetwork.train_network(path_to_db=input_path, model=model, frame_size=frame_size,
                                          rescale=1./255, validation_set_percentage=80, batch_size=20, epochs=30,
                                          optimizer=optimizer, loss=loss, metrics=metrics)
    NeuralNetwork.save_training_history(history, model_name, output_path)
    
    """
    # ------------------------------------------------------------------------------------------------------------------
    # if you have a saved training history and want to visualise it:
    # ------------------------------------------------------------------------------------------------------------------
    # load and plot training history:
    history = NeuralNetwork.load_training_history(output_path, model_name)
    Plotter.plot_history(history_dic=history)

    # ------------------------------------------------------------------------------------------------------------------
    # if you have a saved model and want to visualise its results:
    # -----------------------------------------------------------------------------------------------------------------

    frames = Dataset.get_class_sets(frames_size=frame_size)
    NeuralNetwork.decode_and_save_predictions(model=model, frames_classes=frames, model_name=model_name,
                                                      output_path=output_path)

    """
    Plotter.full_model_visualisation(model, frames[0][0][0])
    """
if __name__ == "__main__":
    main(sys.argv[1:])

