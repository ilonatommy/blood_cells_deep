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
    # use functions form old_NeuralNetwork to create a model:
    frame_size = (120, 160)
    activation = 'relu'
    output_activation = 'softmax'

    """
    # using basic pretrained networks makes sense only if we add some top layes which adapt it to our classifier (out=4)
    simple_model = NeuralNetwork.create_model_from_scratch((frame_size[0], frame_size[1], 3), activation, output_activation)
    NeuralNetwork.save_model(simple_model, 'simple_model', output_path)
    MobileNet_basic = NeuralNetwork.create_MobileNet(frame_size, basic=True)
    NeuralNetwork.save_model(MobileNet_basic, 'MobileNet_basic', output_path)
    VGG16_basic = NeuralNetwork.create_VGG16(frame_size, basic=True)
    NeuralNetwork.save_model(VGG16_basic, 'VGG16_basic', output_path)
    MobileNet_extended = NeuralNetwork.create_MobileNet(frame_size, basic=False, freeze_layers=range(0, 70))
    NeuralNetwork.save_model(MobileNet_extended, 'MobileNet_extended', output_path)
    VGG16_extended = NeuralNetwork.create_VGG16(frame_size, basic=False, freeze_layers=range(0, 10))
    NeuralNetwork.save_model(VGG16_extended, 'VGG16_extended', output_path)
    """
    VGG16_extended = NeuralNetwork.create_VGG16(frame_size, basic=False, freeze_layers=range(0, 0))
    NeuralNetwork.save_model(VGG16_extended, 'VGG16_extended', output_path)
    # ------------------------------------------------------------------------------------------------------------------
    # if you have a saved model and want to train it:
    # ------------------------------------------------------------------------------------------------------------------

    model_name = 'simple_model'
    optimizer = optimizers.RMSprop(lr=1e-4)
    loss = 'categorical_crossentropy'
    metrics = ['acc']
    model = NeuralNetwork.load_model(output_path, model_name)
    model.summary()

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
    """
    history = NeuralNetwork.load_training_history(output_path, model_name)
    Plotter.plot_history(history_dic=history)
    """
    # ------------------------------------------------------------------------------------------------------------------
    # if you have a saved model and want to visualise its results:
    # -----------------------------------------------------------------------------------------------------------------

    frames = Dataset.get_class_sets(frames_size=frame_size)
    # activation visualisation does not work for pretrained models so far -
    # look at comment in NeuralNetwork.get_activations line 127
    activations = NeuralNetwork.get_activations(model, frames[0][0])
    Plotter.visualise_image_channels(model.layers, activations, 16)

    # when model is ened with Sequential layer then I have problems with my functions
    # I need to end models with dense layers
    return activations

if __name__ == "__main__":
    f = main(sys.argv[1:])

