from Plotter import *
from keras import models
from keras.applications import VGG16
from keras.applications import MobileNet


def train_and_save_new_model():
    data_set = Dataset('/home/ilona/Desktop/Literatura/Bachelors_Thesis/blood_cells/blood-cells_db/dataset2-master')
    train, validation, test = data_set.get_data(size=(80, 60), rescale=1. / 255, validaion_set_percentage=80)

    data_batch, labels_batch = train[0]

    CNN_model = NeuralNetwork.create_cnn_model(data_shape=data_batch[0].shape)
    history = CNN_model.fit_generator(
        train,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation,
        validation_steps=50
    )

    CNN_model.save('blood_cells_1.h5')
    Plotter.plot_history(history=history)


def train_and_save_pretrained_network_model(size, extended_version=False):
    data_set = Dataset('/home/ilona/Desktop/Literatura/Bachelors_Thesis/blood_cells/blood-cells_db/dataset2-master')
    train, validation, test = data_set.get_data(size=size, rescale=1. / 255, validaion_set_percentage=80, batch_size=20)

    data_batch, labels_batch = train[0]

    conv_base = MobileNet(weights='imagenet',
                      include_top=False, # I will add my own classifier defining my 4 classes
                      input_shape=data_batch[0].shape)

    # check the shape of the last layer there (e.g. block5_pool for VGG16 base is (None, 5, 3, 512)):
    # conv_base.summary()

    if extended_version:
        # merge the base and classifier together and then run it on the dataset
        model = NeuralNetwork.merge_classifier_and_conv_base(conv_base=conv_base, freeze_layers=range(0, 70))
        model.save('blood_cells_pretrained_extended.h5')
        history = model.fit_generator(train, steps_per_epoch=100, epochs=30, validation_data=validation,
                                      validation_steps=50)
    else:
        # running a pretrained base on the dataset, save the output, then put it to standalone Dense classifier
        train_features, train_labels = NeuralNetwork.extract_features(conv_base=conv_base,
                                                                      dataset=train,
                                                                      shape=(len(train), 5, 3, 1024),  # 5, 3, 512),
                                                                      batch_size=20)
        validation_features, validation_labels = NeuralNetwork.extract_features(conv_base=conv_base,
                                                                                dataset=validation,
                                                                                shape=(len(validation), 5, 3, 1024),
                                                                                # 5, 3, 512),
                                                                                batch_size=20)
        test_features, test_labels = NeuralNetwork.extract_features(conv_base=conv_base,
                                                                    dataset=test,
                                                                    shape=(len(test), 5, 3, 1024),  # 5, 3, 512),
                                                                    batch_size=20)

        model = NeuralNetwork.create_classifier_for_conv_base(shape=(5, 3, 1024), dropout=0.5)  # 5, 3, 512), dropout=0)
        model.save('blood_cells_pretrained_basic.h5')
        history = model.fit(train_features, train_labels,
                            epochs=30,
                            batch_size=20,
                            validation_data=(validation_features, validation_labels))
    Plotter.plot_history(history)


def visualisation(model_name, size):
    # load model
    model = models.load_model(model_name)

    data_set = Dataset('/home/ilona/Desktop/Literatura/Bachelors_Thesis/blood_cells/blood-cells_db/dataset2-master')
    test_simple_frames_e = data_set.get_frames(path=os.path.join(
        data_set.base_path, 'images/TEST_SIMPLE/EOSINOPHIL'), size=size, rescale=1. / 255)
    test_simple_frames_l = data_set.get_frames(path=os.path.join(
        data_set.base_path, 'images/TEST_SIMPLE/LYMPHOCYTE'), size=size, rescale=1. / 255)
    test_simple_frames_m = data_set.get_frames(path=os.path.join(
        data_set.base_path, 'images/TEST_SIMPLE/MONOCYTE'), size=size, rescale=1. / 255)
    test_simple_frames_n = data_set.get_frames(path=os.path.join(
        data_set.base_path, 'images/TEST_SIMPLE/NEUTROPHIL'), size=size, rescale=1. / 255)

    # visualise channels:

    # version for non-pretrained network or pretrained network other than MobileNet
    # activations = NeuralNetwork.get_activations(model=model, image=test_simple_frames_e[0])

    # version for MobileNet:
    mn_model = model.layers[0]
    mn_model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])
    activations = NeuralNetwork.get_activations(model=mn_model, image=test_simple_frames_e[0])

    Plotter.visualise_image_channels(layers=model.layers, activations=activations, images_per_row=8)

    # visualise filters (we have only 32 of them):
    Plotter.visualise_filters(layers=model.layers[:8], model=model, shape=(test_simple_frames_e[0]).shape, test_img=(test_simple_frames_e[0]))


# train_and_save_pretrained_network_model(size=(160, 120), extended_version=True)
visualisation('blood_cells_pretrained_extended.h5', size=(160, 120))
