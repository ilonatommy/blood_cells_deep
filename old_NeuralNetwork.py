import pickle
from keras import layers
from keras import models
from keras import optimizers
from Dataset import *
from keras.applications import VGG16
from keras.applications import MobileNet
import json


class old_NeuralNetwork:
    @staticmethod
    def create_cnn_model(data_shape):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=data_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(4, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.RMSprop(lr=1e-4),
                      metrics=['acc'])
        return model

    @staticmethod
    def create_classifier_for_conv_base(shape, dropout):
        model = models.Sequential()
        model.add(layers.Dense(256, activation='relu', input_dim=shape[0] * shape[1] * shape[2]))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(4, activation='softmax'))

        model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        return model

    @staticmethod
    def add_top_to_model(conv_base, freeze_layers, weights_path):
        top_model = models.Sequential()
        # freeze certain layers to prevent changes in its weights during the training:
        conv_base.trainable = True
        for layer_index in freeze_layers:
            layer = conv_base.layers[layer_index]
            layer.trainable = False

        conv_base.trainable = True
        top_model.add(layers.Flatten())
        top_model.add(layers.Dense(256, activation='relu'))
        top_model.add(layers.Dense(4, activation='softmax'))
        weights = conv_base.get_weights()
        top_model.set_weights(weights)
        conv_base.add(top_model)

        conv_base.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        conv_base.summary()
        return conv_base

    @staticmethod
    def get_activations(model, image):
        # layer_outputs = model.layers.get_output_at(node_index=0)
        layer_outputs = [layer.get_output_at(-1) for layer in model.layers[:8]]
        activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(image)
        return activations

    @staticmethod
    def get_layer_names(layers):
        layer_names = []
        for layer in layers:
            layer_names.append(layer.name)
        return layer_names

    @staticmethod
    def extract_features(conv_base, dataset, shape, batch_size, flatten):
        sample_count_dividable_by_batch_size = (len(dataset) // batch_size) * batch_size
        shape = (sample_count_dividable_by_batch_size, shape[1], shape[2], shape[3])
        features = np.zeros(shape=shape)
        labels = np.zeros(shape=(shape[0], 4))
        counter = 0
        for inputs_batch, labels_batch in dataset:
            features_batch = conv_base.predict(inputs_batch)
            if (counter + 1) * 20 > sample_count_dividable_by_batch_size:
                break
            features[counter * batch_size: (counter + 1) * batch_size] = features_batch
            labels[counter * batch_size: (counter + 1) * batch_size] = labels_batch
            counter += 1
        if flatten:
            features = np.reshape(features, (sample_count_dividable_by_batch_size, shape[1] * shape[2] * shape[3]))
        return features, labels

    @staticmethod
    def train_and_save_new_model(path_to_db, model_name, size, validation_set_percentage, epochs, rescale, batch_size):
        data_set = Dataset(path_to_db)
        train, validation, test = data_set.get_train_val_test_sets(size=size,
                                                    rescale=rescale,
                                                    validaion_set_percentage=validation_set_percentage,
                                                    batch_size=batch_size)

        data_batch, labels_batch = train[0]

        CNN_model = NeuralNetwork.create_cnn_model(data_shape=data_batch[0].shape)
        history = CNN_model.fit_generator(
            train,
            steps_per_epoch=100,
            epochs=epochs,
            validation_data=validation,
            validation_steps=50
        )

        CNN_model.save(str(model_name) + '.h5')
        return history

    @staticmethod
    def __choose_pretrained_model_version(extended_version, conv_base, model_name, train, validation, test, epochs,
                                          batch_size, shape, dropout, freeze_layers):
        if extended_version:
            # merge the base and classifier together and then run it on the dataset
            model = NeuralNetwork.add_top_to_model(conv_base=conv_base, freeze_layers=freeze_layers)
            model.save(str(model_name) + '_extended.h5')
            history = model.fit_generator(train, steps_per_epoch=100, epochs=epochs, validation_data=validation,
                                          validation_steps=50)
        else:
            # running a pretrained base on the dataset, save the output, then put it to standalone Dense classifier
            train_features, train_labels = NeuralNetwork.extract_features(conv_base=conv_base,
                                                                          dataset=train,
                                                                          shape=shape,
                                                                          batch_size=batch_size,
                                                                          flatten=True)
            validation_features, validation_labels = NeuralNetwork.extract_features(conv_base=conv_base,
                                                                                    dataset=validation,
                                                                                    shape=shape,
                                                                                    # 5, 3, 512),
                                                                                    batch_size=batch_size,
                                                                                    flatten=True)
            test_features, test_labels = NeuralNetwork.extract_features(conv_base=conv_base,
                                                                        dataset=test,
                                                                        shape=shape,
                                                                        batch_size=batch_size,
                                                                        flatten=True)
            model = NeuralNetwork.create_classifier_for_conv_base(shape=(shape[1:4]),
                                                                  dropout=dropout)
            model.save(str(model_name) + '_basic.h5')
            history = model.fit(train_features, train_labels,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(validation_features, validation_labels))
        return history


    # if basic version chosen then no guarantee that other functions will work properly on that model
    @staticmethod
    def train_and_save_MobileNet(path_to_db, model_name, size, rescale, validation_set_percentage,
                                            batch_size, epochs, extended_version=True):
        data_set = Dataset(path_to_db)
        train, validation, test = data_set.get_train_val_test_sets(size=size, rescale=rescale,
                                                                   validaion_set_percentage=validation_set_percentage,
                                                                   batch_size=batch_size)

        data_batch, labels_batch = train[0]

        conv_base = MobileNet(weights='imagenet',
                              include_top=False,  # I will add my own classifier defining my 4 classes
                              input_shape=data_batch[0].shape)

        # check the shape of the last layer there (e.g. block5_pool for VGG16 base is (None, 5, 3, 512)):
        # conv_base.summary()
        history = NeuralNetwork.__choose_pretrained_model_version(extended_version=extended_version,
                                                                  train=train,
                                                                  test=test,
                                                                  validation=validation,
                                                                  conv_base=conv_base,
                                                                  model_name=str(model_name) + '_MobileNet',
                                                                  batch_size=batch_size,
                                                                  epochs=epochs,
                                                                  shape=(len(validation), 5, 3, 1024),
                                                                  dropout=0.5,
                                                                  freeze_layers=range(0, 70))

        return history

    # if basic version chosen then no guarantee that other functions will work properly on that model
    @staticmethod
    def train_and_save_VGG16(path_to_db, model_name, size, rescale, validation_set_percentage,
                                        batch_size, epochs, extended_version=True):
        data_set = Dataset(path_to_db)
        train, validation, test = data_set.get_train_val_test_sets(size=size, rescale=rescale,
                                                                   validaion_set_percentage=validation_set_percentage,
                                                                   batch_size=batch_size)

        data_batch, labels_batch = train[0]

        conv_base = VGG16(weights='imagenet',
                              include_top=False,  # I will add my own classifier defining my 4 classes
                              input_shape=data_batch[0].shape)

        # check the shape of the last layer there (e.g. block5_pool for VGG16 base is shape=(None, 3, 5, 512)):
        # conv_base.summary()
        history = NeuralNetwork.__choose_pretrained_model_version(extended_version=extended_version,
                                                                  train=train,
                                                                  test=test,
                                                                  validation=validation,
                                                                  conv_base=conv_base,
                                                                  epochs=epochs,
                                                                  batch_size=batch_size,
                                                                  model_name=str(model_name) + '_VGG16',
                                                                  shape=(len(train), 3, 5, 512),
                                                                  dropout=0.5,
                                                                  freeze_layers=range(0, 19))

        return history

    @staticmethod
    def save_training_history(history, your_history_path):
        history_dict = history.history
        json.dump(history_dict, open(your_history_path, 'w'))

    @staticmethod
    def load_training_history(your_history_path):
        history = json.load(open(your_history_path, 'r'))
        return history

    @staticmethod
    def decode_predictions_for_classes(model, frames_classes, print_detailed_predictions):
        for frames_class in frames_classes:
            results = NeuralNetwork.decode_predictions_for_one_class(model=model, frames=frames_class,
                                                                     print_detailed_predictions=False)
            print(results)

    @staticmethod
    def decode_predictions_for_one_class(model, frames, print_detailed_predictions):
        results = []
        for frame in frames:
            preds = model.predict(frame)
            results.append(preds[0])
            if print_detailed_predictions:
                print('Predicted: ' +
                        '\neosinophil:    ' + str(preds[0][0]) +
                        '\nlymphocyte:    ' + str(preds[0][1]) +
                        '\nmonocyte:      ' + str(preds[0][2]) +
                        '\nneutrophil:    ' + str(preds[0][3]))
        mean_result = np.mean(results, axis=0)
        print('Index of frame with highest probability: ', np.argmax(results, axis=0))
        return mean_result
