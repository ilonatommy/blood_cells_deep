import json
from keras import models, layers
from keras.applications import VGG16, MobileNet
from Dataset import Dataset
import os
import numpy as np


class NeuralNetwork:
    @staticmethod
    def create_model_from_scratch(input_data_shape, activation, out_activation, kernel_size, dense_units,
                                  features_num_sequence):
        model = models.Sequential()
        for features_num in features_num_sequence:
            model.add(layers.Conv2D(features_num, kernel_size, activation=activation, input_shape=input_data_shape))
            model.add(layers.MaxPooling2D((kernel_size[0] - 1, kernel_size[1] - 1)))
            model.add(layers.Dropout(0.2))
        model.add(layers.Flatten())
        model.add(layers.Dense(dense_units, activation=activation))
        model.add(layers.Dense(4, activation=out_activation))
        return model

    @staticmethod
    def create_keras_kernel_model(input_data_shape):
        input = layers.Input(shape=input_data_shape)
        x = layers.Conv2D(16, 5, strides=(2, 2), padding='same', activation='relu')(input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(8, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(4, 5, strides=(2, 2), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(8, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(4, activation='softmax')(x)
        model = models.Model(inputs=input, outputs=x)
        return model

    @staticmethod
    def create_article_1_model(input_data_shape, kernel_size):
        input = layers.Input(shape=input_data_shape)
        x = layers.Conv2D(64, kernel_size=kernel_size, activation='relu')(input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.7)(x)
        x = layers.Conv2D(64, kernel_size=kernel_size, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.7)(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), border_mode='same')(x)

        x = layers.Conv2D(128, kernel_size=kernel_size, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.7)(x)
        x = layers.Conv2D(128, kernel_size=kernel_size, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.7)(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), border_mode='same')(x)

        x = layers.Conv2D(256, kernel_size=kernel_size, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.7)(x)
        x = layers.Conv2D(256, kernel_size=kernel_size, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.7)(x)
        x = layers.Conv2D(256, kernel_size=kernel_size, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.7)(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), border_mode='same')(x)

        x = layers.Conv2D(512, kernel_size=kernel_size, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.7)(x)
        x = layers.Conv2D(512, kernel_size=kernel_size, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.7)(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), border_mode='same')(x)

        x = layers.Conv2D(512, kernel_size=kernel_size, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.7)(x)
        x = layers.Conv2D(512, kernel_size=kernel_size, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.7)(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), border_mode='same')(x)

        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(4, activation='softmax')(x)

        model = models.Model(inputs=input, outputs=x)
        return model

    @staticmethod
    def __create_top_model(input_shape, activation, dropout=0.5, output_shape=4, output_activation='softmax'):
        top_model = models.Sequential()
        merging_layer = layers.Flatten(input_shape=input_shape)
        top_model.add(merging_layer)
        top_model.add(layers.Dense(256, activation=activation))
        top_model.add(layers.Dropout(dropout))
        top_model.add(layers.Dense(output_shape, activation=output_activation))
        return top_model

    @staticmethod
    def add_top_to_base_model(base_model, freeze_layers, activation, dropout=0.5, output_shape=4,
                              output_activation='softmax'):
        # freeze certain layers to prevent changes in its weights during the training:
        base_model.trainable = True
        for layer_index in freeze_layers:
            layer = base_model.layers[layer_index]
            layer.trainable = False

        top = layers.Flatten()(base_model.output)
        top = layers.Dropout(dropout)(top)
        top = layers.Dense(256, activation=activation)(top)
        # why it pays off to add BatchNormalisation:
        # https://towardsdatascience.com/batch-normalization-in-neural-networks-1ac91516821c
        top = layers.BatchNormalization()(top)
        pred_layer = layers.Dense(output_shape, activation=output_activation)(top)

        model = models.Model(input=base_model.input, output=pred_layer)
        return model

    @staticmethod
    def create_VGG16(frame_size, basic=True, freeze_layers=range(0), activation='relu', dropout=0.5, output_shape=4,
                     output_activation='softmax'):
        input_shape = (frame_size[0], frame_size[1], 3)
        if basic:
            # When setting `include_top=True` and loading `imagenet` weights, `input_shape` should be (224, 224, 3)
            return VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3), output_shape=(4,))
        else:
            base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
            return NeuralNetwork.add_top_to_base_model(base, freeze_layers, activation, dropout, output_shape,
                                                       output_activation)

    @staticmethod
    def create_MobileNet(frame_size, basic=True, freeze_layers=range(0), activation='relu', dropout=0.5, output_shape=4,
                         output_activation='softmax'):
        input_shape = (frame_size[0], frame_size[1], 3)
        if basic:
            # When setting `include_top=True` and loading `imagenet` weights, `input_shape` should be (224, 224, 3)
            return MobileNet(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
        else:
            base = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
            return NeuralNetwork.add_top_to_base_model(base, freeze_layers, activation, dropout, output_shape,
                                                       output_activation)

    @staticmethod
    def save_summary(model, model_name, output_path):
        if not os.path.exists(os.path.join(output_path, model_name)):
            os.makedirs(os.path.join(output_path, model_name))
        with open(os.path.join(output_path, model_name, str(model_name) + '_summary.txt'), 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

    @staticmethod
    def save_model(model, model_name, output_path):
        if not os.path.exists(os.path.join(output_path, model_name)):
            os.makedirs(os.path.join(output_path, model_name))
        model.save(os.path.join(output_path, model_name, str(model_name) + '.h5'))
        print('Model', model_name, 'was saved.')

    @staticmethod
    def load_model(output_path, model_name):
        model = models.load_model(os.path.join(output_path, model_name, str(model_name) + '.h5'))
        return model

    @staticmethod
    def save_training_history(history, model_name, output_path):
        if not os.path.exists(os.path.join(output_path, model_name)):
            os.makedirs(os.path.join(output_path, model_name))
        history_dict = history.history
        json.dump(history_dict, open(os.path.join(output_path, model_name, str(model_name) + '_training'), 'w'))
        print('Training', str(model_name) + '_training', 'was saved.')

    @staticmethod
    def load_training_history(output_path, model_name):
        history = json.load(open(os.path.join(output_path, model_name, str(model_name) + '_training'), 'r'))
        return history

    @staticmethod
    def train_network(path_to_db, model, frame_size, rescale, validation_set_percentage, batch_size, epochs, optimizer,
                      loss='categorical_crossentropy', metrics=['acc']):
        data_set = Dataset(path_to_db)
        train, validation, test = data_set.get_train_val_test_sets(size=frame_size, rescale=rescale,
                                                                   validaion_set_percentage=validation_set_percentage,
                                                                   batch_size=batch_size)
        # configure the model for training:
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        history = model.fit_generator(train, steps_per_epoch=100, epochs=epochs, validation_data=validation,
                                      validation_steps=50)
        return history

    @staticmethod
    def get_activations(model, frame, num_of_visualised_layers=8):
        dim_m = len(model.input.get_shape())
        dim_f = len(frame.shape)
        if dim_f != dim_m:
            print('Image dim does not fit the model input dim. Check function parameters.')
            return []
        layer_outputs = [layer.output for layer in model.layers[:8]]
        activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
        # flollowing line gives an error for pretrained models (for built from scratch it does not):
        # "input_1_1:0 is both fed and fetched."
        # which is caused by the first layer of pretrained network. How to modify those models to avoid this error?
        activations = activation_model.predict(frame)
        return activations

    @staticmethod
    def get_layer_names(model):
        layer_names = []
        for layer in model.layers:
            layer_names.append(layer.name)
        return layer_names

    @staticmethod
    def get_conv_layers(model):
        conv_layers = []
        for layer in model.layers:
            if 'conv' in layer.name:
                conv_layers.append(layer)
        return conv_layers

    @staticmethod
    def __decode_predictions_for_one_class(model, frames, file_to_save):
        results = []
        for frame in frames:
            preds = model.predict(frame)
            results.append(preds[0])
            file_to_save.write(str(preds[0]))
            file_to_save.write('\n--------------------- ' +
                               '\neosinophil:    ' + str(preds[0][0]) +
                               '\nlymphocyte:    ' + str(preds[0][1]) +
                               '\nmonocyte:      ' + str(preds[0][2]) +
                               '\nneutrophil:    ' + str(preds[0][3]) +
                               '\n')
        mean_result = np.mean(results, axis=0)
        file_to_save.write('\n' + str(mean_result))
        file_to_save.write('\nIndex of frame with highest probability: ' + str(np.argmax(results, axis=0)))
        return results

    @staticmethod
    def decode_and_save_predictions(model, frames_classes, model_name, output_path):
        if not os.path.exists(os.path.join(output_path, model_name)):
            os.makedirs(os.path.join(output_path, model_name))
        f = open(os.path.join(output_path, model_name, str(model_name) + '_predictions.txt'), "w+")

        results = []
        cnt = 0
        for frames_class in frames_classes:
            f.write("\n----------------------\nclass " + cnt)
            results = NeuralNetwork.__decode_predictions_for_one_class(model=model, frames=frames_class, file_to_save=f)
            cnt = cnt + 1

        f.close()
        print('Predictions', str(model_name) + '_predictions', 'was saved.')