from keras import layers
from keras import models
from keras import optimizers
from keras import backend
import numpy as np


class NeuralNetwork:
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
                      loss='binary_crossentropy',
                      metrics=['acc'])
        return model

    @staticmethod
    def merge_classifier_and_conv_base(conv_base, freeze_layers):
        model = models.Sequential()
        # freeze certain layers to prevent changes in its weights during the training:
        conv_base.trainable = True
        for layer_index in freeze_layers:
            layer = conv_base.layers[layer_index]
            layer.trainable = False

        conv_base.trainable = True
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(4, activation='softmax'))

        model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                      loss='binary_crossentropy',
                      metrics=['acc'])
        return model

    @staticmethod
    def get_activations(model, image):
        layer_outputs = [layer.output for layer in model.layers[:8]]
        activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
        activations = activation_model.predict(image)
        return activations

    @staticmethod
    def get_loss_tensor(model, layer_name, filter_index):
        layer_output = model.get_layer(layer_name).output
        loss_tensor = backend.mean(layer_output[:, :, :, filter_index])
        return loss_tensor

    @staticmethod
    def get_loss_tensor_grad(model, layer_name, filter_index):
        loss_tensor = NeuralNetwork.get_loss_tensor(model=model, layer_name=layer_name, filter_index=filter_index)
        loss_tensor_grad = backend.gradients(loss_tensor, model.input)[0]
        loss_tensor_grad /= backend.sqrt(backend.mean(backend.square(loss_tensor_grad)) + 1e-5)
        return loss_tensor_grad

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
