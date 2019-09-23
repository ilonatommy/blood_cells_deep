import matplotlib.pyplot as plt
from Maths import *
from skimage.transform import resize
from keras import models
from Dataset import *
from old_NeuralNetwork import *
import cv2


class Plotter:
    @staticmethod
    def plot_history(history_dic):
        acc = history_dic['acc']
        val_acc = history_dic['val_acc']
        loss = history_dic['loss']
        val_loss = history_dic['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()

    @staticmethod
    def visualise_image_channels(layers, activations, images_per_row):
        print("Channels visualisation started.")
        layer_names = old_NeuralNetwork.get_layer_names(layers=layers)
        for layer_name, layer_activation in zip(layer_names, activations):
            n_features = layer_activation.shape[-1]
            size_1 = layer_activation.shape[1]
            size_2 = layer_activation.shape[2]
            n_cols = n_features // images_per_row
            display_grid = np.zeros((size_1 * n_cols, images_per_row * size_2))

            for col in range(n_cols):
                for row in range(images_per_row):
                    channel_image = layer_activation[0, :, :, col * images_per_row + row]
                    channel_image = Dataset.normalise_before_display(channel_image)
                    display_grid[col * size_1: (col + 1) * size_1,
                    row * size_2: (row + 1) * size_2] = channel_image

            scale_1 = 1. / size_1
            scale_2 = 1. / size_2
            plt.figure(figsize=(scale_1 * display_grid.shape[1], scale_2 * display_grid.shape[0]))
            plt.title(layer_name)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')

        plt.show()

    @staticmethod
    def _get_layer_filter(model, layer_name, filter_index, shape):
        print(filter_index)
        loss_tensor = Maths.get_loss_tensor(model=model, layer_name=layer_name, filter_index=filter_index)
        loss_tensor_grad = Maths.get_loss_tensor_grad(model=model, layer_name=layer_name,
                                                              filter_index=filter_index)
        iterate = backend.function([model.input], [loss_tensor, loss_tensor_grad])
        gray_image_with_noise = np.random.random(shape) * 20 + 128.
        step = 1.
        for i in range(40):
            loss_tensor, loss_tensor_grad = iterate([gray_image_with_noise])
            gray_image_with_noise += loss_tensor_grad * step
        img = gray_image_with_noise[0]
        img = Dataset.normalise_before_display(img)
        return img

    @staticmethod
    def visualise_filters(layers, model, shape, test_img):
        print("Filters visualisation started.")
        layer_names = old_NeuralNetwork.get_layer_names(layers=layers)

        # layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
        # what do they mean by looking only on the first layer but on the first 64 filters in this layer?
        for index in range(len(layer_names)):
            size_w = shape[1]
            size_h = shape[2]
            margin = 1
            rows = 4
            cols = 8
            results = np.zeros((rows * size_w + (rows - 1) * margin, cols * size_h + (cols - 1) * margin, 3))

            # we look only on 32 filters
            for width in range(rows):
                for height in range(cols):
                    layer_filter = Plotter._get_layer_filter(model=model,
                                                            layer_name=layer_names[index],
                                                            filter_index=width + (height * 4),
                                                            shape=shape)

                    layer_filter = resize(image=layer_filter, output_shape=(size_w, size_h, 3))
                    results[width * size_w + width * margin: width * size_w + width * margin + size_w,
                    height * size_h + height * margin: height * size_h + height * margin + size_h, :] = layer_filter
            plt.imshow(results)
            plt.title(layer_names[index])
            plt.show()

    @staticmethod
    def full_model_visualisation(model, size):
        set_e, set_l, set_m, set_n = Dataset.get_class_sets(frames_size=size)
        # ------------------------------------------------------------------------------------------------------------------
        # visualise channels:

        # version for non-pretrained network or pretrained network other than MobileNet
        # activations = old_NeuralNetwork.get_activations(model=model, image=test_simple_frames_e[0])

        # version for MobileNet - not working. Why I cannot visualise this model?:
        # mn_model = model.layers[0]
        # mn_model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])
        # activations = old_NeuralNetwork.get_activations(model=mn_model, image=test_simple_frames_e[0])

        # Plotter.visualise_image_channels(layers=model.layers, activations=activations, images_per_row=8)
        # ------------------------------------------------------------------------------------------------------------------
        # visualise filters (we have only 32 of them):
        Plotter.visualise_filters(layers=model.layers[:8], model=model, shape=(set_e[0]).shape, test_img=(set_e[0]))

    @staticmethod
    def visualise_heatmap(frame, heatmap):
        width = frame.shape[1]
        height = frame.shape[0]
        display_grid = np.zeros((height, width * 2, 3))
        heatmap = resize(heatmap, (height, width))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        result = heatmap * 0.001 + frame * 0.999
        display_grid[0: height, 0: width, :] = result
        display_grid[0: height, width: 2*width, :] = frame
        plt.imshow(display_grid)
        plt.show()
