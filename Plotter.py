import matplotlib.pyplot as plt
from Maths import *
from skimage.transform import resize
from Dataset import *
from NeuralNetwork import *
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
    def normalise_before_display(image):
        result = np.copy(image)
        result -= result.mean(axis=0)
        result /= result.std(axis=0) + 1e-5
        result *= 0.1

        result += 0.5
        result = np.clip(result, 0, 1)

        result *= 255
        result = np.clip(result, 0, 255).astype('uint8')
        return result

    @staticmethod
    def visualise_model_filters(model, frame_size):
        print("Filters visualisation started.")
        accepted_layers = ['conv']
        layer_names = NeuralNetwork.get_layer_names(model)
        figure_index = 1
        for layer_name in layer_names:
            # if the layer name contains accepted keyword
            if any(elem in layer_name for elem in accepted_layers):
                Plotter.visualise_layer_filters(model=model, layer_name=layer_name, frame_size=frame_size,
                                                  figure_index=figure_index)
                figure_index += 1
        plt.show()

    @staticmethod
    def visualise_layer_filters(model, layer_name, frame_size, figure_index):
        print('Filters visualisation for layer', layer_name, 'started.')
        layer_output = model.get_layer(layer_name).output
        cols = layer_output.shape[-1] // 4.
        rows = layer_output.shape[-1] // cols
        margin = 2

        size_h = frame_size[0]
        size_w = frame_size[1]
        results = np.zeros((rows * size_h + (rows - 1) * margin, cols * size_w + (cols - 1) * margin, 3))
        for width in range(cols):
            for height in range(rows):
                filter_img = Plotter.__generate_pattern(model, width + (height * rows), frame_size, layer_name)
                v_start = width * size_w + width * margin
                v_end = v_start + size_w
                h_start = height * size_h + height * margin
                h_end = h_start + size_h
                results[h_start: h_end, v_start: v_end, :] = filter_img / 255
        plt.figure(figure_index)
        plt.imshow(results)
        plt.title(layer_name)

    @staticmethod
    def __generate_pattern(model, filter_index, frame_size, layer_name):
        loss = Maths.get_loss_tensor(model, layer_name, filter_index)
        grads = Maths.get_loss_tensor_grad(loss_tensor=loss, model=model)
        iterate = backend.function([model.input], [loss, grads])
        input_img = np.random.random((1, frame_size[0], frame_size[1], 3)) * 20 + 128.
        step = 1.
        for i in range(40):
            loss_tensor, loss_tensor_grad = iterate([input_img])
            input_img += loss_tensor_grad * step
        img = input_img[0]
        img = Plotter.normalise_before_display(img)
        return img

    @staticmethod
    def full_model_visualisation(model, frame):
        """
        # activations:
        # activation visualisation does not work for pretrained models so far -
        # look at comment in NeuralNetwork.get_activations line 127
        activations = NeuralNetwork.get_activations(model, frame)
        Plotter.visualise_image_channels(model.layers, activations, 4)

        # filters:
        # takes a lot of time, but works - uncomment if needed:
        Plotter.visualise_model_filters(model=model, frame_size=frame.shape)
        """
        # heatmaps:
        conv_layers = NeuralNetwork.get_conv_layers(model)
        for conv_layer in conv_layers:
            hm = Maths.get_heatmap(model, [frame], conv_layer, 1)
            Plotter.visualise_heatmap(frame, hm)

    @staticmethod
    def visualise_heatmap(frame, heatmap):
        width = frame.shape[1]
        height = frame.shape[0]
        display_grid = np.zeros((height, width * 2, 3))
        heatmap = resize(heatmap, (height, width))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = heatmap /255
        result = heatmap * 0.4 + frame * 0.6
        display_grid[0: height, 0: width, :] = result
        display_grid[0: height, width: 2*width, :] = frame
        plt.imshow(display_grid)
        plt.text(60, -10, 'in order of increasing weight: blue - yellow - red', style='italic')
        plt.show()