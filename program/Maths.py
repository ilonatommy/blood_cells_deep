from keras import backend
from keras.callbacks import *


class Maths:
    @staticmethod
    def get_loss_tensor(model, layer_name, filter_index):
        layer_output = model.get_layer(layer_name).output #get_output_at(-1)
        loss_tensor = backend.mean(layer_output[:, :, :, filter_index])
        return loss_tensor

    @staticmethod
    def get_loss_tensor_grad(loss_tensor, model):
        loss_tensor_grad = backend.gradients(loss_tensor, model.input)[0]
        loss_tensor_grad /= backend.sqrt(backend.mean(backend.square(loss_tensor_grad)) + 1e-5)
        return loss_tensor_grad

    @staticmethod
    def get_heatmap(model, frame, layer_4d, class_number):
        class_output = model.output[:, class_number]
        grads = backend.gradients(class_output, layer_4d.output)[0]
        pooled_grad = backend.mean(grads, axis=(0, 1, 2))
        iterate = backend.function([model.input], [pooled_grad, layer_4d.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([frame])
        for i in range(32):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-5)
        return heatmap

