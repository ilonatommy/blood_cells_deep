from keras import backend
import numpy as np


class Maths:
    @staticmethod
    def get_loss_tensor(model, layer_name, filter_index):
        layer_output = model.get_layer(layer_name).get_output_at(-1)
        print(layer_output.shape)
        loss_tensor = backend.mean(layer_output[:, :, :, filter_index])
        return loss_tensor

    @staticmethod
    def get_loss_tensor_grad(model, layer_name, filter_index):
        loss_tensor = Maths.get_loss_tensor(model=model, layer_name=layer_name, filter_index=filter_index)
        loss_tensor_grad = backend.gradients(loss_tensor, model.input)[0]
        loss_tensor_grad /= backend.sqrt(backend.mean(backend.square(loss_tensor_grad)) + 1e-5)
        return loss_tensor_grad

    @staticmethod
    def get_heatmap(model, layer_name, frame, class_number):
        what_is_it = model.output[:, class_number]
        last_conv_layer = model.get_layer(layer_name)
        grads = backend.gradients(what_is_it, last_conv_layer.output)[0]
        pooled_grad = backend.mean(grads, axis=(0, 1, 2))
        iterate = backend.function([model.input], [pooled_grad, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([frame])
        for i in range(128):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap