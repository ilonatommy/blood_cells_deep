from keras import backend


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
