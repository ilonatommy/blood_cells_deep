from Dataset import *
from NeuralNetwork import *
from Plotter import *

data_set = Dataset('/home/ilona/Desktop/Literatura/Bachelors_Thesis/blood_cells/blood-cells_db/dataset2-master')
train, validation, test = data_set.get_data()

data_batch, labels_batch = train[0]

CNN_model = NeuralNetwork.create_cnn_model(data_shape=data_batch[0].shape)
history = CNN_model.fit_generator(
    train,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation,
    validation_steps=50
)

CNN_model.save('blood_cells_1')
Plotter.plot_history(history=history)

# conclusions:
# saw-shape of the plot - probably too many layers.
# I also decreased the size of image to shorten the training time.
# still acc not higher than 80%...
