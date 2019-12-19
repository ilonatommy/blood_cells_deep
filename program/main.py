import json
import glob
import shutil
import random
from keras.applications import VGG16, MobileNet, Xception, ResNet50, InceptionV3

import numpy as np
import math, cv2, os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import models, layers
from keras.layers import Input, Activation, Dense, Conv2D, Reshape, concatenate, BatchNormalization, MaxPooling2D, \
    GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from keras import optimizers
from IPython.display import clear_output
#matplotlib inline
"""
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
"""
from random import randint
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def __create_validation_set(percentage, categories, train_dataset_path, validation_dataset_path):
    print("Creation of validation set...")
    if not os.path.exists(validation_dataset_path):
        print("Creation of validation dir...")
        os.mkdir(os.path.join(validation_dataset_path))
    for cat in categories:
        print(cat)
        # create a dir for validation files which will be moved
        category_path = os.path.join(train_dataset_path, cat)
        if not os.path.exists(os.path.join(validation_dataset_path, cat)):
            print("Creation of " + str(cat) + " dir...")
            os.mkdir(os.path.join(validation_dataset_path, cat))

        category_frames = [os.path.join(category_path, f)
                           for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        random.shuffle(category_frames)
        num_elem_category = int(round(len(category_frames) * percentage / 100))
        validation_set = category_frames[num_elem_category:]
        print(len(validation_set))
        for moved_frame in validation_set:
            shutil.move(moved_frame,
                        os.path.join(
                            validation_dataset_path,
                            cat))
    print("Validation set ready.")


# ------------------saving and loading functions--------------------------------
def save_model(model, model_name, output_path):
    if not os.path.exists(os.path.join(output_path, model_name)):
        os.makedirs(os.path.join(output_path, model_name))
    model.save(os.path.join(output_path, model_name, str(model_name) + '.h5'))
    print('Model', model_name, 'was saved.')


def load_model(output_path, model_name):
    model = models.load_model(os.path.join(output_path, model_name, str(model_name) + '.h5'))
    return model


def save_training_history(history, model_name, output_path):
    if not os.path.exists(os.path.join(output_path, model_name)):
        os.makedirs(os.path.join(output_path, model_name))
    history_dict = history.history
    json.dump(str(history_dict), open(os.path.join(output_path, model_name, str(model_name) + '_training'), 'w'))
    print('Training', str(model_name) + '_training', 'was saved.')


def load_training_history(output_path, model_name):
    history = json.load(open(os.path.join(output_path, model_name, str(model_name) + '_training'), 'r'))
    return history


def save_summary(model, model_name, output_path):
    if not os.path.exists(os.path.join(output_path, model_name)):
        os.makedirs(os.path.join(output_path, model_name))
    with open(os.path.join(output_path, model_name, str(model_name) + '_summary.txt'), 'w') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


# -----------------------------data loading funtions----------------------------
def get_frames(path, size, rescale):
    assert isinstance(path, str), 'Argument of wrong type! Expected string.'
    print("Loading data from " + path)

    frames = []
    for filename in glob.glob(os.path.join(path, '*.jpeg')):
        im = image.load_img(os.path.join(path, filename), target_size=size)
        img_arr = image.img_to_array(im)
        img_arr = np.expand_dims(img_arr, axis=0)
        img_arr = img_arr * rescale
        frames.append(img_arr)

    print("Finished. Loaded " + str(len(frames)) + " frames.")
    return frames


def get_class_sets(db_path, frames_size, rescale, categories):
    results = []
    for cat in categories:
        frames = get_frames(path=
                            os.path.join(db_path, cat), size=frames_size, rescale=rescale)
        results.append(frames)
    return results


# ----------------------predictions and testing----------------------------------
def __decode_predictions_for_one_class(model, frames, file_to_save, class_number, th):
    results = []
    correctly_classified_cnt = 0
    all_frames_cnt = 0
    if frames != []:
        for frame in frames:
            preds = model.predict(frame)
            results.append(preds[0])
            file_to_save.write(str(preds[0]))
            file_to_save.write(
                '\neosinophil:    ' + str(preds[0][0]) +
                '\nlymphocyte:    ' + str(preds[0][1]) +
                '\nmonocyte:      ' + str(preds[0][2]) +
                '\nneutrophil:    ' + str(preds[0][3]) +
                '\n')
            if preds[0][class_number] > (th / 100):
                correctly_classified_cnt = correctly_classified_cnt + 1
            all_frames_cnt = all_frames_cnt + 1
        mean_result = np.mean(results, axis=0)
        file_to_save.write('\n' + str(mean_result))
        file_to_save.write('\nIndex of frame with the highest probability: ' + str(np.argmax(results, axis=0)))
        file_to_save.write(
            '\nPercentage of correctly classified frames: ' + str(correctly_classified_cnt / all_frames_cnt * 100))
    return results


def decode_and_save_predictions(model, frames_classes, model_name, output_path, th):
    if not os.path.exists(os.path.join(output_path, model_name)):
        os.makedirs(os.path.join(output_path, model_name))
    f = open(os.path.join(output_path, model_name, str(model_name) + '_predictions_th' + str(th) + '.txt'), "w+")

    results = []
    cnt = 0
    for frames_class in frames_classes:
        f.write("\n----------------------\nclass " + str(cnt))
        f.write('\n----------------------\n')
        results = __decode_predictions_for_one_class(model=model, frames=frames_class, file_to_save=f, class_number=cnt,
                                                     th=th)
        cnt = cnt + 1

    f.close()
    print('Predictions', str(model_name) + '_predictions_th' + str(th), 'was saved.')


def get_confusion_matrix(model, generator, categories, generator_type):
    if not os.path.exists(os.path.join(output_path, model_name)):
        os.makedirs(os.path.join(output_path, model_name))
    f = open(os.path.join(output_path, model_name, str(model_name) + '_conf_matrix_' + str(generator_type) + '.txt'),
             "w+")

    Y_pred = model.predict_generator(generator)
    y_pred = np.argmax(Y_pred, axis=1)
    c_m = confusion_matrix(generator.classes, y_pred)
    c_r = classification_report(generator.classes, y_pred, target_names=categories)
    f.write('Confusion Matrix\n' + str(c_m) + "\nClassification Report\n" + str(c_r))
    print('Confusion Matrix\n' + str(c_m))
    print('Classification Report\n' + str(c_r))


def test_network(test_dataset, model, frame_size, rescale, batch_size):
    print('\n# Evaluation on test data')
    results = model.evaluate(test_dataset)
    return results


# ------------------------creating model functions-------------------------------

# ----------kaggle:
# https://www.kaggle.com/drobchak1988/blood-cell-images-acc-92-val-acc-90
def fire(x, squeeze, expand, bnmomemtum=0.85):
    y = Conv2D(filters=squeeze, kernel_size=1, activation='relu', padding='same')(x)
    y = BatchNormalization(momentum=bnmomemtum)(y)
    y1 = Conv2D(filters=expand // 2, kernel_size=1, activation='relu', padding='same')(y)
    y1 = BatchNormalization(momentum=bnmomemtum)(y1)
    y3 = Conv2D(filters=expand // 2, kernel_size=3, activation='relu', padding='same')(y)
    y3 = BatchNormalization(momentum=bnmomemtum)(y3)
    return concatenate([y1, y3])


def fire_module(squeeze, expand):
    return lambda x: fire(x, squeeze, expand)


def model_231(input_shape, bnmomemtum=0.85):
    x = Input(shape=input_shape)
    y = BatchNormalization(center=True, scale=False)(x)
    y = Activation('relu')(y)
    y = Conv2D(kernel_size=5, filters=12, padding='same', use_bias=True, activation='relu')(x)
    y = BatchNormalization(momentum=bnmomemtum)(y)

    y = fire_module(12, 24)(y)
    y = MaxPooling2D(pool_size=2)(y)

    y = fire_module(24, 48)(y)
    y = MaxPooling2D(pool_size=2)(y)

    y = fire_module(32, 64)(y)
    y = MaxPooling2D(pool_size=2)(y)

    y = fire_module(24, 48)(y)
    y = MaxPooling2D(pool_size=2)(y)

    y = fire_module(18, 36)(y)
    y = MaxPooling2D(pool_size=2)(y)

    y = fire_module(12, 24)(y)

    y = GlobalAveragePooling2D()(y)
    y = Dense(4, activation='softmax')(y)
    model = Model(x, y)
    return model


# -----------------------program and model parameters---------------------------------------------------------------------------
# w 3 jest przetrenowany na zbiorze treningowym + testowy (sic!)
# w 4 jest przetrenowany na zbiorze trningowym + walidacyjnym
print(tf.__version__)

output_path = "/home/ilona/Desktop/Literatura/Bachelors_Thesis/blood_cells/blood_cells_deep_network/program/displayforecastio/kaggle/models_2.3.1."
db_path = "/home/ilona/Desktop/Literatura/Bachelors_Thesis/blood_cells/databases/blood-cells_3/dataset2-master/dataset2-master/images"
train_dataset_path = os.path.join(db_path, "TRAIN")
validation_dataset_path = os.path.join(db_path, "VALIDATION")
test_dataset_path = os.path.join(db_path, "TEST")
pred_dataset_path = os.path.join(db_path, "TEST_SIMPLE")

IMG_SIZE = 128
data_list = os.listdir(train_dataset_path)
NUM_CLASSES = len(data_list)
BATCH_SIZE = 32
EPOCHS = 25
CATEGORIES = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

input_shape = (IMG_SIZE, IMG_SIZE, 3)
rescale = 1. / 255
model_name = "model_231_original"

# -----------------------------create validation set if not existing-------------------------------------------------------------
__create_validation_set(80, CATEGORIES, train_dataset_path, validation_dataset_path)
# ------------------------------normalisation, loading data-----------------------------------------------------------------------

train_datagen = ImageDataGenerator(rescale=rescale)
validation_datagen = ImageDataGenerator(rescale=rescale)
test_datagen = ImageDataGenerator(rescale=rescale)
pred_datagen = ImageDataGenerator(rescale=rescale)

# ----------------------------------building the model------------------------------------------------------------------

model = model_231(input_shape)
model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
save_model(model, model_name, output_path)
save_summary(model, model_name, output_path)

# -------------------------------training-----------------------------------------------------------------------

train_generator = train_datagen.flow_from_directory(
        train_dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=None,
        class_mode="categorical")

validation_generator = validation_datagen.flow_from_directory(
        validation_dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=None,
        class_mode="categorical")

STEP_SIZE_TRAIN=train_generator.n // train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n // validation_generator.batch_size

history = model.fit_generator(
      train_generator,
      steps_per_epoch=STEP_SIZE_TRAIN,
      epochs=EPOCHS,
      validation_data=validation_generator,
      validation_steps=STEP_SIZE_VALID,
      workers=1,
      use_multiprocessing=False)
      #callbacks=[plot_losses, callback_is_nan]) # callback_learning_rate,
save_training_history(history, model_name, output_path)
history = history.history
get_confusion_matrix(model, validation_generator, CATEGORIES, 'val')

# -----------------------------plotting results-----------------------------------------------------------------

#model = load_model(output_path, model_name)
#history = load_training_history(output_path, model_name)

pred_generator = test_datagen.flow_from_directory(
        pred_dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=None,
        class_mode="categorical")

pred_images = get_class_sets(pred_dataset_path, (IMG_SIZE, IMG_SIZE), rescale, CATEGORIES)
preds = decode_and_save_predictions(model=model, frames_classes=pred_images, model_name=model_name, output_path=output_path, th=25)
"""
accuracy = history['acc']
loss = history['loss']
val_accuracy = history['val_acc']
val_loss = history['val_loss']

print(f'Training Accuracy: {np.max(accuracy)}')
print(f'Training Loss: {np.min(loss)}')
print(f'Validation Accuracy: {np.max(val_accuracy)}')
print(f'Validation Loss: {np.min(val_loss)}')

plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
"""
# -----------------------------testing evaluation---------------------------------------------------------------

#model = load_model(output_path, model_name)
test_generator = test_datagen.flow_from_directory(
    test_dataset_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=None,
    class_mode="categorical")

get_confusion_matrix(model, test_generator, CATEGORIES, 'test')

test_network(test_generator, model, (IMG_SIZE, IMG_SIZE), rescale, BATCH_SIZE)
