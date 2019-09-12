import numpy as np
import matplotlib.image as mpimg
import glob
import os
from keras.preprocessing.image import ImageDataGenerator
import shutil


class Dataset:
    base_path = ''
    test_dir = None
    test_sub_dirs = []
    validation_dir = None
    validation_sub_dirs = []
    train_dir = None
    train_sub_dirs = []

    def __init__(self, path):
        assert isinstance(path, str), 'Argument of wrong type! Expected string.'
        self.base_path = path

    @staticmethod
    def __get_frames(path):
        assert isinstance(path, str), 'Argument of wrong type! Expected string.'
        print("Loading data from " + path)

        frames = []
        for filename in glob.glob(os.path.join(path, '*.jpeg')):
            im = mpimg.imread(filename)
            frames.append(im)
        print("Finished.")
        return frames

    def __create_validation_set(self, percentage):
        print("Creation of validation set...")
        for category_path in self.train_sub_dirs:
            # create a dir for validation files which will be moved
            category_name = os.path.basename(os.path.normpath(category_path))
            os.mkdir(os.path.join(self.validation_dir, category_name))

            category_frames = [os.path.join(category_path, f)
                               for f in os.listdir(category_path) if os.isfile(os.path.join(category_path, f))]
            num_elem_category = int(round(len(category_frames) * percentage / 100))
            validation_set = category_frames[num_elem_category:]
            for moved_frame in validation_set:
                shutil.move(moved_frame,
                            os.path.join(
                                self.validation_dir,
                                category_name,
                                os.path.basename(
                                    os.path.normpath(moved_frame)
                                )))
        print("Validation set ready.")

    def get_data(self):
        print("Loading data...")

        frames_path = os.path.join(self.base_path, 'images')
        self.train_dir = os.path.join(frames_path, 'TRAIN')
        self.test_dir = os.path.join(frames_path, "TEST")
        self.validation_dir = os.path.join(frames_path, "VALIDATION")
        if not os.path.exists(self.validation_dir):
            os.makedirs(self.validation_dir)

        self.train_sub_dirs = [f.path for f in os.scandir(self.train_dir) if f.is_dir()]
        self.test_sub_dirs = [f.path for f in os.scandir(self.test_dir) if f.is_dir()]

        # prepare validation set if self.validation_dir is empty
        if not os.listdir(self.validation_dir):
            self.__create_validation_set(percentage=90)

        self.validation_sub_dirs = [f.path for f in os.scandir(self.validation_dir) if f.is_dir()]

        # decode JPEG to RGB, convert into floating-point tensors, rescale the values to range [0:1]
        print("Preprocessing data...")
        """
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        test_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        validation_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        """
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        validation_datagen = ImageDataGenerator(rescale=1. / 255)

        train_datagen = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(320, 240),  # originally: 320x240
            batch_size=20,
            class_mode='categorical'
        )

        test_datagen = test_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(320, 240),
            batch_size=20,
            class_mode='categorical'
        )

        validation_generator = validation_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(320, 240),
            batch_size=20,
            class_mode='categorical'
        )

        print("Finished loading and preparing data.")
        return train_datagen, validation_generator, test_datagen

    @staticmethod
    def normalise_set(dataset):
        results = []
        for key in dataset:
            for data in dataset[key]:
                result = np.copy(data)
                result = np.asarray(result, dtype='float32')
                mean = result.mean(axis=0)
                std = result.std(axis=0)
                result -= mean
                result /= std
                results.append(result)
        return results


