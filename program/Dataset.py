from os.path import isfile, join

import numpy as np
import glob
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import shutil
import random


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
    def shuffle_dataset(path='/home/ilona/Desktop/Literatura/Bachelors_Thesis/blood_cells/blood-cells_db/dataset2-shuffled',
                        percentage_tr_val_test=[60,15,25]):
        if sum(percentage_tr_val_test) == 100:
            # do not make dirs if they already exist - instead stop execution
            if os.path.exists(os.path.join(path, 'TRAIN')) or \
                    os.path.exists(os.path.join(path, 'VALIDATION')) or \
                    os.path.exists(os.path.join(path, 'TEST')):
                print("Delete TRAIN, VALIDATION and TEST directories from the main dir first.")
                return
            os.mkdir(os.path.join(path, 'TRAIN'))
            os.mkdir(os.path.join(path, 'VALIDATION'))
            os.mkdir(os.path.join(path, 'TEST'))

            categories = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
            for category in categories:
                category_path = os.path.join(path, category)
                category_frames = [os.path.join(category_path, f) for f in os.listdir(category_path) if isfile(join(category_path, f))]
                random.shuffle(category_frames)
                # shuffled, so high time to move into dirs according to percentages:
                category_size = len(category_frames)
                number_tr_val_test = [(category_size * percentage_tr_val_test[0]) //100,
                                      (category_size * percentage_tr_val_test[1]) //100,
                                      (category_size * percentage_tr_val_test[2]) //100]

                print(category_size)
                print(number_tr_val_test)

                train_set = category_frames[0:number_tr_val_test[1]]
                validation_set = category_frames[number_tr_val_test[1]:(number_tr_val_test[1] + number_tr_val_test[2])]
                test_set = category_frames[number_tr_val_test[1] + number_tr_val_test[2]:]

                os.mkdir(os.path.join(path, 'TRAIN', category))
                for moved_frame in train_set:
                    shutil.copyfile(moved_frame,
                                os.path.join(path, 'TRAIN', category,
                                             os.path.basename(
                                    os.path.normpath(moved_frame)
                                )))

                os.mkdir(os.path.join(path, 'VALIDATION', category))
                for moved_frame in validation_set:
                    shutil.copyfile(moved_frame,
                                os.path.join(path, 'VALIDATION', category,
                                             os.path.basename(
                                    os.path.normpath(moved_frame)
                                )))

                os.mkdir(os.path.join(path, 'TEST', category))
                for moved_frame in test_set:
                    shutil.copyfile(moved_frame,
                                os.path.join(path, 'TEST', category,
                                             os.path.basename(
                                    os.path.normpath(moved_frame)
                                )))

                print("Created dirs for " + str(category))



    @staticmethod
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

        print("Finished.")
        return frames

    def __create_validation_set(self, percentage):
        print("Creation of validation set...")
        for category_path in self.train_sub_dirs:
            # create a dir for validation files which will be moved
            category_name = os.path.basename(os.path.normpath(category_path))
            os.mkdir(os.path.join(self.validation_dir, category_name))

            category_frames = [os.path.join(category_path, f)
                               for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
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

    @staticmethod
    def get_class_sets(frames_size):
        data_set = Dataset('/home/ilona/Desktop/Literatura/Bachelors_Thesis/blood_cells/blood-cells_db/dataset2-master')
        test_simple_frames_e = data_set.get_frames(path=os.path.join(
            data_set.base_path, 'images/VALIDATION_SUBSET/E'), size=frames_size, rescale=1. / 255)
        test_simple_frames_l = data_set.get_frames(path=os.path.join(
            data_set.base_path, 'images/VALIDATION_SUBSET/L'), size=frames_size, rescale=1. / 255)
        test_simple_frames_m = data_set.get_frames(path=os.path.join(
            data_set.base_path, 'images/VALIDATION_SUBSET/M'), size=frames_size, rescale=1. / 255)
        test_simple_frames_n = data_set.get_frames(path=os.path.join(
            data_set.base_path, 'images/VALIDATION_SUBSET/N'), size=frames_size, rescale=1. / 255)
        return test_simple_frames_e, test_simple_frames_l, test_simple_frames_m, test_simple_frames_n

    def get_train_val_test_sets(self, size, rescale, validaion_set_percentage, batch_size):
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
            self.__create_validation_set(percentage=validaion_set_percentage)

        self.validation_sub_dirs = [f.path for f in os.scandir(self.validation_dir) if f.is_dir()]

        # decode JPEG to RGB, convert into floating-point tensors, rescale the values to range [0:1]
        print("Preprocessing data...")

        # train_datagen, test_datagen, validation_datagen = self.augmentation()

        train_datagen = ImageDataGenerator(rescale=rescale)
        test_datagen = ImageDataGenerator(rescale=rescale)
        validation_datagen = ImageDataGenerator(rescale=rescale)

        train_datagen = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=size,  # originally: 320x240
            batch_size=batch_size,
            class_mode='categorical'
        )

        test_datagen = test_datagen.flow_from_directory(
            self.validation_dir,
            target_size=size,
            batch_size=batch_size,
            class_mode='categorical'
        )

        validation_generator = validation_datagen.flow_from_directory(
            self.validation_dir,
            target_size=size,
            batch_size=batch_size,
            class_mode='categorical'
        )

        print("Finished loading and preparing data.")
        return train_datagen, validation_generator, test_datagen

    @staticmethod
    def augmentation():
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
        return train_datagen, test_datagen, validation_datagen
