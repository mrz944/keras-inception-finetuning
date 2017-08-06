# -*- coding: utf-8 -*-
"""Train model using transfer learning."""

# https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator

import os
import re
import glob
import hashlib
import argparse
import warnings

import six
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from keras.models import Model
from keras import backend as K
from keras.optimizers import SGD
from keras.layers import Dense, GlobalAveragePooling2D, Input
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import (ImageDataGenerator, Iterator,
                                       array_to_img, img_to_array, load_img)
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

RANDOM_SEED = 0
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
VALID_IMAGE_FORMATS = frozenset(['jpg', 'jpeg', 'JPG', 'JPEG'])
# we chose to train the top 2 inception blocks
BATCH_SIZE = 100
TRAINABLE_LAYERS = 172
INCEPTIONV3_BASE_LAYERS = len(InceptionV3(weights=None, include_top=False).layers)

STEPS_PER_EPOCH = 625
VALIDATION_STEPS = 100
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
FC_LAYER_SIZE = 1024

# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath='./output/checkpoints/inception.{epoch:03d}-{val_loss:.2f}.hdf5',
    verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir='./output/')


def as_bytes(bytes_or_text, encoding='utf-8'):
    """Converts bytes or unicode to `bytes`, using utf-8 encoding for text.

    # Arguments
        bytes_or_text: A `bytes`, `str`, or `unicode` object.
        encoding: A string indicating the charset for encoding unicode.

    # Returns
        A `bytes` object.

    # Raises
        TypeError: If `bytes_or_text` is not a binary or unicode string.
    """
    if isinstance(bytes_or_text, six.text_type):
        return bytes_or_text.encode(encoding)
    elif isinstance(bytes_or_text, bytes):
        return bytes_or_text
    else:
        raise TypeError('Expected binary or unicode string, got %r' %
                        (bytes_or_text,))


class CustomImageDataGenerator(ImageDataGenerator):
    def flow_from_image_lists(self, image_lists,
                              category, image_dir,
                              target_size=(256, 256), color_mode='rgb',
                              class_mode='categorical',
                              batch_size=32, shuffle=True, seed=None,
                              save_to_dir=None,
                              save_prefix='',
                              save_format='jpeg'):
        return ImageListIterator(
            image_lists, self,
            category, image_dir,
            target_size=target_size, color_mode=color_mode,
            class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format)


class ImageListIterator(Iterator):
    """Iterator capable of reading images from a directory on disk.

    # Arguments
        image_lists: Dictionary of training images for each label.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        target_size: tuple of integers, dimensions to resize input images to.
        color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
        classes: Optional list of strings, names of sudirectories
            containing images from each class (e.g. `["dogs", "cats"]`).
            It will be computed automatically if not set.
        class_mode: Mode for yielding the targets:
            `"binary"`: binary targets (if there are only two classes),
            `"categorical"`: categorical targets,
            `"sparse"`: integer targets,
            `None`: no targets get yielded (only input images are yielded).
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, image_lists, image_data_generator,
                 category, image_dir,
                 target_size=(256, 256), color_mode='rgb',
                 class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if data_format is None:
            data_format = K.image_data_format()

        classes = list(image_lists.keys())
        self.category = category
        self.num_class = len(classes)
        self.image_lists = image_lists
        self.image_dir = image_dir

        how_many_files = 0
        for label_name in classes:
            for _ in self.image_lists[label_name][category]:
                how_many_files += 1

        self.samples = how_many_files
        self.class2id = dict(zip(classes, range(len(classes))))
        self.id2class = dict((v, k) for k, v in self.class2id.items())
        self.classes = np.zeros((self.samples,), dtype='int32')

        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size

        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        i = 0
        self.filenames = []
        for label_name in classes:
            for j, _ in enumerate(self.image_lists[label_name][category]):
                self.classes[i] = self.class2id[label_name]
                img_path = get_image_path(self.image_lists,
                                          label_name,
                                          j,
                                          self.image_dir,
                                          self.category)
                self.filenames.append(img_path)
                i += 1

        print("Found {} {} files".format(len(self.filenames), category))
        super(ImageListIterator, self).__init__(self.samples, batch_size, shuffle,
                                                seed)

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape,
                           dtype=K.floatx())
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            img = load_img(self.filenames[j],
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=current_index + i,
                    hash=np.random.randint(10000),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.num_class),
                               dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, batch_y


# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
def create_image_lists(image_dir, validation_pct=10):
    """Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    # Arguments
        image_dir: string path to a folder containing subfolders of images.
        validation_pct: integer percentage of images reserved for validation.

    # Returns
        dictionary of label subfolder, with images split into training
        and validation sets within each label.
    """
    if not os.path.isdir(image_dir):
        raise ValueError("Image directory {} not found.".format(image_dir))
    image_lists = {}
    sub_dirs = [x[0] for x in os.walk(image_dir)]
    sub_dirs_without_root = sub_dirs[1:]  # first element is root directory
    for sub_dir in sub_dirs_without_root:
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        print("Looking for images in '{}'".format(dir_name))
        for extension in VALID_IMAGE_FORMATS:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            warnings.warn('No files found')
            continue
        if len(file_list) < 20:
            warnings.warn('Folder has less than 20 images, which may cause '
                          'issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            warnings.warn('WARNING: Folder {} has more than {} images. Some '
                          'images will never be selected.'
                          .format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # Get the hash of the file name and perform variant assignment.
            hash_name = hashlib.sha1(as_bytes(base_name)).hexdigest()
            hash_pct = ((int(hash_name, 16) % (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                        (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if hash_pct < validation_pct:
                validation_images.append(base_name)
            else:
                training_images.append(base_name)
        image_lists[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'validation': validation_images,
        }
    return image_lists


# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py
def get_image_path(image_lists, label_name, index, image_dir, category):
    """"Returns a path to an image for a label at the given index.

    # Arguments
      image_lists: Dictionary of training images for each label.
      label_name: Label string we want to get an image for.
      index: Int offset of the image we want. This will be moduloed by the
      available number of images for the label, so it can be arbitrarily large.
      image_dir: Root folder string of the subfolders containing the training
      images.
      category: Name string of set to pull images from - training, testing, or
      validation.

    # Returns
      File system path string to an image that meets the requested parameters.
    """
    if label_name not in image_lists:
        raise ValueError('Label does not exist ', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        raise ValueError('Category does not exist ', category)
    category_list = label_lists[category]
    if not category_list:
        raise ValueError('Label %s has no images in the category %s.',
                         label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_generators(image_lists, image_dir):
    train_datagen = CustomImageDataGenerator(rescale=1. / 255,
                                             horizontal_flip=True)

    test_datagen = CustomImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_image_lists(
        image_lists=image_lists,
        category='training',
        image_dir=image_dir,
        target_size=(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=RANDOM_SEED)

    validation_generator = test_datagen.flow_from_image_lists(
        image_lists=image_lists,
        category='validation',
        image_dir=image_dir,
        target_size=(MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=RANDOM_SEED)

    return train_generator, validation_generator


def get_model(num_classes, weights='imagenet'):
    # create the base pre-trained model
    # , input_tensor=input_tensor
    base_model = InceptionV3(weights=weights, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(FC_LAYER_SIZE, activation='relu')(x)
    # and a logistic layer -- let's say we have 2 classes
    predictions = Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=[base_model.input], outputs=[predictions])
    return model


def get_top_layer_model(model):
    """Used to train just the top layers of the model."""
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in model.layers[:INCEPTIONV3_BASE_LAYERS]:
        layer.trainable = False
    for layer in model.layers[INCEPTIONV3_BASE_LAYERS:]:
        layer.trainable = True

    # compile the model (should be done after setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def get_mid_layer_model(model):
    """After we fine-tune the dense layers, train deeper."""
    # freeze the first TRAINABLE_LAYER_INDEX layers and unfreeze the rest
    for layer in model.layers[:TRAINABLE_LAYERS]:
        layer.trainable = False
    for layer in model.layers[TRAINABLE_LAYERS:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model(model, epochs, generators, callbacks=None):
    train_generator, validation_generator = generators
    model.fit_generator(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_generator,
        validation_steps=VALIDATION_STEPS,
        epochs=epochs,
        callbacks=callbacks)
    return model


def main(image_dir, validation_pct):
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    num_classes = len(sub_dirs) - 1
    print("Number of classes found: {}".format(num_classes))

    model = get_model(num_classes)

    print("Using validation percent of %{}".format(validation_pct))
    image_lists = create_image_lists(image_dir, validation_pct)

    generators = get_generators(image_lists, image_dir)

    # Get and train the top layers.
    model = get_top_layer_model(model)
    model = train_model(model, epochs=10, generators=generators)

    # Get and train the mid layers.
    model = get_mid_layer_model(model)
    _ = train_model(model, epochs=100, generators=generators,
                    callbacks=[checkpointer, early_stopper, tensorboard])

    # save model
    model.save('./output/model.hdf5', overwrite=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', required=True, help='data directory')
    parser.add_argument('--validation-pct', default=10, help='validation percentage')
    args = parser.parse_args()

    os.makedirs('./output/checkpoints/', exist_ok=True)

    main(**vars(args))
