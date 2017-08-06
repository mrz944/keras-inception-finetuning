from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

# dimensions of our images.
img_width, img_height = 299, 299

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 3064
nb_validation_samples = 400
train_epochs = 20
fine_tune_epochs = 20
batch_size = 32

# create the base pre-trained model
base_model = applications.InceptionV3(weights='imagenet', include_top=False)
print('Model loaded.')

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 2 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
#               metrics=['accuracy'])

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=0.001),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# train the model on the new data for a few epochs
tensorboard = TensorBoard(log_dir='./logs/training',
                          histogram_freq=1,
                          write_graph=True,
                          write_images=True)

model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=train_epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    callbacks=[tensorboard])

model.save_weights('seeds_split.h5')

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=0.0001),
              metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
tensorboard = TensorBoard(log_dir='./logs/fine-tuning',
                          histogram_freq=1,
                          write_graph=True,
                          write_images=True)

model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=fine_tune_epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    callbacks=[tensorboard])

model.save_weights('seeds_split_fine_tuned.h5')
