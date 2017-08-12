from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model
from keras.layers import Activation, Conv2D, Dense, Dropout, GlobalAveragePooling2D, Reshape

# Settings

train_directory = './data/train'
validation_directory = './data/validation'

img_width, img_height = 224, 224
batch_size = 16
train_epochs = 30
fine_tune_epochs = 60
train_samples = 3064
validation_samples = 400

# Data generators & augmentation

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

train_generator = datagen.flow_from_directory(
    train_directory,
    target_size=(img_height, img_width),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    seed=123)

validation_generator = datagen.flow_from_directory(
    validation_directory,
    target_size=(img_height, img_width),
    color_mode='rgb',
    classes=None,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    seed=123)

# Loading pre-trained model and adding custom layers
base_model = applications.mobilenet.MobileNet(weights='imagenet',
                                              include_top=False,
                                              input_shape=(img_height, img_width, 3))
print('Model loaded.')

# Custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Reshape((1, 1, 1024))(x)
x = Dropout(1e-3)(x)
x = Conv2D(2, (1, 1), padding='same')(x)
x = Activation('softmax')(x)
predictions = Reshape((2,))(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=0.001, decay=0.00004),
    metrics=['accuracy'])

# train the model on the new data for a few epochs
csv_logger = CSVLogger('./output/logs/training.csv', separator=';')

tensorboard = TensorBoard(
    log_dir='./output/logs/training',
    histogram_freq=1,
    write_graph=True,
    write_images=True)

model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples // batch_size,
    epochs=train_epochs,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size,
    verbose=1,
    callbacks=[csv_logger, tensorboard])

model.save_weights('./output/mobilenet_1_0_224_30_epochs.h5')

for layer in model.layers:
    layer.trainable = True

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(lr=0.0001, decay=0.00004),
    metrics=['accuracy'])

csv_logger = CSVLogger('./output/logs/fine_tuning.csv', separator=';')

checkpointer = ModelCheckpoint(
    filepath='./output/checkpoints/mobilenet_1_0_224_fine_tuned_epoch_{epoch:02d}_acc_{val_acc:.5f}.h5',
    monitor='val_acc',
    mode='max',
    verbose=1,
    save_best_only=True)

# early_stopper = EarlyStopping(patience=10)

tensorboard = TensorBoard(
    log_dir='./output/logs/fine_tuning',
    histogram_freq=1,
    write_graph=True,
    write_images=True)

model.fit_generator(
    train_generator,
    steps_per_epoch=train_samples // batch_size,
    epochs=fine_tune_epochs,
    validation_data=validation_generator,
    validation_steps=validation_samples // batch_size,
    verbose=1,
    callbacks=[csv_logger, checkpointer, tensorboard])

model.save_weights('./output/mobilenet_1_0_224_fine_tuned_90_epochs.h5')

# serialize model to JSON
model_json = model.to_json()
with open('./output/mobilenet_1_0_224_fine_tuned.json', 'w') as json_file:
    json_file.write(model_json)
