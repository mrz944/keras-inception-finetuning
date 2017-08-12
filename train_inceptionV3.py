from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D

# based on
# https://www.kaggle.com/ogurtsov/0-99-with-r-and-keras-inception-v3-fine-tune/code

# Settings

train_directory = './data/train'
validation_directory = './data/validation'

img_width, img_height = 299, 299
batch_size = 16
train_epochs = 40
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

base_model = applications.InceptionV3(weights='imagenet',
                                      include_top=False,
                                      input_shape=(img_height, img_width, 3))
print('Model loaded.')

# Custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(2048, activation='relu')(x)
x = Dropout(0.8)(x)
predictions = Dense(2, activation='softmax')(x)
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

model.save_weights('./output/inceptionV3_40_epochs.h5')

for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.SGD(lr=0.0001,
                             momentum=0.9,
                             decay=0.00004),
    metrics=['accuracy'])

csv_logger = CSVLogger('./output/logs/fine_tuning.csv', separator=';')

checkpointer = ModelCheckpoint(
    filepath='./output/checkpoints/inceptionV3_fine_tuned_epoch_{epoch:02d}_acc_{val_acc:.5f}.h5',
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

model.save_weights('./output/inceptionV3_fine_tuned_60_epochs.h5')

# serialize model to JSON
model_json = model.to_json()
with open('./output/inceptionV3_fine_tuned.json', 'w') as json_file:
    json_file.write(model_json)
