import sys
from keras import applications
from keras import optimizers
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
import time

time1 = time.time()

test_directory = './data/test'

img_width, img_height = 299, 299
batch_size = 1
test_samples = 20

datagen = image.ImageDataGenerator()

test_generator = datagen.flow_from_directory(
    test_directory,
    target_size=(img_height, img_width),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False)

time2 = time.time()

# base_model = InceptionV3(weights='imagenet',
#                          include_top=False,
#                          input_shape=(img_height, img_width, 3))
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
# predictions = Dense(2, activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=predictions)
#
# model.load_weights('./RESULTS/seeds_split_11082017/inceptionV3_fine_tuned_60_epochs.h5')
#
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer=optimizers.RMSprop(lr=0.0001, decay=0.00004),
#     metrics=['accuracy'])

model = load_model('./output/checkpoints/mobilenet_1_0_224_fine_tuned_epoch_26_acc_0.99250.h5', custom_objects={
                   'relu6': applications.mobilenet.relu6,
                   'DepthwiseConv2D': applications.mobilenet.DepthwiseConv2D})

time3 = time.time()

score = model.evaluate_generator(test_generator, test_samples // batch_size)
time4 = time.time()
print "Test fraction correct (Accuracy) = {:.2f}".format(score[1])

prediction = model.predict_generator(test_generator, test_samples // batch_size)
time5 = time.time()
print prediction

# img_path = sys.argv[1]
# img = image.load_img(img_path, target_size=(img_width, img_height))
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = applications.mobilenet.MobileNet.preprocess_input(x)
#
# preds = model.predict(x, batch_size=1)
#
# print preds
# print preds.flatten()
# print np.argmax(preds.flatten())

print 'Init time = %0.3f ms' % ((time2-time1)*1000.0)
print 'Model prepare time = %0.3f ms' % ((time3-time2)*1000.0)
print 'Evaluation time = %0.3f ms' % ((time4-time3)*1000.0)
print 'Prediction time = %0.3f ms' % ((time5-time4)*1000.0)
