from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import tensorflow as tf
import matplotlib.pyplot as plt


def get_datasets():
    train_data = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0
    )
    train_generator = train_data.flow_from_directory("./data/train/", target_size=(64, 64), color_mode="grayscale",
                                                     batch_size=16, shuffle=True, class_mode="categorical")

    test_data = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0
    )
    test_generator = test_data.flow_from_directory("./data/validation/", target_size=(64, 64), color_mode="grayscale",
                                                   batch_size=16, shuffle=False, class_mode="categorical")

    train_number = train_generator.samples
    test_number = test_generator.samples

    return train_generator, test_generator, train_number, test_number


def get_model():
    model = tf.keras.Sequential([

        tf.keras.layers.ZeroPadding2D((1,1),input_shape=(64, 64, 1)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.ZeroPadding2D((1,1)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same",
                               activation=tf.keras.activations.relu),

        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same"),  
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.ZeroPadding2D((1,1)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.ZeroPadding2D((1,1)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same"),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.ZeroPadding2D((1,1)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.ZeroPadding2D((1,1)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.ZeroPadding2D((1,1)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same"),
        tf.keras.layers.BatchNormalization(),
        
        tf.keras.layers.ZeroPadding2D((1,1)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.ZeroPadding2D((1,1)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.ZeroPadding2D((1,1)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same",
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same"),
        tf.keras.layers.BatchNormalization(),
        


        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=256, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(units=5, activation=tf.keras.activations.softmax)
    ])

    return model


def model_evaluate():
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    train_generator, test_generator, train_number, test_number = get_datasets()
    model = get_model()
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                  metrics=['accuracy'])
    model.summary()

    history = model.fit_generator(train_generator, epochs=10, steps_per_epoch=train_number // 16,
                                  validation_data=test_generator, validation_steps=test_number // 16)
    model.save("./model/model.h5")

    model_evaluate()
