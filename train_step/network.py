import tensorflow as tf

class cnnlite():
    def __init__(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(input_shape=[32,32,3],filters= 3,kernel_size=5, strides=(1,1),activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3)))
        model.add(tf.keras.layers.Flatten(1))
        model.add(tf.keras.layers.Dense(1000))
        model.add(tf.keras.layers.ReLU())
        model.add(tf.keras.layers.Dense(1000,2))

        self.model = model


