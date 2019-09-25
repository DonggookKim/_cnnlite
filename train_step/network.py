import tensorflow as tf

class cnnlite():
    def __init__(self):
        self.cnn_model = tf.keras.Sequential()
        # data_format : default = channel_last
        self.cnn_model.add(tf.keras.layers.Conv2D(input_shape=[32,32,3],filters=6, kernel_size=5, activation='relu'))
        self.cnn_model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=1))

        self.classifier_model = tf.keras.Sequential()
        self.classifier_model.add(tf.keras.layers.Dense(1000))
        self.classifier_model.add(tf.keras.layers.ReLU())
        self.classifier_model.add(tf.keras.layers.Dense(2))

    def forward(self, input_layer):
        print("CNN layer")
        c1 = self.cnn_model(input_layer)
        print(c1.shape)
        print("flatten layer")
        c2 = tf.reshape(c1, [-1, 26*26*6])
        print(c2.shape)
        print("classifier layer")
        c3 = self.classifier_model(c2)
        print(c3.shape)
        return c3

## Example of using keras layers
input_layer1 = tf.random_uniform([100, 32, 32, 3])
t1 = tf.keras.layers.Conv2D(input_shape=[32,32,3],filters=6, kernel_size=5, activation='relu')(input_layer1)
print(t1.shape)
t2 = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=1)(t1)
print(t2.shape)

## Example for use cnnlite
print("DEFINE START")
cnnL = cnnlite()
print("DEFINE END")

print("INPUT LAYER DEF")
input_layer = tf.random_uniform([100, 32, 32, 3])
print("INPUT LAYER DEF END")
print("CALCULATE INPUT LAYER")
print(cnnL.forward(input_layer))
