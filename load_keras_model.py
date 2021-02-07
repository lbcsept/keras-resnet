
import tensorflow as tf

model = tf.keras.models.load_model("./resnet18_cifar10_1000x3epochs.h5")
model.summary()


model = tf.keras.models.load_model("./resnet18_cifar10_1000x3epochs_custom_activation.h5")
model.summary()