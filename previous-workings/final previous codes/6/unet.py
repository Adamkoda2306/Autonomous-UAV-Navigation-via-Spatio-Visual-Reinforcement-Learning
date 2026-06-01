import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def build_unet(inputs):
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    p1 = tf.keras.layers.MaxPooling2D(2,2)(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    p2 = tf.keras.layers.MaxPooling2D(2,2)(x)

    u1 = tf.keras.layers.UpSampling2D(2)(p2)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u1)
    u2 = tf.keras.layers.UpSampling2D(2)(x)

    return tf.keras.layers.Conv2D(1,1,activation='sigmoid')(u2)