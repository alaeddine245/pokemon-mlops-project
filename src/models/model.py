import tensorflow as tf
from keras.applications.efficientnet_v2 import EfficientNetV2B2
import keras

def transfer_learning(model, number_classes, learning_rate):
  for i in model.layers:
    i.trainable = False
  x=tf.keras.layers.Flatten()(model.output)
  pred=tf.keras.layers.Dense(number_classes, activation='softmax')(x)
  model=tf.keras.Model(inputs=model.input, outputs=pred)
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.Precision(), keras.metrics.Recall(), keras.metrics.F1Score, keras.metrics.AUC()])
  return model

model=EfficientNetV2B2(input_shape=(224,224,3),weights='imagenet',include_top=False)
model = transfer_learning(model, 151, 0.001)