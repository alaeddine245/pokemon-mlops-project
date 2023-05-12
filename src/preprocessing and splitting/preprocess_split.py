import tensorflow as tf

img_height = 224
img_width = 224
batch_size=32
data_dir= '../../data/PokemonData/'


train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  seed=123,
  label_mode='categorical',
  shuffle=True,
  validation_split=0.2,
  subset="training",
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  seed=123,
  label_mode='categorical',
  shuffle=True,
  validation_split=0.2,
  subset="validation",
  image_size=(img_height, img_width),
  batch_size=batch_size)
