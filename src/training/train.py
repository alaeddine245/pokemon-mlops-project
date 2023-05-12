from models.model import model
from preprocessing.preprocess_split.py import train_ds,val_ds
import tensorflow as tf
import mlflow

# import argparse
# parser = argparse.ArgumentParser(
#                     prog='train.py',
#                     description='Train the model using specified arguments',
#                     )

# parser.add_argument('--epochs')          
# parser.add_argument('--learning_rate')      
# args = parser.parse_args()
# print(args)
def train(epochs=6,
          learning_rate=0.001,
          optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
          loss='categorical_crossentropy'
          ):
  with mlflow.start_run():
    mlflow.keras.log_model(model, "model")
   
    model.compile(optimizer, loss, metrics=[tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    history=model.fit(train_ds,validation_data=val_ds,epochs=6)
    mlflow.log_metric("train_history", history)
    mlflow.sklearn.log_model(model, "model")
    # mlflow.sklearn.save_model(model, modelpath)
    
