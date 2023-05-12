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
def train(model, train_ds, val_ds, epochs=6):
  with mlflow.start_run():
    mlflow.keras.log_model(model, "model")
    history=model.fit(train_ds,validation_data=val_ds,epochs=epochs)
    mlflow.log_metric("train_history", history)
    mlflow.sklearn.log_model(model, "model")
    # mlflow.sklearn.save_model(model, modelpath)
    return model
    
