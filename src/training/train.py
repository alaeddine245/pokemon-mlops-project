import tensorflow as tf
import mlflow

def train(model, train_ds, val_ds, epochs=6):
  with mlflow.start_run():
    mlflow.keras.log_model(model, "model")
    mlflow.keras.autolog()
    history=model.fit(train_ds,validation_data=val_ds,epochs=epochs)
    mlflow.sklearn.log_model(model, "model")
    # mlflow.sklearn.save_model(model, modelpath)
    return model, history
