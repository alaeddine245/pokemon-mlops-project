import mlflow
import mlflow.keras
import tensorflow as tf

def evaluate_model(model: tf.keras.Model, X_test: list, y_test: list):
    with mlflow.start_run():
        # Log the model architecture
        mlflow.keras.log_model(model, "model")
        
        # Evaluate the model on the test set
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Test loss: {loss:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Log the evaluation metrics
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", accuracy)