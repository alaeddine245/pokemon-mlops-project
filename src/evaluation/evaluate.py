import mlflow
import mlflow.keras
import tensorflow as tf


def extract_features_labels(example):

    x = example['image']
    y = example['label']
    return x, y

def evaluate_model(model: tf.keras.Model, val_ds: tf.data.Dataset):

    X_test, y_test = zip(*val_ds.map(extract_features_labels))

    with mlflow.start_run():
        # Log the model architecture
        mlflow.keras.log_model(model, "model")
        
        # Evaluate the model on the test set
        loss, accuracy, precision, recall, f1, auc = model.evaluate(X_test, y_test)
        print(f"Test loss: {loss:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Test precision: {precision:.4f}")
        print(f"Test recall: {recall:.4f}")
        print(f"Test f1: {f1:.4f}")
        print(f"Test auc: {auc:.4f}")
        
        # Log the evaluation metrics
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_auc", auc)