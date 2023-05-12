from constants import *
from models.model import load_model
from preprocessing.preprocess_split import preprocess_split_data
from training.train import train
from evaluation.evaluate import evaluate_model
from visualization.plot_results import plot_results
from mlflow_setup import setup_mlflow

if __name__ == '__main__':
    # Setup MLflow
    setup_mlflow(EXPERIMENT_NAME)

    # Load the model
    model = load_model(IMG_WIDTH, IMG_HEIGHT, NUMBER_CLASSES, CHANNELS, LEARNING_RATE)

    # Preprocess the data
    train_ds, val_ds = preprocess_split_data(
        data_dir=DATA_DIR,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT
    )

    # Train the model
    trained_model, history = train(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=EPOCHS
    )

    # Evaluate the model
    evaluate_model(model=trained_model, val_ds=val_ds)

    # Visualize the results
    plot_results(history=history, epochs=EPOCHS)
    