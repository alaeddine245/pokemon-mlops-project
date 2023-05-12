from constants import *
from models import load_model
from preprocessing.preprocess_split import preprocess_split_data
from training.train import train
from evaluation.evaluate import evaluate_model
from visualization.plot_results import plot_results

import mlflow

if __name__ == '__main__':
    # Load the model
    model = load_model.load_model()

    # Preprocess the data
    train_ds, val_ds = preprocess_split_data(
        data_dir=DATA_DIR,
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT
    )

    # Train the model
    trained_model = train(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=EPOCHS
    )

    # Evaluate the model
    evaluate_model()

    # Visualize the results
    plot_results()