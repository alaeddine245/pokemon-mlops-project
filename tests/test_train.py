import pytest
import tensorflow as tf
from unittest.mock import Mock, patch

from src.training.train import train


@pytest.fixture
def model():
    # create a mock model object
    model = Mock(spec=tf.keras.Model)
    return model


@pytest.fixture
def train_ds():
    # create a mock training dataset
    train_ds = Mock(spec=tf.data.Dataset)
    return train_ds


@pytest.fixture
def val_ds():
    # create a mock validation dataset
    val_ds = Mock(spec=tf.data.Dataset)
    return val_ds

def test_train_calls_fit(model, train_ds, val_ds):
    # call the train function
    train(model, train_ds, val_ds)

    # assert that the model's fit method was called with the correct arguments
    model.fit.assert_called_with(train_ds, validation_data=val_ds, epochs=6)

