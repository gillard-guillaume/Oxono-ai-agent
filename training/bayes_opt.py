import optuna
import tensorflow as tf
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from keras import backend as K
import gc


tf.random.set_seed(42)
np.random.seed(42)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from training.value_network import ValueNetwork

DEBUG = False

def objective(trial: optuna.trial.Trial) -> float:
    """
    Objective function for Bayesian hyperparameter optimization

    Args:
        trial (optuna.trial.Trial):
            Optuna trial object used to sample hyperparameters

    Returns:
        float:
            Validation loss (mean squared error) of the trained model
            Lower values indicate better performance
    """
    net = None

    try :
        K.clear_session() 
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        optimizer = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])
        huber_delta = trial.suggest_float("huber_delta", 0.3, 1.7)
        loss = tf.keras.losses.Huber(delta=huber_delta)
        conv_number = trial.suggest_int("conv_number", 1, 4)
        filters = trial.suggest_int("filters", 32, 128)
        kernel_size = (3, 3)
        dense_size = trial.suggest_int("dense_size", 32, 256)
        activation = "relu"
        if DEBUG:
            batch_size = trial.suggest_int("batch_size", 4, 16)
        else:
            batch_size = trial.suggest_int("batch_size", 32, 128)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=3, #number of epochs with no improvemnt
                min_delta=1e-4, #delta needed to be consider improvement
                restore_best_weights=True,
                verbose=0
            )
        ]

        net = ValueNetwork(
            lr=lr,
            optimizer=optimizer,
            loss=loss,
            conv_number=conv_number,
            filters=filters,
            kernel_size=kernel_size,
            dense_size=dense_size,
            activation=activation
        )

        history = net.train_supervised(
            X_train,
            y_train,
            sample_weight=w_train,
            batch_size=batch_size,
            epochs=20,
            callbacks=callbacks,
            validation_data=(X_val, y_val, w_val)
        )

        #val_loss = net.model.evaluate(X_val, y_val, sample_weight=w_val, verbose=0)
        val_loss = min(history.history["val_loss"])
        return val_loss

    except tf.errors.ResourceExhaustedError:
        return float("inf") 

    finally :
        tf.keras.backend.clear_session()
        if net is not None:
            del net
        gc.collect()

 




if __name__ == "__main__":
    """
    Run Bayesian optimization for ValueNetwork hyperparameters

    Usage:
        python bayes_opt.py --trials 20
        python bayes_opt.py --debug
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()


    data = np.load("datasets/ab5_vs_ab1_t6.0_g50000_20260417_2236.npz")

    X = data["X"]; y = data["y"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    w_train = np.abs(y_train) ** 2 + 0.1
    w_train = w_train / np.mean(w_train)

    w_val = np.abs(y_val) ** 2 + 0.1
    w_val = w_val / np.mean(w_val)

    DEBUG = args.debug
    if DEBUG:
        args.trials = 2
        X_train = X_train[:20]; y_train = y_train[:20]
        X_val = X_val[:5]; y_val = y_val[:5]
        w_train = w_train[:20]; w_val = w_val[:5]

    study_name = "value_net_bo_ab5_vs_ab1_50k_huber_delta"

    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///studies/{study_name}.db",
        study_name=study_name,
        load_if_exists=True
    )

    #study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    study.optimize(objective, timeout=5*3600, show_progress_bar=True)


    print("\n=== BEST RESULT ===")
    print("Best loss:", study.best_value)
    print("Best params:", study.best_params)
    print("\nTotal trials:", len(study.trials))