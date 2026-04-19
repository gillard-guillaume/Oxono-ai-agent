import argparse
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm
import optuna
import tensorflow as tf

from training.value_network import ValueNetwork

import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass


# systemd-inhibit python -m training.train_idefix --data datasets/ab5_vs_ab1_t6.0_g50000_20260417_2236.npz




def load_data(path):
    data = np.load(path)
    X = data["X"].astype(np.float32)
    y = np.clip(data["y"].astype(np.float32), -1, 1)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # === LOAD DATA ===
    X, y = load_data(args.data)

    split = int(0.9 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # === DEBUG MODE ===
    if args.debug:
        print("DEBUG MODE ON")
        X_train = X_train[:10]
        y_train = y_train[:10]
        X_val = X_val[:10]
        y_val = y_val[:10]
        args.epochs = 1
        batch_size = 4
    else:
        batch_size = 64

    # === SAMPLE WEIGHTS ===
    w_train = np.abs(y_train) ** 2 + 0.1
    w_train /= np.mean(w_train)

    w_val = np.abs(y_val) ** 2 + 0.1
    w_val /= np.mean(w_val)

    # === LOAD OPTUNA ===
    study_name = "value_net_bo_ab5_vs_ab1_50k_huber_delta"

    study = optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///studies/{study_name}.db"
    )

    p = study.best_params
    print("Best params:", p)

    # === MODEL ===
    loss = tf.keras.losses.Huber(delta=p["huber_delta"])

    filters = 32 if args.debug else 128

    model = ValueNetwork(
        lr=p["lr"],
        optimizer=p["optimizer"],
        loss=loss,
        conv_number=p["conv_number"],
        filters=filters,
        dense_size=p["dense_size"]
    )

    # === SAVE SETUP ===
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dataset_name = os.path.basename(args.data).replace(".npz", "")

    base_name = (
        f"idefix_cnn_{dataset_name}_"
        f"c{p['conv_number']}_f128_d{p['dense_size']}_"
        f"lr{p['lr']:.1e}_hd{p['huber_delta']:.2f}_"
        f"b{batch_size}_n{len(X)}_{timestamp}"
    )

    # === TRAIN LOOP ===
    best_val = float("inf")
    pbar = tqdm(range(args.epochs), desc="Training", unit="epoch")

    for epoch in pbar:
        hist = model.model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            batch_size=batch_size,
            epochs=1,
            validation_data=(X_val, y_val, w_val),
            verbose=0
        )

        train_loss = hist.history["loss"][0]
        val_loss = hist.history["val_loss"][0]

        pbar.set_postfix({
            "train": f"{train_loss:.4f}",
            "val": f"{val_loss:.4f}",
            "best": f"{best_val:.4f}"
        })

        # save best
        if val_loss < best_val:
            best_val = val_loss
            best_path = f"models/{base_name}_best.keras"
            model.model.save(best_path)

        # backup
        if (epoch + 1) % args.save_every == 0:
            path = f"models/{base_name}_ep{epoch+1}.keras"
            model.model.save(path)

    # final save
    final_path = f"models/{base_name}_final.keras"
    model.model.save(final_path)

    print(f"\nSaved final model: {final_path}")


if __name__ == "__main__":
    main()