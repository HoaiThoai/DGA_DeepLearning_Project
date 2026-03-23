import logging
import os
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping

from src.model import build_model

logger = logging.getLogger(__name__)


def _model_builder(config: dict):
    """Return a closure compatible with KerasTuner's ``build_model``
    interface, injecting architecture hyper-parameters from *config*.
    """
    model_cfg = config["model"]
    tuner_cfg = config["training"]["tuner"]

    def _build(hp):
        return build_model(
            hp=hp,
            vocab_size=model_cfg["vocab_size"],
            max_seq_len=model_cfg["max_sequence_length"],
            embedding_dim=model_cfg["embedding_dim"],
            cnn_filters=model_cfg["cnn_filters"],
            cnn_kernel_size=model_cfg["cnn_kernel_size"],
            lstm_units=model_cfg["lstm_units"],
            dense_units=model_cfg["dense_units"],
            use_bidirectional=model_cfg.get("use_bidirectional", True),
            default_dropout=model_cfg.get("dropout_rate", 0.3),
            default_lr=config["training"].get("learning_rate", 0.001),
            lr_min=tuner_cfg["learning_rate_min"],
            lr_max=tuner_cfg["learning_rate_max"],
            dropout_min=tuner_cfg["dropout_min"],
            dropout_max=tuner_cfg["dropout_max"],
        )
    return _build


def train(config: dict,
          X_train, y_train,
          X_val, y_val) -> "keras.Model":
    """Run hyperparameter search, train the best model, and save it.

    Parameters
    ----------
    config : dict
        Full configuration dictionary (parsed from YAML).
    X_train, y_train : np.ndarray
        Training features and labels.
    X_val, y_val : np.ndarray
        Validation features and labels.

    Returns
    -------
    tensorflow.keras.Model
        The best trained model (already saved to disk).
    """
    train_cfg = config["training"]
    tuner_cfg = train_cfg["tuner"]
    eval_cfg = config["evaluation"]

    # Ensure output directory exists ------------------------------------------
    os.makedirs(eval_cfg["output_dir"], exist_ok=True)

    # ---- 1. KerasTuner: Hyperparameter Search --------------------------------
    logger.info("=" * 60)
    logger.info("Starting KerasTuner RandomSearch …")
    logger.info("=" * 60)

    tuner = kt.RandomSearch(
        hypermodel=_model_builder(config),
        objective="val_accuracy",
        max_trials=tuner_cfg["max_trials"],
        executions_per_trial=tuner_cfg["executions_per_trial"],
        directory=tuner_cfg["directory"],
        project_name=tuner_cfg["project_name"],
        overwrite=True,
    )

    # Early Stopping callback --------------------------------------------------
    es_cfg = train_cfg["early_stopping"]
    early_stop = EarlyStopping(
        monitor=es_cfg["monitor"],
        patience=es_cfg["patience"],
        restore_best_weights=es_cfg["restore_best_weights"],
        verbose=1,
    )

    # Run the search -----------------------------------------------------------
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        callbacks=[early_stop],
        verbose=1,
    )

    # ---- 2. Retrieve the best hyperparameters --------------------------------
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    logger.info("Best hyperparameters found:")
    logger.info("  learning_rate = %.6f", best_hp.get("learning_rate"))
    logger.info("  dropout_rate  = %.2f", best_hp.get("dropout_rate"))

    # ---- 3. Train the best model fully ---------------------------------------
    logger.info("Training the best model on the full training set …")
    best_model = tuner.hypermodel.build(best_hp)

    best_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        callbacks=[early_stop],
        verbose=1,
    )

    # ---- 4. Save the model ---------------------------------------------------
    save_path = eval_cfg["model_save_path"]
    best_model.save(save_path)
    logger.info("Best model saved to %s", save_path)

    # Print model summary for reference ----------------------------------------
    best_model.summary(print_fn=logger.info)

    return best_model
