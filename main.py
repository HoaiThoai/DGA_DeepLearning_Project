import argparse
import ast
import logging
import sys
import yaml
from pathlib import Path

# Configure Root Logger --------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(name)-20s : %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")

# Import Modules ---------------------------------------------------------------
from src import preprocessing, train, evaluate


def load_config(cfg_path: str = "configs/config.yaml") -> dict:
    """Parse the YAML configuration file."""
    logger.info("Parsing configuration from %s …", cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def main(config_path: str):
    logger.info("=== DGA Detection Pipeline Started ===")
    config = load_config(config_path)
    
    # --------------------------------------------------------------------------
    # 1. Data Ingestion & Preprocessing
    # --------------------------------------------------------------------------
    logger.info("\n=== Phase 1: Data Ingestion & Preprocessing ===")
    
    data_cfg = config["data"]
    df = preprocessing.load_data(
        filepath=data_cfg["filepath"],
        domain_col=data_cfg["domain_column"],
        label_col=data_cfg["label_column"],
        positive_label=data_cfg["positive_label"],
    )
    
    # Character-level vectorisation (Option A)
    max_len = config["model"]["max_sequence_length"]
    X = preprocessing.vectorize_domains(df["domain"], max_len=max_len)
    y = df["label"].values
    
    # Data splitting
    splits = preprocessing.split_data(
        X, y,
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        test_ratio=data_cfg["test_ratio"],
        random_seed=data_cfg["random_seed"],
    )
    
    # Display split sizes
    logger.info("Split Sizes - Train: %d, Val: %d, Test: %d", 
                len(splits['X_train']), len(splits['X_val']), len(splits['X_test']))
    
    # Apply SMOTE to training data if enabled
    if data_cfg.get("smote", {}).get("enabled", False):
        rs = data_cfg["smote"].get("random_state", 42)
        X_train, y_train = preprocessing.apply_smote(
            splits["X_train"], splits["y_train"], random_state=rs
        )
    else:
        X_train, y_train = splits["X_train"], splits["y_train"]

    # --------------------------------------------------------------------------
    # 2. Training & Hyperparameter Tuning
    # --------------------------------------------------------------------------
    logger.info("\n=== Phase 2: Training & Hyperparameter Tuning ===")
    
    # Disable tf/keras logging clutter
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    # Train and get the best model
    best_model = train.train(
        config=config,
        X_train=X_train,
        y_train=y_train,
        X_val=splits["X_val"],
        y_val=splits["y_val"],
    )

    # --------------------------------------------------------------------------
    # 3. Evaluation
    # --------------------------------------------------------------------------
    logger.info("\n=== Phase 3: Evaluation ===")
    
    eval_cfg = config["evaluation"]
    X_test, y_test = splits["X_test"], splits["y_test"]
    
    # Evaluate model (classification report, confusion matrix)
    _, y_pred_probs = evaluate.evaluate_model(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        output_dir=eval_cfg["output_dir"],
        confusion_matrix_path=eval_cfg["confusion_matrix_path"],
    )
    
    # Plot ROC curve
    evaluate.plot_roc_curve(
        y_test=y_test,
        y_pred_probs=y_pred_probs,
        roc_curve_path=eval_cfg["roc_curve_path"],
    )

    logger.info("\n=== DGA Detection Pipeline Completed successfully ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DGA Detection Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Path to the config.yaml file")
    args = parser.parse_args()
    main(args.config)
