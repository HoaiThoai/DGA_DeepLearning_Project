import logging
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                    # non-interactive backend
import matplotlib.pyplot as plt          # noqa: E402
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

logger = logging.getLogger(__name__)


def evaluate_model(model,
                   X_test: np.ndarray,
                   y_test: np.ndarray,
                   output_dir: str = "models",
                   confusion_matrix_path: str = "models/confusion_matrix.png"):
    """Evaluate the model on the held-out test set.

    Prints a full classification report and saves a confusion-matrix
    heat-map to *confusion_matrix_path*.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Trained model.
    X_test, y_test : np.ndarray
        Test features and labels.
    output_dir : str
        Directory for saving artefacts.
    confusion_matrix_path : str
        File path for the confusion-matrix PNG.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(y_pred_classes, y_pred_probs)`` — predicted class labels and
        raw sigmoid probabilities.
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Evaluating model on the TEST set (%d samples) …", len(y_test))
    logger.info("=" * 60)

    # Predict probabilities and derive class labels ----------------------------
    y_pred_probs = model.predict(X_test, verbose=0).ravel()
    y_pred_classes = (y_pred_probs >= 0.5).astype(int)

    # Classification report (Precision, Recall, F1-Score) ----------------------
    report = classification_report(
        y_test, y_pred_classes,
        target_names=["Legit (0)", "DGA (1)"],
    )
    logger.info("\nClassification Report:\n%s", report)
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(report)

    # Confusion Matrix ---------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred_classes)
    logger.info("Confusion Matrix:\n%s", cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Legit", "DGA"])
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    ax.set_title("Confusion Matrix — DGA Detection")
    fig.tight_layout()
    fig.savefig(confusion_matrix_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", confusion_matrix_path)

    return y_pred_classes, y_pred_probs


def plot_roc_curve(y_test: np.ndarray,
                   y_pred_probs: np.ndarray,
                   roc_curve_path: str = "models/roc_curve.png"):
    """Plot and save the Receiver Operating Characteristic (ROC) curve.

    Parameters
    ----------
    y_test : np.ndarray
        True binary labels.
    y_pred_probs : np.ndarray
        Predicted probabilities from the sigmoid output.
    roc_curve_path : str
        File path for the saved ROC curve PNG.
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2563eb", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--",
            label="Random baseline")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — DGA Detection", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(roc_curve_path, dpi=150)
    plt.close(fig)

    logger.info("ROC/AUC curve saved to %s  (AUC = %.4f)", roc_curve_path, roc_auc)
    print(f"\nROC AUC Score: {roc_auc:.4f}")
    print(f"ROC curve saved to: {roc_curve_path}")
