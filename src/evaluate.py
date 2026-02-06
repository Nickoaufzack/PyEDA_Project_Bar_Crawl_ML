from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


def compute_metrics(y_test, y_pred):
    """
    Compute basic classification metrics. 
    """
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    return {"accuracy": acc,
            "report": report,
            "confusion_matrix": cm, }


def plot_confusion_matrix(cm, save_path):
    """
    Plot and save confusion matrix.
    """
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Sober", "Intoxicated"]
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_test, y_proba, save_path):
    """
    Plot ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_precision_recall_curve(y_test, y_proba, save_path):
    """
    Plot Precision–Recall curve.
    """
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_model(y_test, y_pred, y_proba, results_dir):
    """
    Run evaluation and generate plots
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(y_test, y_pred)

    # plot paths
    cm_path = results_dir / "confusion_matrix.png"
    roc_path = results_dir / "roc_curve.png"
    pr_path = results_dir / "precision_recall_curve.png"

    plot_confusion_matrix(metrics["confusion_matrix"], cm_path)
    plot_roc_curve(y_test, y_proba, roc_path)
    plot_precision_recall_curve(y_test, y_proba, pr_path)

    print("\nEvaluation complete")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
