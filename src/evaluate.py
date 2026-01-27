import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def evaluate_model(y_true, y_pred, output_path):
    """
    Evaluate model predictions and generate a PDF report.
    """

    with PdfPages(output_path) as pdf:

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 10))
        ax.axis("off")

        text = (
            "Model Evaluation Report\n\n"
            f"Accuracy: {accuracy:.4f}\n\n"
            "Classification Report:\n"
            f"{report}"
        )

        ax.text(0.01, 0.99, text, va="top", ha="left", fontsize=10)
        pdf.savefig(fig)
        plt.close(fig)

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Sober", "Intoxicated"]
        )
        disp.plot(ax=ax, cmap="Blues", colorbar=False)

        ax.set_title("Confusion Matrix")
        pdf.savefig(fig)
        plt.close(fig)
