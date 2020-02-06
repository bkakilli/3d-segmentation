import numpy as np
from sklearn import metrics as sk_metrics


def get_segmentation_metrics(labels, preds):

    confusion_matrix = sk_metrics.confusion_matrix(labels, preds)
    confusion_matrix = confusion_matrix.astype(np.float32)
    num_classes = len(confusion_matrix)
    matrix_diagonal = confusion_matrix.diag()

    row_sum = confusion_matrix.sum(axis=1)
    col_sum = confusion_matrix.sum(axis=0)
    all_sum = row_sum.sum()
    re = np.sum(matrix_diagonal / np.maximum(row_sum, 1))

    errors_summed_by_row = row_sum - matrix_diagonal
    errors_summed_by_column = col_sum - matrix_diagonal
    divisor = errors_summed_by_row + errors_summed_by_column + matrix_diagonal
    divisor[matrix_diagonal == 0] = 1

    class_seen = ((errors_summed_by_row+errors_summed_by_column) != 0).sum()

    metrics = {}
    metrics["overall_accuracy"] = matrix_diagonal.sum() / all_sum
    metrics["mean_class_accuracy"] = re / num_classes
    metrics["iou_per_class"] = matrix_diagonal / divisor
    metrics["iou_average"] = np.sum(metrics["iou_per_class"]) / class_seen

    return metrics

