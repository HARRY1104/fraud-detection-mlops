"""Financial Loss Matrix: Computes threshold that minimizes total monetary loss."""
import numpy as np

def calculate_optimal_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    tx_amounts: np.ndarray,
    chargeback_fee: float = 25.0,
    false_positive_friction_cost: float = 30.0,
) -> dict:
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.50
    min_total_loss = float("inf")
    loss_curve = []

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        
        # False Negatives: Fraud approved -> Loss = Stolen Amount + Chargeback Processing Fee
        fn_mask = (y_true == 1) & (preds == 0)
        fn_loss = np.sum(tx_amounts[fn_mask] + chargeback_fee)
        
        # False Positives: Good user blocked -> Loss = Customer Friction / Churn Risk Cost
        fp_mask = (y_true == 0) & (preds == 1)
        fp_loss = np.sum(fp_mask) * false_positive_friction_cost
        
        total_loss = fn_loss + fp_loss
        loss_curve.append({"threshold": round(t, 2), "loss": round(total_loss, 2)})
        
        if total_loss < min_total_loss:
            min_total_loss = total_loss
            best_threshold = t

    return {
        "optimal_threshold": round(best_threshold, 2),
        "minimized_financial_loss": round(min_total_loss, 2),
        "loss_curve": loss_curve,
    }