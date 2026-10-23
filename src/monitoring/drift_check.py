from pathlib import Path
import numpy as np
from scipy.stats import ks_2samp
from src.data.make_dataset import load_or_create_dataset

ROOT_DIR = Path(__file__).resolve().parents[2]
DRIFT_REPORT_PATH = ROOT_DIR / "outputs" / "drift_report.html"

MONITORED_FEATURES = [
    "transaction_amount",
    "transaction_hour",
    "distance_from_home",
    "ratio_to_median_purchase_price",
]


def run_drift_check(alpha: float = 0.05, return_data: bool = False):
    """Compares reference training distribution against shifted production traffic using KS-test."""
    reference_df = load_or_create_dataset()
    current_df = reference_df.sample(n=min(1500, len(reference_df)), random_state=99).copy()

    # Simulate production drift (e.g., inflation in transaction amounts and distance)
    rng = np.random.default_rng(99)
    current_df["transaction_amount"] = current_df["transaction_amount"] * rng.uniform(1.35, 1.80, len(current_df))
    current_df["distance_from_home"] = current_df["distance_from_home"] + rng.exponential(15.0, len(current_df))

    results = {}
    rows_html = ""
    for col in MONITORED_FEATURES:
        stat, p_val = ks_2samp(reference_df[col], current_df[col])
        drifted = bool(p_val < alpha)
        results[col] = {
            "ks_stat": round(float(stat), 4),
            "p_value": round(float(p_val), 6),
            "drift_detected": drifted,
            "ref_mean": round(float(reference_df[col].mean()), 2),
            "current_mean": round(float(current_df[col].mean()), 2),
        }
        badge = "<span style='color:#ef4444;font-weight:bold;'>DRIFT DETECTED</span>" if drifted else "<span style='color:#10b981;font-weight:bold;'>STABLE</span>"
        rows_html += f"<tr><td>{col}</td><td>{reference_df[col].mean():.2f}</td><td>{current_df[col].mean():.2f}</td><td>{stat:.4f}</td><td>{p_val:.6f}</td><td>{badge}</td></tr>"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Data Drift Report</title>
    <style>
        body {{ font-family: sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; background: #1e293b; }}
        th, td {{ padding: 0.75rem 1rem; border: 1px solid #334155; text-align: left; }}
        th {{ background: #0f172a; }}
    </style>
</head>
<body>
    <h2>Production Data Drift Report (Kolmogorov-Smirnov Test)</h2>
    <table>
        <thead>
            <tr><th>Feature</th><th>Ref Mean</th><th>Current Mean</th><th>KS Stat</th><th>p-value</th><th>Status</th></tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
</body>
</html>"""

    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRIFT_REPORT_PATH.write_text(html_content, encoding="utf-8")
    print(f"Drift report generated at {DRIFT_REPORT_PATH}")
    if return_data:
        return results, reference_df, current_df
    return results


if __name__ == "__main__":
    run_drift_check()
