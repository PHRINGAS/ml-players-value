import os
import argparse

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(ROOT, "reports")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data drift report with Evidently")
    parser.add_argument("reference", help="Ruta al CSV de referencia (entrenamiento)")
    parser.add_argument("current", help="Ruta al CSV actual a comparar")
    parser.add_argument(
        "--output",
        default=os.path.join(REPORTS_DIR, "drift_report.html"),
        help="Ruta de salida del reporte HTML",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ref = pd.read_csv(args.reference)
    cur = pd.read_csv(args.current)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)
    report.save_html(args.output)

    print(f"Reporte de drift generado: {args.output}")


if __name__ == "__main__":
    main()
