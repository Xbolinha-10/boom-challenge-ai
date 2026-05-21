import pandas as pd
from pathlib import Path

from config import OUTPUTS_DIR


report_path = OUTPUTS_DIR / "technical_report.md"

validation_path = OUTPUTS_DIR / "validation_results.csv"
cv_path = OUTPUTS_DIR / "cross_validation_results.csv"
inverse_path = OUTPUTS_DIR / "inverse_design.csv"

report = []

report.append("# Technical Report - Boom Challenge AI Solution\n")

report.append("## Model Validation\n")

if validation_path.exists():
    validation = pd.read_csv(validation_path)
    report.append(validation.to_markdown(index=False))
    report.append("\n")

report.append("## Cross Validation\n")

if cv_path.exists():
    cv = pd.read_csv(cv_path)
    report.append(cv.to_markdown(index=False))
    report.append("\n")
    report.append(f"Mean MAE: {cv['mae'].mean():.4f}\n")
    report.append(f"Std MAE: {cv['mae'].std():.4f}\n")

report.append("## Best Model\n")
report.append("The best performing model is CatBoost optimized with Optuna.\n")

report.append("## Physics-Informed Features\n")
report.append("""
The model uses derived physical features such as:

- effective_energy
- fragmentation_index
- energy_log
- angle_sin
- angle_cos
""")

report.append("## Inverse Design\n")

if inverse_path.exists():
    inverse = pd.read_csv(inverse_path)
    report.append("Top optimized scenarios:\n")
    report.append(inverse.head(20).to_markdown(index=False))
    report.append("\n")

report.append("## Final Outputs\n")
report.append("""
Generated files:

- outputs/submission.csv
- outputs/inverse_design.csv
- outputs/validation_results.csv
- outputs/cross_validation_results.csv
- outputs/feature_importance.csv
- outputs/feature_importance.png
""")

OUTPUTS_DIR.mkdir(exist_ok=True)

report_path.write_text("\n".join(report), encoding="utf-8")

print("Relatório criado em:")
print(report_path)