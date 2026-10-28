import sys
from pathlib import Path

import joblib
import pandas as pd


def predict_secret_houses(input_csv: Path) -> None:
    root_dir = Path(__file__).resolve().parent.parent
    bundle_path = root_dir / "outputs" / "models" / "house_prices_bundle.joblib"

    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Bundle de modelos não encontrado em {bundle_path}. Execute src/generate_report.py primeiro."
        )

    bundle = joblib.load(bundle_path)
    processor = bundle["processor"]
    baseline_model = bundle["baseline_model"]
    competition_model = bundle["competition_model_full"]

    df_secret = pd.read_csv(input_csv)
    X_secret = processor.transform_new_data(df_secret)

    preds_comp = competition_model.predict(X_secret)
    preds_base = baseline_model.predict(X_secret)

    result_df = pd.DataFrame(
        {
            "Id": df_secret["Id"] if "Id" in df_secret.columns else range(1, len(df_secret) + 1),
            "SalePrice_Competicao_Otimizado": preds_comp.round(2),
            "SalePrice_Baseline_Linear": preds_base.round(2),
        }
    )

    out_csv = root_dir / "outputs" / "predicoes_5_imoveis_secretos.csv"
    result_df.to_csv(out_csv, index=False)

    print("\n=== PREDIÇÃO DOS IMÓVEIS SECRETOS (COMPETIÇÃO) ===")
    print(result_df.to_string(index=False))
    print(f"\nArquivo salvo em: {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: .venv/bin/python src/predict_competition.py <caminho_csv_5_imoveis.csv>")
        sys.exit(1)
    predict_secret_houses(Path(sys.argv[1]))
