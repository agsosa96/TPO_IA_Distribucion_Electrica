"""
Entrenamiento del modelo de riesgo para transformadores.

Modelo: Random Forest Classifier
Objetivo: clasificar transformadores en riesgo Bajo, Medio o Alto.

Uso desde la raíz del proyecto:

python src/entrenar_modelo.py

Entrada esperada:
data/processed/transformadores_dataset_final.csv

Salidas:
models/modelo_riesgo_transformadores.joblib
outputs/metrics/reporte_clasificacion.txt
outputs/metrics/matriz_confusion.csv
outputs/metrics/importancia_variables.csv
data/processed/transformadores_con_prediccion.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "riesgo_clase"

NUMERIC_FEATURES = [
    "lat",
    "lon",
    "densidad_poblacional",
    "temperatura_actual",
    "dias_desde_mantenimiento",
    "antiguedad_anios",
    "capacidad_kva",
    "hogares_estimados",
    "carga_estimada_kw",
    "porcentaje_carga",
    "fallas_ultimos_12_meses",
]

CATEGORICAL_FEATURES = [
    "region",
]

CLASS_ORDER = ["Bajo", "Medio", "Alto"]


def validar_columnas(df: pd.DataFrame) -> None:
    columnas_necesarias = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    faltantes = [col for col in columnas_necesarias if col not in df.columns]

    if faltantes:
        raise ValueError(
            "Faltan columnas necesarias en el dataset: "
            + ", ".join(faltantes)
        )


def preparar_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {path}")

    df = pd.read_csv(path)
    validar_columnas(df)

    df = df.copy()

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET] = df[TARGET].astype(str).str.strip()

    clases_validas = set(CLASS_ORDER)
    clases_encontradas = set(df[TARGET].unique())

    clases_invalidas = clases_encontradas - clases_validas
    if clases_invalidas:
        raise ValueError(
            "Se encontraron clases de riesgo inválidas: "
            + ", ".join(clases_invalidas)
        )

    df = df.dropna(subset=[TARGET]).copy()

    if df.empty:
        raise ValueError("El dataset quedó vacío luego de validar la variable objetivo.")

    return df


def construir_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def obtener_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]

    numeric_names = NUMERIC_FEATURES

    encoder = (
        preprocessor
        .named_transformers_["cat"]
        .named_steps["encoder"]
    )

    categorical_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()

    return numeric_names + categorical_names


def guardar_metricas(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = pipeline.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    reporte = classification_report(
        y_test,
        y_pred,
        labels=CLASS_ORDER,
        zero_division=0,
    )

    matriz = confusion_matrix(
        y_test,
        y_pred,
        labels=CLASS_ORDER,
    )

    reporte_texto = (
        "REPORTE DE CLASIFICACIÓN - MODELO DE RIESGO\n"
        "================================================\n\n"
        f"Accuracy: {accuracy:.4f}\n\n"
        "Clases:\n"
        "- Bajo\n"
        "- Medio\n"
        "- Alto\n\n"
        "Reporte:\n"
        f"{reporte}\n"
    )

    (output_dir / "reporte_clasificacion.txt").write_text(
        reporte_texto,
        encoding="utf-8",
    )

    matriz_df = pd.DataFrame(
        matriz,
        index=[f"Real_{c}" for c in CLASS_ORDER],
        columns=[f"Predicho_{c}" for c in CLASS_ORDER],
    )

    matriz_df.to_csv(
        output_dir / "matriz_confusion.csv",
        index=True,
        encoding="utf-8-sig",
    )

    feature_names = obtener_feature_names(pipeline)
    importances = pipeline.named_steps["model"].feature_importances_

    importancia_df = pd.DataFrame(
        {
            "variable": feature_names,
            "importancia": importances,
        }
    ).sort_values("importancia", ascending=False)

    importancia_df.to_csv(
        output_dir / "importancia_variables.csv",
        index=False,
        encoding="utf-8-sig",
    )


def guardar_predicciones(
    pipeline: Pipeline,
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    predicciones = pipeline.predict(x)
    probabilidades = pipeline.predict_proba(x)

    clases_modelo = pipeline.named_steps["model"].classes_

    df_salida = df.copy()
    df_salida["riesgo_predicho"] = predicciones

    for i, clase in enumerate(clases_modelo):
        df_salida[f"probabilidad_{clase.lower()}"] = probabilidades[:, i].round(4)

    df_salida["probabilidad_maxima"] = probabilidades.max(axis=1).round(4)

    df_salida.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena un modelo Random Forest para clasificar riesgo de transformadores."
    )

    parser.add_argument(
        "--input",
        default="data/processed/transformadores_dataset_final.csv",
        help="Ruta del dataset final generado.",
    )

    parser.add_argument(
        "--model-output",
        default="models/modelo_riesgo_transformadores.joblib",
        help="Ruta donde se guardará el modelo entrenado.",
    )

    parser.add_argument(
        "--predictions-output",
        default="data/processed/transformadores_con_prediccion.csv",
        help="Ruta donde se guardará el dataset con predicciones.",
    )

    parser.add_argument(
        "--metrics-output",
        default="outputs/metrics",
        help="Carpeta donde se guardarán las métricas.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    model_output_path = Path(args.model_output)
    predictions_output_path = Path(args.predictions_output)
    metrics_output_dir = Path(args.metrics_output)

    df = preparar_dataset(input_path)

    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipeline = construir_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)

    artefacto = {
        "pipeline": pipeline,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "class_order": CLASS_ORDER,
        "accuracy_test": accuracy,
    }

    joblib.dump(artefacto, model_output_path)

    guardar_metricas(
        pipeline=pipeline,
        x_test=x_test,
        y_test=y_test,
        output_dir=metrics_output_dir,
    )

    guardar_predicciones(
        pipeline=pipeline,
        df=df,
        output_path=predictions_output_path,
    )

    print("Modelo entrenado correctamente")
    print(f"Dataset usado: {input_path}")
    print(f"Registros totales: {len(df)}")
    print(f"Registros entrenamiento: {len(x_train)}")
    print(f"Registros prueba: {len(x_test)}")
    print(f"Accuracy test: {accuracy:.4f}")
    print(f"Modelo guardado en: {model_output_path}")
    print(f"Predicciones guardadas en: {predictions_output_path}")
    print(f"Métricas guardadas en: {metrics_output_dir}")

    print("\nDistribución real:")
    print(y.value_counts().to_string())

    print("\nDistribución predicha en test:")
    print(pd.Series(y_pred).value_counts().to_string())


if __name__ == "__main__":
    main()
