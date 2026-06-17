"""
Agente recomendador para transformadores.

Objetivo:
Convertir la predicción de riesgo del modelo en una recomendación operativa
para mantenimiento preventivo.

Uso desde la raíz del proyecto:

python src/agente_recomendador.py

Entrada esperada:
data/processed/transformadores_con_prediccion.csv

Salida:
data/processed/transformadores_con_recomendaciones.csv
outputs/metrics/resumen_recomendaciones.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


INPUT_DEFAULT = "data/processed/transformadores_con_prediccion.csv"
OUTPUT_DEFAULT = "data/processed/transformadores_con_recomendaciones.csv"
SUMMARY_DEFAULT = "outputs/metrics/resumen_recomendaciones.csv"


def obtener_riesgo(row: pd.Series) -> str:
    """
    Usa riesgo_predicho si existe.
    Si no existe, usa riesgo_clase.
    """
    if "riesgo_predicho" in row.index and pd.notna(row["riesgo_predicho"]):
        return str(row["riesgo_predicho"]).strip()

    if "riesgo_clase" in row.index and pd.notna(row["riesgo_clase"]):
        return str(row["riesgo_clase"]).strip()

    return "Sin dato"


def recomendar_accion(row: pd.Series) -> dict:
    riesgo = obtener_riesgo(row)

    temperatura = float(row.get("temperatura_actual", 0))
    porcentaje_carga = float(row.get("porcentaje_carga", 0))
    dias_mantenimiento = int(row.get("dias_desde_mantenimiento", 0))
    fallas = int(row.get("fallas_ultimos_12_meses", 0))

    motivos = []

    if riesgo == "Alto":
        prioridad = 1
        accion = "Mantenimiento urgente"
        ventana_horas = 24
        tipo_intervencion = "Inspección térmica y revisión de carga"

        motivos.append("riesgo alto predicho por el modelo")

        if temperatura >= 36:
            motivos.append("temperatura elevada")
        if porcentaje_carga >= 0.90:
            motivos.append("carga cercana o superior al límite operativo")
        if dias_mantenimiento >= 300:
            motivos.append("muchos días desde el último mantenimiento")
        if fallas >= 2:
            motivos.append("historial reciente de fallas")

    elif riesgo == "Medio":
        if porcentaje_carga >= 0.85 or temperatura >= 35 or dias_mantenimiento >= 300:
            prioridad = 2
            accion = "Inspección prioritaria"
            ventana_horas = 72
            tipo_intervencion = "Revisión preventiva en campo"
        else:
            prioridad = 3
            accion = "Revisión programada"
            ventana_horas = 168
            tipo_intervencion = "Control preventivo"

        motivos.append("riesgo medio predicho por el modelo")

        if temperatura >= 35:
            motivos.append("temperatura moderadamente elevada")
        if porcentaje_carga >= 0.75:
            motivos.append("nivel de carga relevante")
        if dias_mantenimiento >= 240:
            motivos.append("mantenimiento próximo a vencer")
        if fallas >= 1:
            motivos.append("registro de fallas recientes")

    elif riesgo == "Bajo":
        prioridad = 4
        accion = "Monitoreo normal"
        ventana_horas = 720
        tipo_intervencion = "Seguimiento operativo"

        motivos.append("riesgo bajo predicho por el modelo")

        if temperatura >= 35 or porcentaje_carga >= 0.80:
            accion = "Monitoreo reforzado"
            prioridad = 3
            ventana_horas = 168
            motivos.append("variable operativa en observación")

    else:
        prioridad = 5
        accion = "Revisión manual"
        ventana_horas = 168
        tipo_intervencion = "Validación de datos"
        motivos.append("no se encontró una clasificación de riesgo válida")

    return {
        "prioridad_operativa": prioridad,
        "accion_recomendada": accion,
        "ventana_atencion_horas": ventana_horas,
        "tipo_intervencion": tipo_intervencion,
        "motivo_recomendacion": "; ".join(motivos),
    }


def validar_columnas(df: pd.DataFrame) -> None:
    columnas_minimas = [
        "temperatura_actual",
        "porcentaje_carga",
        "dias_desde_mantenimiento",
        "fallas_ultimos_12_meses",
    ]

    tiene_riesgo = "riesgo_predicho" in df.columns or "riesgo_clase" in df.columns

    if not tiene_riesgo:
        raise ValueError(
            "El dataset debe tener una columna 'riesgo_predicho' o 'riesgo_clase'."
        )

    faltantes = [col for col in columnas_minimas if col not in df.columns]

    if faltantes:
        raise ValueError(
            "Faltan columnas necesarias: " + ", ".join(faltantes)
        )


def aplicar_recomendaciones(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    recomendaciones = df.apply(recomendar_accion, axis=1)
    recomendaciones_df = pd.DataFrame(recomendaciones.tolist())

    df_resultado = pd.concat([df, recomendaciones_df], axis=1)

    return df_resultado


def guardar_resumen(df: pd.DataFrame, summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    resumen = (
        df.groupby(["accion_recomendada", "prioridad_operativa"])
        .size()
        .reset_index(name="cantidad_transformadores")
        .sort_values(["prioridad_operativa", "cantidad_transformadores"])
    )

    resumen.to_csv(summary_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera recomendaciones operativas para transformadores."
    )

    parser.add_argument(
        "--input",
        default=INPUT_DEFAULT,
        help="Ruta del CSV con predicciones del modelo.",
    )

    parser.add_argument(
        "--output",
        default=OUTPUT_DEFAULT,
        help="Ruta de salida del CSV con recomendaciones.",
    )

    parser.add_argument(
        "--summary-output",
        default=SUMMARY_DEFAULT,
        help="Ruta de salida del resumen de recomendaciones.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de entrada: {input_path}. "
            "Primero ejecutá: python src/entrenar_modelo.py"
        )

    df = pd.read_csv(input_path)
    validar_columnas(df)

    df_resultado = aplicar_recomendaciones(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_resultado.to_csv(output_path, index=False, encoding="utf-8-sig")

    guardar_resumen(df_resultado, summary_path)

    print("Recomendaciones generadas correctamente")
    print(f"Archivo de entrada: {input_path}")
    print(f"Archivo de salida:  {output_path}")
    print(f"Resumen guardado:   {summary_path}")
    print(f"Registros procesados: {len(df_resultado)}")

    print("\nResumen por acción:")
    print(
        df_resultado["accion_recomendada"]
        .value_counts()
        .to_string()
    )

    print("\nPrimeras recomendaciones:")
    columnas_mostrar = [
        "transformador_id",
        "riesgo_predicho" if "riesgo_predicho" in df_resultado.columns else "riesgo_clase",
        "temperatura_actual",
        "porcentaje_carga",
        "dias_desde_mantenimiento",
        "accion_recomendada",
        "prioridad_operativa",
        "ventana_atencion_horas",
    ]

    print(df_resultado[columnas_mostrar].head(10).to_string(index=False))


if __name__ == "__main__":
    main()