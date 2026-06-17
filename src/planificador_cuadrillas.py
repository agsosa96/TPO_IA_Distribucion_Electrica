"""
Planificador de cuadrillas para transformadores críticos.

Objetivo:
Tomar los transformadores con mayor prioridad operativa y asignarlos a cuadrillas
según cercanía geográfica, usando clustering + ordenamiento por distancia.

Uso desde la raíz del proyecto:

python src/planificador_cuadrillas.py

Entrada:
data/processed/transformadores_con_recomendaciones.csv

Salidas:
data/processed/plan_cuadrillas.csv
outputs/metrics/resumen_cuadrillas.csv
"""

from __future__ import annotations

import argparse
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


INPUT_DEFAULT = "data/processed/transformadores_con_recomendaciones.csv"
OUTPUT_DEFAULT = "data/processed/plan_cuadrillas.csv"
SUMMARY_DEFAULT = "outputs/metrics/resumen_cuadrillas.csv"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcula distancia aproximada entre dos puntos geográficos en kilómetros.
    """
    radio_tierra_km = 6371.0

    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2) ** 2
    )

    c = 2 * asin(sqrt(a))

    return radio_tierra_km * c


def validar_columnas(df: pd.DataFrame) -> None:
    columnas_necesarias = [
        "transformador_id",
        "lat",
        "lon",
        "prioridad_operativa",
        "accion_recomendada",
        "ventana_atencion_horas",
        "temperatura_actual",
        "porcentaje_carga",
        "dias_desde_mantenimiento",
    ]

    faltantes = [col for col in columnas_necesarias if col not in df.columns]

    if faltantes:
        raise ValueError("Faltan columnas necesarias: " + ", ".join(faltantes))

    if "riesgo_predicho" not in df.columns and "riesgo_clase" not in df.columns:
        raise ValueError("Debe existir la columna 'riesgo_predicho' o 'riesgo_clase'.")


def seleccionar_transformadores_criticos(
    df: pd.DataFrame,
    max_transformadores: int,
) -> pd.DataFrame:
    """
    Selecciona transformadores críticos para la planificación.
    Criterio principal: prioridad operativa 1 y 2.
    """
    df = df.copy()

    df["prioridad_operativa"] = pd.to_numeric(
        df["prioridad_operativa"],
        errors="coerce",
    )

    df["ventana_atencion_horas"] = pd.to_numeric(
        df["ventana_atencion_horas"],
        errors="coerce",
    )

    df["temperatura_actual"] = pd.to_numeric(
        df["temperatura_actual"],
        errors="coerce",
    )

    df["porcentaje_carga"] = pd.to_numeric(
        df["porcentaje_carga"],
        errors="coerce",
    )

    df["dias_desde_mantenimiento"] = pd.to_numeric(
        df["dias_desde_mantenimiento"],
        errors="coerce",
    )

    criticos = df[df["prioridad_operativa"].isin([1, 2])].copy()

    if criticos.empty:
        criticos = df.sort_values(
            by=["prioridad_operativa", "ventana_atencion_horas"],
            ascending=[True, True],
        ).head(max_transformadores).copy()
    else:
        criticos = criticos.sort_values(
            by=[
                "prioridad_operativa",
                "ventana_atencion_horas",
                "temperatura_actual",
                "porcentaje_carga",
            ],
            ascending=[True, True, False, False],
        ).head(max_transformadores).copy()

    criticos = criticos.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    if criticos.empty:
        raise ValueError("No hay transformadores críticos con coordenadas válidas.")

    return criticos


def asignar_cuadrillas(
    df_criticos: pd.DataFrame,
    cantidad_cuadrillas: int,
) -> pd.DataFrame:
    """
    Agrupa transformadores críticos por cercanía geográfica usando KMeans.
    """
    df = df_criticos.copy()

    cantidad_clusters = min(cantidad_cuadrillas, len(df))

    if cantidad_clusters <= 0:
        raise ValueError("La cantidad de cuadrillas debe ser mayor a cero.")

    if cantidad_clusters == 1:
        df["cuadrilla_id"] = "Cuadrilla 1"
        return df

    coordenadas = df[["lat", "lon"]].to_numpy()

    kmeans = KMeans(
        n_clusters=cantidad_clusters,
        random_state=42,
        n_init=10,
    )

    df["cluster"] = kmeans.fit_predict(coordenadas)

    # Ordenamos clusters por latitud/longitud para que la numeración sea estable.
    centroides = (
        df.groupby("cluster")[["lat", "lon"]]
        .mean()
        .reset_index()
        .sort_values(["lat", "lon"])
        .reset_index(drop=True)
    )

    mapa_cluster_cuadrilla = {
        int(row["cluster"]): f"Cuadrilla {i + 1}"
        for i, row in centroides.iterrows()
    }

    df["cuadrilla_id"] = df["cluster"].map(mapa_cluster_cuadrilla)

    df = df.drop(columns=["cluster"])

    return df


def ordenar_ruta_cuadrilla(df_cuadrilla: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena la ruta de una cuadrilla.
    Primero atiende mayor prioridad.
    Dentro de cada prioridad, usa una heurística greedy por cercanía.
    """
    pendientes = df_cuadrilla.copy().reset_index(drop=True)

    lat_actual = pendientes["lat"].mean()
    lon_actual = pendientes["lon"].mean()

    ruta = []
    orden = 1
    distancia_total = 0.0

    while not pendientes.empty:
        prioridad_minima = pendientes["prioridad_operativa"].min()
        candidatas = pendientes[
            pendientes["prioridad_operativa"] == prioridad_minima
        ].copy()

        candidatas["distancia_desde_actual_km"] = candidatas.apply(
            lambda row: haversine_km(
                lat_actual,
                lon_actual,
                float(row["lat"]),
                float(row["lon"]),
            ),
            axis=1,
        )

        idx_elegido = candidatas.sort_values(
            by=["distancia_desde_actual_km", "ventana_atencion_horas"],
            ascending=[True, True],
        ).index[0]

        elegido = pendientes.loc[idx_elegido].copy()
        distancia = haversine_km(
            lat_actual,
            lon_actual,
            float(elegido["lat"]),
            float(elegido["lon"]),
        )

        distancia_total += distancia

        elegido["orden_visita"] = orden
        elegido["distancia_desde_anterior_km"] = round(distancia, 2)
        elegido["distancia_acumulada_km"] = round(distancia_total, 2)

        ruta.append(elegido)

        lat_actual = float(elegido["lat"])
        lon_actual = float(elegido["lon"])

        pendientes = pendientes.drop(index=idx_elegido).reset_index(drop=True)
        orden += 1

    return pd.DataFrame(ruta)


def generar_plan(df_asignado: pd.DataFrame) -> pd.DataFrame:
    rutas = []

    for _, grupo in df_asignado.groupby("cuadrilla_id"):
        ruta = ordenar_ruta_cuadrilla(grupo)
        rutas.append(ruta)

    plan = pd.concat(rutas, ignore_index=True)

    riesgo_col = "riesgo_predicho" if "riesgo_predicho" in plan.columns else "riesgo_clase"

    columnas_salida = [
        "cuadrilla_id",
        "orden_visita",
        "transformador_id",
        "lat",
        "lon",
        riesgo_col,
        "accion_recomendada",
        "prioridad_operativa",
        "ventana_atencion_horas",
        "temperatura_actual",
        "porcentaje_carga",
        "dias_desde_mantenimiento",
        "distancia_desde_anterior_km",
        "distancia_acumulada_km",
    ]

    columnas_existentes = [col for col in columnas_salida if col in plan.columns]

    plan = plan[columnas_existentes].sort_values(
        by=["cuadrilla_id", "orden_visita"]
    )

    return plan


def guardar_resumen(plan: pd.DataFrame, summary_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    resumen = (
        plan.groupby("cuadrilla_id")
        .agg(
            cantidad_transformadores=("transformador_id", "count"),
            prioridad_minima=("prioridad_operativa", "min"),
            ventana_minima_horas=("ventana_atencion_horas", "min"),
            distancia_total_estimada_km=("distancia_desde_anterior_km", "sum"),
            temperatura_promedio=("temperatura_actual", "mean"),
            carga_promedio=("porcentaje_carga", "mean"),
        )
        .reset_index()
    )

    acciones = (
        plan.pivot_table(
            index="cuadrilla_id",
            columns="accion_recomendada",
            values="transformador_id",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )

    resumen = resumen.merge(acciones, on="cuadrilla_id", how="left")

    resumen["distancia_total_estimada_km"] = resumen[
        "distancia_total_estimada_km"
    ].round(2)

    resumen["temperatura_promedio"] = resumen["temperatura_promedio"].round(1)
    resumen["carga_promedio"] = resumen["carga_promedio"].round(3)

    resumen.to_csv(summary_path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Planifica cuadrillas para transformadores críticos."
    )

    parser.add_argument(
        "--input",
        default=INPUT_DEFAULT,
        help="Ruta del CSV con recomendaciones.",
    )

    parser.add_argument(
        "--output",
        default=OUTPUT_DEFAULT,
        help="Ruta de salida del plan de cuadrillas.",
    )

    parser.add_argument(
        "--summary-output",
        default=SUMMARY_DEFAULT,
        help="Ruta de salida del resumen de cuadrillas.",
    )

    parser.add_argument(
        "--cuadrillas",
        type=int,
        default=4,
        help="Cantidad de cuadrillas disponibles.",
    )

    parser.add_argument(
        "--max-transformadores",
        type=int,
        default=60,
        help="Cantidad máxima de transformadores críticos a planificar.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de entrada: {input_path}. "
            "Primero ejecutá: python src/agente_recomendador.py"
        )

    if args.cuadrillas <= 0:
        raise ValueError("La cantidad de cuadrillas debe ser mayor a cero.")

    if args.max_transformadores <= 0:
        raise ValueError("La cantidad máxima de transformadores debe ser mayor a cero.")

    df = pd.read_csv(input_path)
    validar_columnas(df)

    criticos = seleccionar_transformadores_criticos(
        df,
        max_transformadores=args.max_transformadores,
    )

    asignado = asignar_cuadrillas(
        criticos,
        cantidad_cuadrillas=args.cuadrillas,
    )

    plan = generar_plan(asignado)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output_path, index=False, encoding="utf-8-sig")

    guardar_resumen(plan, summary_path)

    print("Plan de cuadrillas generado correctamente")
    print(f"Archivo de entrada: {input_path}")
    print(f"Archivo de salida:  {output_path}")
    print(f"Resumen guardado:   {summary_path}")
    print(f"Cuadrillas usadas:  {args.cuadrillas}")
    print(f"Transformadores planificados: {len(plan)}")

    print("\nAsignación por cuadrilla:")
    print(plan["cuadrilla_id"].value_counts().sort_index().to_string())

    print("\nPrimeras visitas planificadas:")
    columnas_mostrar = [
        "cuadrilla_id",
        "orden_visita",
        "transformador_id",
        "accion_recomendada",
        "prioridad_operativa",
        "ventana_atencion_horas",
        "distancia_desde_anterior_km",
    ]

    print(plan[columnas_mostrar].head(15).to_string(index=False))


if __name__ == "__main__":
    main()