"""
Generador de dataset simulado para TP de IA - Transformadores.

Uso desde la raíz del proyecto:

python src/generar_dataset.py --input data/raw/transformadores_base.csv --output data/processed/transformadores_dataset_final.csv

El CSV de entrada puede tener:
- columnas lat/lon
- columnas latitud/longitud
- o una columna geojson con coordenadas tipo Point
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


RANDOM_SEED = 42


def leer_csv(path: Path) -> pd.DataFrame:
    opciones = [
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ",", "encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ";", "encoding": "latin1"},
        {"sep": "|", "encoding": "utf-8"},
        {"sep": "|", "encoding": "latin1"},
    ]

    for opcion in opciones:
        try:
            df = pd.read_csv(path, **opcion)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass

    raise ValueError(f"No se pudo leer el archivo CSV: {path}")


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("á", "a", regex=False)
        .str.replace("é", "e", regex=False)
        .str.replace("í", "i", regex=False)
        .str.replace("ó", "o", regex=False)
        .str.replace("ú", "u", regex=False)
    )
    return df


def detectar_columna(df: pd.DataFrame, candidatos: list[str]) -> Optional[str]:
    for col in candidatos:
        if col in df.columns:
            return col
    return None


def extraer_lat_lon_desde_geojson(valor: object) -> Tuple[Optional[float], Optional[float]]:
    if pd.isna(valor):
        return None, None

    try:
        geo = json.loads(str(valor))
    except json.JSONDecodeError:
        return None, None

    tipo = geo.get("type")
    coords = geo.get("coordinates")

    if not coords:
        return None, None

    try:
        if tipo == "Point":
            lon, lat = coords[0], coords[1]
            return float(lat), float(lon)

        if tipo == "LineString":
            puntos = np.array(coords, dtype=float)
            lon, lat = puntos.mean(axis=0)
            return float(lat), float(lon)

        if tipo == "MultiLineString":
            puntos = np.array([p for linea in coords for p in linea], dtype=float)
            lon, lat = puntos.mean(axis=0)
            return float(lat), float(lon)

    except Exception:
        return None, None

    return None, None


def asegurar_coordenadas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    lat_col = detectar_columna(df, ["lat", "latitud", "latitude", "y"])
    lon_col = detectar_columna(df, ["lon", "long", "lng", "longitud", "longitude", "x"])

    if lat_col and lon_col:
        df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
        df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")

    elif "geojson" in df.columns:
        coords = df["geojson"].apply(extraer_lat_lon_desde_geojson)
        df["lat"] = coords.apply(lambda x: x[0])
        df["lon"] = coords.apply(lambda x: x[1])

    else:
        raise ValueError(
            "El archivo debe tener columnas lat/lon, latitud/longitud o una columna geojson."
        )

    df = df.dropna(subset=["lat", "lon"]).copy()

    # Filtro amplio para quedarse con coordenadas razonables del AMBA.
    df = df[
        (df["lat"].between(-35.50, -34.30))
        & (df["lon"].between(-59.20, -57.80))
    ].copy()

    if df.empty:
        raise ValueError("No quedaron registros válidos después de filtrar coordenadas AMBA.")

    return df


def clasificar_region(lat: float, lon: float) -> str:
    # Aproximación simple para la demo.
    es_caba = (-34.72 <= lat <= -34.52) and (-58.55 <= lon <= -58.32)
    return "CABA" if es_caba else "Provincia"


def normalizar_serie(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    minimo = s.min()
    maximo = s.max()

    if pd.isna(minimo) or pd.isna(maximo) or maximo == minimo:
        return pd.Series(np.zeros(len(s)), index=s.index)

    return (s - minimo) / (maximo - minimo)


def crear_dataset_simulado(df: pd.DataFrame, n_max: Optional[int] = None) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    df = df.copy().reset_index(drop=True)

    if n_max is not None and len(df) > n_max:
        df = df.sample(n=n_max, random_state=RANDOM_SEED).reset_index(drop=True)

    n = len(df)

    if "transformador_id" not in df.columns:
        df.insert(0, "transformador_id", [f"TR-{i:05d}" for i in range(1, n + 1)])

    if "region" not in df.columns:
        df["region"] = [
            clasificar_region(lat, lon)
            for lat, lon in zip(df["lat"], df["lon"])
        ]

    densidad_col = detectar_columna(
        df,
        ["densidad", "densidad_poblacional", "population_density"]
    )

    if densidad_col:
        df["densidad_poblacional"] = pd.to_numeric(df[densidad_col], errors="coerce")
        df["densidad_poblacional"] = df["densidad_poblacional"].fillna(
            df["densidad_poblacional"].median()
        )
    else:
        densidad_caba = rng.normal(loc=14500, scale=3500, size=n)
        densidad_provincia = rng.normal(loc=7200, scale=3000, size=n)

        df["densidad_poblacional"] = np.where(
            df["region"].eq("CABA"),
            densidad_caba,
            densidad_provincia
        )

        df["densidad_poblacional"] = (
            np.clip(df["densidad_poblacional"], 500, 25000)
            .round(0)
        )

    temp_col = detectar_columna(df, ["temperatura_actual", "temperatura", "temp"])

    if temp_col:
        df["temperatura_actual"] = pd.to_numeric(df[temp_col], errors="coerce")
        df["temperatura_actual"] = df["temperatura_actual"].fillna(
            df["temperatura_actual"].median()
        )
    else:
        base_temp = rng.normal(loc=33.5, scale=3.2, size=n)
        ajuste_region = np.where(df["region"].eq("CABA"), 1.0, 0.2)

        df["temperatura_actual"] = (
            np.clip(base_temp + ajuste_region, 24, 43)
            .round(1)
        )

    df["dias_desde_mantenimiento"] = np.where(
        df["region"].eq("CABA"),
        rng.gamma(shape=4.0, scale=45.0, size=n),
        rng.gamma(shape=4.8, scale=55.0, size=n),
    )

    df["dias_desde_mantenimiento"] = (
        np.clip(df["dias_desde_mantenimiento"], 5, 720)
        .round(0)
        .astype(int)
    )

    df["antiguedad_anios"] = (
        np.clip(rng.normal(loc=14, scale=7, size=n), 1, 38)
        .round(0)
        .astype(int)
    )

    capacidades = np.array([160, 250, 315, 400, 500, 630, 800, 1000])
    probabilidades = np.array([0.08, 0.14, 0.18, 0.22, 0.14, 0.12, 0.08, 0.04])

    df["capacidad_kva"] = rng.choice(
        capacidades,
        size=n,
        p=probabilidades
    )

    densidad_normalizada = normalizar_serie(df["densidad_poblacional"])

    df["hogares_estimados"] = (
        35
        + densidad_normalizada * 260
        + rng.normal(0, 25, size=n)
    ).round(0)

    df["hogares_estimados"] = (
        np.clip(df["hogares_estimados"], 10, 420)
        .astype(int)
    )

    factor_calor = np.maximum(df["temperatura_actual"] - 30, 0) * 0.035

    consumo_hogar = (
        rng.normal(loc=1.15, scale=0.18, size=n)
        + factor_calor
    )

    consumo_hogar = np.clip(consumo_hogar, 0.65, 2.20)

    df["carga_estimada_kw"] = (
        df["hogares_estimados"]
        * consumo_hogar
        * rng.normal(loc=1.15, scale=0.12, size=n)
    ).round(1)

    df["porcentaje_carga"] = (
        df["carga_estimada_kw"]
        / (df["capacidad_kva"] * 0.92)
    ).round(3)

    lambda_fallas = (
        0.10
        + normalizar_serie(df["porcentaje_carga"]) * 1.10
        + normalizar_serie(df["temperatura_actual"]) * 0.45
        + normalizar_serie(df["dias_desde_mantenimiento"]) * 0.45
    )

    df["fallas_ultimos_12_meses"] = rng.poisson(lam=lambda_fallas)
    df["fallas_ultimos_12_meses"] = df["fallas_ultimos_12_meses"].clip(0, 8)

    score = (
        0.35 * normalizar_serie(df["porcentaje_carga"])
        + 0.25 * normalizar_serie(df["temperatura_actual"])
        + 0.15 * normalizar_serie(df["dias_desde_mantenimiento"])
        + 0.15 * normalizar_serie(df["fallas_ultimos_12_meses"])
        + 0.10 * normalizar_serie(df["antiguedad_anios"])
    )

    score = np.clip(score + rng.normal(0, 0.035, size=n), 0, 1)

    df["riesgo_score_simulado"] = score.round(3)

    condicion_alto = (
        (df["riesgo_score_simulado"] >= 0.52)
        | (
            (df["porcentaje_carga"] >= 0.95)
            & (df["temperatura_actual"] >= 36)
        )
        | (df["fallas_ultimos_12_meses"] >= 3)
    )

    condicion_medio = (
        (df["riesgo_score_simulado"] >= 0.32)
        | (
            (df["porcentaje_carga"] >= 0.75)
            & (df["temperatura_actual"] >= 34)
        )
        | (df["dias_desde_mantenimiento"] >= 300)
    )

    df["riesgo_clase"] = np.select(
        [condicion_alto, condicion_medio],
        ["Alto", "Medio"],
        default="Bajo"
    )

    df["requiere_revision"] = df["riesgo_clase"].isin(["Medio", "Alto"]).astype(int)

    columnas_finales = [
        "transformador_id",
        "lat",
        "lon",
        "region",
        "densidad_poblacional",
        "temperatura_actual",
        "dias_desde_mantenimiento",
        "antiguedad_anios",
        "capacidad_kva",
        "hogares_estimados",
        "carga_estimada_kw",
        "porcentaje_carga",
        "fallas_ultimos_12_meses",
        "riesgo_score_simulado",
        "riesgo_clase",
        "requiere_revision",
    ]

    return df[columnas_finales].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera dataset simulado de transformadores para el TP de IA."
    )

    parser.add_argument(
        "--input",
        default="data/raw/transformadores_base.csv",
        help="Ruta del CSV base."
    )

    parser.add_argument(
        "--output",
        default="data/processed/transformadores_dataset_final.csv",
        help="Ruta de salida del dataset final."
    )

    parser.add_argument(
        "--n-max",
        type=int,
        default=1000,
        help="Cantidad máxima de registros para la demo. Usar 0 para no limitar."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"No existe el archivo de entrada: {input_path}. "
            "Copialo en data/raw/ o indicá otra ruta con --input."
        )

    df_base = leer_csv(input_path)
    df_base = normalizar_columnas(df_base)
    df_base = asegurar_coordenadas(df_base)

    n_max = None if args.n_max == 0 else args.n_max

    df_final = crear_dataset_simulado(df_base, n_max=n_max)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Dataset generado correctamente")
    print(f"Archivo de entrada: {input_path}")
    print(f"Archivo de salida:  {output_path}")
    print(f"Registros: {len(df_final)}")

    print("\nDistribución de riesgo:")
    print(df_final["riesgo_clase"].value_counts().to_string())

    print("\nPrimeras filas:")
    print(df_final.head(5).to_string(index=False))


if __name__ == "__main__":
    main()