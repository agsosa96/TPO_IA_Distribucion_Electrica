"""
Demo Streamlit - Sistema Inteligente de Riesgo de Transformadores

Ejecutar desde la raíz del proyecto:

streamlit run app.py

Archivos esperados:
data/processed/transformadores_con_recomendaciones.csv
data/processed/plan_cuadrillas.csv
outputs/metrics/resumen_cuadrillas.csv
outputs/metrics/resumen_recomendaciones.csv
"""

from __future__ import annotations

from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium


TRANSFORMADORES_PATH = Path("data/processed/transformadores_con_recomendaciones.csv")
PLAN_CUADRILLAS_PATH = Path("data/processed/plan_cuadrillas.csv")
RESUMEN_CUADRILLAS_PATH = Path("outputs/metrics/resumen_cuadrillas.csv")
RESUMEN_RECOMENDACIONES_PATH = Path("outputs/metrics/resumen_recomendaciones.csv")


st.set_page_config(
    page_title="IA - Riesgo de Transformadores",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data
def cargar_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def validar_archivos() -> None:
    archivos_requeridos = [
        TRANSFORMADORES_PATH,
        PLAN_CUADRILLAS_PATH,
    ]

    faltantes = [str(path) for path in archivos_requeridos if not path.exists()]

    if faltantes:
        st.error("Faltan archivos necesarios para ejecutar la demo.")
        st.write("Archivos faltantes:")
        for archivo in faltantes:
            st.code(archivo)

        st.write("Ejecutá en orden:")
        st.code(
            """
python src/generar_dataset.py
python src/entrenar_modelo.py
python src/agente_recomendador.py
python src/planificador_cuadrillas.py
streamlit run app.py
            """.strip()
        )
        st.stop()


def obtener_columna_riesgo(df: pd.DataFrame) -> str:
    if "riesgo_predicho" in df.columns:
        return "riesgo_predicho"

    if "riesgo_clase" in df.columns:
        return "riesgo_clase"

    raise ValueError("No existe columna de riesgo.")


def color_por_riesgo(riesgo: str) -> str:
    colores = {
        "Alto": "red",
        "Medio": "orange",
        "Bajo": "green",
    }

    return colores.get(str(riesgo), "gray")


def radio_por_prioridad(prioridad: int) -> int:
    if prioridad == 1:
        return 8

    if prioridad == 2:
        return 6

    if prioridad == 3:
        return 5

    return 4


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    columnas_numericas = [
        "lat",
        "lon",
        "temperatura_actual",
        "porcentaje_carga",
        "dias_desde_mantenimiento",
        "prioridad_operativa",
        "ventana_atencion_horas",
        "riesgo_score_simulado",
        "probabilidad_maxima",
    ]

    for col in columnas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["lat", "lon"]).copy()

    return df


def crear_mapa(df: pd.DataFrame, riesgo_col: str) -> folium.Map:
    centro_lat = df["lat"].mean()
    centro_lon = df["lon"].mean()

    mapa = folium.Map(
        location=[centro_lat, centro_lon],
        zoom_start=10,
        tiles="OpenStreetMap",
    )

    cluster = MarkerCluster(name="Transformadores").add_to(mapa)

    for _, row in df.iterrows():
        riesgo = row[riesgo_col]
        prioridad = int(row.get("prioridad_operativa", 4))

        popup_html = f"""
        <b>Transformador:</b> {row.get("transformador_id", "Sin ID")}<br>
        <b>Riesgo:</b> {riesgo}<br>
        <b>Acción:</b> {row.get("accion_recomendada", "Sin dato")}<br>
        <b>Prioridad:</b> {row.get("prioridad_operativa", "Sin dato")}<br>
        <b>Temperatura:</b> {row.get("temperatura_actual", "Sin dato")} °C<br>
        <b>Carga:</b> {row.get("porcentaje_carga", "Sin dato")}<br>
        <b>Días sin mantenimiento:</b> {row.get("dias_desde_mantenimiento", "Sin dato")}<br>
        <b>Motivo:</b> {row.get("motivo_recomendacion", "Sin dato")}
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radio_por_prioridad(prioridad),
            color=color_por_riesgo(riesgo),
            fill=True,
            fill_color=color_por_riesgo(riesgo),
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=350),
        ).add_to(cluster)

    folium.LayerControl().add_to(mapa)

    return mapa


def aplicar_filtros(df: pd.DataFrame, riesgo_col: str) -> pd.DataFrame:
    df_filtrado = df.copy()

    with st.sidebar:
        st.header("Filtros")

        if "region" in df_filtrado.columns:
            regiones = sorted(df_filtrado["region"].dropna().unique().tolist())
            regiones_sel = st.multiselect(
                "Región",
                options=regiones,
                default=regiones,
            )

            df_filtrado = df_filtrado[df_filtrado["region"].isin(regiones_sel)]

        riesgos = ["Bajo", "Medio", "Alto"]
        riesgos_presentes = [
            r for r in riesgos if r in df_filtrado[riesgo_col].dropna().unique()
        ]

        riesgos_sel = st.multiselect(
            "Riesgo",
            options=riesgos_presentes,
            default=riesgos_presentes,
        )

        df_filtrado = df_filtrado[df_filtrado[riesgo_col].isin(riesgos_sel)]

        if "accion_recomendada" in df_filtrado.columns:
            acciones = sorted(
                df_filtrado["accion_recomendada"].dropna().unique().tolist()
            )

            acciones_sel = st.multiselect(
                "Acción recomendada",
                options=acciones,
                default=acciones,
            )

            df_filtrado = df_filtrado[
                df_filtrado["accion_recomendada"].isin(acciones_sel)
            ]

        prioridad_max = int(df_filtrado["prioridad_operativa"].max()) if not df_filtrado.empty else 4

        prioridad_sel = st.slider(
            "Prioridad máxima a mostrar",
            min_value=1,
            max_value=5,
            value=min(4, prioridad_max),
        )

        df_filtrado = df_filtrado[
            df_filtrado["prioridad_operativa"] <= prioridad_sel
        ]

        max_puntos = st.slider(
            "Máximo de puntos en el mapa",
            min_value=50,
            max_value=1000,
            value=500,
            step=50,
        )

    if len(df_filtrado) > max_puntos:
        df_filtrado = df_filtrado.sample(
            n=max_puntos,
            random_state=42,
        )

    return df_filtrado


def mostrar_metricas(df: pd.DataFrame, plan: pd.DataFrame, riesgo_col: str) -> None:
    total = len(df)
    riesgo_alto = int((df[riesgo_col] == "Alto").sum())
    riesgo_medio = int((df[riesgo_col] == "Medio").sum())
    urgentes = int((df["accion_recomendada"] == "Mantenimiento urgente").sum())
    inspecciones = int((df["accion_recomendada"] == "Inspección prioritaria").sum())
    planificados = len(plan)
    cuadrillas = plan["cuadrilla_id"].nunique() if "cuadrilla_id" in plan.columns else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Transformadores analizados", f"{total:,}".replace(",", "."))
    col2.metric("Riesgo alto", riesgo_alto)
    col3.metric("Mantenimiento urgente", urgentes)
    col4.metric("Cuadrillas planificadas", cuadrillas)

    col5, col6, col7, col8 = st.columns(4)

    col5.metric("Riesgo medio", riesgo_medio)
    col6.metric("Inspecciones prioritarias", inspecciones)
    col7.metric("Transformadores en plan", planificados)
    col8.metric(
        "Temperatura promedio",
        f"{df['temperatura_actual'].mean():.1f} °C",
    )


def mostrar_graficos(df: pd.DataFrame, riesgo_col: str) -> None:
    col1, col2 = st.columns(2)

    with col1:
        conteo_riesgo = (
            df[riesgo_col]
            .value_counts()
            .rename_axis("riesgo")
            .reset_index(name="cantidad")
        )

        fig_riesgo = px.bar(
            conteo_riesgo,
            x="riesgo",
            y="cantidad",
            title="Distribución de riesgo predicho",
            text="cantidad",
        )

        st.plotly_chart(fig_riesgo, use_container_width=True)

    with col2:
        conteo_accion = (
            df["accion_recomendada"]
            .value_counts()
            .rename_axis("accion")
            .reset_index(name="cantidad")
        )

        fig_accion = px.bar(
            conteo_accion,
            x="accion",
            y="cantidad",
            title="Acciones recomendadas por el agente",
            text="cantidad",
        )

        fig_accion.update_layout(xaxis_tickangle=-25)

        st.plotly_chart(fig_accion, use_container_width=True)


def mostrar_top_criticos(df: pd.DataFrame, riesgo_col: str) -> None:
    columnas = [
        "transformador_id",
        riesgo_col,
        "accion_recomendada",
        "prioridad_operativa",
        "ventana_atencion_horas",
        "temperatura_actual",
        "porcentaje_carga",
        "dias_desde_mantenimiento",
        "fallas_ultimos_12_meses",
        "motivo_recomendacion",
    ]

    columnas = [col for col in columnas if col in df.columns]

    top = (
        df.sort_values(
            by=[
                "prioridad_operativa",
                "ventana_atencion_horas",
                "temperatura_actual",
                "porcentaje_carga",
            ],
            ascending=[True, True, False, False],
        )
        .head(10)
        .copy()
    )

    st.dataframe(
        top[columnas],
        use_container_width=True,
        hide_index=True,
    )


def mostrar_detalle_transformador(df: pd.DataFrame, riesgo_col: str) -> None:
    st.subheader("Consulta individual de transformador")

    ids = df["transformador_id"].dropna().astype(str).tolist()

    seleccionado = st.selectbox(
        "Seleccionar transformador",
        options=ids,
    )

    row = df[df["transformador_id"].astype(str) == seleccionado].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Riesgo", row[riesgo_col])
    col2.metric("Acción", row["accion_recomendada"])
    col3.metric("Prioridad", int(row["prioridad_operativa"]))

    st.write("**Motivo de recomendación:**")
    st.info(row.get("motivo_recomendacion", "Sin dato"))

    columnas_detalle = [
        "temperatura_actual",
        "porcentaje_carga",
        "dias_desde_mantenimiento",
        "antiguedad_anios",
        "capacidad_kva",
        "hogares_estimados",
        "fallas_ultimos_12_meses",
        "probabilidad_maxima",
    ]

    columnas_detalle = [col for col in columnas_detalle if col in df.columns]

    st.dataframe(
        pd.DataFrame(row[columnas_detalle]).rename(columns={row.name: "valor"}),
        use_container_width=True,
    )


def mostrar_plan_cuadrillas(plan: pd.DataFrame) -> None:
    st.subheader("Plan de cuadrillas")

    if plan.empty:
        st.warning("No hay plan de cuadrillas disponible.")
        return

    cuadrillas = sorted(plan["cuadrilla_id"].dropna().unique().tolist())

    cuadrilla_sel = st.selectbox(
        "Seleccionar cuadrilla",
        options=["Todas"] + cuadrillas,
    )

    plan_filtrado = plan.copy()

    if cuadrilla_sel != "Todas":
        plan_filtrado = plan_filtrado[plan_filtrado["cuadrilla_id"] == cuadrilla_sel]

    columnas = [
        "cuadrilla_id",
        "orden_visita",
        "transformador_id",
        "riesgo_predicho",
        "accion_recomendada",
        "prioridad_operativa",
        "ventana_atencion_horas",
        "temperatura_actual",
        "porcentaje_carga",
        "distancia_desde_anterior_km",
        "distancia_acumulada_km",
    ]

    columnas = [col for col in columnas if col in plan_filtrado.columns]

    st.dataframe(
        plan_filtrado[columnas],
        use_container_width=True,
        hide_index=True,
    )


def mostrar_resumen_cuadrillas() -> None:
    st.subheader("Resumen por cuadrilla")

    if not RESUMEN_CUADRILLAS_PATH.exists():
        st.warning("No se encontró el resumen de cuadrillas.")
        return

    resumen = cargar_csv(RESUMEN_CUADRILLAS_PATH)

    st.dataframe(
        resumen,
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    validar_archivos()

    df = cargar_csv(TRANSFORMADORES_PATH)
    plan = cargar_csv(PLAN_CUADRILLAS_PATH)

    df = preparar_datos(df)
    plan = preparar_datos(plan)

    riesgo_col = obtener_columna_riesgo(df)

    st.title("⚡ Sistema Inteligente de Riesgo de Transformadores")
    st.caption(
        "Demo académica: clasificación de riesgo, agente recomendador y planificación de cuadrillas."
    )

    mostrar_metricas(df, plan, riesgo_col)

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Mapa de riesgo",
            "Análisis y recomendaciones",
            "Plan de cuadrillas",
            "Detalle individual",
        ]
    )

    with tab1:
        st.subheader("Mapa de transformadores")

        df_filtrado = aplicar_filtros(df, riesgo_col)

        if df_filtrado.empty:
            st.warning("No hay transformadores para mostrar con los filtros actuales.")
        else:
            mapa = crear_mapa(df_filtrado, riesgo_col)

            st_folium(
                mapa,
                width=None,
                height=600,
            )

            st.caption(
                "Colores: rojo = riesgo alto, naranja = riesgo medio, verde = riesgo bajo."
            )

    with tab2:
        mostrar_graficos(df, riesgo_col)

        st.subheader("Top 10 transformadores críticos")
        mostrar_top_criticos(df, riesgo_col)

        st.subheader("Dataset con recomendaciones")
        columnas_recomendaciones = [
            "transformador_id",
            riesgo_col,
            "accion_recomendada",
            "prioridad_operativa",
            "ventana_atencion_horas",
            "temperatura_actual",
            "porcentaje_carga",
            "dias_desde_mantenimiento",
            "motivo_recomendacion",
        ]

        columnas_recomendaciones = [
            col for col in columnas_recomendaciones if col in df.columns
        ]

        st.dataframe(
            df[columnas_recomendaciones].head(100),
            use_container_width=True,
            hide_index=True,
        )

    with tab3:
        mostrar_resumen_cuadrillas()
        mostrar_plan_cuadrillas(plan)

    with tab4:
        mostrar_detalle_transformador(df, riesgo_col)


if __name__ == "__main__":
    main()