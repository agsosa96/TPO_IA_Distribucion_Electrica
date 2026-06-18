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
import joblib


TRANSFORMADORES_PATH = Path("data/processed/transformadores_con_recomendaciones.csv")
PLAN_CUADRILLAS_PATH = Path("data/processed/plan_cuadrillas.csv")
RESUMEN_CUADRILLAS_PATH = Path("outputs/metrics/resumen_cuadrillas.csv")
RESUMEN_RECOMENDACIONES_PATH = Path("outputs/metrics/resumen_recomendaciones.csv")
MODELO_PATH = Path("models/modelo_riesgo_transformadores.joblib")
IMPORTANCIA_VARIABLES_PATH = Path("outputs/metrics/importancia_variables.csv")


st.set_page_config(
    page_title="IA - Riesgo de Transformadores",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data
def cargar_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def cargar_modelo(path: Path) -> dict:
    return joblib.load(path)


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


def recomendar_accion_simulada(
    riesgo: str,
    temperatura: float,
    porcentaje_carga: float,
    dias_mantenimiento: int,
    fallas: int,
) -> dict:
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

    else:
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

    return {
        "accion": accion,
        "prioridad": prioridad,
        "ventana_horas": ventana_horas,
        "tipo_intervencion": tipo_intervencion,
        "motivo": "; ".join(motivos),
    }


def predecir_escenario_what_if(row_simulada: pd.Series) -> dict:
    if not MODELO_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el modelo entrenado. Ejecutá primero: python src/entrenar_modelo.py"
        )

    artefacto = cargar_modelo(MODELO_PATH)

    pipeline = artefacto["pipeline"]
    numeric_features = artefacto["numeric_features"]
    categorical_features = artefacto["categorical_features"]

    features = numeric_features + categorical_features

    x_simulado = pd.DataFrame([row_simulada[features].to_dict()])

    riesgo_predicho = pipeline.predict(x_simulado)[0]
    probabilidades = pipeline.predict_proba(x_simulado)[0]
    clases = pipeline.named_steps["model"].classes_

    probabilidad_maxima = float(probabilidades.max())

    probabilidades_dict = {
        f"probabilidad_{clase.lower()}": round(float(prob), 4)
        for clase, prob in zip(clases, probabilidades)
    }

    return {
        "riesgo_predicho": riesgo_predicho,
        "probabilidad_maxima": round(probabilidad_maxima, 4),
        **probabilidades_dict,
    }


def mostrar_simulador_what_if(df: pd.DataFrame, riesgo_col: str) -> None:
    st.subheader("Simulador What-if")

    st.write(
        "Este simulador permite modificar variables operativas de un transformador "
        "y volver a ejecutar el modelo para ver cómo cambiaría el riesgo."
    )

    if not MODELO_PATH.exists():
        st.error("No se encontró el modelo entrenado.")
        st.code("python src/entrenar_modelo.py")
        return

    ids = df["transformador_id"].dropna().astype(str).tolist()

    transformador_sel = st.selectbox(
        "Seleccionar transformador para simular",
        options=ids,
        key="what_if_transformador",
    )

    row_original = df[df["transformador_id"].astype(str) == transformador_sel].iloc[0].copy()

    st.write("### Valores originales")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Riesgo actual", row_original[riesgo_col])
    col2.metric("Temperatura", f"{float(row_original['temperatura_actual']):.1f} °C")
    col3.metric("Carga", f"{float(row_original['porcentaje_carga']):.2f}")
    col4.metric("Días sin mantenimiento", int(row_original["dias_desde_mantenimiento"]))

    st.write("### Modificar escenario")

    temperatura_max = max(45.0, float(df["temperatura_actual"].max()) + 2)
    carga_max = max(2.5, float(df["porcentaje_carga"].max()) + 0.2)
    dias_max = max(800, int(df["dias_desde_mantenimiento"].max()) + 30)

    col_temp, col_carga, col_mant = st.columns(3)

    temperatura_simulada = col_temp.slider(
        "Temperatura simulada °C",
        min_value=15.0,
        max_value=float(temperatura_max),
        value=float(row_original["temperatura_actual"]),
        step=0.5,
    )

    carga_simulada = col_carga.slider(
        "Porcentaje de carga simulado",
        min_value=0.10,
        max_value=float(carga_max),
        value=float(row_original["porcentaje_carga"]),
        step=0.05,
    )

    dias_simulados = col_mant.slider(
        "Días desde mantenimiento simulados",
        min_value=0,
        max_value=int(dias_max),
        value=int(row_original["dias_desde_mantenimiento"]),
        step=10,
    )

    row_simulada = row_original.copy()

    row_simulada["temperatura_actual"] = temperatura_simulada
    row_simulada["porcentaje_carga"] = carga_simulada
    row_simulada["dias_desde_mantenimiento"] = dias_simulados

    if "capacidad_kva" in row_simulada.index and pd.notna(row_simulada["capacidad_kva"]):
        row_simulada["carga_estimada_kw"] = (
            carga_simulada * float(row_simulada["capacidad_kva"]) * 0.92
        )

    if st.button("Ejecutar simulación", type="primary"):
        resultado = predecir_escenario_what_if(row_simulada)

        fallas = int(row_simulada.get("fallas_ultimos_12_meses", 0))

        recomendacion = recomendar_accion_simulada(
            riesgo=resultado["riesgo_predicho"],
            temperatura=temperatura_simulada,
            porcentaje_carga=carga_simulada,
            dias_mantenimiento=dias_simulados,
            fallas=fallas,
        )

        st.write("### Resultado simulado")

        col_a, col_b, col_c, col_d = st.columns(4)

        col_a.metric(
            "Riesgo original",
            row_original[riesgo_col],
        )

        col_b.metric(
            "Riesgo simulado",
            resultado["riesgo_predicho"],
        )

        col_c.metric(
            "Probabilidad máxima",
            f"{resultado['probabilidad_maxima'] * 100:.1f}%",
        )

        col_d.metric(
            "Prioridad simulada",
            recomendacion["prioridad"],
        )

        if str(row_original[riesgo_col]) != str(resultado["riesgo_predicho"]):
            st.warning(
                f"El escenario cambia el riesgo de {row_original[riesgo_col]} "
                f"a {resultado['riesgo_predicho']}."
            )
        else:
            st.success("El escenario no modifica la clase de riesgo predicha.")

        st.write("### Recomendación del agente")

        st.info(
            f"""
            **Acción recomendada:** {recomendacion["accion"]}

            **Tipo de intervención:** {recomendacion["tipo_intervencion"]}

            **Ventana de atención:** {recomendacion["ventana_horas"]} horas

            **Motivo:** {recomendacion["motivo"]}
            """.strip()
        )

        st.write("### Probabilidades por clase")

        probs_mostrar = {
            key.replace("probabilidad_", "").capitalize(): value
            for key, value in resultado.items()
            if key.startswith("probabilidad_")
        }

        probs_df = pd.DataFrame(
            {
                "riesgo": list(probs_mostrar.keys()),
                "probabilidad": list(probs_mostrar.values()),
            }
        )

        fig = px.bar(
            probs_df,
            x="riesgo",
            y="probabilidad",
            text="probabilidad",
            title="Probabilidad estimada por clase de riesgo",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write("### Comparación de variables")

        comparacion = pd.DataFrame(
            {
                "variable": [
                    "temperatura_actual",
                    "porcentaje_carga",
                    "dias_desde_mantenimiento",
                    "carga_estimada_kw",
                ],
                "valor_original": [
                    row_original.get("temperatura_actual"),
                    row_original.get("porcentaje_carga"),
                    row_original.get("dias_desde_mantenimiento"),
                    row_original.get("carga_estimada_kw"),
                ],
                "valor_simulado": [
                    row_simulada.get("temperatura_actual"),
                    row_simulada.get("porcentaje_carga"),
                    row_simulada.get("dias_desde_mantenimiento"),
                    row_simulada.get("carga_estimada_kw"),
                ],
            }
        )

        st.dataframe(comparacion, use_container_width=True, hide_index=True)


def mostrar_explicabilidad_modelo() -> None:
    st.subheader("Explicabilidad del modelo")

    st.write(
        "Esta sección muestra qué variables tuvieron mayor peso en el modelo "
        "Random Forest al momento de clasificar el riesgo de los transformadores."
    )

    if not IMPORTANCIA_VARIABLES_PATH.exists():
        st.warning("No se encontró el archivo de importancia de variables.")
        st.write("Ejecutá nuevamente el entrenamiento del modelo:")
        st.code("python src/entrenar_modelo.py")
        return

    importancia = cargar_csv(IMPORTANCIA_VARIABLES_PATH)

    columnas_necesarias = {"variable", "importancia"}

    if not columnas_necesarias.issubset(importancia.columns):
        st.error(
            "El archivo importancia_variables.csv no tiene el formato esperado. "
            "Debe contener las columnas 'variable' e 'importancia'."
        )
        return

    importancia = importancia.sort_values("importancia", ascending=False).copy()

    top_n = st.slider(
        "Cantidad de variables a mostrar",
        min_value=5,
        max_value=min(20, len(importancia)),
        value=min(10, len(importancia)),
    )

    top_importancia = importancia.head(top_n).copy()

    fig = px.bar(
        top_importancia.sort_values("importancia", ascending=True),
        x="importancia",
        y="variable",
        orientation="h",
        text="importancia",
        title="Variables más importantes del modelo",
    )

    fig.update_traces(texttemplate="%{text:.3f}")
    fig.update_layout(yaxis_title="Variable", xaxis_title="Importancia")

    st.plotly_chart(fig, use_container_width=True)

    st.write("### Tabla de importancia")

    st.dataframe(
        importancia,
        use_container_width=True,
        hide_index=True,
    )

    st.write("### Cómo interpretar este gráfico")

    st.info(
        """
        Las variables con mayor importancia son las que más ayudaron al modelo a separar
        los casos de riesgo Bajo, Medio y Alto.

        Por ejemplo, si `porcentaje_carga`, `temperatura_actual` o
        `dias_desde_mantenimiento` aparecen arriba, significa que el modelo las usó con
        más peso para decidir el nivel de riesgo.

        Esto permite explicar que el modelo no funciona como una caja negra completa:
        podemos observar qué factores técnicos influyen más en la clasificación.
        """.strip()
    )

    st.write("### Lectura técnica para la exposición")

    st.success(
        """
        El modelo Random Forest no solo genera una predicción, sino que también permite
        analizar la contribución relativa de cada variable. Esto nos ayuda a validar si
        el comportamiento del modelo tiene sentido operativo: un transformador con alta
        carga, alta temperatura, muchos días sin mantenimiento o fallas recientes debería
        tener mayor probabilidad de ser clasificado como crítico.
        """.strip()
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


    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
        "Mapa de riesgo",
        "Análisis y recomendaciones",
        "Plan de cuadrillas",
        "Detalle individual",
        "Simulador What-if",
        "Explicabilidad",
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

    with tab5:
        mostrar_simulador_what_if(df, riesgo_col)

    with tab6:
        mostrar_explicabilidad_modelo()

if __name__ == "__main__":
    main()