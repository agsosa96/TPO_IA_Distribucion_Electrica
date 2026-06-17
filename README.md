Sistema Inteligente de Riesgo y Mantenimiento de Transformadores
Descripción del proyecto
Este proyecto tiene como objetivo desarrollar una solución de Inteligencia Artificial orientada a estimar el riesgo operativo de transformadores eléctricos en el AMBA y CABA, tomando como base datos geográficos, climáticos, poblacionales y operativos.
El trabajo parte de un proyecto previo de Ciencia de Datos sobre distribución eléctrica y lo extiende incorporando componentes de Inteligencia Artificial para clasificar niveles de riesgo, recomendar acciones de mantenimiento y asistir en la planificación de cuadrillas técnicas.
Problema a resolver
Las redes de distribución eléctrica pueden verse afectadas por sobrecarga, altas temperaturas, falta de mantenimiento y aumento de la demanda en zonas densamente pobladas.
El problema principal es que muchas intervenciones se realizan de forma reactiva, una vez que el corte o falla ya ocurrió. Este proyecto propone una herramienta de apoyo a la decisión para pasar de un enfoque reactivo a uno preventivo.
Hipótesis
A partir de variables como temperatura ambiente, densidad poblacional, carga estimada, antigüedad del transformador y días desde el último mantenimiento, es posible estimar un nivel de riesgo operativo y priorizar acciones preventivas sobre los activos más críticos.
Objetivo general
Desarrollar una demo funcional de un sistema inteligente capaz de:
•	estimar el riesgo de sobrecarga o sobrecalentamiento de transformadores;
•	clasificar transformadores en riesgo bajo, medio o alto;
•	recomendar acciones de mantenimiento mediante un agente basado en reglas;
•	priorizar zonas críticas;
•	asignar cuadrillas técnicas para revisión preventiva.
Alcance
El proyecto es una prueba de concepto académica. Utiliza datos reales disponibles públicamente y datos simulados para completar variables operativas no disponibles, como carga estimada, antigüedad, historial de fallas y días desde mantenimiento.
Los resultados no deben interpretarse como una predicción real de fallas eléctricas, sino como una demostración técnica de cómo podría construirse una solución de IA aplicada al mantenimiento predictivo.
Fuentes de datos
El proyecto utiliza o toma como referencia:
•	datos geográficos de infraestructura eléctrica;
•	coordenadas estimadas de transformadores;
•	densidad poblacional como proxy de demanda;
•	temperatura actual o simulada;
•	variables operativas simuladas para entrenamiento del modelo.
Componentes principales
1. Preparación de datos
Consolidación de datos geográficos, climáticos, poblacionales y operativos en un único dataset final.
2. Modelo de IA
Entrenamiento de un modelo de clasificación para estimar el nivel de riesgo de cada transformador.
Salida esperada:
•	Riesgo bajo
•	Riesgo medio
•	Riesgo alto
3. Agente recomendador
Sistema basado en reglas que interpreta la predicción del modelo y recomienda acciones operativas, por ejemplo:
•	monitoreo normal;
•	revisión programada;
•	inspección prioritaria;
•	mantenimiento urgente.
4. Planificador de cuadrillas
Módulo simple de asignación de cuadrillas a transformadores críticos, considerando prioridad y cercanía geográfica.
5. Demo visual
Aplicación en Streamlit para visualizar:
•	mapa de transformadores;
•	clasificación de riesgo;
•	top de activos críticos;
•	recomendación del agente;
•	asignación de cuadrillas.
Tecnologías utilizadas
•	Python
•	Pandas
•	NumPy
•	Scikit-learn
•	Streamlit
•	Folium
•	Plotly
•	GitHub
Estructura del proyecto
proyecto_ia_transformadores/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── 01_preparacion_datos.ipynb
│   └── 02_modelo_riesgo.ipynb
│
├── src/
│   ├── generar_dataset.py
│   ├── entrenar_modelo.py
│   ├── agente_recomendador.py
│   ├── planificador_cuadrillas.py
│   └── utils.py
│
├── models/
├── outputs/
│   ├── figures/
│   └── maps/
│
├── docs/
├── app.py
├── requirements.txt
└── README.md
Instalación
Crear un entorno virtual:
python -m venv venv
Activar el entorno virtual en Windows:
venv\Scripts\activate
Instalar dependencias:
pip install -r requirements.txt
Ejecución de la demo
Una vez preparado el dataset y entrenado el modelo, ejecutar:
streamlit run app.py
Estado del proyecto
Proyecto en desarrollo para presentación académica del Trabajo Práctico Obligatorio de Inteligencia Artificial.
Integrantes
•	Integrante 1 - Rol funcional / negocio
•	Integrante 2 - Ingeniería de datos
•	Integrante 3 - Datos geoespaciales
•	Integrante 4 - Modelo de IA
•	Integrante 5 - Validación y métricas
•	Integrante 6 - Agente recomendador
•	Integrante 7 - Demo y planificación operativa
