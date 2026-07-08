# Data Science Requirements: Breast Cancer Diagnostic Dataset

Proyecto academico reproducible para una practica de maestria en Ciencia de Datos. El repositorio documenta la seleccion tecnica de un dataset, su caracterizacion estadistica y la definicion de requerimientos funcionales y no funcionales para un sistema analitico biomédico.

## Objetivo

Desarrollar y publicar un proyecto profesional, reproducible y auditado que cubre las fases iniciales de un ciclo de Ciencia de Datos: busqueda y comparacion de datasets, seleccion cuantitativa, descripcion exhaustiva del dataset seleccionado, analisis exploratorio y especificacion de requerimientos.

## Descripcion de la practica

La practica evalua tres datasets locales incluidos en scikit-learn y selecciona el mas adecuado mediante criterios cuantitativos. El trabajo se enfoca en calidad de datos, interpretabilidad, completitud, estructura estadistica, valor academico y trazabilidad metodologica.

No se entrena un modelo predictivo porque el alcance solicitado corresponde a las fases 1 a 3: seleccion, contextualizacion, analisis descriptivo y requerimientos. Esto reduce consumo computacional y evita procesos innecesarios.

## Dataset seleccionado

**Breast Cancer Wisconsin Diagnostic**

- Fuente: datasets incluidos localmente en scikit-learn.
- Observaciones: 569.
- Variables predictoras: 30.
- Variable objetivo: diagnostico benigno o maligno.
- Valores faltantes: 0%.
- Puntaje ponderado de seleccion: 97.7355 sobre 100.

El dataset fue seleccionado por su combinacion de impacto, interpretabilidad, completitud, tamano muestral y riqueza dimensional. La evidencia estadistica muestra separacion relevante entre clases en variables morfologicas como `worst concave points`, `worst perimeter`, `mean concave points` y `worst radius`.

## Metodologia

1. Busqueda local de tres datasets candidatos disponibles en scikit-learn.
2. Comparacion tecnica con criterios normalizados: tamano, dimensionalidad, completitud, preparacion analitica, interpretabilidad e impacto.
3. Seleccion del dataset con mayor puntaje ponderado.
4. Construccion de diccionario exhaustivo de variables.
5. Analisis de calidad de datos: faltantes, duplicados, balance de clases y atipicos por regla IQR.
6. Analisis estadistico exploratorio: descriptivos, correlaciones y tamano de efecto entre clases.
7. Formulacion de requerimientos funcionales y no funcionales.
8. Generacion de tablas, figuras editoriales y reporte HTML reproducible.

## Estructura del repositorio

```text
data-science-requirements-breast-cancer/
├── data/
├── figures/
├── report/
├── src/
├── tables/
├── README.md
├── requirements.txt
└── .gitignore
```

## Principales resultados

- El dataset **Breast Cancer Wisconsin Diagnostic** obtuvo el mayor puntaje de seleccion frente a Wine Recognition y Diabetes Progression.
- La completitud del dataset es 100%, por lo que no se requiere imputacion en esta fase.
- La clase maligna representa una proporcion suficiente para analisis exploratorio, aunque un modelado futuro deberia usar particiones estratificadas.
- Se identificaron multiples variables con tamano de efecto alto entre diagnosticos.
- El heatmap de correlaciones evidencia redundancia entre variables geometricas, un aspecto clave para fases posteriores de seleccion de variables o regularizacion.
- Se definieron 10 requerimientos funcionales y 10 no funcionales con prioridad, categoria, justificacion e impacto.

## Reporte HTML

El informe final con estilo editorial esta disponible en:

[report/final_report.html](report/final_report.html)

> Nota: si se visualiza desde GitHub, algunas restricciones del navegador pueden afectar rutas locales. Para una lectura completa, descargar o clonar el repositorio y abrir el archivo HTML localmente.

## Como reproducir el analisis

Instalar dependencias permitidas:

```bash
pip install -r requirements.txt
```

Ejecutar desde la raiz del repositorio:

```bash
python src/build_project.py
```

El script genera nuevamente los artefactos finales en:

- `data/`
- `tables/`
- `figures/`
- `report/`

## Restricciones de reproducibilidad

Este proyecto cumple las restricciones academicas definidas:

- No usa modelos externos de IA.
- No realiza llamadas a APIs durante el analisis.
- No utiliza servicios en la nube durante el analisis.
- No descarga datasets.
- No regenera artefactos salvo que se ejecute explicitamente `src/build_project.py`.
- Usa solamente Python, Pandas, NumPy, Matplotlib y scikit-learn.

## Archivos clave

- `src/build_project.py`: script reproducible principal.
- `data/breast_cancer_wisconsin_diagnostic.csv`: dataset final seleccionado.
- `tables/dataset_comparison.csv`: comparacion cuantitativa de datasets.
- `tables/data_dictionary.csv`: diccionario completo de variables.
- `figures/`: visualizaciones editoriales finales.
- `report/final_report.html`: reporte academico final.
- `report/reproducibility_manifest.json`: manifiesto de reproducibilidad.

## Licencia y uso academico

Repositorio preparado para fines academicos y de demostracion metodologica. El analisis no debe interpretarse como herramienta clinica ni como recomendacion medica individual.
