from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"

PDF_NAME = "final_report.pdf"
TEX_NAME = "final_report.tex"
PREVIEW_NAME = "final_report_preview.png"
BUILD_STEMS = ["final_report"]


def read_csv(name: str) -> list[dict[str, str]]:
    with (TABLES_DIR / name).open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def tex_escape(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def fmt_number(value: str) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if abs(number) >= 100:
        return f"{number:,.1f}"
    if abs(number) >= 10:
        return f"{number:,.2f}"
    return f"{number:,.4f}".rstrip("0").rstrip(".")


def select_columns(rows: list[dict[str, str]], columns: list[str]) -> list[list[str]]:
    return [[fmt_number(row.get(column, "")) for column in columns] for row in rows]


def longtable(
    caption: str,
    label: str,
    columns: list[str],
    rows: list[list[str]],
    widths: list[float],
    size: str = r"\scriptsize",
    landscape: bool = False,
) -> str:
    target_width = 0.88 if landscape else 0.90
    total_width = sum(widths)
    if total_width:
        scale = min(1.0, target_width / total_width)
        widths = [width * scale for width in widths]
    spec = " ".join(r">{\raggedright\arraybackslash}p{" + f"{width:.3f}" + r"\linewidth}" for width in widths)
    display_headers = {
        "Observaciones": "Obs.",
        "Variables predictoras": "Variables",
        "Valores faltantes (%)": "Faltantes (%)",
        "Puntaje total ponderado": "Puntaje total",
        "Puntaje tamaño": "Tamaño",
        "Puntaje dimensionalidad": "Dimensión",
        "Puntaje completitud": "Completitud",
        "Puntaje preparación analítica": "Prep. analítica",
        "Puntaje interpretabilidad": "Interpretabilidad",
        "Puntaje impacto": "Impacto",
    }
    header = " & ".join(tex_escape(display_headers.get(column, column)) for column in columns) + r" \\"
    body_lines = [" & ".join(tex_escape(cell) for cell in row) + r" \\" for row in rows]
    body = "\n".join(body_lines)
    table = rf"""
{size}
\setlength{{\tabcolsep}}{{1.5pt}}
\renewcommand{{\arraystretch}}{{1.16}}
\begin{{longtable}}{{{spec}}}
\caption{{{tex_escape(caption)}}}\label{{{label}}}\\
\toprule
{header}
\midrule
\endfirsthead
\caption[]{{{tex_escape(caption)} (continuación)}}\\
\toprule
{header}
\midrule
\endhead
\midrule
\multicolumn{{{len(columns)}}}{{r}}{{Continúa en la siguiente página}}\\
\midrule
\endfoot
\bottomrule
\endlastfoot
{body}
\end{{longtable}}
\normalsize
"""
    if landscape:
        return "\\begin{landscape}\n" + table + "\n\\end{landscape}\n"
    return table


def figure_block(number: int, filename: str, caption: str, label: str, interpretation: str) -> str:
    path = f"figures/{filename}"
    return rf"""
\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth,height=0.62\textheight,keepaspectratio]{{{path}}}
\caption{{{tex_escape(caption)}}}
\label{{{label}}}
\vspace{{0.2em}}
\begin{{minipage}}{{0.92\linewidth}}
\footnotesize \textbf{{Interpretación.}} {tex_escape(interpretation)}
\end{{minipage}}
\end{{figure}}
"""


def make_preview() -> None:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="#fbfaf7")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.08, 0.10), 0.84, 0.80, fill=False, linewidth=1.2, edgecolor="#d9d6ce"))
    ax.add_patch(plt.Rectangle((0.08, 0.83), 0.84, 0.015, color="#a64f4f"))
    ax.text(0.12, 0.78, "Informe científico", fontsize=15, color="#a64f4f", weight="bold")
    ax.text(
        0.12,
        0.64,
        "Selección, caracterización\n"
        "y requerimientos para un\n"
        "sistema analítico biomédico",
        fontsize=28,
        color="#22272b",
        weight="bold",
        linespacing=1.12,
    )
    ax.text(0.12, 0.48, "Breast Cancer Wisconsin Diagnostic Dataset", fontsize=14, color="#536f8a")
    ax.text(0.12, 0.38, "Monica Cueto Tapia", fontsize=13, color="#22272b", weight="bold")
    ax.text(
        0.12,
        0.31,
        "Maestría en Inteligencia Artificial,\nMachine Learning y Data Science\nUniversidad Pública de El Alto (UPEA)",
        fontsize=11.5,
        color="#69737a",
        linespacing=1.35,
    )
    ax.text(0.12, 0.17, "PDF académico reproducible | 7 de julio de 2026", fontsize=10, color="#69737a")
    fig.savefig(REPORT_DIR / PREVIEW_NAME, dpi=180, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def build_latex() -> str:
    comparison = read_csv("dataset_comparison.csv")
    quality = read_csv("data_quality_summary.csv")
    data_dictionary = read_csv("data_dictionary.csv")
    descriptive = read_csv("descriptive_statistics.csv")
    effects = read_csv("variable_effect_sizes.csv")
    functional = read_csv("functional_requirements.csv")
    nonfunctional = read_csv("nonfunctional_requirements.csv")

    comparison_summary_cols = [
        "Dataset",
        "Dominio",
        "Tarea",
        "Observaciones",
        "Variables predictoras",
        "Valores faltantes (%)",
        "Puntaje total ponderado",
    ]
    comparison_score_cols = [
        "Dataset",
        "Puntaje tamaño",
        "Puntaje dimensionalidad",
        "Puntaje completitud",
        "Puntaje preparación analítica",
        "Puntaje interpretabilidad",
        "Puntaje impacto",
        "Puntaje total ponderado",
    ]
    dictionary_cols = [
        "Nombre",
        "Tipo",
        "Descripción",
        "Rango / valores",
        "Unidad",
        "Importancia",
        "Posibles valores atípicos",
        "Observaciones",
    ]
    descriptive_position_cols = [
        "Variable",
        "n",
        "Media",
        "Desv. estándar",
        "Mínimo",
        "P1",
        "P25",
        "Mediana",
        "P75",
        "P99",
        "Máximo",
    ]
    descriptive_signal_cols = [
        "Variable",
        "Asimetría",
        "|d|",
        "Dirección",
        "Atípicos IQR",
        "Atípicos IQR (%)",
    ]
    effect_cols = ["Variable", "Media maligno", "Media benigno", "Diferencia estandarizada (d)", "|d|", "Dirección"]
    req_f_cols = ["ID", "Requerimiento funcional", "Categoría", "Prioridad", "Justificación", "Impacto"]
    req_nf_cols = ["ID", "Requerimiento no funcional", "Prioridad", "Justificación", "Impacto"]

    figures = [
        (12, "12_executive_dashboard.png", "Dashboard ejecutivo del dataset seleccionado", "fig:dashboard", "El panel resume que el conjunto es completo, moderadamente desbalanceado y con variables de fuerte separación diagnóstica."),
        (1, "01_project_flow.png", "Flujo metodológico del proyecto", "fig:project-flow", "El flujo separa selección, análisis y comunicación para que cada conclusión se origine en evidencia verificable."),
        (2, "02_dataset_comparison_matrix.png", "Matriz comparativa de datasets candidatos", "fig:comparison-matrix", "El dataset de cáncer de mama domina por tamaño, profundidad de variables, impacto e interpretabilidad; la completitud es equivalente en los tres."),
        (3, "03_dataset_radar.png", "Perfil multicriterio de selección", "fig:dataset-radar", "La decisión no depende de un único criterio: el dataset seleccionado mantiene desempeño alto en seis dimensiones."),
        (4, "04_missingness_map.png", "Diagrama de completitud por familia de variable", "fig:missingness", "La completitud uniforme del 100% elimina la necesidad de imputación en esta fase y reduce incertidumbre por datos faltantes."),
        (5, "05_target_distribution.png", "Distribución de la variable objetivo", "fig:target-distribution", "La clase benigna es mayoritaria, pero la clase maligna representa más de un tercio del conjunto; el desbalance es manejable para análisis exploratorio."),
        (6, "06_effect_size_lollipop.png", "Variables con mayor separación entre diagnósticos", "fig:effect-size", "Las medidas de concavidad, puntos cóncavos y perímetro/área concentran la mayor diferencia estandarizada entre masas malignas y benignas."),
        (7, "07_correlation_heatmap.png", "Heatmap de correlación entre variables de mayor señal", "fig:correlation", "La señal predictiva convive con redundancia fuerte entre medidas geométricas; esto sugiere cautela con multicolinealidad en fases de modelado."),
        (8, "08_distribution_panel.png", "Distribuciones comparadas de variables prioritarias", "fig:distributions", "Las distribuciones muestran desplazamientos sistemáticos entre clases, especialmente en variables de borde y concavidad."),
        (9, "09_boxplot_panel.png", "Boxplots robustos por diagnóstico", "fig:boxplots", "La escala robusta permite comparar variables con unidades diferentes y evidencia medianas más altas en malignidad para variables clave."),
        (10, "10_architecture_pipeline.png", "Arquitectura reproducible del proyecto", "fig:architecture", "La arquitectura mantiene una sola ruta de datos y separa artefactos finales por función, evitando duplicación y procesamiento innecesario."),
        (11, "11_project_timeline.png", "Cronograma ejecutivo de la práctica", "fig:timeline", "El cronograma prioriza selección y comprensión de datos antes de cualquier modelado, consistente con una práctica de formulación analítica."),
    ]

    figure_tex = "\n".join(figure_block(*item) for item in figures)

    return rf"""
\documentclass[11pt,a4paper]{{article}}
\usepackage[a4paper,margin=2.4cm,headheight=15pt]{{geometry}}
\usepackage{{fontspec}}
\setmainfont{{Segoe UI}}
\setsansfont{{Segoe UI}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{float}}
\usepackage{{lscape}}
\usepackage{{hyperref}}
\graphicspath{{{{./}}{{figures/}}{{../figures/}}}}
\definecolor{{Ink}}{{HTML}}{{22272B}}
\definecolor{{Muted}}{{HTML}}{{69737A}}
\definecolor{{Accent}}{{HTML}}{{A64F4F}}
\definecolor{{Rule}}{{HTML}}{{D9D6CE}}
\definecolor{{Paper}}{{HTML}}{{FBFAF7}}
\hypersetup{{
  colorlinks=true,
  linkcolor=Accent,
  urlcolor=Accent,
  citecolor=Accent,
  pdftitle={{Selección, caracterización y requerimientos para un sistema analítico biomédico}},
  pdfauthor={{Monica Cueto Tapia}}
}}
\renewcommand{{\contentsname}}{{Tabla de contenidos}}
\renewcommand{{\listfigurename}}{{Índice de figuras}}
\renewcommand{{\listtablename}}{{Índice de tablas}}
\renewcommand{{\figurename}}{{Figura}}
\renewcommand{{\tablename}}{{Tabla}}
\renewcommand{{\refname}}{{Bibliografía}}
\makeatletter
\def\ps@academic{{%
  \def\@oddhead{{\small Proyecto de Ciencia de Datos\hfil UPEA}}%
  \def\@evenhead{{\small UPEA\hfil Proyecto de Ciencia de Datos}}%
  \def\@oddfoot{{\hfil\thepage\hfil}}%
  \def\@evenfoot{{\hfil\thepage\hfil}}%
}}
\makeatother
\pagestyle{{academic}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.72em}}

\begin{{document}}

\begin{{titlepage}}
\thispagestyle{{empty}}
\pagecolor{{Paper}}
\begin{{center}}
\vspace*{{1.2cm}}
{{\Large \textcolor{{Accent}}{{Informe científico reproducible}}}}\\[1.2cm]
{{\Huge \bfseries Selección, caracterización y requerimientos para un sistema analítico biomédico}}\\[0.45cm]
{{\Large Breast Cancer Wisconsin Diagnostic Dataset}}\\[1.6cm]
\rule{{0.78\textwidth}}{{0.6pt}}\\[1.2cm]
{{\Large \bfseries Monica Cueto Tapia}}\\[0.5cm]
{{\large Maestría en Inteligencia Artificial, Machine Learning y Data Science}}\\[0.25cm]
{{\large Universidad Pública de El Alto (UPEA)}}\\[1.2cm]
{{\large 7 de julio de 2026}}\\[2.1cm]
\begin{{minipage}}{{0.78\textwidth}}
\small
Documento académico preparado como evidencia de un proyecto reproducible de Ciencia de Datos. El informe conserva los análisis, tablas, gráficos e interpretaciones del reporte HTML original, reorganizados en formato de investigación con numeración, referencias cruzadas, índices y anexos.
\end{{minipage}}
\vfill
{{\small Sin APIs, sin nube, sin descargas de datasets y sin modelos externos de inteligencia artificial.}}
\end{{center}}
\end{{titlepage}}
\nopagecolor

\pagenumbering{{roman}}
\tableofcontents
\clearpage
\listoffigures
\clearpage
\listoftables
\clearpage

\pagenumbering{{arabic}}
\section{{Resumen ejecutivo}}
Se evaluaron tres datasets disponibles localmente en scikit-learn. El dataset \textbf{{Breast Cancer Wisconsin Diagnostic}} obtuvo el mayor puntaje ponderado (97.7355/100) al combinar tamaño muestral, riqueza dimensional, completitud, preparación analítica, interpretabilidad e impacto. La selección es apropiada para una práctica de maestría porque permite discutir calidad de datos, separación estadística entre clases, redundancia de variables y requerimientos de un flujo analítico reproducible sin depender de servicios externos.

El conjunto seleccionado contiene 569 observaciones, 30 variables predictoras y 0\% de valores faltantes. La evidencia univariada más fuerte aparece en \textit{{worst concave points}} (|d|=2.6926), mientras que el análisis de correlación evidencia redundancia fuerte entre medidas geométricas. El tablero ejecutivo de la Figura~\ref{{fig:dashboard}} sintetiza estos hallazgos.

{figure_block(12, "12_executive_dashboard.png", "Dashboard ejecutivo del dataset seleccionado", "fig:dashboard", "El panel resume que el conjunto es completo, moderadamente desbalanceado y con variables de fuerte separación diagnóstica.")}

\section{{Fase 1. Búsqueda, comparación y selección del dataset}}
Para cumplir la restricción de no usar internet, APIs ni servicios en la nube, la búsqueda se limitó al catálogo local de datasets incluidos en scikit-learn. Esta decisión reduce el consumo computacional, elimina riesgos de disponibilidad externa y mantiene la reproducibilidad. Los candidatos fueron Breast Cancer Wisconsin Diagnostic, Wine Recognition y Diabetes Progression.

Como se muestra en la Figura~\ref{{fig:project-flow}}, el flujo separa selección, análisis y comunicación. La comparación cuantitativa se resume en las Tablas~\ref{{tab:comparison-summary}} y~\ref{{tab:comparison-scores}}; las Figuras~\ref{{fig:comparison-matrix}} y~\ref{{fig:dataset-radar}} complementan la decisión visualmente.

{figure_block(1, "01_project_flow.png", "Flujo metodológico del proyecto", "fig:project-flow", "El flujo separa selección, análisis y comunicación para que cada conclusión se origine en evidencia verificable.")}

{longtable("Comparación técnica resumida de datasets candidatos", "tab:comparison-summary", comparison_summary_cols, select_columns(comparison, comparison_summary_cols), [0.18, 0.19, 0.14, 0.10, 0.11, 0.12, 0.13], r"\scriptsize")}

{longtable("Puntajes normalizados usados para la selección del dataset", "tab:comparison-scores", comparison_score_cols, select_columns(comparison, comparison_score_cols), [0.18, 0.11, 0.13, 0.12, 0.15, 0.13, 0.09, 0.12], r"\scriptsize", True)}

{figure_block(2, "02_dataset_comparison_matrix.png", "Matriz comparativa de datasets candidatos", "fig:comparison-matrix", "El dataset de cáncer de mama domina por tamaño, profundidad de variables, impacto e interpretabilidad; la completitud es equivalente en los tres.")}

{figure_block(3, "03_dataset_radar.png", "Perfil multicriterio de selección", "fig:dataset-radar", "La decisión no depende de un único criterio: el dataset seleccionado mantiene desempeño alto en seis dimensiones.")}

La ponderación priorizó evidencia verificable: 25\% tamaño muestral, 20\% dimensionalidad, 15\% completitud, 15\% preparación analítica, 15\% interpretabilidad y 10\% impacto. El dataset seleccionado domina por volumen, número de variables, completitud total y relevancia biomédica. Wine Recognition es técnicamente limpio, pero tiene menor escala e impacto para una discusión socio-técnica. Diabetes Progression posee impacto alto, aunque su objetivo continuo y variables estandarizadas reducen interpretabilidad variable por variable.

\section{{Fase 2. Contexto, problema, objetivo y alcance}}
\subsection{{Contexto}}
El análisis biomédico basado en imágenes requiere transformar mediciones morfológicas en evidencia interpretable. Este dataset resume características geométricas y de textura de núcleos celulares obtenidas a partir de aspiración con aguja fina de masas mamarias.

\subsection{{Problema}}
Antes de plantear un sistema predictivo, es necesario demostrar que los datos son completos, comprensibles, estadísticamente informativos y gobernables. Sin esa base, cualquier modelo posterior podría amplificar redundancias, sesgos de clase o interpretaciones clínicas débiles.

\subsection{{Objetivo}}
Seleccionar y caracterizar un dataset biomédico reproducible, identificar sus variables críticas, evaluar calidad estadística y definir requerimientos de un sistema analítico que pueda sostener futuras fases de modelado con rigor académico.

\subsection{{Alcance}}
El alcance cubre selección de dataset, análisis exploratorio, diccionario de datos, visualizaciones editoriales y requerimientos. No incluye entrenamiento de modelos clínicos ni recomendaciones médicas individuales, porque las fases solicitadas no lo exigen y porque se prioriza eficiencia.

\subsection{{Beneficiarios}}
Los beneficiarios son estudiantes e investigadores de ciencia de datos, docentes evaluadores, equipos de analítica biomédica y responsables de gobernanza de datos que requieran una base reproducible para decisiones metodológicas.

\subsection{{Resultados esperados}}
Se espera una selección defendible, un inventario de variables accionable, evidencia de calidad de datos, detección de señales estadísticas relevantes, visualizaciones publicables y especificación clara de requerimientos para continuidad del proyecto.

{longtable("Resumen de calidad del dataset seleccionado", "tab:quality", ["Indicador", "Resultado", "Interpretación técnica"], select_columns(quality, ["Indicador", "Resultado", "Interpretación técnica"]), [0.20, 0.21, 0.53], r"\small")}

La Figura~\ref{{fig:missingness}} confirma completitud total; la Figura~\ref{{fig:target-distribution}} muestra un desbalance moderado, no extremo. La señal univariada se observa en la Figura~\ref{{fig:effect-size}} y se complementa con la Tabla~\ref{{tab:effect-sizes}}.

{figure_block(4, "04_missingness_map.png", "Diagrama de completitud por familia de variable", "fig:missingness", "La completitud uniforme del 100% elimina la necesidad de imputación en esta fase y reduce incertidumbre por datos faltantes.")}

{figure_block(5, "05_target_distribution.png", "Distribución de la variable objetivo", "fig:target-distribution", "La clase benigna es mayoritaria, pero la clase maligna representa más de un tercio del conjunto; el desbalance es manejable para análisis exploratorio.")}

{figure_block(6, "06_effect_size_lollipop.png", "Variables con mayor separación entre diagnósticos", "fig:effect-size", "Las medidas de concavidad, puntos cóncavos y perímetro/área concentran la mayor diferencia estandarizada entre masas malignas y benignas.")}

{longtable("Tamaños de efecto por variable entre diagnósticos", "tab:effect-sizes", effect_cols, select_columns(effects, effect_cols), [0.20, 0.13, 0.13, 0.16, 0.09, 0.17], r"\scriptsize", True)}

La Figura~\ref{{fig:correlation}} evidencia que la señal predictiva convive con multicolinealidad potencial. Las Figuras~\ref{{fig:distributions}} y~\ref{{fig:boxplots}} muestran los desplazamientos de distribución y mediana entre clases para las variables prioritarias.

{figure_block(7, "07_correlation_heatmap.png", "Heatmap de correlación entre variables de mayor señal", "fig:correlation", "La señal predictiva convive con redundancia fuerte entre medidas geométricas; esto sugiere cautela con multicolinealidad en fases de modelado.")}

{figure_block(8, "08_distribution_panel.png", "Distribuciones comparadas de variables prioritarias", "fig:distributions", "Las distribuciones muestran desplazamientos sistemáticos entre clases, especialmente en variables de borde y concavidad.")}

{figure_block(9, "09_boxplot_panel.png", "Boxplots robustos por diagnóstico", "fig:boxplots", "La escala robusta permite comparar variables con unidades diferentes y evidencia medianas más altas en malignidad para variables clave.")}

\section{{Fase 3. Requerimientos del sistema analítico}}
Los requerimientos se formulan para un sistema de análisis reproducible, no para un producto clínico operativo. La prioridad se asigna según riesgo metodológico, trazabilidad y valor para decisiones académicas posteriores. La arquitectura propuesta se observa en la Figura~\ref{{fig:architecture}} y el cronograma ejecutivo en la Figura~\ref{{fig:timeline}}.

{figure_block(10, "10_architecture_pipeline.png", "Arquitectura reproducible del proyecto", "fig:architecture", "La arquitectura mantiene una sola ruta de datos y separa artefactos finales por función, evitando duplicación y procesamiento innecesario.")}

{figure_block(11, "11_project_timeline.png", "Cronograma ejecutivo de la práctica", "fig:timeline", "El cronograma prioriza selección y comprensión de datos antes de cualquier modelado, consistente con una práctica de formulación analítica.")}

{longtable("Requerimientos funcionales", "tab:functional", req_f_cols, select_columns(functional, req_f_cols), [0.08, 0.29, 0.13, 0.09, 0.21, 0.16], r"\scriptsize", True)}

{longtable("Requerimientos no funcionales", "tab:nonfunctional", req_nf_cols, select_columns(nonfunctional, req_nf_cols), [0.09, 0.22, 0.10, 0.30, 0.23], r"\scriptsize", True)}

\section{{Conclusiones}}
\begin{{itemize}}
\item La selección del dataset es cuantitativamente defendible: Breast Cancer Wisconsin Diagnostic alcanza 97.7355/100 y supera a los candidatos por balance entre escala, dimensionalidad, interpretabilidad e impacto.
\item La completitud del 100\% evita imputación en esta fase; por tanto, las diferencias observadas entre clases no están condicionadas por estrategias de reemplazo de datos faltantes.
\item La clase maligna conserva representación analítica suficiente para exploración, aunque un modelado posterior debería controlar el desbalance con particiones estratificadas.
\item La presencia de múltiples variables con |d| $\geq$ 1 indica separación estadística fuerte; no obstante, la correlación alta entre medidas geométricas confirma información redundante.
\item La fase de requerimientos debe priorizar reproducibilidad, trazabilidad, independencia de red y comunicación visual, porque esos atributos hacen defendible el proyecto en un contexto de maestría.
\end{{itemize}}

\section{{Bibliografía}}
\begin{{thebibliography}}{{9}}
\bibitem{{pedregosa2011}} Pedregosa, F. et al. (2011). \textit{{Scikit-learn: Machine Learning in Python}}. Journal of Machine Learning Research, 12, 2825--2830.
\bibitem{{street1993}} Street, W. N., Wolberg, W. H., \& Mangasarian, O. L. (1993). \textit{{Nuclear feature extraction for breast tumor diagnosis}}. IS\&T/SPIE International Symposium on Electronic Imaging.
\bibitem{{wolberg1995}} Wolberg, W. H., Street, W. N., \& Mangasarian, O. L. (1995). \textit{{Breast cancer diagnosis and prognosis via linear programming}}. Operations Research, 43(4), 570--577.
\end{{thebibliography}}

\clearpage
\appendix
\section{{Anexo A. Diccionario exhaustivo de variables}}
La Tabla~\ref{{tab:data-dictionary}} reproduce el diccionario completo de variables del proyecto, con tipo, descripción, rango observado, unidad, importancia, posibles valores atípicos y observaciones técnicas.

{longtable("Diccionario exhaustivo de variables", "tab:data-dictionary", dictionary_cols, select_columns(data_dictionary, dictionary_cols), [0.12, 0.10, 0.20, 0.10, 0.12, 0.09, 0.16, 0.16], r"\tiny", True)}

\section{{Anexo B. Estadística descriptiva completa}}
Las Tablas~\ref{{tab:descriptive-position}} y~\ref{{tab:descriptive-signal}} presentan los estadísticos de posición, dispersión, asimetría, atípicos IQR y señal univariada conservados desde el reporte HTML.

{longtable("Estadísticos descriptivos de posición y dispersión", "tab:descriptive-position", descriptive_position_cols, select_columns(descriptive, descriptive_position_cols), [0.16, 0.06, 0.08, 0.09, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08], r"\tiny", True)}

{longtable("Asimetría, señal univariada y atípicos IQR", "tab:descriptive-signal", descriptive_signal_cols, select_columns(descriptive, descriptive_signal_cols), [0.24, 0.12, 0.10, 0.18, 0.12, 0.14], r"\scriptsize", True)}

\section{{Anexo C. Reproducibilidad}}
El proyecto se reproduce desde la raíz del repositorio con:

\begin{{verbatim}}
python src/build_project.py
\end{{verbatim}}

El PDF se compone a partir de los artefactos finales con:

\begin{{verbatim}}
python src/build_pdf_report.py
\end{{verbatim}}

Durante la generación de este documento no se recalcularon estadísticas ni se regeneraron gráficos analíticos. Solo se transformaron los artefactos existentes en una versión académica en PDF.

\end{{document}}
"""


def compile_pdf() -> None:
    tex_path = REPORT_DIR / TEX_NAME
    tex_path.write_text(build_latex(), encoding="utf-8")

    command = [
        "xelatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(REPORT_DIR),
        str(tex_path),
    ]
    for _ in range(3):
        subprocess.run(command, cwd=ROOT, check=True)

    for suffix in [".aux", ".log", ".out", ".toc", ".lof", ".lot"]:
        path = REPORT_DIR / f"{BUILD_STEMS[0]}{suffix}"
        if path.exists():
            path.unlink()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    make_preview()
    compile_pdf()
    print(f"PDF generado: {REPORT_DIR / PDF_NAME}")
    print(f"Vista previa generada: {REPORT_DIR / PREVIEW_NAME}")


if __name__ == "__main__":
    main()






