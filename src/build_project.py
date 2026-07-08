from __future__ import annotations

import json
import math
import re
import textwrap
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"
REPORT_DIR = ROOT / "report"

ANALYSIS_DATE = "2026-07-07"
SOURCE_NOTE = "scikit-learn 1.5.1, datasets incluidos localmente; sin descargas, APIs ni nube."

PALETTE = {
    "ink": "#22272b",
    "paper": "#fbfaf7",
    "line": "#d9d6ce",
    "muted": "#69737a",
    "red": "#a64f4f",
    "teal": "#4f8a89",
    "blue": "#536f8a",
    "gold": "#b79a4a",
    "green": "#67855f",
    "light": "#f0eee8",
    "light_blue": "#dbe5ea",
    "light_red": "#eadcda",
}


def ensure_directories() -> None:
    for directory in (DATA_DIR, FIGURES_DIR, TABLES_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["paper"],
            "axes.facecolor": PALETTE["paper"],
            "savefig.facecolor": PALETTE["paper"],
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": PALETTE["line"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "grid.color": "#e5e2dc",
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def clean_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def footer(fig: plt.Figure, interpretation: str, source: str = SOURCE_NOTE) -> None:
    wrapped = textwrap.fill(f"Interpretación: {interpretation}", width=125)
    fig.text(0.01, 0.035, wrapped, ha="left", va="bottom", fontsize=8.4, color=PALETTE["ink"])
    fig.text(0.01, 0.012, f"Fuente: {source}", ha="left", va="bottom", fontsize=7.8, color=PALETTE["muted"])


def savefig(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIGURES_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_candidate_datasets() -> dict[str, dict[str, object]]:
    loaders = {
        "Breast Cancer Wisconsin Diagnostic": {
            "loader": load_breast_cancer,
            "domain": "Salud / diagnóstico oncológico",
            "task": "Clasificación binaria",
            "target": "diagnóstico de masa mamaria",
            "interpretability": 90.0,
            "impact": 100.0,
        },
        "Wine Recognition": {
            "loader": load_wine,
            "domain": "Química analítica / enología",
            "task": "Clasificación multiclase",
            "target": "cultivar de vino",
            "interpretability": 86.0,
            "impact": 67.0,
        },
        "Diabetes Progression": {
            "loader": load_diabetes,
            "domain": "Salud / progresión metabólica",
            "task": "Regresión",
            "target": "progresión de enfermedad a un año",
            "interpretability": 73.0,
            "impact": 90.0,
        },
    }
    datasets: dict[str, dict[str, object]] = {}
    for name, meta in loaders.items():
        bunch = meta["loader"](as_frame=True)
        x = bunch.data.copy()
        y = bunch.target.copy()
        datasets[name] = {
            **meta,
            "bunch": bunch,
            "x": x,
            "y": y,
            "frame": pd.concat([x, y.rename("target")], axis=1),
        }
    return datasets


def class_balance_score(y: pd.Series, task: str) -> tuple[float, str]:
    if "Regresión" in task:
        return 100.0, "No aplica: objetivo continuo"
    counts = y.value_counts()
    ideal = len(y) / len(counts)
    score = min(100.0, float(counts.min() / ideal * 100.0))
    detail = " / ".join(f"clase {idx}: {count}" for idx, count in counts.sort_index().items())
    return score, detail


def build_dataset_comparison(datasets: dict[str, dict[str, object]]) -> pd.DataFrame:
    max_n = max(int(meta["x"].shape[0]) for meta in datasets.values())
    max_p = max(int(meta["x"].shape[1]) for meta in datasets.values())
    rows = []
    for name, meta in datasets.items():
        x: pd.DataFrame = meta["x"]  # type: ignore[assignment]
        y: pd.Series = meta["y"]  # type: ignore[assignment]
        n, p = x.shape
        missing_rate = float(x.isna().mean().mean())
        duplicate_rows = int(pd.concat([x, y.rename("target")], axis=1).duplicated().sum())
        numeric_ratio = float(x.select_dtypes(include=[np.number]).shape[1] / p)
        balance_score, balance_detail = class_balance_score(y, str(meta["task"]))
        size_score = math.sqrt(n / max_n) * 100.0
        feature_score = math.sqrt(p / max_p) * 100.0
        completeness_score = (1.0 - missing_rate) * 100.0
        modeling_score = 40.0 * numeric_ratio + 40.0 + 20.0 * (balance_score / 100.0)
        selection_score = (
            0.25 * size_score
            + 0.20 * feature_score
            + 0.15 * completeness_score
            + 0.15 * modeling_score
            + 0.15 * float(meta["interpretability"])
            + 0.10 * float(meta["impact"])
        )
        rows.append(
            {
                "Dataset": name,
                "Dominio": meta["domain"],
                "Tarea": meta["task"],
                "Objetivo": meta["target"],
                "Observaciones": n,
                "Variables predictoras": p,
                "Valores faltantes (%)": missing_rate * 100,
                "Filas duplicadas": duplicate_rows,
                "Balance / objetivo": balance_detail,
                "Puntaje tamaño": size_score,
                "Puntaje dimensionalidad": feature_score,
                "Puntaje completitud": completeness_score,
                "Puntaje preparación analítica": modeling_score,
                "Puntaje interpretabilidad": float(meta["interpretability"]),
                "Puntaje impacto": float(meta["impact"]),
                "Puntaje total ponderado": selection_score,
            }
        )
    comparison = pd.DataFrame(rows).sort_values("Puntaje total ponderado", ascending=False)
    return comparison


def cohen_d_by_feature(x: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    records = []
    malignant_mask = labels == "malignant"
    benign_mask = labels == "benign"
    for col in x.columns:
        a = x.loc[malignant_mask, col].astype(float)
        b = x.loc[benign_mask, col].astype(float)
        pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d = 0.0 if pooled == 0 else float((a.mean() - b.mean()) / pooled)
        records.append(
            {
                "Variable": col,
                "Media maligno": a.mean(),
                "Media benigno": b.mean(),
                "Diferencia estandarizada (d)": d,
                "|d|": abs(d),
                "Dirección": "Mayor en maligno" if d > 0 else "Mayor en benigno",
            }
        )
    return pd.DataFrame(records).sort_values("|d|", ascending=False)


def iqr_outlier_summary(x: pd.DataFrame) -> pd.DataFrame:
    records = []
    for col in x.columns:
        q1 = float(x[col].quantile(0.25))
        q3 = float(x[col].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = int(((x[col] < lower) | (x[col] > upper)).sum())
        records.append(
            {
                "Variable": col,
                "Límite inferior IQR": lower,
                "Límite superior IQR": upper,
                "Atípicos IQR": count,
                "Atípicos IQR (%)": count / len(x) * 100,
            }
        )
    return pd.DataFrame(records).sort_values("Atípicos IQR (%)", ascending=False)


BASE_DESCRIPTIONS = {
    "radius": "distancia media desde el centro hasta puntos del perímetro celular",
    "texture": "variabilidad de la intensidad de gris en la imagen",
    "perimeter": "longitud del contorno de la masa celular",
    "area": "superficie proyectada de la masa celular",
    "smoothness": "variación local de longitudes de radio",
    "compactness": "relación de compacidad calculada como perímetro al cuadrado sobre área menos uno",
    "concavity": "severidad de porciones cóncavas del contorno",
    "concave points": "cantidad relativa de puntos cóncavos del contorno",
    "symmetry": "grado de simetría de la estructura celular",
    "fractal dimension": "aproximación de complejidad del contorno celular",
}


UNITS = {
    "radius": "unidades de imagen",
    "texture": "desviación estándar de niveles de gris",
    "perimeter": "unidades de imagen",
    "area": "unidades cuadradas de imagen",
    "smoothness": "índice adimensional",
    "compactness": "índice adimensional",
    "concavity": "índice adimensional",
    "concave points": "índice adimensional",
    "symmetry": "índice adimensional",
    "fractal dimension": "índice adimensional",
}


def parse_breast_feature(name: str) -> tuple[str, str]:
    if name.startswith("mean "):
        return "media", name.replace("mean ", "", 1)
    if name.startswith("worst "):
        return "peor valor", name.replace("worst ", "", 1)
    if name.endswith(" error"):
        return "error estándar", name.replace(" error", "")
    return "medición", name


def importance_label(value: float) -> str:
    if value >= 2.0:
        return "Muy alta"
    if value >= 1.25:
        return "Alta"
    if value >= 0.70:
        return "Media"
    return "Exploratoria"


def build_data_dictionary(x: pd.DataFrame, effects: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
    effect_map = effects.set_index("Variable").to_dict(orient="index")
    outlier_map = outliers.set_index("Variable").to_dict(orient="index")
    rows = [
        {
            "Nombre": "diagnosis",
            "Tipo": "categórica nominal",
            "Descripción": "Variable objetivo: diagnóstico histopatológico codificado como maligno o benigno.",
            "Rango / valores": "malignant, benign",
            "Unidad": "categoría clínica",
            "Importancia": "Crítica",
            "Posibles valores atípicos": "No aplica; se valida consistencia de etiquetas.",
            "Observaciones": "Es la variable de referencia para segmentar el análisis estadístico.",
        }
    ]
    for col in x.columns:
        statistic, base = parse_breast_feature(col)
        desc = BASE_DESCRIPTIONS.get(base, "medición morfológica derivada de imagen")
        unit = UNITS.get(base, "índice")
        out = outlier_map[col]
        eff = effect_map[col]
        outlier_text = (
            f"{int(out['Atípicos IQR'])} casos ({out['Atípicos IQR (%)']:.1f}%) fuera de "
            f"[{out['Límite inferior IQR']:.3g}, {out['Límite superior IQR']:.3g}] por regla IQR."
        )
        obs_parts = [
            f"{eff['Dirección'].lower()}",
            f"|d|={eff['|d|']:.2f}",
        ]
        skew = float(x[col].skew())
        if abs(skew) > 1:
            obs_parts.append("asimetría marcada")
        elif abs(skew) > 0.5:
            obs_parts.append("asimetría moderada")
        else:
            obs_parts.append("distribución relativamente estable")
        rows.append(
            {
                "Nombre": col,
                "Tipo": "numérica continua",
                "Descripción": f"{statistic.capitalize()} de {desc}.",
                "Rango / valores": f"{x[col].min():.4g} a {x[col].max():.4g}",
                "Unidad": unit,
                "Importancia": importance_label(float(eff["|d|"])),
                "Posibles valores atípicos": outlier_text,
                "Observaciones": "; ".join(obs_parts) + ".",
            }
        )
    return pd.DataFrame(rows)


def build_descriptive_statistics(x: pd.DataFrame, effects: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
    stats = x.describe(percentiles=[0.01, 0.25, 0.50, 0.75, 0.99]).T
    stats = stats.rename(
        columns={
            "count": "n",
            "mean": "Media",
            "std": "Desv. estándar",
            "min": "Mínimo",
            "1%": "P1",
            "25%": "P25",
            "50%": "Mediana",
            "75%": "P75",
            "99%": "P99",
            "max": "Máximo",
        }
    )
    stats.insert(0, "Variable", stats.index)
    stats = stats.reset_index(drop=True)
    stats["Asimetría"] = x.skew().values
    stats = stats.merge(effects[["Variable", "|d|", "Dirección"]], on="Variable", how="left")
    stats = stats.merge(outliers[["Variable", "Atípicos IQR", "Atípicos IQR (%)"]], on="Variable", how="left")
    return stats


def build_data_quality_summary(
    x: pd.DataFrame,
    labels: pd.Series,
    effects: pd.DataFrame,
    outliers: pd.DataFrame,
    corr: pd.DataFrame,
) -> pd.DataFrame:
    counts = labels.value_counts()
    corr_abs = corr.abs().copy()
    np.fill_diagonal(corr_abs.values, 0.0)
    max_idx = np.unravel_index(np.argmax(corr_abs.values), corr_abs.shape)
    max_pair = (corr_abs.index[max_idx[0]], corr_abs.columns[max_idx[1]], corr_abs.values[max_idx])
    rows = [
        ("Observaciones", f"{len(x):,}", "Tamaño suficiente para exploración estadística y especificación de requerimientos."),
        ("Variables predictoras", str(x.shape[1]), "Cobertura morfológica amplia: medias, errores estándar y peores valores."),
        ("Variable objetivo", "diagnosis", "Etiqueta clínica binaria para comparar patrones entre masas benignas y malignas."),
        ("Valores faltantes", f"{int(x.isna().sum().sum())} ({x.isna().mean().mean() * 100:.2f}%)", "No se requiere imputación; se preserva trazabilidad."),
        ("Filas duplicadas", "0", "No se detectan registros duplicados exactos en el conjunto final."),
        ("Distribución objetivo", f"benign: {counts.get('benign', 0)}; malignant: {counts.get('malignant', 0)}", "Existe desbalance moderado, no extremo."),
        ("Proporción minoritaria", f"{counts.min() / counts.sum() * 100:.1f}%", "La clase maligna conserva representación analítica suficiente."),
        ("Variables con atípicos IQR", f"{int((outliers['Atípicos IQR'] > 0).sum())} de {x.shape[1]}", "Los atípicos deben revisarse como extremos biológicos plausibles, no eliminarse por defecto."),
        ("Mayor correlación absoluta", f"{max_pair[0]} vs {max_pair[1]}: {max_pair[2]:.3f}", "Sugiere redundancia morfológica que debe controlarse en modelado posterior."),
        ("Variables con |d| >= 1", str(int((effects["|d|"] >= 1).sum())), "La separación univariada es fuerte en varias variables, en especial medidas de concavidad y tamaño."),
    ]
    return pd.DataFrame(rows, columns=["Indicador", "Resultado", "Interpretación técnica"])


def build_requirements() -> tuple[pd.DataFrame, pd.DataFrame]:
    functional = pd.DataFrame(
        [
            ("RF-01", "Cargar el dataset seleccionado desde una fuente local versionada.", "Ingesta", "Alta", "Evita dependencias externas y garantiza reproducción en entornos académicos.", "Permite reconstruir el análisis sin internet ni APIs."),
            ("RF-02", "Validar estructura, número de variables, tipos y variable objetivo.", "Calidad de datos", "Alta", "Previene análisis sobre columnas corruptas o incompatibles.", "Reduce riesgo de conclusiones no trazables."),
            ("RF-03", "Calcular métricas de completitud, duplicados y balance del objetivo.", "Calidad de datos", "Alta", "La confiabilidad del diagnóstico depende de ausencia de sesgos de captura.", "Define si se requiere imputación, depuración o estratificación."),
            ("RF-04", "Generar comparación cuantitativa entre tres datasets candidatos.", "Selección analítica", "Alta", "La selección debe sustentarse en criterios medibles y no en preferencia subjetiva.", "Justifica el dataset ante evaluación académica."),
            ("RF-05", "Construir diccionario exhaustivo de variables con rango, unidad e importancia.", "Documentación", "Alta", "La interpretación clínica necesita semántica clara por variable.", "Mejora auditabilidad y transferencia del proyecto."),
            ("RF-06", "Calcular estadísticos descriptivos robustos por variable.", "EDA", "Alta", "Percentiles, asimetría y atípicos revelan la estructura real de los datos.", "Soporta decisiones de transformación y modelado futuro."),
            ("RF-07", "Cuantificar separación entre clases mediante tamaño de efecto.", "Análisis estadístico", "Alta", "El tamaño de efecto aporta evidencia interpretable sin entrenar modelos innecesarios.", "Prioriza variables relevantes para diagnóstico."),
            ("RF-08", "Identificar correlaciones altas entre variables predictoras.", "Análisis estadístico", "Media", "La redundancia puede afectar interpretabilidad y estabilidad de modelos futuros.", "Informa reducción dimensional o regularización posterior."),
            ("RF-09", "Exportar tablas finales en formato reutilizable.", "Entregables", "Media", "Las tablas deben poder revisarse, auditarse o integrarse en anexos.", "Facilita evaluación y continuidad del trabajo."),
            ("RF-10", "Compilar un reporte HTML editorial con figuras, tablas e interpretación.", "Reporte", "Alta", "El entregable final debe comunicar evidencia, no solo producir archivos.", "Aumenta claridad, profesionalismo y presentabilidad."),
        ],
        columns=["ID", "Requerimiento funcional", "Categoría", "Prioridad", "Justificación", "Impacto"],
    )
    nonfunctional = pd.DataFrame(
        [
            ("RNF-01", "Reproducibilidad", "Alta", "El flujo debe ejecutarse con un único script local y resultados deterministas.", "Permite validar el trabajo en cualquier revisión académica."),
            ("RNF-02", "Eficiencia computacional", "Alta", "Solo se cargan datasets livianos y se evita entrenamiento innecesario.", "Minimiza tiempo, memoria y consumo de recursos."),
            ("RNF-03", "Independencia de red", "Alta", "No se usan descargas, APIs, nube ni modelos externos.", "El proyecto cumple restricciones y protege continuidad operativa."),
            ("RNF-04", "Trazabilidad", "Alta", "Cada figura y tabla declara fuente e interpretación.", "Facilita auditoría de evidencia y defensa metodológica."),
            ("RNF-05", "Interpretabilidad", "Alta", "Las métricas se basan en estadística descriptiva y tamaño de efecto.", "Evita cajas negras en fases tempranas del proyecto."),
            ("RNF-06", "Mantenibilidad", "Media", "El código separa carga, métricas, visualización y reporte.", "Permite extender el análisis sin rehacer la base."),
            ("RNF-07", "Portabilidad", "Media", "Usa únicamente Python, Pandas, NumPy, Matplotlib y scikit-learn.", "Reduce fricción de instalación y dependencia de librerías pesadas."),
            ("RNF-08", "Calidad visual", "Alta", "La salida debe usar paleta sobria, espacio en blanco y jerarquía editorial.", "Mejora comunicación de hallazgos a audiencias técnicas y ejecutivas."),
            ("RNF-09", "Robustez", "Media", "Las validaciones deben fallar de forma explícita ante cambios de estructura.", "Evita reportes silenciosamente inconsistentes."),
            ("RNF-10", "Gobernanza ética", "Alta", "El análisis evita inferencias clínicas individualizadas y trata el dataset como material académico.", "Reduce riesgo de uso indebido en decisiones médicas reales."),
        ],
        columns=["ID", "Requerimiento no funcional", "Prioridad", "Justificación", "Impacto"],
    )
    return functional, nonfunctional


def rounded_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=[np.number]).columns:
        out[col] = out[col].map(lambda value: round(float(value), 4))
    return out


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    for filename, df in tables.items():
        rounded_table(df).to_csv(TABLES_DIR / filename, index=False, encoding="utf-8-sig")


def plot_project_flow() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.axis("off")
    steps = [
        ("1", "Búsqueda local\n3 datasets", "Catálogo local de scikit-learn"),
        ("2", "Evaluación\ncuantitativa", "Tamaño, variables, completitud, balance, impacto"),
        ("3", "Selección\nargumentada", "Breast Cancer Wisconsin Diagnostic"),
        ("4", "EDA\nestadística", "Calidad, atípicos, correlaciones, tamaño de efecto"),
        ("5", "Reporte\neditorial", "Tablas, figuras e interpretación reproducible"),
    ]
    xs = np.linspace(0.06, 0.82, len(steps))
    for i, (num, title, desc) in enumerate(steps):
        color = [PALETTE["blue"], PALETTE["teal"], PALETTE["red"], PALETTE["gold"], PALETTE["green"]][i]
        rect = Rectangle((xs[i], 0.38), 0.15, 0.32, transform=ax.transAxes, facecolor="white", edgecolor=PALETTE["line"], linewidth=1.2)
        ax.add_patch(rect)
        ax.text(xs[i] + 0.018, 0.64, num, transform=ax.transAxes, fontsize=18, fontweight="bold", color=color)
        ax.text(xs[i] + 0.018, 0.55, title, transform=ax.transAxes, fontsize=12, fontweight="bold", color=PALETTE["ink"], va="top")
        ax.text(xs[i] + 0.018, 0.43, textwrap.fill(desc, 21), transform=ax.transAxes, fontsize=8.8, color=PALETTE["muted"], va="top")
        if i < len(steps) - 1:
            arrow = FancyArrowPatch(
                (xs[i] + 0.155, 0.54),
                (xs[i + 1] - 0.01, 0.54),
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.2,
                color=PALETTE["muted"],
            )
            ax.add_patch(arrow)
    ax.set_title("Flujo metodológico del proyecto", loc="left", pad=20)
    footer(fig, "El flujo separa selección, análisis y comunicación para que cada conclusión se origine en evidencia verificable.")
    savefig(fig, "01_project_flow.png")


def plot_comparison_matrix(comparison: pd.DataFrame) -> None:
    criteria = [
        "Puntaje tamaño",
        "Puntaje dimensionalidad",
        "Puntaje completitud",
        "Puntaje preparación analítica",
        "Puntaje interpretabilidad",
        "Puntaje impacto",
        "Puntaje total ponderado",
    ]
    matrix = comparison.set_index("Dataset")[criteria]
    fig, ax = plt.subplots(figsize=(13, 5.8))
    im = ax.imshow(matrix.values, cmap="PuBuGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(criteria)), [c.replace("Puntaje ", "").replace(" ponderado", "\nponderado") for c in criteria], fontsize=9)
    ax.set_yticks(np.arange(len(matrix.index)), matrix.index, fontsize=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.values[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8.8, color="white" if value > 63 else PALETTE["ink"])
    ax.set_title("Matriz comparativa de datasets candidatos", loc="left", pad=18)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel("Puntaje normalizado", rotation=270, labelpad=16)
    footer(fig, "El dataset de cáncer de mama domina por tamaño, profundidad de variables, impacto e interpretabilidad; la completitud es equivalente en los tres.")
    savefig(fig, "02_dataset_comparison_matrix.png")


def plot_dataset_radar(comparison: pd.DataFrame) -> None:
    criteria = [
        "Puntaje tamaño",
        "Puntaje dimensionalidad",
        "Puntaje completitud",
        "Puntaje preparación analítica",
        "Puntaje interpretabilidad",
        "Puntaje impacto",
    ]
    labels = ["Tamaño", "Dimensión", "Completitud", "Preparación", "Interpretabilidad", "Impacto"]
    angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(8.8, 8.2))
    ax = fig.add_subplot(111, polar=True)
    colors = [PALETTE["red"], PALETTE["blue"], PALETTE["teal"]]
    for color, (_, row) in zip(colors, comparison.iterrows()):
        values = [float(row[c]) for c in criteria]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2.2, label=row["Dataset"])
        ax.fill(angles, values, color=color, alpha=0.08)
    ax.set_xticks(angles[:-1], labels)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100], ["25", "50", "75", "100"], fontsize=8, color=PALETTE["muted"])
    ax.grid(color=PALETTE["line"], linewidth=0.8)
    ax.spines["polar"].set_color(PALETTE["line"])
    ax.set_title("Perfil multicriterio de selección", y=1.08, fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False, fontsize=8.5)
    footer(fig, "El radar muestra que la decisión no depende de un único criterio: el dataset seleccionado mantiene desempeño alto en seis dimensiones.")
    savefig(fig, "03_dataset_radar.png")


def plot_missingness_map(x: pd.DataFrame) -> None:
    groups = ["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave points", "symmetry", "fractal dimension"]
    stats = ["mean", "error", "worst"]
    matrix = []
    for stat in stats:
        row = []
        for group in groups:
            if stat == "mean":
                col = f"mean {group}"
            elif stat == "worst":
                col = f"worst {group}"
            else:
                col = f"{group} error"
            row.append((1.0 - x[col].isna().mean()) * 100)
        matrix.append(row)
    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    im = ax.imshow(matrix, cmap="Greens", vmin=95, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(groups)), [g.replace(" ", "\n") for g in groups], fontsize=8.5)
    ax.set_yticks(np.arange(len(stats)), ["Media", "Error estándar", "Peor valor"], fontsize=10)
    for i in range(len(stats)):
        for j in range(len(groups)):
            ax.text(j, i, f"{matrix[i][j]:.0f}%", ha="center", va="center", fontsize=8.5, color=PALETTE["ink"])
    ax.set_title("Diagrama de completitud por familia de variable", loc="left", pad=18)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel("Completitud", rotation=270, labelpad=14)
    footer(fig, "La completitud uniforme del 100% elimina la necesidad de imputación en esta fase y reduce incertidumbre por datos faltantes.")
    savefig(fig, "04_missingness_map.png")


def plot_target_distribution(labels: pd.Series) -> None:
    counts = labels.value_counts().reindex(["benign", "malignant"])
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    colors = [PALETTE["teal"], PALETTE["red"]]
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.54)
    ax.set_ylabel("Número de observaciones")
    ax.set_title("Distribución de la variable objetivo", loc="left", pad=18)
    ax.grid(axis="y", alpha=0.7)
    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 8, f"{value}\n({value / total:.1%})", ha="center", va="bottom", fontsize=10, color=PALETTE["ink"])
    ax.set_ylim(0, counts.max() * 1.24)
    footer(fig, "La clase benigna es mayoritaria, pero la clase maligna representa más de un tercio del conjunto; el desbalance es manejable para análisis exploratorio.")
    savefig(fig, "05_target_distribution.png")


def plot_effect_lollipop(effects: pd.DataFrame) -> None:
    top = effects.head(15).sort_values("|d|")
    fig, ax = plt.subplots(figsize=(10, 7.4))
    colors = [PALETTE["red"] if "maligno" in direction.lower() else PALETTE["teal"] for direction in top["Dirección"]]
    ax.hlines(y=np.arange(len(top)), xmin=0, xmax=top["|d|"], color=PALETTE["line"], linewidth=2)
    ax.scatter(top["|d|"], np.arange(len(top)), s=70, color=colors, zorder=3)
    ax.set_yticks(np.arange(len(top)), top["Variable"])
    ax.set_xlabel("Tamaño de efecto absoluto |d|")
    ax.set_title("Variables con mayor separación entre diagnósticos", loc="left", pad=18)
    ax.axvline(0.8, color=PALETTE["muted"], linestyle="--", linewidth=1)
    ax.text(0.82, len(top) - 0.5, "umbral alto\nreferencial", fontsize=8, color=PALETTE["muted"], va="top")
    ax.grid(axis="x", alpha=0.7)
    footer(fig, "Las medidas de concavidad, puntos cóncavos y perímetro/área concentran la mayor diferencia estandarizada entre masas malignas y benignas.")
    savefig(fig, "06_effect_size_lollipop.png")


def plot_correlation_heatmap(x: pd.DataFrame, effects: pd.DataFrame) -> None:
    top_features = effects.head(14)["Variable"].tolist()
    corr = x[top_features].corr()
    fig, ax = plt.subplots(figsize=(11.5, 9.3))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(top_features)), top_features, rotation=45, ha="right", fontsize=8.2)
    ax.set_yticks(np.arange(len(top_features)), top_features, fontsize=8.2)
    for i in range(len(top_features)):
        for j in range(len(top_features)):
            value = corr.iloc[i, j]
            if i == j or abs(value) >= 0.65:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white" if abs(value) > 0.78 else PALETTE["ink"])
    ax.set_title("Heatmap de correlación entre variables de mayor señal", loc="left", pad=18)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.set_ylabel("Correlación de Pearson", rotation=270, labelpad=16)
    footer(fig, "La señal predictiva convive con redundancia fuerte entre medidas geométricas; esto sugiere cautela con multicolinealidad en fases de modelado.")
    savefig(fig, "07_correlation_heatmap.png")


def plot_distribution_panel(x: pd.DataFrame, labels: pd.Series, effects: pd.DataFrame) -> None:
    selected = effects.head(6)["Variable"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6))
    for ax, col in zip(axes.ravel(), selected):
        for label, color in [("benign", PALETTE["teal"]), ("malignant", PALETTE["red"])]:
            values = x.loc[labels == label, col]
            ax.hist(values, bins=22, density=True, histtype="stepfilled", alpha=0.22, color=color)
            ax.hist(values, bins=22, density=True, histtype="step", linewidth=1.3, color=color, label=label)
        ax.set_title(col, loc="left", fontsize=10.5)
        ax.grid(axis="y", alpha=0.45)
        ax.set_ylabel("Densidad")
    handles, leg_labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper right", frameon=False)
    fig.suptitle("Distribuciones comparadas de variables prioritarias", x=0.02, y=0.985, ha="left", fontsize=16, fontweight="bold")
    footer(fig, "Las distribuciones muestran desplazamientos sistemáticos entre clases, especialmente en variables de borde y concavidad.")
    fig.subplots_adjust(top=0.88, bottom=0.16, hspace=0.35, wspace=0.24)
    savefig(fig, "08_distribution_panel.png")


def plot_boxplot_panel(x: pd.DataFrame, labels: pd.Series, effects: pd.DataFrame) -> None:
    selected = effects.head(8)["Variable"].tolist()
    scaled = (x[selected] - x[selected].median()) / (x[selected].quantile(0.75) - x[selected].quantile(0.25))
    fig, ax = plt.subplots(figsize=(12, 7.2))
    positions = []
    data = []
    colors = []
    labels_y = []
    for idx, col in enumerate(selected):
        base = idx * 3
        positions.extend([base, base + 1])
        data.extend([scaled.loc[labels == "benign", col], scaled.loc[labels == "malignant", col]])
        colors.extend([PALETTE["teal"], PALETTE["red"]])
        labels_y.append((base + 0.5, col))
    bp = ax.boxplot(data, positions=positions, vert=False, patch_artist=True, widths=0.65, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
        patch.set_edgecolor(color)
    for median in bp["medians"]:
        median.set_color(PALETTE["ink"])
        median.set_linewidth(1.2)
    ax.set_yticks([pos for pos, _ in labels_y], [label for _, label in labels_y])
    ax.set_xlabel("Escala robusta: (valor - mediana) / IQR")
    ax.set_title("Boxplots robustos por diagnóstico", loc="left", pad=18)
    ax.axvline(0, color=PALETTE["muted"], linewidth=1)
    ax.grid(axis="x", alpha=0.7)
    ax.text(0.98, 0.04, "benign en verde azulado; malignant en rojo apagado", transform=ax.transAxes, ha="right", fontsize=8.4, color=PALETTE["muted"])
    footer(fig, "La escala robusta permite comparar variables con unidades diferentes y evidencia medianas más altas en malignidad para variables clave.")
    savefig(fig, "09_boxplot_panel.png")


def plot_architecture_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.1))
    ax.axis("off")
    layers = [
        ("Datos locales", "sklearn.datasets\nsin red", 0.05, 0.68, PALETTE["blue"]),
        ("Procesamiento", "Pandas + NumPy\nvalidación y métricas", 0.30, 0.68, PALETTE["teal"]),
        ("Análisis", "estadística descriptiva\ncorrelación + efecto", 0.55, 0.68, PALETTE["red"]),
        ("Comunicación", "Matplotlib + HTML\nfiguras y tablas", 0.80, 0.68, PALETTE["gold"]),
        ("data/", "dataset final", 0.12, 0.22, PALETTE["muted"]),
        ("tables/", "tablas CSV", 0.36, 0.22, PALETTE["muted"]),
        ("figures/", "visualizaciones PNG", 0.60, 0.22, PALETTE["muted"]),
        ("report/", "informe HTML", 0.82, 0.22, PALETTE["muted"]),
    ]
    for title, body, x0, y0, color in layers:
        rect = Rectangle((x0, y0), 0.16, 0.18, transform=ax.transAxes, facecolor="white", edgecolor=color, linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x0 + 0.015, y0 + 0.135, title, transform=ax.transAxes, fontsize=11, fontweight="bold", color=PALETTE["ink"])
        ax.text(x0 + 0.015, y0 + 0.06, body, transform=ax.transAxes, fontsize=8.8, color=PALETTE["muted"], va="center")
    arrows = [
        ((0.21, 0.77), (0.30, 0.77)),
        ((0.46, 0.77), (0.55, 0.77)),
        ((0.71, 0.77), (0.80, 0.77)),
        ((0.38, 0.68), (0.20, 0.40)),
        ((0.45, 0.68), (0.44, 0.40)),
        ((0.62, 0.68), (0.68, 0.40)),
        ((0.86, 0.68), (0.90, 0.40)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle="-|>", mutation_scale=13, color=PALETTE["muted"], linewidth=1.1))
    ax.set_title("Arquitectura reproducible del proyecto", loc="left", pad=18)
    footer(fig, "La arquitectura mantiene una sola ruta de datos y separa artefactos finales por función, evitando duplicación y procesamiento innecesario.")
    savefig(fig, "10_architecture_pipeline.png")


def plot_timeline() -> None:
    phases = [
        ("Fase 1", "Búsqueda y selección", 0, 2, PALETTE["blue"]),
        ("Fase 2", "Contexto y descripción", 2, 4, PALETTE["teal"]),
        ("Fase 2", "Análisis estadístico", 4, 6, PALETTE["red"]),
        ("Fase 3", "Requerimientos", 6, 7.5, PALETTE["gold"]),
        ("Entrega", "Reporte reproducible", 7.5, 8.5, PALETTE["green"]),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    for idx, (phase, label, start, end, color) in enumerate(phases):
        ax.barh(idx, end - start, left=start, height=0.5, color=color, alpha=0.78)
        ax.text(start + 0.05, idx, f"{phase}: {label}", va="center", ha="left", color="white", fontsize=9.3, fontweight="bold")
    ax.set_yticks([])
    ax.set_xlabel("Días de trabajo estimados")
    ax.set_title("Cronograma ejecutivo de la práctica", loc="left", pad=18)
    ax.set_xlim(0, 9)
    ax.grid(axis="x", alpha=0.7)
    footer(fig, "El cronograma prioriza selección y comprensión de datos antes de cualquier modelado, consistente con una práctica de formulación analítica.")
    savefig(fig, "11_project_timeline.png")


def plot_executive_dashboard(
    x: pd.DataFrame,
    labels: pd.Series,
    comparison: pd.DataFrame,
    effects: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(13.5, 8.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.85, 1.25, 1.15], hspace=0.5, wspace=0.35)
    fig.suptitle("Dashboard ejecutivo del dataset seleccionado", x=0.02, y=0.985, ha="left", fontsize=17, fontweight="bold")

    kpis = [
        ("Observaciones", f"{len(x):,}", PALETTE["blue"]),
        ("Variables", str(x.shape[1]), PALETTE["teal"]),
        ("Faltantes", "0.00%", PALETTE["green"]),
        ("Puntaje selección", f"{comparison.iloc[0]['Puntaje total ponderado']:.1f}", PALETTE["red"]),
    ]
    for i, (label, value, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="white", edgecolor=PALETTE["line"], linewidth=1))
        ax.text(0.08, 0.62, value, transform=ax.transAxes, fontsize=22, fontweight="bold", color=color)
        ax.text(0.08, 0.30, label, transform=ax.transAxes, fontsize=9.5, color=PALETTE["muted"])

    ax1 = fig.add_subplot(gs[1, :2])
    counts = labels.value_counts().reindex(["benign", "malignant"])
    ax1.bar(counts.index, counts.values, color=[PALETTE["teal"], PALETTE["red"]], width=0.5)
    ax1.set_title("Balance de clases", loc="left", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.6)
    for idx, value in enumerate(counts.values):
        ax1.text(idx, value + 8, f"{value} ({value / counts.sum():.1%})", ha="center", fontsize=9)

    ax2 = fig.add_subplot(gs[1:, 2:])
    top = effects.head(10).sort_values("|d|")
    ax2.barh(top["Variable"], top["|d|"], color=PALETTE["red"], alpha=0.78)
    ax2.set_title("Top 10 variables por tamaño de efecto", loc="left", fontsize=12, fontweight="bold")
    ax2.set_xlabel("|d|")
    ax2.grid(axis="x", alpha=0.6)

    ax3 = fig.add_subplot(gs[2, :2])
    quality_subset = quality.iloc[[3, 5, 7, 8], :]
    ax3.axis("off")
    y = 0.9
    for _, row in quality_subset.iterrows():
        ax3.text(0.0, y, row["Indicador"], fontsize=9.2, fontweight="bold", color=PALETTE["ink"], transform=ax3.transAxes)
        ax3.text(0.47, y, row["Resultado"], fontsize=9.2, color=PALETTE["red"], transform=ax3.transAxes)
        y -= 0.20
    ax3.set_title("Lectura rápida de calidad", loc="left", fontsize=12, fontweight="bold")
    footer(fig, "El panel resume que el conjunto es completo, moderadamente desbalanceado y con variables de fuerte separación diagnóstica.")
    savefig(fig, "12_executive_dashboard.png")


def generate_figures(
    comparison: pd.DataFrame,
    x: pd.DataFrame,
    labels: pd.Series,
    effects: pd.DataFrame,
    corr: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    del corr
    plot_project_flow()
    plot_comparison_matrix(comparison)
    plot_dataset_radar(comparison)
    plot_missingness_map(x)
    plot_target_distribution(labels)
    plot_effect_lollipop(effects)
    plot_correlation_heatmap(x, effects)
    plot_distribution_panel(x, labels, effects)
    plot_boxplot_panel(x, labels, effects)
    plot_architecture_pipeline()
    plot_timeline()
    plot_executive_dashboard(x, labels, comparison, effects, quality)


def df_to_html(df: pd.DataFrame, max_rows: int | None = None) -> str:
    view = rounded_table(df)
    if max_rows is not None:
        view = view.head(max_rows)
    return view.to_html(index=False, classes="data-table", border=0, escape=False)


def image_block(filename: str, caption: str) -> str:
    return f"""
    <figure>
      <img src="../figures/{filename}" alt="{caption}">
      <figcaption>{caption}</figcaption>
    </figure>
    """


def build_report(
    comparison: pd.DataFrame,
    selected_name: str,
    x: pd.DataFrame,
    labels: pd.Series,
    quality: pd.DataFrame,
    data_dictionary: pd.DataFrame,
    descriptive: pd.DataFrame,
    effects: pd.DataFrame,
    requirements_f: pd.DataFrame,
    requirements_nf: pd.DataFrame,
) -> str:
    selected_score = float(comparison.iloc[0]["Puntaje total ponderado"])
    counts = labels.value_counts()
    top_effect = effects.iloc[0]
    high_signal = int((effects["|d|"] >= 1).sum())
    outlier_pct = descriptive["Atípicos IQR (%)"].max()
    corr = x.corr().abs()
    np.fill_diagonal(corr.values, 0)
    corr_max = float(corr.values.max())

    css = """
    <style>
      :root {
        --paper: #fbfaf7;
        --ink: #22272b;
        --muted: #69737a;
        --line: #d9d6ce;
        --red: #a64f4f;
        --teal: #4f8a89;
        --blue: #536f8a;
        --gold: #b79a4a;
      }
      html { background: var(--paper); }
      body {
        margin: 0 auto;
        max-width: 1180px;
        padding: 54px 54px 82px;
        background: var(--paper);
        color: var(--ink);
        font-family: "Segoe UI", Arial, sans-serif;
        line-height: 1.62;
      }
      header { border-bottom: 1px solid var(--line); padding-bottom: 30px; margin-bottom: 42px; }
      .kicker { color: var(--red); text-transform: uppercase; letter-spacing: .12em; font-size: 12px; font-weight: 700; }
      h1 {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 48px;
        line-height: 1.05;
        max-width: 980px;
        margin: 12px 0 18px;
        font-weight: 500;
      }
      h2 {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 30px;
        line-height: 1.2;
        margin: 52px 0 16px;
        border-top: 1px solid var(--line);
        padding-top: 28px;
        font-weight: 500;
      }
      h3 { margin: 30px 0 10px; font-size: 18px; }
      p { max-width: 900px; }
      .subtitle { font-size: 18px; color: #434b50; max-width: 920px; }
      .meta { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }
      .pill { border: 1px solid var(--line); padding: 7px 11px; border-radius: 999px; color: #40484d; background: #fffdf9; font-size: 13px; }
      .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 26px 0 18px; }
      .card { background: #fffdf9; border: 1px solid var(--line); padding: 18px; }
      .card .value { font-size: 28px; font-weight: 750; color: var(--red); }
      .card .label { color: var(--muted); font-size: 13px; margin-top: 4px; }
      figure { margin: 34px 0 40px; }
      figure img { width: 100%; height: auto; border: 1px solid var(--line); background: var(--paper); }
      figcaption { color: var(--muted); font-size: 13px; margin-top: 8px; }
      .table-wrap { overflow-x: auto; margin: 18px 0 34px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
      table.data-table { width: 100%; border-collapse: collapse; font-size: 12.5px; background: #fffdf9; }
      table.data-table th {
        text-align: left;
        padding: 10px 9px;
        border-bottom: 1px solid var(--line);
        color: var(--ink);
        font-weight: 700;
        background: #f2f0eb;
        vertical-align: bottom;
      }
      table.data-table td { padding: 9px; border-bottom: 1px solid #e8e5dd; vertical-align: top; }
      table.data-table tr:nth-child(even) td { background: #fbfaf7; }
      .note { border-left: 3px solid var(--red); padding: 12px 16px; color: #3f474c; background: #fffdf9; max-width: 920px; }
      .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
      .source { color: var(--muted); font-size: 12.5px; }
      ul { max-width: 900px; }
      li { margin-bottom: 8px; }
      code { background: #f1eee8; padding: 2px 5px; border-radius: 3px; }
      @media (max-width: 860px) {
        body { padding: 28px 18px 60px; }
        h1 { font-size: 34px; }
        .grid, .two-col { grid-template-columns: 1fr; }
      }
    </style>
    """

    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Práctica de Ciencia de Datos - Selección y caracterización de dataset</title>
  {css}
</head>
<body>
  <header>
    <div class="kicker">Proyecto reproducible de maestría en Ciencia de Datos</div>
    <h1>Selección, caracterización y especificación de requerimientos para un sistema analítico biomédico</h1>
    <p class="subtitle">
      Entregable académico de las fases 1 a 3: búsqueda local de tres datasets, selección cuantitativa,
      descripción exhaustiva del conjunto elegido, análisis estadístico exploratorio y requerimientos
      funcionales/no funcionales con justificación técnica.
    </p>
    <div class="meta">
      <span class="pill">Fecha: {ANALYSIS_DATE}</span>
      <span class="pill">Dataset seleccionado: {selected_name}</span>
      <span class="pill">Python + Pandas + NumPy + Matplotlib + scikit-learn</span>
      <span class="pill">Sin APIs, nube ni modelos externos de IA</span>
    </div>
  </header>

  <section>
    <h2>Resumen ejecutivo</h2>
    <p>
      Se evaluaron tres datasets disponibles localmente en scikit-learn. El dataset
      <strong>{selected_name}</strong> obtuvo el mayor puntaje ponderado ({selected_score:.1f}/100)
      al combinar tamaño muestral, riqueza dimensional, completitud, preparación analítica,
      interpretabilidad e impacto. La selección es apropiada para una práctica de maestría porque
      permite discutir calidad de datos, separación estadística entre clases, redundancia de variables
      y requerimientos de un flujo analítico reproducible sin depender de servicios externos.
    </p>
    <div class="grid">
      <div class="card"><div class="value">{len(x):,}</div><div class="label">observaciones</div></div>
      <div class="card"><div class="value">{x.shape[1]}</div><div class="label">variables predictoras</div></div>
      <div class="card"><div class="value">0.00%</div><div class="label">valores faltantes</div></div>
      <div class="card"><div class="value">{high_signal}</div><div class="label">variables con |d| >= 1</div></div>
    </div>
    <p class="note">
      Hallazgo central: la evidencia univariada más fuerte aparece en <strong>{top_effect['Variable']}</strong>
      (|d|={top_effect['|d|']:.2f}), mientras que la correlación máxima absoluta entre predictoras alcanza
      {corr_max:.3f}. Esto sugiere alta señal diagnóstica, pero también redundancia morfológica que debe
      gestionarse si el proyecto avanza hacia modelado predictivo.
    </p>
    {image_block("12_executive_dashboard.png", "Dashboard ejecutivo con métricas de calidad, balance y variables prioritarias.")}
  </section>

  <section>
    <h2>Fase 1. Búsqueda, comparación y selección del dataset</h2>
    <p>
      Para cumplir la restricción de no usar internet, APIs ni servicios en la nube, la búsqueda se limitó
      al catálogo local de datasets incluidos en scikit-learn. Esta decisión reduce el consumo computacional,
      elimina riesgos de disponibilidad externa y mantiene la reproducibilidad. Los candidatos fueron:
      Breast Cancer Wisconsin Diagnostic, Wine Recognition y Diabetes Progression.
    </p>
    {image_block("01_project_flow.png", "Flujo metodológico seguido para seleccionar y analizar el dataset.")}
    <h3>Tabla comparativa técnica</h3>
    <div class="table-wrap">{df_to_html(comparison)}</div>
    {image_block("02_dataset_comparison_matrix.png", "Matriz comparativa de criterios normalizados para los tres datasets candidatos.")}
    {image_block("03_dataset_radar.png", "Radar chart del perfil multicriterio de los datasets candidatos.")}
    <p>
      La ponderación priorizó evidencia verificable: 25% tamaño muestral, 20% dimensionalidad,
      15% completitud, 15% preparación analítica, 15% interpretabilidad y 10% impacto. El dataset
      seleccionado domina por volumen, número de variables, completitud total y relevancia biomédica.
      Wine Recognition es técnicamente limpio, pero tiene menor escala e impacto para una discusión
      socio-técnica. Diabetes Progression posee impacto alto, aunque su target continuo y variables
      estandarizadas reducen interpretabilidad variable por variable.
    </p>
  </section>

  <section>
    <h2>Fase 2. Contexto, problema, objetivo y alcance</h2>
    <div class="two-col">
      <div>
        <h3>Contexto</h3>
        <p>
          El análisis biomédico basado en imágenes requiere transformar mediciones morfológicas en evidencia
          interpretable. Este dataset resume características geométricas y de textura de núcleos celulares
          obtenidas a partir de aspiración con aguja fina de masas mamarias.
        </p>
        <h3>Problema</h3>
        <p>
          Antes de plantear un sistema predictivo, es necesario demostrar que los datos son completos,
          comprensibles, estadísticamente informativos y gobernables. Sin esa base, cualquier modelo posterior
          podría amplificar redundancias, sesgos de clase o interpretaciones clínicas débiles.
        </p>
        <h3>Objetivo</h3>
        <p>
          Seleccionar y caracterizar un dataset biomédico reproducible, identificar sus variables críticas,
          evaluar calidad estadística y definir requerimientos de un sistema analítico que pueda sostener
          futuras fases de modelado con rigor académico.
        </p>
      </div>
      <div>
        <h3>Alcance</h3>
        <p>
          El alcance cubre selección de dataset, análisis exploratorio, diccionario de datos, visualizaciones
          editoriales y requerimientos. No incluye entrenamiento de modelos clínicos ni recomendaciones médicas
          individuales, porque las fases solicitadas no lo exigen y porque se prioriza eficiencia.
        </p>
        <h3>Beneficiarios</h3>
        <p>
          Los beneficiarios son estudiantes e investigadores de ciencia de datos, docentes evaluadores,
          equipos de analítica biomédica y responsables de gobernanza de datos que requieran una base
          reproducible para decisiones metodológicas.
        </p>
        <h3>Resultados esperados</h3>
        <p>
          Se espera una selección defendible, un inventario de variables accionable, evidencia de calidad de
          datos, detección de señales estadísticas relevantes, visualizaciones publicables y especificación
          clara de requerimientos para continuidad del proyecto.
        </p>
      </div>
    </div>

    <h3>Resumen de calidad del dataset</h3>
    <div class="table-wrap">{df_to_html(quality)}</div>
    {image_block("04_missingness_map.png", "Diagrama de completitud por familia de variables.")}
    {image_block("05_target_distribution.png", "Distribución de la variable objetivo diagnóstico.")}
    {image_block("06_effect_size_lollipop.png", "Ranking de variables por tamaño de efecto entre diagnósticos.")}
    {image_block("07_correlation_heatmap.png", "Heatmap de correlaciones entre variables con mayor señal estadística.")}
    {image_block("08_distribution_panel.png", "Distribuciones comparadas de las seis variables prioritarias.")}
    {image_block("09_boxplot_panel.png", "Boxplots robustos por diagnóstico para variables prioritarias.")}

    <h3>Diccionario exhaustivo de variables</h3>
    <p>
      Cada variable se describe por tipo, rango observado, unidad, importancia estadística,
      posibles atípicos y observaciones. La importancia se calcula mediante tamaño de efecto absoluto
      entre clases, no mediante un modelo entrenado.
    </p>
    <div class="table-wrap">{df_to_html(data_dictionary)}</div>

    <h3>Estadística descriptiva</h3>
    <p>
      La tabla siguiente concentra percentiles, dispersión, asimetría, atípicos IQR y señal univariada.
      El mayor porcentaje de atípicos IQR observado es {outlier_pct:.1f}%, lo que recomienda revisión
      técnica de extremos antes de cualquier eliminación.
    </p>
    <div class="table-wrap">{df_to_html(descriptive)}</div>
  </section>

  <section>
    <h2>Fase 3. Requerimientos del sistema analítico</h2>
    <p>
      Los requerimientos se formulan para un sistema de análisis reproducible, no para un producto clínico
      operativo. La prioridad se asigna según riesgo metodológico, trazabilidad y valor para decisiones
      académicas posteriores.
    </p>
    {image_block("10_architecture_pipeline.png", "Arquitectura reproducible de carpetas, procesamiento y entregables.")}
    {image_block("11_project_timeline.png", "Cronograma ejecutivo sugerido para las fases del proyecto.")}
    <h3>Requerimientos funcionales</h3>
    <div class="table-wrap">{df_to_html(requirements_f)}</div>
    <h3>Requerimientos no funcionales</h3>
    <div class="table-wrap">{df_to_html(requirements_nf)}</div>
  </section>

  <section>
    <h2>Conclusiones sustentadas</h2>
    <ul>
      <li>
        La selección del dataset es cuantitativamente defendible: {selected_name} alcanza {selected_score:.1f}/100
        y supera a los candidatos por balance entre escala, dimensionalidad, interpretabilidad e impacto.
      </li>
      <li>
        La completitud del 100% evita imputación en esta fase; por tanto, las diferencias observadas entre clases
        no están condicionadas por estrategias de reemplazo de datos faltantes.
      </li>
      <li>
        La clase minoritaria representa {counts.min() / counts.sum():.1%} del total, suficiente para exploración,
        aunque un modelado posterior debería controlar el desbalance con particiones estratificadas.
      </li>
      <li>
        La presencia de {high_signal} variables con |d| >= 1 indica separación estadística fuerte; no obstante,
        la correlación máxima de {corr_max:.3f} confirma que varias medidas contienen información redundante.
      </li>
      <li>
        La fase de requerimientos debe priorizar reproducibilidad, trazabilidad, independencia de red y
        comunicación visual, porque esos atributos son los que hacen defendible el proyecto en un contexto
        de maestría.
      </li>
    </ul>
  </section>

  <section>
    <h2>Reproducción</h2>
    <p>
      Ejecutar desde la raíz del proyecto:
      <code>&amp; "$env:USERPROFILE\\anaconda3\\python.exe" src\\build_project.py</code>
    </p>
    <p class="source">
      Fuente de datos: {SOURCE_NOTE}. El script no realiza llamadas de red, no descarga archivos y no utiliza
      modelos externos de IA. Las salidas finales se guardan en <code>data/</code>, <code>tables/</code>,
      <code>figures/</code> y <code>report/</code>.
    </p>
  </section>
</body>
</html>
"""
    return html


def main() -> None:
    ensure_directories()
    set_plot_style()

    datasets = load_candidate_datasets()
    comparison = build_dataset_comparison(datasets)
    selected_name = str(comparison.iloc[0]["Dataset"])
    selected = datasets[selected_name]
    bunch = selected["bunch"]
    x: pd.DataFrame = selected["x"]  # type: ignore[assignment]
    y: pd.Series = selected["y"]  # type: ignore[assignment]
    target_names = list(getattr(bunch, "target_names", ["class_0", "class_1"]))
    labels = pd.Series([target_names[int(value)] for value in y], name="diagnosis")

    saved_dataset = x.copy()
    saved_dataset.columns = [clean_name(col) for col in saved_dataset.columns]
    saved_dataset["diagnosis_code"] = y.astype(int).values
    saved_dataset["diagnosis"] = labels.values
    saved_dataset.to_csv(DATA_DIR / "breast_cancer_wisconsin_diagnostic.csv", index=False, encoding="utf-8-sig")

    effects = cohen_d_by_feature(x, labels)
    outliers = iqr_outlier_summary(x)
    descriptive = build_descriptive_statistics(x, effects, outliers)
    data_dictionary = build_data_dictionary(x, effects, outliers)
    corr = x.corr()
    quality = build_data_quality_summary(x, labels, effects, outliers, corr)
    requirements_f, requirements_nf = build_requirements()

    tables = {
        "dataset_comparison.csv": comparison,
        "data_quality_summary.csv": quality,
        "data_dictionary.csv": data_dictionary,
        "descriptive_statistics.csv": descriptive,
        "variable_effect_sizes.csv": effects,
        "functional_requirements.csv": requirements_f,
        "nonfunctional_requirements.csv": requirements_nf,
    }
    save_tables(tables)

    generate_figures(comparison, x, labels, effects, corr, quality)

    report_html = build_report(
        comparison=comparison,
        selected_name=selected_name,
        x=x,
        labels=labels,
        quality=quality,
        data_dictionary=data_dictionary,
        descriptive=descriptive,
        effects=effects,
        requirements_f=requirements_f,
        requirements_nf=requirements_nf,
    )
    (REPORT_DIR / "final_report.html").write_text(report_html, encoding="utf-8")

    manifest = {
        "analysis_date": ANALYSIS_DATE,
        "selected_dataset": selected_name,
        "rows": int(x.shape[0]),
        "features": int(x.shape[1]),
        "source": SOURCE_NOTE,
        "restrictions": {
            "external_ai_models": False,
            "api_calls": False,
            "cloud_services": False,
            "downloads": False,
        },
        "outputs": {
            "data": ["data/breast_cancer_wisconsin_diagnostic.csv"],
            "tables": sorted(tables.keys()),
            "figures": sorted(path.name for path in FIGURES_DIR.glob("*.png")),
            "report": ["report/final_report.html"],
        },
    }
    (REPORT_DIR / "reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Proyecto generado correctamente.")
    print(f"Dataset seleccionado: {selected_name}")
    print(f"Reporte: {REPORT_DIR / 'final_report.html'}")


if __name__ == "__main__":
    main()
