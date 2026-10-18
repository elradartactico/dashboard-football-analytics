# =============================================================================
#  utils/data.py — Carga y pre-procesamiento de datos
#  No modificar. El alumno solo cambia config.py.
# =============================================================================

from pathlib import Path
import json
import os
import re
import subprocess
import sys
import threading
import numpy as np
import pandas as pd
import streamlit as st

# Ruta a la carpeta data/ relativa a la raiz del proyecto (donde esta app.py)
_DATA = Path(__file__).parent.parent / "data"

_PARTICULAS = {"de", "del", "di", "da", "dos", "van", "von", "mac", "al", "el", "la", "le", "ter", "den", "der"}


def nombre_corto(nombre: str) -> str:
    """Apellido para etiquetas de graficos: 'Lionel Messi' -> 'Messi',
    'Rodrigo De Paul' -> 'De Paul', 'Alexis Mac Allister' -> 'Mac Allister'."""
    partes = str(nombre).split()
    if len(partes) <= 1:
        return str(nombre)
    if len(partes) >= 3 and partes[-2].lower() in _PARTICULAS:
        return " ".join(partes[-2:])
    return partes[-1]


# Orden de fases para torneos de copa (World Cup, Euros, AFCON, etc.)
STAGE_ORDER: dict[str, int] = {
    "Group Stage": 1,
    "Round of 16": 2,
    "Quarter-finals": 3,
    "Semi-finals": 4,
    "3rd Place Final": 5,
    "Final": 6,
    # Los mismos nombres ya traducidos (asi se muestran en el dashboard)
    "Fase de grupos": 1,
    "Octavos de final": 2,
    "Cuartos de final": 3,
    "Semifinales": 4,
    "Tercer puesto": 5,
}

ETAPAS_ES: dict[str, str] = {
    "Group Stage": "Fase de grupos",
    "Round of 16": "Octavos de final",
    "Quarter-finals": "Cuartos de final",
    "Semi-finals": "Semifinales",
    "3rd Place Final": "Tercer puesto",
    "Final": "Final",
    "Regular Season": "Temporada regular",
}


_ARCHIVOS = ["matches.parquet", "events.parquet", "lineups.parquet", "players.parquet", "teams.parquet"]
_CANDADO = threading.Lock()  # evita dos descargas a la vez si se abren varias paginas


def _torneo_configurado() -> dict | None:
    """Lee del BLOQUE 2 de prepare_data.py que torneo quiere el alumno."""
    script = _DATA.parent / "prepare_data.py"
    if not script.exists():
        return None
    texto = script.read_text(encoding="utf-8")
    valores = {}
    for clave, patron in [("competition_id", r"^COMPETITION_ID\s*:\s*int\s*=\s*(\d+)"),
                          ("season_id", r"^SEASON_ID\s*:\s*int\s*=\s*(\d+)"),
                          ("datos_completos", r"^DATOS_COMPLETOS\s*:\s*bool\s*=\s*(True|False)"),
                          ("modo_prueba", r"^MODO_PRUEBA\s*:\s*bool\s*=\s*(True|False)")]:
        m = re.search(patron, texto, flags=re.MULTILINE)
        if not m:
            return None
        valores[clave] = int(m.group(1)) if m.group(1).isdigit() else m.group(1) == "True"
    m = re.search(r"^VERSION_DATOS\s*:\s*int\s*=\s*(\d+)", texto, flags=re.MULTILINE)
    if m:
        valores["version"] = int(m.group(1))
    return valores


def _pantalla_de_espera():
    """Mientras se descargan los datos se muestra el logo y el nombre del
    analista, en vez del menu en blanco de Streamlit."""
    try:
        from config import NOMBRE_ANALISTA, TITULO
    except Exception:
        NOMBRE_ANALISTA, TITULO = "", "Dashboard"
    try:  # colores propios: asi un logo blanco no queda sobre fondo blanco
        from config import COLOR_FONDO, COLOR_TEXTO
    except Exception:
        COLOR_FONDO, COLOR_TEXTO = "#FFFFFF", "#0D0D0D"
    hueco = st.empty()
    with hueco.container():
        st.markdown("<style>[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'], "
                    "[data-testid='collapsedControl'] {display: none !important;}"
                    f".stApp, [data-testid='stAppViewContainer'], [data-testid='stHeader'] "
                    f"{{background: {COLOR_FONDO} !important;}}"
                    f".stApp, .stApp p, .stApp h2, [data-testid='stSpinner'] * {{color: {COLOR_TEXTO} !important;}}"
                    "</style>",
                    unsafe_allow_html=True)
        _, centro, _ = st.columns([2, 1, 2])
        logo = _DATA.parent / "assets" / "logo.png"
        if logo.exists():
            centro.image(str(logo), use_container_width=True)
        st.markdown(f"<h2 style='text-align:center; margin-bottom:0'>{TITULO}</h2>"
                    f"<p style='text-align:center; opacity:0.7'>{NOMBRE_ANALISTA}</p>",
                    unsafe_allow_html=True)
    return hueco


def _minutos_jugados(lineups: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """Minutos de cada jugador en el torneo, a partir de las alineaciones."""
    if not {"player_name", "positions", "match_id"} <= set(lineups.columns):
        return pd.Series(dtype=float)
    fin = (events.groupby("match_id")["minute"].max() + 1).to_dict() if "minute" in events.columns else {}

    def _a_min(reloj) -> float:
        try:
            mm, ss = str(reloj).split(":")[:2]
            return int(mm) + int(ss) / 60
        except Exception:
            return np.nan

    def _de_fila(fila) -> float:
        pos = fila["positions"]
        try:
            pos = json.loads(pos) if isinstance(pos, str) else pos
        except ValueError:
            return 0.0
        if not isinstance(pos, (list, np.ndarray)) or len(pos) == 0:
            return 0.0
        ini = _a_min(pos[0].get("from"))
        ult = pos[-1].get("to")
        end = _a_min(ult) if ult else float(fin.get(fila["match_id"], 90))
        return max(end - ini, 0.0) if not (np.isnan(ini) or np.isnan(end)) else 0.0

    minutos = lineups.apply(_de_fila, axis=1)
    return minutos.groupby(lineups["player_name"]).sum()


def _tabla_ppda(events: pd.DataFrame, matches: pd.DataFrame) -> pd.Series:
    """PPDA por equipo en todo el torneo (definicion habitual).

    Pases del rival en su 60 % de campo mas cercano a su porteria, divididos
    entre las acciones defensivas del equipo en esa misma zona (entradas,
    intercepciones, faltas y regates sufridos). Menor = presion mas alta.
    """
    if not {"type", "team", "match_id", "location"} <= set(events.columns):
        return pd.Series(dtype=float)
    x = pd.to_numeric(events["location"].astype(str).str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
    tipo = events["type"]
    if "duel_type" in events.columns:
        entrada = (tipo == "Duel") & (events["duel_type"] == "Tackle")
    else:  # datos descargados con una version anterior: todos los duelos
        entrada = tipo == "Duel"
    es_def  = (entrada | tipo.isin(["Interception", "Foul Committed", "Dribbled Past"])) & (x >= 48)
    es_pase = (tipo == "Pass") & (x <= 72)
    p = events[es_pase].groupby(["match_id", "team"]).size().reset_index(name="n")
    d = events[es_def].groupby(["match_id", "team"]).size().reset_index(name="n")
    filas = {}
    for equipo in pd.concat([matches["home_team"], matches["away_team"]]).unique():
        mids = matches[(matches["home_team"] == equipo) | (matches["away_team"] == equipo)]["match_id"]
        pases_rival = p[(p["team"] != equipo) & (p["match_id"].isin(mids))]["n"].sum()
        defensivas  = d[(d["team"] == equipo) & (d["match_id"].isin(mids))]["n"].sum()
        if defensivas > 0:
            filas[equipo] = pases_rival / defensivas
    return pd.Series(filas, dtype=float)


def _datos_listos() -> bool:
    """True si estan los cinco archivos y son del torneo configurado."""
    if not all((_DATA / nombre).exists() for nombre in _ARCHIVOS):
        return False
    marca, configurado = _DATA / "torneo.json", _torneo_configurado()
    if not marca.exists() or configurado is None:
        return True  # datos descargados con una version anterior: se aceptan
    try:
        return json.loads(marca.read_text(encoding="utf-8")) == configurado
    except (ValueError, OSError):
        return True


def _descargar_si_faltan() -> bool:
    """Si faltan los datos (por ejemplo, en la app recien publicada), ejecuta
    prepare_data.py una sola vez. Los datos NO se suben al repositorio: la app
    los descarga de StatsBomb Open Data y los guarda en su propia carpeta data/.
    """
    if _datos_listos():
        return True
    script = _DATA.parent / "prepare_data.py"
    if not script.exists():
        return False
    with _CANDADO:
        if _datos_listos():
            return True
        hueco = _pantalla_de_espera()
        with st.spinner("Descargando los datos del torneo desde StatsBomb Open Data. "
                        "Solo ocurre la primera vez y puede tardar unos minutos..."):
            proceso = subprocess.run(
                [sys.executable, str(script)], cwd=str(script.parent),
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        hueco.empty()
        if proceso.returncode != 0 or not _datos_listos():
            st.error("No se pudieron descargar los datos. Revisa COMPETITION_ID y SEASON_ID en prepare_data.py.")
            st.code((proceso.stdout or "")[-1500:] + (proceso.stderr or "")[-1500:])
            return False
        # Datos nuevos: se descartan los calculos guardados del torneo anterior
        st.cache_data.clear()
        st.cache_resource.clear()
    return True


def load_data() -> dict | None:
    """Devuelve los datos del torneo. Si no estan en disco, los descarga primero.

    Devuelve None si no hay datos y no se pudieron descargar.
    """
    if not _descargar_si_faltan():
        return None
    return _leer_datos(json.dumps(_torneo_configurado()))


@st.cache_resource(show_spinner="Cargando datos del torneo...")  # una sola copia en memoria, compartida
def _leer_datos(torneo: str) -> dict:
    """Carga todos los parquets y devuelve un dict con DataFrames pre-procesados.

    `torneo` solo sirve para que la cache se renueve si cambias de torneo.
    """

    # -- Archivos base --------------------------------------------------------
    matches  = pd.read_parquet(_DATA / "matches.parquet")
    events   = pd.read_parquet(_DATA / "events.parquet")
    lineups  = pd.read_parquet(_DATA / "lineups.parquet")
    players  = pd.read_parquet(_DATA / "players.parquet")
    teams    = pd.read_parquet(_DATA / "teams.parquet")

    # Las tandas de penales (periodo 5 en StatsBomb) no cuentan como goles ni
    # tiros del partido: se excluyen para que las cifras coincidan con las oficiales.
    if "period" in events.columns:
        events = events[events["period"] < 5].reset_index(drop=True)

    # -- Nombres de uso comun -------------------------------------------------
    # StatsBomb guarda el nombre legal ("Lionel Andrés Messi Cuccittini"). Las
    # alineaciones traen tambien el apodo ("Lionel Messi"): se usa ese en todo
    # el dashboard, salvo que dos jugadores distintos compartan apodo.
    if {"player_name", "player_nickname"} <= set(lineups.columns):
        _nombres = (lineups[["player_name", "player_nickname"]].dropna()
                    .drop_duplicates("player_name"))
        _nombres = _nombres[_nombres["player_nickname"].astype(str).str.strip() != ""]
        _repetidos = _nombres["player_nickname"].duplicated(keep=False)
        _apodo = dict(zip(_nombres.loc[~_repetidos, "player_name"],
                          _nombres.loc[~_repetidos, "player_nickname"]))
        if _apodo:
            for _col in ("player", "pass_recipient"):
                if _col in events.columns:
                    events[_col] = events[_col].map(lambda n: _apodo.get(n, n))
            lineups["player_name"] = lineups["player_name"].map(lambda n: _apodo.get(n, n))
            if "player_name" in players.columns:
                players["player_name"] = players["player_name"].map(lambda n: _apodo.get(n, n))

    # -- Sub-sets frecuentes --------------------------------------------------
    shots = events[events["type"] == "Shot"].copy()
    goals = shots[shots["shot_outcome"] == "Goal"].copy()
    passes = events[events["type"] == "Pass"].copy()

    # -- Enriquecer matches con stage_sort para ordenar correctamente ---------
    if "competition_stage" in matches.columns:
        matches["stage_sort"] = (
            matches["competition_stage"].map(STAGE_ORDER).fillna(0).astype(int)
        )


    # Fases en español (el orden ya quedo guardado en stage_sort)
    if "competition_stage" in matches.columns:
        matches["competition_stage"] = matches["competition_stage"].map(lambda e: ETAPAS_ES.get(e, e))

    ppda    = _tabla_ppda(events, matches)
    minutos = _minutos_jugados(lineups, events)

    return {
        "ppda": ppda,
        "minutos": minutos,
        "matches": matches,
        "events":  events,
        "lineups": lineups,
        "players": players,
        "teams":   teams,
        "shots":   shots,
        "goals":   goals,
        "passes":  passes,
    }


def parse_location(series: pd.Series) -> pd.DataFrame:
    """
    Parsea la columna 'location' de StatsBomb a un DataFrame con columnas x e y.
    prepare_data.py serializa todas las listas como JSON strings antes de guardar
    en parquet, por lo que es necesario deserializar con json.loads.
    Maneja: string JSON, list, tuple, numpy array y None/NaN.
    """
    xs, ys = [], []
    for v in series:
        try:
            if isinstance(v, str):
                v = json.loads(v)
            xs.append(float(v[0]))
            ys.append(float(v[1]))
        except Exception:
            xs.append(np.nan)
            ys.append(np.nan)
    return pd.DataFrame({"x": xs, "y": ys}, index=series.index)


def filter_by_team(df: pd.DataFrame, equipo: str, col: str = "team") -> pd.DataFrame:
    """Filtra un DataFrame por equipo. Devuelve df completo si equipo == 'Todos'."""
    if equipo == "Todos" or not equipo:
        return df
    return df[df[col] == equipo].copy() if col in df.columns else df


def get_competition_info(matches: pd.DataFrame) -> tuple[str, str]:
    """Extrae nombre del torneo y temporada del primer registro de matches."""
    comp = matches.get("competition_name", matches.get("competition", pd.Series(["Torneo"]))).iloc[0]
    season = matches.get("season", pd.Series([""])).iloc[0]
    return str(comp), str(season)


def top_scorers(goals: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Devuelve tabla de los n maximos goleadores."""
    if goals.empty or "player" not in goals.columns:
        return pd.DataFrame(columns=["Jugador", "Equipo", "Goles", "xG"])

    _xg_col = "shot_statsbomb_xg"
    agg: dict = {"Goles": ("player", "count")}
    if _xg_col in goals.columns:
        agg["xG"] = (_xg_col, "sum")

    result = (
        goals.dropna(subset=["player"])
        .groupby("player")
        .agg(
            Equipo=("team", lambda x: x.mode().iloc[0] if not x.mode().empty else ""),
            **{k: v for k, v in agg.items()},
        )
        .sort_values("Goles", ascending=False)
        .head(n)
        .reset_index()
        .rename(columns={"player": "Jugador"})
    )
    if "xG" in result.columns:
        result["xG"] = result["xG"].round(2)
    result.index = range(1, len(result) + 1)
    return result


def match_results(matches: pd.DataFrame) -> pd.DataFrame:
    """Devuelve tabla de resultados ordenada por fase y fecha."""
    cols = ["match_date", "competition_stage", "home_team", "home_score", "away_score", "away_team"]
    cols = [c for c in cols if c in matches.columns]

    res = matches[cols].copy()
    if "stage_sort" in matches.columns:
        res["_sort"] = matches["stage_sort"]
        res = res.sort_values(["_sort", "match_date"]).drop("_sort", axis=1)
    else:
        res = res.sort_values("match_date")

    res["Resultado"] = (
        res["home_score"].astype(str) + "  -  " + res["away_score"].astype(str)
    )
    rename = {
        "match_date": "Fecha",
        "competition_stage": "Etapa",
        "home_team": "Local",
        "away_team": "Visitante",
    }
    res = res.rename(columns=rename)
    keep = ["Fecha", "Etapa", "Local", "Resultado", "Visitante"]
    res = res[[c for c in keep if c in res.columns]]
    res.index = range(1, len(res) + 1)
    return res
