# =============================================================================
#  MASTER EN FOOTBALL ANALYTICS: Eventing, Tracking, Scouting e IA
#  Radar Tactico -- Formacion en habla hispana
# =============================================================================
#  Modulo 7  : Construccion del Dashboard en Streamlit
#  Leccion   : 14 -- pages/2_Jugador.py: Analisis individual de jugador
#  Graficos  : Radar · Pizza · Heatmap · Shot Map · xG/partido
#              Mapa de Pases · Distribucion de Acciones · Actividad por Minuto
# =============================================================================

import sys
import warnings
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")

import json as _json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from config import COLOR_PRIMARIO, COLOR_SECUNDARIO, NOMBRE_ANALISTA
try:
    from config import COLOR_TEXTO, COLOR_FONDO
except ImportError:
    COLOR_TEXTO = "#e8eaf0"
    COLOR_FONDO = "#0d1117"

from utils.data import load_data, get_competition_info
from utils.sidebar import render_sidebar


# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================

st.set_page_config(page_title="Análisis de Jugador", page_icon="👤", layout="wide")

st.markdown(f"""
<style>
    #MainMenu {{ visibility: hidden; }}
    footer    {{ visibility: hidden; }}
    .section-header {{
        font-size: 1rem; font-weight: 600;
        color: {COLOR_PRIMARIO};
        text-transform: uppercase; letter-spacing: 1.5px;
        border-bottom: 2px solid {COLOR_PRIMARIO}44;
        padding-bottom: 0.3rem; margin-bottom: 0.8rem;
    }}
    .kpi-card {{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px; padding: 1rem 1.2rem; text-align: center;
    }}
    .kpi-label {{ font-size: 0.75rem; color: inherit; opacity: 0.62; text-transform: uppercase; letter-spacing: 1px; font-size: clamp(0.58rem, 0.9vw, 0.75rem); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .kpi-value {{ font-size: 2rem; font-weight: 700; color: {COLOR_PRIMARIO}; line-height: 1.2; }}
    .kpi-sub   {{ font-size: 0.8rem; color: inherit; opacity: 0.7; }}
    .player-header {{
        background: linear-gradient(135deg, {COLOR_PRIMARIO}22, {COLOR_SECUNDARIO}11);
        border-left: 4px solid {COLOR_PRIMARIO};
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1.2rem; margin-bottom: 1rem;
    }}
    .player-name {{ font-size: 1.4rem; font-weight: 700; color: {COLOR_PRIMARIO}; }}
    .player-meta {{ font-size: 0.85rem; color: inherit; opacity: 0.7; margin-top: 0.2rem; }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CARGA DE DATOS
# =============================================================================

data = load_data()
if data is None:
    st.error("⚠️  No se encontró la carpeta **data/**. Ejecuta `prepare_data.py` primero.")
    st.stop()

matches = data["matches"]
events  = data["events"]
shots   = data["shots"]
goals   = data["goals"]
passes  = data["passes"]
teams   = data["teams"]
players = data.get("players", pd.DataFrame())
lineups = data.get("lineups", pd.DataFrame())


# =============================================================================
# SIDEBAR
# =============================================================================

equipo_sel = render_sidebar(teams["team"].tolist())


# =============================================================================
# HELPERS
# =============================================================================

def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

_CSCALE_PRI = [[0.0, COLOR_FONDO], [0.4, _rgba(COLOR_PRIMARIO, 0.5)], [1.0, COLOR_PRIMARIO]]
_CSCALE_SEC = [[0.0, COLOR_FONDO], [0.4, _rgba(COLOR_SECUNDARIO, 0.5)], [1.0, COLOR_SECUNDARIO]]

def _parse_loc(series: pd.Series) -> pd.DataFrame:
    xs, ys = [], []
    for v in series:
        try:
            if isinstance(v, str):
                v = _json.loads(v)
            xs.append(float(v[0]))
            ys.append(float(v[1]))
        except Exception:
            xs.append(np.nan)
            ys.append(np.nan)
    return pd.DataFrame({"x": xs, "y": ys}, index=series.index)

_tipos_es = {
    "Ball Receipt*": "Recepción", "Carry": "Conducción", "Pass": "Pase", "Pressure": "Presión",
        "Dribble": "Regate", "Ball Recovery": "Recuperación", "Shot": "Tiro", "Dispossessed": "Pérdida (robo)",
        "Foul Won": "Falta recibida", "Duel": "Duelo", "Miscontrol": "Mal control",
        "Foul Committed": "Falta cometida", "Dribbled Past": "Regateado", "Block": "Bloqueo",
        "Injury Stoppage": "Parón por lesión", "Interception": "Intercepción", "Clearance": "Despeje",
        "Goal Keeper": "Acción de portero", "Substitution": "Sustitución", "Offside": "Fuera de juego",
        "50/50": "Balón dividido", "Shield": "Protección de balón", "Error": "Error",
        "Bad Behaviour": "Tarjeta", "Half Start": "Inicio de tiempo", "Half End": "Fin de tiempo",
        "Starting XI": "Once inicial", "Tactical Shift": "Cambio táctico", "Player Off": "Sale del campo",
        "Player On": "Vuelve al campo", "Referee Ball-Drop": "Bote neutral", "Own Goal Against": "Autogol",
}


def _pct_rank(values: np.ndarray, score: float) -> float:
    if len(values) == 0:
        return 50.0
    # Rango medio: quien empata con muchos (p. ej. 0 goles) no sale arriba
    return float((np.sum(values < score) + np.sum(values == score) / 2) / len(values) * 100)

def _pitch_shapes(half: bool = False) -> list:
    lc = _rgba(COLOR_TEXTO, 0.5)
    kw = dict(line=dict(color=lc, width=1.5), fillcolor="rgba(0,0,0,0)", layer="above")
    x0 = 60 if half else 0
    sh = [
        dict(type="rect",   x0=x0,  y0=0,  x1=120, y1=80,  **kw),
        dict(type="rect",   x0=102, y0=18, x1=120, y1=62,   **kw),
        dict(type="rect",   x0=114, y0=30, x1=120, y1=50,   **kw),
        dict(type="rect",   x0=120, y0=36, x1=122, y1=44,   **kw),
    ]
    if not half:
        sh += [
            dict(type="line",   x0=60,  y0=0,  x1=60,  y1=80,  **kw),
            dict(type="circle", x0=50,  y0=30, x1=70,  y1=50,  **kw),
            dict(type="rect",   x0=0,   y0=18, x1=18,  y1=62,  **kw),
            dict(type="rect",   x0=0,   y0=30, x1=6,   y1=50,  **kw),
            dict(type="rect",   x0=-2,  y0=36, x1=0,   y1=44,  **kw),
        ]
    return sh

def _pitch_layout(half: bool = False, height: int = 440, title: str = "") -> dict:
    xr = [58, 123] if half else [-3, 123]
    return dict(
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=10),
        shapes=_pitch_shapes(half=half),
        xaxis=dict(range=xr, showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-3, 83], showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(t=40, b=30, l=10, r=10),
        height=height,
        title=dict(text=title.replace("  (", "<br>("), font=dict(color=COLOR_TEXTO, size=10), x=0.5),
    )


# =============================================================================
# ESTADÍSTICAS GLOBALES POR JUGADOR (para percentiles — se cachean)
# =============================================================================

@st.cache_data(show_spinner=False)
def _build_player_stats(_events: pd.DataFrame, _shots: pd.DataFrame,
                        _passes: pd.DataFrame) -> pd.DataFrame:
    """Computa métricas por jugador sobre el torneo completo para percentiles."""
    n_matches = _events.groupby("player")["match_id"].nunique().rename("n_matches")

    n_goals  = (_shots[_shots["shot_outcome"] == "Goal"]
                .groupby("player").size().rename("goals"))
    n_shots  = _shots.groupby("player").size().rename("shots")
    xg_sum   = (_shots.groupby("player")["shot_statsbomb_xg"].sum().rename("xg")
                if "shot_statsbomb_xg" in _shots.columns else pd.Series(dtype=float, name="xg"))

    n_passes = _passes.groupby("player").size().rename("passes")
    if "pass_outcome" in _passes.columns:
        n_complete = (_passes[_passes["pass_outcome"].isna()]
                      .groupby("player").size().rename("passes_ok"))
    else:
        n_complete = pd.Series(dtype=float, name="passes_ok")

    if "dribble_outcome" in _events.columns:
        n_drib = (_events[(_events["type"] == "Dribble") &
                          (_events["dribble_outcome"] == "Complete")]
                  .groupby("player").size().rename("dribbles"))
    else:
        n_drib = (_events[_events["type"] == "Dribble"]
                  .groupby("player").size().rename("dribbles"))

    n_press = (_events[_events["type"] == "Pressure"]
               .groupby("player").size().rename("pressures"))
    n_def   = (_events[_events["type"].isin({"Duel","Interception","Clearance","Ball Recovery"})]
               .groupby("player").size().rename("defensive"))

    df = pd.DataFrame(n_matches)
    for s in [n_goals, n_shots, xg_sum, n_passes, n_complete, n_drib, n_press, n_def]:
        if s is not None and not s.empty:
            df = df.join(s, how="left")
    df = df.fillna(0)
    df = df[df["n_matches"] >= 1].copy()

    def _pm(col):
        return df.get(col, pd.Series(0, index=df.index)) / df["n_matches"]

    df["goals_pm"]     = _pm("goals")
    df["shots_pm"]     = _pm("shots")
    df["xg_pm"]        = _pm("xg")
    df["passes_pm"]    = _pm("passes")
    df["dribbles_pm"]  = _pm("dribbles")
    df["pressures_pm"] = _pm("pressures")
    df["defensive_pm"] = _pm("defensive")
    df["pass_pct"]     = np.where(df.get("passes", 0) > 0,
                                   df.get("passes_ok", 0) / df.get("passes", 1) * 100, 0)
    return df

_all_player_stats = _build_player_stats(events, shots, passes)

_RADAR_KEYS    = ["goals_pm", "shots_pm", "xg_pm", "passes_pm",
                  "pass_pct",  "dribbles_pm", "pressures_pm", "defensive_pm"]
_RADAR_LABELS  = ["Goles\n/partido", "Tiros\n/partido", "xG\n/partido",
                  "Pases\n/partido", "% Pases\ncompletados",
                  "Regates\n/partido", "Presiones\n/partido", "Defensiva\n/partido"]
_RADAR_LABELS_SHORT = ["Goles/P", "Tiros/P", "xG/P",
                       "Pases/P", "% Pases", "Regates/P", "Presiones/P", "Defensiva/P"]


def _compute_filtered_stats(ev_j: pd.DataFrame, sh_j: pd.DataFrame,
                             pa_j: pd.DataFrame, n_matches: int) -> dict:
    """Calcula métricas por partido desde los datos YA FILTRADOS del jugador."""
    if n_matches == 0:
        return {k: 0.0 for k in _RADAR_KEYS}

    _xc = "shot_statsbomb_xg"
    n_goals   = int((sh_j["shot_outcome"] == "Goal").sum()) if not sh_j.empty else 0
    n_shots   = len(sh_j)
    xg_total  = float(sh_j[_xc].sum()) if (_xc in sh_j.columns and not sh_j.empty) else 0.0
    n_passes  = len(pa_j)
    n_ok      = int(pa_j["pass_outcome"].isna().sum()) if ("pass_outcome" in pa_j.columns and not pa_j.empty) else 0
    n_drib    = (int(((ev_j["type"] == "Dribble") & (ev_j["dribble_outcome"] == "Complete")).sum())
                 if "dribble_outcome" in ev_j.columns and not ev_j.empty
                 else int((ev_j["type"] == "Dribble").sum()) if not ev_j.empty else 0)
    n_press   = int((ev_j["type"] == "Pressure").sum()) if not ev_j.empty else 0
    n_def     = int(ev_j["type"].isin({"Duel","Interception","Clearance","Ball Recovery"}).sum()) if not ev_j.empty else 0

    return {
        "goals_pm":     n_goals   / n_matches,
        "shots_pm":     n_shots   / n_matches,
        "xg_pm":        xg_total  / n_matches,
        "passes_pm":    n_passes  / n_matches,
        "pass_pct":     (n_ok / n_passes * 100) if n_passes > 0 else 0.0,
        "dribbles_pm":  n_drib    / n_matches,
        "pressures_pm": n_press   / n_matches,
        "defensive_pm": n_def     / n_matches,
    }


def _get_percentiles_from_vals(vals: dict) -> list[float]:
    """Percentiles (0-100) para cada clave usando la distribución del torneo completo."""
    pcts = []
    for key in _RADAR_KEYS:
        if key in _all_player_stats.columns and key in vals:
            dist = _all_player_stats[key].dropna().values
            pcts.append(_pct_rank(dist, vals[key]))
        else:
            pcts.append(50.0)
    return pcts


# =============================================================================
# ENCABEZADO DE PÁGINA
# =============================================================================

comp_name, season_name = get_competition_info(matches)
st.title("👤 Análisis de Jugador")
st.caption(f"{comp_name} {season_name}")

if equipo_sel == "Todos":
    st.info("ℹ️  Selecciona un equipo en el sidebar para filtrar los jugadores.")

st.divider()


# =============================================================================
# FILTROS
# =============================================================================

_fc0, _fc1, _fc2, _fc3, _fc4 = st.columns([2, 3, 2, 2, 4])

# ── Filtro 0: Equipo ──────────────────────────────────────────────────────────
with _fc0:
    st.markdown("**🛡️ Equipo**")
    _team_list = sorted(teams["team"].tolist())
    _def_idx   = _team_list.index(equipo_sel) if equipo_sel in _team_list else 0
    _equipo    = st.selectbox("Equipo", _team_list, index=_def_idx,
                              label_visibility="collapsed", key="eq_jugador")

# Derivar datos del equipo seleccionado
_team_mask   = (matches["home_team"] == _equipo) | (matches["away_team"] == _equipo)
_team_mids   = matches[_team_mask]["match_id"].unique()
_team_events = events[(events["team"] == _equipo) & (events["match_id"].isin(_team_mids))]
_player_list = sorted(_team_events["player"].dropna().unique().tolist())

if not _player_list:
    st.warning(f"No se encontraron jugadores para **{_equipo}**.")
    st.stop()

with _fc1:
    st.markdown("**👤 Jugador**")
    # Por defecto, el maximo goleador del equipo (y si nadie marco, el mas activo)
    _goles_eq = goals[(goals["team"] == _equipo) & (goals["match_id"].isin(_team_mids))]["player"].value_counts()
    _dest     = _goles_eq.index[0] if len(_goles_eq) else _team_events["player"].value_counts().index[0]
    _jugador  = st.selectbox("Jugador", _player_list,
                             index=_player_list.index(_dest) if _dest in _player_list else 0,
                             label_visibility="collapsed")

with _fc2:
    st.markdown("**📅 Partido**")
    _team_matches = matches[_team_mask].sort_values("match_date")
    _match_labels = {"Todos los partidos": None}
    for _row in _team_matches.itertuples():
        _lbl = (f"{str(getattr(_row,'match_date',''))[:10]} · "
                f"{getattr(_row,'home_team','?')} "
                f"{getattr(_row,'home_score','')}–{getattr(_row,'away_score','')} "
                f"{getattr(_row,'away_team','?')}")
        _match_labels[_lbl] = _row.match_id
    _match_sel  = st.selectbox("Partido", list(_match_labels.keys()), label_visibility="collapsed")
    _match_id_f = _match_labels[_match_sel]

with _fc3:
    st.markdown("**⏱️ Período**")
    _period_opts = {"Completo": None, "1er Tiempo": [1], "2do Tiempo": [2]}
    if "period" in events.columns and (events["period"] >= 3).any():
        _period_opts["Prórroga"] = [3, 4]
    _period_sel  = st.radio("Período", list(_period_opts.keys()),
                            horizontal=False, label_visibility="collapsed")
    _period_val  = _period_opts[_period_sel]

# Detección de portero para catálogo de gráficos (antes de _fc4)
_is_gk_catalog = False
if not lineups.empty and "player_name" in lineups.columns and "position_name" in lineups.columns:
    _lu_q = lineups[lineups["player_name"] == _jugador]
    if not _lu_q.empty:
        _is_gk_catalog = str(_lu_q["position_name"].iloc[0]).strip().lower() == "goalkeeper"
# Fallback: position_name puede estar vacío si prepare_data serializó positions como JSON
# antes de parsearla — se detecta por presencia de eventos "Goal Keeper" en el equipo
if not _is_gk_catalog and not _team_events.empty and "type" in _team_events.columns:
    _pl_ev_check = _team_events[_team_events["player"] == _jugador]
    _is_gk_catalog = bool((_pl_ev_check["type"] == "Goal Keeper").any())

with _fc4:
    if _is_gk_catalog:
        _ALL_CHARTS = [
            "🥅 Paradas por Partido",
            "🗺️ Tiros Recibidos",
            "📤 Distribución (Pases)",
            "🧤 Acciones del Portero",
            "🔥 Heatmap de Acciones",
            "⏱️ Actividad por Minuto",
        ]
    else:
        _ALL_CHARTS = [
            "📊 Radar de Métricas",
            "🍕 Pizza Chart",
            "🔥 Heatmap Individual",
            "🎯 Shot Map",
            "📈 xG por Partido",
            "➡️ Mapa de Pases",
            "📋 Distribución de Acciones",
            "⏱️ Actividad por Minuto",
        ]
    _DEFAULT_CHARTS = _ALL_CHARTS[:4]
    _MAX_CHARTS = 6
    st.markdown(f"**📊 Gráficos a mostrar** *(máx. {_MAX_CHARTS})*")
    _charts_sel = st.multiselect(
        "Gráficos",
        options=_ALL_CHARTS,
        default=_DEFAULT_CHARTS,
        max_selections=_MAX_CHARTS,
        label_visibility="collapsed",
    )

if not _charts_sel:
    st.info("Selecciona al menos un gráfico arriba.")
    st.stop()


# =============================================================================
# APLICAR FILTROS
# =============================================================================

_active_ids = np.array([_match_id_f]) if _match_id_f else _team_mids

_ev_j  = events[(events["player"] == _jugador) & (events["match_id"].isin(_active_ids))].copy()
_sh_j  = shots [(shots ["player"] == _jugador) & (shots ["match_id"].isin(_active_ids))].copy()
_go_j  = goals [(goals ["player"] == _jugador) & (goals ["match_id"].isin(_active_ids))].copy()
_pa_j  = passes[(passes["player"] == _jugador) & (passes["match_id"].isin(_active_ids))].copy()

if _period_val is not None:
    for _df, _name in [(_ev_j,"_ev_j"), (_sh_j,"_sh_j"), (_go_j,"_go_j"), (_pa_j,"_pa_j")]:
        if "period" in _df.columns:
            _df.drop(_df.index[~_df["period"].isin(_period_val)], inplace=True)

_n_matches_j  = _ev_j["match_id"].nunique()
_n_goals_j    = len(_go_j)
_xg_col       = "shot_statsbomb_xg"
_xg_j         = _sh_j[_xg_col].sum() if _xg_col in _sh_j.columns else 0
_n_shots_j    = len(_sh_j)
_n_passes_j   = len(_pa_j)
_n_assists_j  = int(_pa_j["pass_goal_assist"].sum()) if "pass_goal_assist" in _pa_j.columns else 0

# Métricas por partido calculadas desde los datos FILTRADOS (alimentan radar y pizza)
_player_vals = _compute_filtered_stats(_ev_j, _sh_j, _pa_j, max(_n_matches_j, 1))


# =============================================================================
# CABECERA DEL JUGADOR
# =============================================================================

# Información de posición desde lineups
_pos_str, _dorsal = "", ""
if not lineups.empty and "player_name" in lineups.columns:
    _lu_row = lineups[lineups["player_name"] == _jugador]
    if not _lu_row.empty:
        # position_name puede estar vacío si prepare_data.py serializó positions
        # como JSON string antes de parsearla — se reconstruye aquí como fallback
        _pn = str(_lu_row["position_name"].iloc[0]) if "position_name" in _lu_row.columns else ""
        if not _pn.strip() and "positions" in _lu_row.columns:
            try:
                _raw = _lu_row["positions"].iloc[0]
                if isinstance(_raw, str):
                    _raw = _json.loads(_raw)
                if isinstance(_raw, list) and _raw:
                    _first = _raw[0]
                    _pn = _first.get("position", "") if isinstance(_first, dict) else str(_first)
            except Exception:
                _pn = ""
        _pos_str = _pn
        if "jersey_number" in _lu_row.columns:
            _dorsal = str(int(_lu_row["jersey_number"].iloc[0]))

# Confirmar si es portero (posición o eventos GK)
_is_gk = _pos_str.strip().lower() == "goalkeeper"
if not _is_gk and not _ev_j.empty and "type" in _ev_j.columns:
    _is_gk = bool((_ev_j["type"] == "Goal Keeper").any())

# Datos específicos de portero
if _is_gk:
    _shots_vs = shots[
        (shots["team"] != _equipo) & (shots["match_id"].isin(_active_ids))
    ].copy()
    if _period_val is not None and "period" in _shots_vs.columns:
        _shots_vs = _shots_vs[_shots_vs["period"].isin(_period_val)]
    _gk_events  = _ev_j[_ev_j["type"] == "Goal Keeper"].copy() if not _ev_j.empty else pd.DataFrame()
    _saves_n    = int((_shots_vs["shot_outcome"] == "Saved").sum()) if not _shots_vs.empty else 0
    _gc_n       = int((_shots_vs["shot_outcome"] == "Goal").sum())  if not _shots_vs.empty else 0
    _shots_fcd  = len(_shots_vs)
    _save_pct   = (_saves_n / _shots_fcd * 100) if _shots_fcd > 0 else 0.0
    _psxg_vs    = float(_shots_vs[_xg_col].sum()) if (_xg_col in _shots_vs.columns and not _shots_vs.empty) else 0.0
    _psxg_diff  = _psxg_vs - _gc_n
else:
    _shots_vs = pd.DataFrame()
    _gk_events = pd.DataFrame()
    _saves_n = _gc_n = _shots_fcd = 0
    _save_pct = _psxg_vs = _psxg_diff = 0.0

_meta_parts = [_equipo]
if _dorsal:  _meta_parts.append(f"#{_dorsal}")
if _pos_str: _meta_parts.append(_pos_str)

st.markdown(f"""
<div class="player-header">
    <div class="player-name">{_jugador}</div>
    <div class="player-meta">{' · '.join(_meta_parts)}</div>
</div>""", unsafe_allow_html=True)

# KPIs
_k1, _k2, _k3, _k4, _k5, _k6 = st.columns(6)
if _is_gk:
    _kpi_rows = [
        (_k1, "Partidos",         _n_matches_j,              "jugados"),
        (_k2, "Paradas",          _saves_n,                  f"{_saves_n/_n_matches_j:.1f}/partido" if _n_matches_j else "—"),
        (_k3, "Goles concedidos", _gc_n,                     f"{_gc_n/_n_matches_j:.2f}/partido" if _n_matches_j else "—"),
        (_k4, "% Paradas",        f"{_save_pct:.1f}%",       f"{_shots_fcd} tiros recibidos"),
        (_k5, "PSxG concedido",   f"{_psxg_vs:.2f}",         "xG de tiros recibidos"),
        (_k6, "PSxG − GC",        f"{_psxg_diff:+.2f}",      "positivo = mejor de lo esperado"),
    ]
else:
    _kpi_rows = [
        (_k1, "Partidos",    _n_matches_j,             "jugados"),
        (_k2, "Goles",       _n_goals_j,               f"{_n_goals_j/_n_matches_j:.2f}/partido" if _n_matches_j else "—"),
        (_k3, "Asistencias", _n_assists_j,             f"{_n_assists_j/_n_matches_j:.2f}/partido" if _n_matches_j else "—"),
        (_k4, "xG total",    f"{_xg_j:.2f}",          f"{_xg_j/_n_matches_j:.2f}/partido" if _n_matches_j else "—"),
        (_k5, "Tiros",       _n_shots_j,              f"{_n_shots_j/_n_matches_j:.1f}/partido" if _n_matches_j else "—"),
        (_k6, "Pases",       _n_passes_j,             f"{_n_passes_j/_n_matches_j:.0f}/partido" if _n_matches_j else "—"),
    ]
for _col, _lbl, _val, _sub in _kpi_rows:
    _col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{_lbl}</div>
        <div class="kpi-value">{_val}</div>
        <div class="kpi-sub">{_sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# FUNCIONES DE GRÁFICOS
# =============================================================================

def _render_radar():
    st.markdown('<div class="section-header">📊 Radar de Métricas</div>', unsafe_allow_html=True)

    pcts     = _get_percentiles_from_vals(_player_vals)
    avg_pcts = [50.0] * len(_RADAR_KEYS)

    # Etiquetas legibles — sin abreviaciones confusas
    _labels = [
        "Goles/partido", "Tiros/partido", "xG/partido",
        "Pases/partido", "% Pases OK",
        "Regates/partido", "Presiones/partido", "Defensiva/partido",
    ]
    # Cerrar polígono repitiendo el primer punto
    cats   = _labels + [_labels[0]]
    vals_p = pcts + [pcts[0]]
    vals_a = avg_pcts + [avg_pcts[0]]

    # Valores reales para hover
    _real = [round(_player_vals.get(k, 0), 2) for k in _RADAR_KEYS] + [round(_player_vals.get(_RADAR_KEYS[0], 0), 2)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_a, theta=cats, fill="toself", name="Promedio torneo",
        fillcolor=_rgba(COLOR_TEXTO, 0.07),
        line=dict(color=_rgba(COLOR_TEXTO, 0.35), width=1.5, dash="dot"),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_p, theta=cats, fill="toself", name=_jugador,
        fillcolor=_rgba(COLOR_PRIMARIO, 0.22),
        line=dict(color=COLOR_PRIMARIO, width=2.5),
        customdata=_real,
        hovertemplate="<b>%{theta}</b><br>Percentil: %{r:.0f}<br>Valor: %{customdata}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=COLOR_FONDO,
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickvals=[25, 50, 75, 100],
                tickfont=dict(color=COLOR_TEXTO, size=7),
                gridcolor=_rgba(COLOR_TEXTO, 0.15),
                linecolor=_rgba(COLOR_TEXTO, 0.15),
            ),
            angularaxis=dict(
                tickfont=dict(color=COLOR_TEXTO, size=9),
                gridcolor=_rgba(COLOR_TEXTO, 0.15),
                linecolor=_rgba(COLOR_TEXTO, 0.2),
                direction="clockwise",
                rotation=90,      # primera métrica arriba (norte)
            ),
        ),
        paper_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=10),
        legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="center", x=0.5,
                    font=dict(color=COLOR_TEXTO, size=9), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=40, b=70, l=80, r=80),
        height=460,
        title=dict(text="Percentil vs torneo  (25 · 50 · 75 · 100)",
                   font=dict(color=COLOR_TEXTO, size=9), x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Hover sobre cada vértice para ver valor real y percentil")


def _render_pizza():
    st.markdown('<div class="section-header">🍕 Pizza Chart</div>', unsafe_allow_html=True)

    pcts   = _get_percentiles_from_vals(_player_vals)

    _labels = [
        "Goles/partido", "Tiros/partido", "xG/partido",
        "Pases/partido", "% Pases OK",
        "Regates/partido", "Presiones/partido", "Defensiva/partido",
    ]
    _real_vals = [round(_player_vals.get(k, 0), 2) for k in _RADAR_KEYS]

    n          = len(_labels)
    _step      = 360 / n                          # 45° por sector
    _bar_w     = _step * 0.88                     # 88% → pequeño gap visual
    # Con rotation=90 + clockwise: theta=0 queda en norte, crece horario
    _thetas    = [i * _step for i in range(n)]   # [0, 45, 90 ... 315]

    def _bar_color(p: float) -> str:
        if p >= 75: return COLOR_PRIMARIO
        if p >= 50: return _rgba(COLOR_PRIMARIO, 0.65)
        if p >= 25: return _rgba(COLOR_PRIMARIO, 0.40)
        return _rgba(COLOR_SECUNDARIO, 0.55)

    # theta en grados numéricos → width se interpreta correctamente en grados
    fig = go.Figure(go.Barpolar(
        r=pcts,
        theta=_thetas,
        width=[_bar_w] * n,
        marker_color=[_bar_color(p) for p in pcts],
        marker_line_color=COLOR_FONDO,
        marker_line_width=2,
        customdata=list(zip(_labels, [f"{p:.0f}" for p in pcts], _real_vals)),
        hovertemplate="<b>%{customdata[0]}</b><br>Percentil: %{customdata[1]}<br>Valor: %{customdata[2]}<extra></extra>",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor=COLOR_FONDO,
            radialaxis=dict(
                range=[0, 105],
                tickvals=[25, 50, 75, 100],
                tickfont=dict(color=COLOR_TEXTO, size=7),
                gridcolor=_rgba(COLOR_TEXTO, 0.15),
                showticklabels=True,
                linecolor=_rgba(COLOR_TEXTO, 0.15),
            ),
            angularaxis=dict(
                # Etiquetas en los mismos grados que los sectores
                tickmode="array",
                tickvals=_thetas,
                ticktext=_labels,
                tickfont=dict(color=COLOR_TEXTO, size=9),
                gridcolor=_rgba(COLOR_TEXTO, 0.12),
                linecolor=_rgba(COLOR_TEXTO, 0.15),
                direction="clockwise",
                rotation=90,
            ),
        ),
        paper_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO),
        showlegend=False,
        margin=dict(t=50, b=50, l=90, r=90),
        height=460,
        title=dict(text="Percentil por métrica (escala 0–100)",
                   font=dict(color=COLOR_TEXTO, size=9), x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Sector largo = alto percentil  ·  Colores: color principal ≥75 · más claro ≥50 y ≥25 · color secundario <25")


def _render_heatmap_j():
    st.markdown('<div class="section-header">🔥 Heatmap Individual</div>', unsafe_allow_html=True)
    if _ev_j.empty or "location" not in _ev_j.columns:
        st.info("Sin datos de posición."); return
    _loc = _parse_loc(_ev_j["location"]).dropna()
    if _loc.empty:
        st.info("Sin coordenadas."); return
    fig = go.Figure()
    fig.add_trace(go.Histogram2d(
        x=_loc["x"], y=_loc["y"], xbins=dict(start=0, end=120, size=5), ybins=dict(start=0, end=80, size=5),
        colorscale=_CSCALE_PRI, showscale=False,
        hovertemplate="x:%{x:.0f} y:%{y:.0f}<br>Acciones: %{z}<extra></extra>",
    ))
    fig.update_layout(**_pitch_layout(False, 440,
        f"Densidad de acciones — {_jugador}  ({len(_loc):,} eventos)"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Mayor intensidad = zona de mayor presencia")


def _render_shot_map_j():
    st.markdown('<div class="section-header">🎯 Shot Map</div>', unsafe_allow_html=True)
    if _sh_j.empty or "location" not in _sh_j.columns:
        st.info("Sin tiros registrados."); return
    _d = _parse_loc(_sh_j["location"])
    _d["outcome"] = _sh_j["shot_outcome"].fillna("Unknown").values
    _d["xg"]      = (_sh_j[_xg_col].fillna(0.05).values if _xg_col in _sh_j.columns else 0.05)
    _d = _d.dropna(subset=["x","y"])

    _cfg = {
        "Goal":    (COLOR_PRIMARIO,   "Gol"),
        "Saved":   (COLOR_SECUNDARIO, "Atajado"),
        "Blocked": ("#9a9a9a",        "Bloqueado"),
        "Off T":   ("#7d7d7d",        "Fuera"),
        "Post":    ("#b5b5b5",        "Poste"),
        "Wayward": ("#6a6a6a",        "Desviado"),
    }
    fig = go.Figure()
    # Se dibuja de menos a mas importante para que los goles queden encima
    for _out, (_color, _lbl) in reversed(list(_cfg.items())):
        _m = _d["outcome"] == _out
        if not _m.any(): continue
        _s = _d[_m]
        fig.add_trace(go.Scatter(
            x=_s["x"], y=_s["y"], mode="markers", name=_lbl,
            marker=dict(size=(_s["xg"] * 40).clip(6, 28), color=_color, opacity=0.9,
                        line=dict(width=1, color=COLOR_FONDO)),
            hovertemplate=f"<b>{_lbl}</b><br>xG: %{{customdata:.3f}}<extra></extra>",
            customdata=_s["xg"].values,
        ))
    fig.update_layout(
        **_pitch_layout(True, 440, f"Tiros — {_jugador}  ({len(_d)} tiros · {_n_goals_j} goles)"),
        legend=dict(orientation="h", yanchor="top", y=-0.04, xanchor="center", x=0.5,
                    font=dict(color=COLOR_TEXTO, size=10), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Tamaño ∝ xG  ·  Zona de ataque (medio campo derecho)")


def _render_xg_partido_j():
    st.markdown('<div class="section-header">📈 xG por Partido</div>', unsafe_allow_html=True)
    if _xg_col not in shots.columns:
        st.info("Sin datos de xG."); return

    _all_sh_j = shots[shots["player"] == _jugador].copy()
    if _all_sh_j.empty:
        st.info("Sin tiros registrados."); return

    _xg_pm = (_all_sh_j.groupby("match_id")[_xg_col].sum().reset_index(name="xG"))
    _go_pm = (_all_sh_j[_all_sh_j["shot_outcome"] == "Goal"]
              .groupby("match_id").size().reset_index(name="goles"))

    _mi = matches[["match_id","match_date","home_team","away_team",
                   "home_score","away_score"]].copy()
    _df = _mi.merge(_xg_pm, on="match_id", how="inner").merge(_go_pm, on="match_id", how="left").fillna(0)
    _df = _df.sort_values("match_date").reset_index(drop=True)   # orden cronologico
    _df["label"] = _df.apply(
        lambda r: f"{str(r.match_date)[:10]}<br>{r.home_team} {r.home_score}–{r.away_score} {r.away_team}", axis=1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=_df["label"], y=_df["xG"].round(2), name="xG",
        marker_color=COLOR_PRIMARIO, opacity=0.85,
        hovertemplate="<b>%{x}</b><br>xG: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=_df["label"], y=_df["goles"], mode="markers", name="Goles",
        marker=dict(size=12, color=COLOR_SECUNDARIO, symbol="star",
                    line=dict(width=1, color=COLOR_FONDO)),
        hovertemplate="<b>%{x}</b><br>Goles: %{y}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=9),
        xaxis=dict(showgrid=False, tickfont=dict(color=COLOR_TEXTO, size=8)),
        yaxis=dict(title="xG / Goles", title_font=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO, 0.12), tickfont=dict(color=COLOR_TEXTO)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    font=dict(color=COLOR_TEXTO), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=70, b=70, l=40, r=10),
        height=440,
        title=dict(y=0.97, text=f"xG y goles por partido — {_jugador}",
                   font=dict(color=COLOR_TEXTO, size=10), x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Barras = xG  ·  Estrellas = goles reales")


def _render_pass_map_j():
    st.markdown('<div class="section-header">➡️ Mapa de Pases</div>', unsafe_allow_html=True)
    _end_col = "pass_end_location"
    if _pa_j.empty or "location" not in _pa_j.columns or _end_col not in _pa_j.columns:
        st.info("Sin datos de pases."); return

    _start = _parse_loc(_pa_j["location"])
    _end   = _parse_loc(_pa_j[_end_col]).rename(columns={"x":"x2","y":"y2"})
    _pm    = pd.concat([_start, _end], axis=1).dropna()

    # Separar por resultado
    _ok_mask  = _pa_j["pass_outcome"].isna() if "pass_outcome" in _pa_j.columns else pd.Series(True, index=_pa_j.index)
    _pm_ok  = _pm[_ok_mask.values[:len(_pm)]] if len(_ok_mask) == len(_pm) else _pm
    _pm_nok = _pm[~_ok_mask.values[:len(_pm)]] if len(_ok_mask) == len(_pm) else pd.DataFrame()

    def _batch_lines(df, max_n=200):
        df = df.sample(min(max_n, len(df)), random_state=42) if len(df) > max_n else df
        lx, ly = [], []
        for r in df.itertuples():
            lx += [r.x, r.x2, None]; ly += [r.y, r.y2, None]
        return lx, ly

    fig = go.Figure()
    if not _pm_nok.empty:
        _lx, _ly = _batch_lines(_pm_nok, 40)
        fig.add_trace(go.Scatter(x=_lx, y=_ly, mode="lines",
            line=dict(color=_rgba(COLOR_SECUNDARIO, 0.55), width=1.3),
            hoverinfo="skip", name="Incompleto", showlegend=True))

    if not _pm_ok.empty:
        _lx, _ly = _batch_lines(_pm_ok, 110)
        fig.add_trace(go.Scatter(x=_lx, y=_ly, mode="lines",
            line=dict(color=_rgba(COLOR_PRIMARIO, 0.5), width=1.3),
            hoverinfo="skip", name="Completado", showlegend=True))

    fig.update_layout(
        **_pitch_layout(False, 440, f"Mapa de pases — {_jugador}  ({len(_pa_j):,} pases)"),
        legend=dict(orientation="h", yanchor="top", y=-0.04, xanchor="center", x=0.5,
                    font=dict(color=COLOR_TEXTO, size=10), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{len(_pa_j)} pases  ·  Color principal = completado  ·  Color secundario = incompleto  ·  (muestra aleatoria de hasta 150 para que se lea)")


def _render_action_dist():
    st.markdown('<div class="section-header">📋 Distribución de Acciones</div>', unsafe_allow_html=True)
    if _ev_j.empty:
        st.info("Sin eventos."); return

    _counts = _ev_j["type"].value_counts().head(15).sort_values()
    _colors = [COLOR_PRIMARIO if t in {"Shot","Pass","Dribble","Carry"} else
               _rgba(COLOR_TEXTO, 0.3) for t in _counts.index]

    fig = go.Figure(go.Bar(
        y=[_tipos_es.get(t, t) for t in _counts.index], x=_counts.values, orientation="h",
        marker_color=_colors, cliponaxis=False,
        text=_counts.values, textposition="outside",
        textfont=dict(color=COLOR_TEXTO, size=8),
        hovertemplate="<b>%{y}</b><br>Total: %{x}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=9),
        xaxis=dict(title="Cantidad", title_font=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO, 0.12), tickfont=dict(color=COLOR_TEXTO)),
        yaxis=dict(showgrid=False, tickfont=dict(color=COLOR_TEXTO, size=9)),
        margin=dict(t=40, b=30, l=10, r=60),
        height=440, showlegend=False,
        title=dict(text=f"Top 15 tipos de acción — {_jugador}",
                   font=dict(color=COLOR_TEXTO, size=10), x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Top 15 tipos de evento más frecuentes del jugador  ·  Color principal = acciones con balón (tiro, pase, regate, conducción)")


def _render_activity_minute():
    st.markdown('<div class="section-header">⏱️ Actividad por Minuto</div>', unsafe_allow_html=True)
    if _ev_j.empty or "minute" not in _ev_j.columns:
        st.info("Sin datos de minuto."); return

    _ev_j2 = _ev_j.copy()
    _ev_j2["minute_bin"] = (_ev_j2["minute"] // 5) * 5
    _bins = _ev_j2.groupby("minute_bin").size().reset_index(name="acciones")

    _colors = [COLOR_SECUNDARIO if b >= 85 else COLOR_PRIMARIO for b in _bins["minute_bin"]]

    fig = go.Figure(go.Bar(
        x=_bins["minute_bin"], y=_bins["acciones"],
        marker_color=_colors, opacity=0.85,
        width=4.5,
        hovertemplate="Min %{x}–%{x+5}<br>Acciones: %{y}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=9),
        xaxis=dict(title="Minuto", title_font=dict(color=COLOR_TEXTO),
                   tickfont=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO, 0.1),
                   dtick=15),
        yaxis=dict(title="Acciones", title_font=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO, 0.12), tickfont=dict(color=COLOR_TEXTO)),
        margin=dict(t=40, b=40, l=40, r=10),
        height=440, showlegend=False,
        title=dict(text=f"Acciones por tramo de 5 minutos — {_jugador}",
                   font=dict(color=COLOR_TEXTO, size=10), x=0.5),
    )
    # Líneas de inicio de parte
    for _xv, _lbl in [(45, "HT"), (90, "FT")]:
        fig.add_vline(x=_xv, line_dash="dot", line_color=_rgba(COLOR_TEXTO, 0.35), line_width=1)
        fig.add_annotation(x=_xv, y=0, yref="paper", text=_lbl, showarrow=False,
                           font=dict(color=COLOR_TEXTO, size=8), yanchor="bottom")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Barras de 5 min  ·  Color secundario = desde el minuto 85 (final del partido y prórroga)  ·  El pico del 45 incluye el añadido del primer tiempo")


# =============================================================================
# GRÁFICOS DE PORTERO
# =============================================================================

def _render_gk_saves_partido():
    st.markdown('<div class="section-header">🥅 Paradas por Partido</div>', unsafe_allow_html=True)
    if _shots_vs.empty or "match_id" not in _shots_vs.columns:
        st.info("Sin tiros recibidos registrados."); return

    _sv_pm  = (_shots_vs[_shots_vs["shot_outcome"] == "Saved"]
               .groupby("match_id").size().rename("paradas"))
    _gc_pm  = (_shots_vs[_shots_vs["shot_outcome"] == "Goal"]
               .groupby("match_id").size().rename("goles_c"))
    _xg_pm2 = (_shots_vs.groupby("match_id")[_xg_col].sum().rename("psxg")
               if _xg_col in _shots_vs.columns else pd.Series(dtype=float, name="psxg"))

    _mi2 = matches[["match_id","match_date","home_team","away_team",
                    "home_score","away_score"]].copy()
    _df2 = _mi2.merge(_sv_pm, on="match_id", how="inner")
    _df2 = _df2.merge(_gc_pm,  on="match_id", how="left")
    _df2 = _df2.merge(_xg_pm2, on="match_id", how="left").fillna(0)
    _df2["label"] = _df2.apply(
        lambda r: f"{str(r.match_date)[:10]}<br>{r.home_team} {r.home_score}–{r.away_score} {r.away_team}", axis=1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=_df2["label"], y=_df2["paradas"], name="Paradas",
        marker_color=COLOR_PRIMARIO, opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Paradas: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=_df2["label"], y=_df2["goles_c"], mode="markers", name="Goles concedidos",
        marker=dict(size=12, color=COLOR_SECUNDARIO, symbol="x",
                    line=dict(width=2.5, color=COLOR_SECUNDARIO)),
        hovertemplate="<b>%{x}</b><br>Goles: %{y}<extra></extra>",
    ))
    if "psxg" in _df2.columns:
        fig.add_trace(go.Scatter(
            x=_df2["label"], y=_df2["psxg"].round(2), mode="lines+markers", name="PSxG",
            line=dict(color=_rgba(COLOR_TEXTO, 0.5), width=1.5, dash="dot"),
            marker=dict(size=5),
            hovertemplate="<b>%{x}</b><br>PSxG: %{y:.2f}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=9),
        xaxis=dict(showgrid=False, tickfont=dict(color=COLOR_TEXTO, size=8)),
        yaxis=dict(title="Cantidad", title_font=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO, 0.12), tickfont=dict(color=COLOR_TEXTO)),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="center", x=0.5,
                    font=dict(color=COLOR_TEXTO, size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20, b=90, l=40, r=10), height=440,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Barras = paradas  ·  × = goles concedidos  ·  Línea = PSxG esperado")


def _render_gk_tiros_recibidos():
    st.markdown('<div class="section-header">🗺️ Tiros Recibidos</div>', unsafe_allow_html=True)
    if _shots_vs.empty or "location" not in _shots_vs.columns:
        st.info("Sin datos de tiros recibidos."); return

    _d = _parse_loc(_shots_vs["location"])
    # StatsBomb normaliza por equipo: tiros del rival en x=60-115 atacando hacia x=120
    # Se muestran en su posición original (origen del tiro, perspectiva del atacante)
    _d["outcome"] = _shots_vs["shot_outcome"].fillna("Unknown").values
    _d["xg"]      = (_shots_vs[_xg_col].fillna(0.05).values
                     if _xg_col in _shots_vs.columns else 0.05)
    _d = _d.dropna(subset=["x","y"])

    _cfg2 = {
        "Goal":    (COLOR_SECUNDARIO, "Gol concedido"),
        "Saved":   (COLOR_PRIMARIO,   "Parada"),
        "Blocked": ("#9a9a9a",        "Bloqueado"),
        "Off T":   ("#7d7d7d",        "Fuera"),
        "Post":    ("#b5b5b5",        "Poste"),
        "Wayward": ("#333333",        "Desviado"),
    }
    fig = go.Figure()
    for _out, (_color, _lbl) in _cfg2.items():
        _m = _d["outcome"] == _out
        if not _m.any(): continue
        _s = _d[_m]
        fig.add_trace(go.Scatter(
            x=_s["x"], y=_s["y"], mode="markers", name=_lbl,
            marker=dict(size=(_s["xg"] * 40).clip(6, 28), color=_color, opacity=0.85,
                        line=dict(width=1, color=COLOR_FONDO)),
            hovertemplate=f"<b>{_lbl}</b><br>xG: %{{customdata:.3f}}<extra></extra>",
            customdata=_s["xg"].values,
        ))
    fig.update_layout(
        **_pitch_layout(True, 440,
            f"Tiros recibidos — {_jugador}  ({len(_d)} tiros · {_gc_n} goles)"),
        legend=dict(orientation="h", yanchor="top", y=-0.04, xanchor="center", x=0.5,
                    font=dict(color=COLOR_TEXTO, size=10), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Origen de los tiros recibidos  ·  Tamaño ∝ xG  ·  Color principal = parada  ·  Color secundario = gol concedido")


def _render_gk_distribucion():
    st.markdown('<div class="section-header">📤 Distribución (Pases)</div>', unsafe_allow_html=True)
    _end_col = "pass_end_location"
    if _pa_j.empty or "location" not in _pa_j.columns or _end_col not in _pa_j.columns:
        st.info("Sin datos de distribución."); return

    _start = _parse_loc(_pa_j["location"])
    _end   = _parse_loc(_pa_j[_end_col]).rename(columns={"x":"x2","y":"y2"})
    _pm    = pd.concat([_start, _end], axis=1).dropna()
    _ok_mk = (_pa_j["pass_outcome"].isna()
              if "pass_outcome" in _pa_j.columns
              else pd.Series(True, index=_pa_j.index))
    _pm_ok  = _pm[_ok_mk.values[:len(_pm)]] if len(_ok_mk) == len(_pm) else _pm
    _pm_nok = _pm[~_ok_mk.values[:len(_pm)]] if len(_ok_mk) == len(_pm) else pd.DataFrame()

    def _blines(df, mx=200):
        df = df.sample(min(mx, len(df)), random_state=42) if len(df) > mx else df
        lx, ly = [], []
        for r in df.itertuples():
            lx += [r.x, r.x2, None]; ly += [r.y, r.y2, None]
        return lx, ly

    fig = go.Figure()
    if not _pm_nok.empty:
        _lx, _ly = _blines(_pm_nok, 80)
        fig.add_trace(go.Scatter(x=_lx, y=_ly, mode="lines",
            line=dict(color=_rgba(COLOR_SECUNDARIO, 0.3), width=1),
            hoverinfo="skip", name="Incompleto"))
    if not _pm_ok.empty:
        _lx, _ly = _blines(_pm_ok, 200)
        fig.add_trace(go.Scatter(x=_lx, y=_ly, mode="lines",
            line=dict(color=_rgba(COLOR_PRIMARIO, 0.4), width=1),
            hoverinfo="skip", name="Completado"))
    fig.update_layout(
        **_pitch_layout(False, 440, f"Distribución — {_jugador}  ({len(_pa_j):,} pases/salidas)"),
        legend=dict(orientation="h", yanchor="top", y=-0.04, xanchor="center", x=0.5,
                    font=dict(color=COLOR_TEXTO, size=10), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{len(_pa_j)} pases/salidas  ·  Color principal = completado  ·  Color secundario = incompleto  ·  (muestra máx. 280)")


def _render_gk_acciones():
    st.markdown('<div class="section-header">🧤 Acciones del Portero</div>', unsafe_allow_html=True)
    if _gk_events.empty:
        st.info("Sin acciones específicas de portero registradas."); return

    _gk_col = "goalkeeper_type" if "goalkeeper_type" in _gk_events.columns else "type"
    _counts = _gk_events[_gk_col].value_counts().head(12).sort_values()
    _gk_highlight = {"Shot Saved","Shot Saved Off Target","Collected","Touched Out","Punch"}
    _colors = [COLOR_PRIMARIO if t in _gk_highlight else _rgba(COLOR_TEXTO, 0.3)
               for t in _counts.index]

    fig = go.Figure(go.Bar(
        y=[_tipos_es.get(t, t) for t in _counts.index], x=_counts.values, orientation="h",
        marker_color=_colors, cliponaxis=False,
        text=_counts.values, textposition="outside",
        textfont=dict(color=COLOR_TEXTO, size=8),
        hovertemplate="<b>%{y}</b><br>Total: %{x}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=9),
        xaxis=dict(title="Cantidad", title_font=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO, 0.12), tickfont=dict(color=COLOR_TEXTO)),
        yaxis=dict(showgrid=False, tickfont=dict(color=COLOR_TEXTO, size=9)),
        margin=dict(t=20, b=30, l=10, r=50),
        height=440, showlegend=False,
        title=dict(text=f"Tipos de acción de portero — {_jugador}",
                   font=dict(color=COLOR_TEXTO, size=10), x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Tipos de evento Goal Keeper según StatsBomb  ·  Azul = acciones de parada principales")


# =============================================================================
# DICCIONARIO DE RENDERIZADO
# =============================================================================

_RENDERERS = {
    "📊 Radar de Métricas":       _render_radar,
    "🍕 Pizza Chart":              _render_pizza,
    "🔥 Heatmap Individual":       _render_heatmap_j,
    "🎯 Shot Map":                 _render_shot_map_j,
    "📈 xG por Partido":           _render_xg_partido_j,
    "➡️ Mapa de Pases":            _render_pass_map_j,
    "📋 Distribución de Acciones": _render_action_dist,
    "⏱️ Actividad por Minuto":     _render_activity_minute,
    # Portero
    "🥅 Paradas por Partido":      _render_gk_saves_partido,
    "🗺️ Tiros Recibidos":          _render_gk_tiros_recibidos,
    "📤 Distribución (Pases)":     _render_gk_distribucion,
    "🧤 Acciones del Portero":     _render_gk_acciones,
    "🔥 Heatmap de Acciones":      _render_heatmap_j,
}


# =============================================================================
# RENDERIZADO EN GRID 2 COLUMNAS
# =============================================================================

_pairs = [_charts_sel[i:i+2] for i in range(0, len(_charts_sel), 2)]

for _pair in _pairs:
    if len(_pair) == 2:
        _cl, _cr = st.columns(2)
        with _cl:
            _RENDERERS[_pair[0]]()
        with _cr:
            _RENDERERS[_pair[1]]()
    else:
        _cl, _ = st.columns([1, 1])
        with _cl:
            _RENDERERS[_pair[0]]()
    st.markdown("<br>", unsafe_allow_html=True)
