# =============================================================================
#  MASTER EN FOOTBALL ANALYTICS: Eventing, Tracking, Scouting e IA
#  Radar Tactico -- Formacion en habla hispana
# =============================================================================
#  Modulo 7  : Construccion del Dashboard en Streamlit
#  Leccion   : 13 -- pages/1_Equipo.py: Analisis de equipo
#  Filtros   : Partido · Período · Selector de hasta 6 gráficos
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

from utils.data import load_data, get_competition_info, nombre_corto, STAGE_ORDER
from utils.sidebar import render_sidebar


# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================

st.set_page_config(page_title="Análisis de Equipo", page_icon="🛡️", layout="wide")

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
    .filter-box {{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 1rem;
    }}
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
lineups = data.get("lineups", pd.DataFrame())


# =============================================================================
# SIDEBAR
# =============================================================================

equipo_sel = render_sidebar(teams["team"].tolist())


# =============================================================================
# HELPERS GLOBALES
# =============================================================================

def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# Escalas de color derivadas automáticamente de los colores del config
_CSCALE_PRI = [[0.0, COLOR_FONDO], [0.4, _rgba(COLOR_PRIMARIO, 0.5)], [1.0, COLOR_PRIMARIO]]
_CSCALE_SEC = [[0.0, COLOR_FONDO], [0.4, _rgba(COLOR_SECUNDARIO, 0.5)], [1.0, COLOR_SECUNDARIO]]

def _parse_loc(series: pd.Series) -> pd.DataFrame:
    """Parsea la columna location (JSON string '[x,y]') de StatsBomb."""
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
# CATÁLOGO DE GRÁFICOS DISPONIBLES
# =============================================================================

_ALL_CHARTS = [
    "🎯 Shot Map",
    "🔥 Heatmap de Acciones",
    "🕸️ Red de Pases",
    "📊 PPDA",
    "💨 Mapa de Presión",
    "🛡️ Acciones Defensivas",
    "📈 xG por Partido",
    "➡️ Mapa de Pases",
    "📉 xG Flow por Minuto",
    "🏃 Pases Progresivos",
    "⚽ Alineación",
]
_DEFAULT_CHARTS = _ALL_CHARTS[:4]
_MAX_CHARTS = 6


# =============================================================================
# ENCABEZADO
# =============================================================================

comp_name, season_name = get_competition_info(matches)
st.title("🛡️ Análisis de Equipo")
_caption_placeholder = st.empty()

if equipo_sel == "Todos":
    st.info("ℹ️  Selecciona un equipo en el sidebar para ver el análisis individual.")

st.divider()


# =============================================================================
# PANEL DE FILTROS
# =============================================================================

# Detectar si el torneo tiene múltiples fases (copas) o una sola (ligas)
_stage_col   = "competition_stage"
_stages_avail = []
if _stage_col in matches.columns:
    _stages_avail = sorted(
        matches[_stage_col].dropna().unique().tolist(),
        key=lambda s: STAGE_ORDER.get(s, 99)
    )
_has_stages = len(_stages_avail) > 1

st.markdown('<div class="filter-box">', unsafe_allow_html=True)

# Layout dinámico: con fase o sin fase
if _has_stages:
    _fc0, _fc1b, _fc1, _fc2, _fc3 = st.columns([2, 2, 3, 2, 5])
else:
    _fc0, _fc1, _fc2, _fc3 = st.columns([2, 3, 2, 5])
    _fc1b = None

# ── Filtro 0: Equipo ──────────────────────────────────────────────────────────
with _fc0:
    st.markdown("**🛡️ Equipo**")
    _team_list  = sorted(teams["team"].tolist())
    _def_idx    = _team_list.index(equipo_sel) if equipo_sel in _team_list else 0
    _equipo     = st.selectbox("Equipo", _team_list, index=_def_idx,
                               label_visibility="collapsed", key="eq_equipo")

_caption_placeholder.caption(f"{comp_name} {season_name}  ·  Equipo: **{_equipo}**")

# Partidos base del equipo
_mask_team    = (matches["home_team"] == _equipo) | (matches["away_team"] == _equipo)
_team_matches = matches[_mask_team].sort_values("match_date")

# ── Filtro 1b: Fase del torneo (solo torneos de copa) ─────────────────────────
if _has_stages and _fc1b is not None:
    with _fc1b:
        st.markdown("**🏆 Fase**")
        _stage_opts = {"Todas": None, **{s: s for s in _stages_avail}}
        _stage_sel  = st.selectbox("Fase", list(_stage_opts.keys()),
                                   label_visibility="collapsed", key="eq_stage")
        _stage_val  = _stage_opts[_stage_sel]
    # Aplicar filtro de fase
    if _stage_val is not None:
        _team_matches = _team_matches[_team_matches[_stage_col] == _stage_val]
else:
    _stage_val = None

_all_match_ids = _team_matches["match_id"].unique()

# ── Filtro 1: Partido ─────────────────────────────────────────────────────────
with _fc1:
    st.markdown("**📅 Partido**")
    _match_labels = {"Todos los partidos": None}
    for _row in _team_matches.itertuples():
        _home = getattr(_row, "home_team", "?")
        _away = getattr(_row, "away_team", "?")
        _hs   = getattr(_row, "home_score", "")
        _as_  = getattr(_row, "away_score", "")
        _dt   = str(getattr(_row, "match_date", ""))[:10]
        _lbl  = f"{_dt} · {_home} {_hs}–{_as_} {_away}"
        _match_labels[_lbl] = _row.match_id

    _match_sel  = st.selectbox("Partido", list(_match_labels.keys()), label_visibility="collapsed")
    _match_id_f = _match_labels[_match_sel]

# ── Filtro 2: Período ─────────────────────────────────────────────────────────
with _fc2:
    st.markdown("**⏱️ Período**")
    _period_opts = {"Completo": None, "1er Tiempo": [1], "2do Tiempo": [2]}
    if "period" in events.columns and (events["period"] >= 3).any():
        _period_opts["Prórroga"] = [3, 4]
    _period_sel  = st.radio("Período", list(_period_opts.keys()),
                            horizontal=False, label_visibility="collapsed")
    _period_val  = _period_opts[_period_sel]

# ── Filtro 3: Selector de gráficos ────────────────────────────────────────────
with _fc3:
    st.markdown(f"**📊 Gráficos a mostrar** *(máx. {_MAX_CHARTS})*")
    _charts_sel = st.multiselect(
        "Gráficos",
        options=_ALL_CHARTS,
        default=_DEFAULT_CHARTS,
        max_selections=_MAX_CHARTS,
        label_visibility="collapsed",
    )

st.markdown('</div>', unsafe_allow_html=True)

if not _charts_sel:
    st.info("Selecciona al menos un gráfico en el filtro de arriba.")
    st.stop()


# =============================================================================
# APLICAR FILTROS A LOS DATOS
# =============================================================================

# Filtro de partidos
_active_ids = np.array([_match_id_f]) if _match_id_f else _all_match_ids

_shots_for   = shots[(shots["team"] == _equipo)   & (shots["match_id"].isin(_active_ids))].copy()
_goals_for   = goals[(goals["team"] == _equipo)   & (goals["match_id"].isin(_active_ids))].copy()
_shots_ag    = shots[(shots["team"] != _equipo)   & (shots["match_id"].isin(_active_ids))].copy()
_goals_ag    = goals[(goals["team"] != _equipo)   & (goals["match_id"].isin(_active_ids))].copy()
_passes_team = passes[(passes["team"] == _equipo) & (passes["match_id"].isin(_active_ids))].copy()
_events_team = events[(events["team"] == _equipo) & (events["match_id"].isin(_active_ids))].copy()

# Filtro de período
if _period_val is not None and "period" in _events_team.columns:
    _shots_for   = _shots_for[_shots_for["period"].isin(_period_val)] if "period" in _shots_for.columns   else _shots_for
    _goals_for   = _goals_for[_goals_for["period"].isin(_period_val)] if "period" in _goals_for.columns   else _goals_for
    _shots_ag    = _shots_ag[_shots_ag["period"].isin(_period_val)] if "period" in _shots_ag.columns    else _shots_ag
    _goals_ag    = _goals_ag[_goals_ag["period"].isin(_period_val)] if "period" in _goals_ag.columns    else _goals_ag
    _passes_team = _passes_team[_passes_team["period"].isin(_period_val)] if "period" in _passes_team.columns else _passes_team
    _events_team = _events_team[_events_team["period"].isin(_period_val)]

_n_matches = len(_active_ids)
_xg_col    = "shot_statsbomb_xg"


# =============================================================================
# KPIs
# =============================================================================

# Los goles en propia puerta no son tiros en StatsBomb: se suman aparte
_og = events[(events["type"] == "Own Goal For") & (events["match_id"].isin(_active_ids))]
if _period_val is not None and "period" in _og.columns:
    _og = _og[_og["period"].isin(_period_val)]
_n_goals_for = len(_goals_for) + int((_og["team"] == _equipo).sum())
_n_goals_ag  = len(_goals_ag)  + int((_og["team"] != _equipo).sum())

_xg_for_total = _shots_for[_xg_col].sum() if _xg_col in _shots_for.columns else 0
_xg_ag_total  = _shots_ag[_xg_col].sum()  if _xg_col in _shots_ag.columns  else 0

_k1, _k2, _k3, _k4, _k5 = st.columns(5)
for _col, _label, _val, _sub in [
    (_k1, "Partidos",         _n_matches,              "jugados"),
    (_k2, "Goles anotados",   _n_goals_for,            f"{_n_goals_for/_n_matches:.1f}/partido" if _n_matches else "—"),
    (_k3, "Goles concedidos", _n_goals_ag,             f"{_n_goals_ag/_n_matches:.1f}/partido"  if _n_matches else "—"),
    (_k4, "xG generado",      f"{_xg_for_total:.1f}", f"{_xg_for_total/_n_matches:.2f}/partido"   if _n_matches else "—"),
    (_k5, "xG concedido",     f"{_xg_ag_total:.1f}",  f"{_xg_ag_total/_n_matches:.2f}/partido"    if _n_matches else "—"),
]:
    _col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{_label}</div>
        <div class="kpi-value">{_val}</div>
        <div class="kpi-sub">{_sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# FUNCIONES DE RENDERIZADO DE GRÁFICOS
# =============================================================================

def _render_shot_map():
    st.markdown('<div class="section-header">🎯 Shot Map</div>', unsafe_allow_html=True)
    if _shots_for.empty or "location" not in _shots_for.columns:
        st.info("Sin datos de tiros."); return

    _d = _parse_loc(_shots_for["location"])
    _d["outcome"] = _shots_for["shot_outcome"].fillna("Unknown").values
    _d["xg"]      = (_shots_for[_xg_col].fillna(0.05).values if _xg_col in _shots_for.columns else 0.05)
    _d["player"]  = _shots_for["player"].fillna("").values
    _d = _d.dropna(subset=["x", "y"])

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
            marker=dict(size=(_s["xg"] * 40).clip(6, 28), color=_color, opacity=0.85,
                        line=dict(width=0.8, color=COLOR_FONDO)),
            customdata=_s[["player", "xg", "outcome"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>xG: %{customdata[1]:.3f}<br>%{customdata[2]}<extra></extra>",
        ))
    fig.update_layout(
        **_pitch_layout(False, 440, f"Tiros — {_equipo}  ({len(_d)} tiros · {len(_goals_for)} goles)"),
        legend=dict(orientation="h", yanchor="top", y=-0.04, xanchor="center", x=0.5,
                    font=dict(color=COLOR_TEXTO, size=10), bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Tamaño ∝ xG  ·  Color según resultado  ·  Hover para detalle")


def _render_heatmap():
    st.markdown('<div class="section-header">🔥 Heatmap de Acciones</div>', unsafe_allow_html=True)
    if _events_team.empty or "location" not in _events_team.columns:
        st.info("Sin datos de posición."); return

    _loc = _parse_loc(_events_team["location"]).dropna()
    fig  = go.Figure()
    fig.add_trace(go.Histogram2d(
        x=_loc["x"], y=_loc["y"], xbins=dict(start=0, end=120, size=5), ybins=dict(start=0, end=80, size=5),
        colorscale=_CSCALE_PRI, showscale=False,
        hovertemplate="Zona x:%{x:.0f} y:%{y:.0f}<br>Acciones: %{z}<extra></extra>",
    ))
    fig.update_layout(**_pitch_layout(False, 440,
        f"Densidad de acciones — {_equipo}  ({len(_loc):,} eventos)"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Mayor intensidad = más acciones  ·  Hover para cantidad")


def _render_pass_network():
    st.markdown('<div class="section-header">🕸️ Red de Pases</div>', unsafe_allow_html=True)
    if _passes_team.empty or "location" not in _passes_team.columns or "pass_recipient" not in _passes_team.columns:
        st.info("Sin datos de pases."); return

    _lp = _parse_loc(_passes_team["location"])
    _lp["player"]    = _passes_team["player"].values
    _lp["recipient"] = _passes_team["pass_recipient"].values
    _lp = _lp.dropna(subset=["x", "y", "player"])

    _avg = (_lp.groupby("player")
            .agg(x=("x","mean"), y=("y","mean"), n=("x","count"))
            .reset_index().nlargest(11, "n"))
    _top = set(_avg["player"])

    _net = _lp[["player","recipient"]].rename(columns={"player":"passer"}).dropna()
    _net = _net[_net["passer"].isin(_top) & _net["recipient"].isin(_top) & (_net["passer"] != _net["recipient"])]
    _conn = _net.groupby(["passer","recipient"]).size().reset_index(name="passes").nlargest(30,"passes")

    _pos = {r.player: (r.x, r.y) for r in _avg.itertuples()}
    _lx, _ly = [], []
    for r in _conn.itertuples():
        if r.passer in _pos and r.recipient in _pos:
            x1, y1 = _pos[r.passer]; x2, y2 = _pos[r.recipient]
            _lx += [x1, x2, None]; _ly += [y1, y2, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_lx, y=_ly, mode="lines",
        line=dict(color=_rgba(COLOR_PRIMARIO, 0.45), width=2),
        hoverinfo="skip", showlegend=False))
    _max_n = _avg["n"].max()
    fig.add_trace(go.Scatter(
        x=_avg["x"], y=_avg["y"], mode="markers+text",
        text=_avg["player"].apply(nombre_corto),
        textposition="top center", textfont=dict(size=8, color=COLOR_TEXTO),
        marker=dict(size=(8 + (_avg["n"] / _max_n) * 22).clip(8, 30),
                    color=COLOR_PRIMARIO, line=dict(width=2, color=COLOR_FONDO)),
        customdata=_avg[["player","n"]].values,
        hovertemplate="<b>%{customdata[0]}</b><br>Pases: %{customdata[1]}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**_pitch_layout(False, 440, f"Red de pases — {_equipo}  (top 11)"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Grosor ∝ pases entre el par  ·  Tamaño ∝ pases totales")


def _render_ppda():
    st.markdown('<div class="section-header">📊 PPDA — Intensidad de Presión</div>', unsafe_allow_html=True)
    # PPDA de todo el torneo, calculado una sola vez en utils/data.py
    _rows = [{"team": _t, "PPDA": round(float(_v), 2)} for _t, _v in data["ppda"].items()]
    if not _rows:
        st.info("Sin datos para PPDA."); return
    _df = pd.DataFrame(_rows).sort_values("PPDA")
    _colors = [COLOR_PRIMARIO if t == _equipo else _rgba(COLOR_TEXTO, 0.25) for t in _df["team"]]
    fig = go.Figure(go.Bar(
        y=_df["team"], x=_df["PPDA"], orientation="h", marker_color=_colors,
        text=_df["PPDA"].apply(lambda v: f"{v:.1f}"), textposition="outside",
        textfont=dict(color=COLOR_TEXTO, size=8),
        hovertemplate="<b>%{y}</b><br>PPDA: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=9),
        xaxis=dict(title="PPDA", title_font=dict(color=COLOR_TEXTO, size=9),
                   tickfont=dict(color=COLOR_TEXTO), gridcolor=_rgba(COLOR_TEXTO, 0.12)),
        yaxis=dict(showgrid=False, tickfont=dict(color=COLOR_TEXTO, size=8)),
        margin=dict(t=20, b=30, l=10, r=50),
        height=440, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"**{_equipo}** resaltado  ·  PPDA = pases del rival en su 60 % de campo por cada acción defensiva en esa zona  ·  Menor = más presionante  ·  Todo el torneo")


def _render_pressure_map():
    st.markdown('<div class="section-header">💨 Mapa de Presión</div>', unsafe_allow_html=True)
    _press = _events_team[_events_team["type"] == "Pressure"].copy()
    if _press.empty or "location" not in _press.columns:
        st.info("Sin datos de presión."); return
    _loc = _parse_loc(_press["location"]).dropna()
    fig  = go.Figure()
    fig.add_trace(go.Histogram2d(
        x=_loc["x"], y=_loc["y"], xbins=dict(start=0, end=120, size=5), ybins=dict(start=0, end=80, size=5),
        colorscale=_CSCALE_PRI, showscale=False,
        hovertemplate="x:%{x:.0f} y:%{y:.0f}<br>Presiones: %{z}<extra></extra>",
    ))
    fig.update_layout(**_pitch_layout(False, 440,
        f"Zonas de presión — {_equipo}  ({len(_loc):,} acciones de pressing)"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Mayor intensidad = más pressing aplicado en esa zona")


def _render_defensive_map():
    st.markdown('<div class="section-header">🛡️ Acciones Defensivas</div>', unsafe_allow_html=True)
    _def_types = {"Duel", "Interception", "Clearance", "Ball Recovery", "Foul Won"}
    _def_ev = _events_team[_events_team["type"].isin(_def_types)].copy()
    if _def_ev.empty or "location" not in _def_ev.columns:
        st.info("Sin datos defensivos."); return
    _loc = _parse_loc(_def_ev["location"]).dropna()
    fig  = go.Figure()
    fig.add_trace(go.Histogram2d(
        x=_loc["x"], y=_loc["y"], xbins=dict(start=0, end=120, size=5), ybins=dict(start=0, end=80, size=5),
        colorscale=_CSCALE_SEC, showscale=False,
        hovertemplate="x:%{x:.0f} y:%{y:.0f}<br>Acciones: %{z}<extra></extra>",
    ))
    fig.update_layout(**_pitch_layout(False, 440,
        f"Acciones defensivas — {_equipo}  ({len(_loc):,} acciones)"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Tackles · Intercepciones · Despejes · Recuperaciones  ·  Mayor densidad = zona de recuperación preferida")


def _render_xg_per_match():
    st.markdown('<div class="section-header">📈 xG por Partido</div>', unsafe_allow_html=True)
    if _xg_col not in shots.columns:
        st.info("Sin datos de xG."); return

    _xg_for_m = (shots[(shots["team"]==_equipo) & (shots["match_id"].isin(_all_match_ids))]
                 .groupby("match_id")[_xg_col].sum().reset_index(name="xG_for"))
    _xg_ag_m  = (shots[(shots["team"]!=_equipo) & (shots["match_id"].isin(_all_match_ids))]
                 .groupby("match_id")[_xg_col].sum().reset_index(name="xG_ag"))

    _match_info = _team_matches[["match_id","match_date","home_team","away_team",
                                  "home_score","away_score"]].copy()
    _df = _match_info.merge(_xg_for_m, on="match_id", how="left") \
                     .merge(_xg_ag_m,  on="match_id", how="left").fillna(0)
    _df["label"] = _df.apply(
        lambda r: f"{str(r.match_date)[:10]}<br>{r.home_team} {r.home_score}–{r.away_score} {r.away_team}", axis=1
    )
    _df["xG_for"] = _df["xG_for"].round(2)
    _df["xG_ag"]  = _df["xG_ag"].round(2)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=_df["label"], y=_df["xG_for"], name="xG generado",
        marker_color=COLOR_PRIMARIO, opacity=0.88,
        hovertemplate="<b>%{x}</b><br>xG generado: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=_df["label"], y=_df["xG_ag"], name="xG concedido",
        marker_color=COLOR_SECUNDARIO, opacity=0.88,
        hovertemplate="<b>%{x}</b><br>xG concedido: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        barmode="group", paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=9),
        xaxis=dict(showgrid=False, tickfont=dict(color=COLOR_TEXTO, size=8)),
        yaxis=dict(title="xG", title_font=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO, 0.12), tickfont=dict(color=COLOR_TEXTO)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    font=dict(color=COLOR_TEXTO), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=70, b=60, l=40, r=10),
        height=440,
        title=dict(y=0.97, text=f"xG por partido — {_equipo}", font=dict(color=COLOR_TEXTO, size=10), x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Barras de xG generado vs xG concedido por partido")


def _render_xg_flow():
    st.markdown('<div class="section-header">📉 xG Flow por Minuto</div>', unsafe_allow_html=True)
    if _match_id_f is None:
        st.info("Selecciona un **partido específico** en el filtro para ver el flujo de xG minuto a minuto.")
        return
    if _xg_col not in shots.columns:
        st.info("Sin datos de xG."); return

    _sh_m = shots[shots["match_id"] == _match_id_f].copy()
    if _sh_m.empty or "minute" not in _sh_m.columns:
        st.info("Sin datos de tiros para este partido."); return

    _mr = matches[matches["match_id"] == _match_id_f].iloc[0]
    _opp = _mr["away_team"] if _mr["home_team"] == _equipo else _mr["home_team"]
    _max_min = max(int(_sh_m["minute"].max()) + 3, 93)
    _mins = list(range(0, _max_min + 1))

    def _cum_xg(team):
        _t = _sh_m[_sh_m["team"] == team].copy()
        _per = _t.groupby("minute")[_xg_col].sum()
        _full = pd.Series(0.0, index=range(0, _max_min + 1))
        _full.update(_per)
        return _full.cumsum().round(3).values

    _xg_team = _cum_xg(_equipo)
    _xg_opp  = _cum_xg(_opp)

    fig = go.Figure()
    for _vals, _color, _name in [(_xg_team, COLOR_PRIMARIO, _equipo), (_xg_opp, COLOR_SECUNDARIO, _opp)]:
        fig.add_trace(go.Scatter(
            x=_mins, y=_vals, name=_name, mode="lines",
            line=dict(color=_color, width=2.5),
            fill="tozeroy", fillcolor=_rgba(_color, 0.12),
            hovertemplate=f"<b>{_name}</b><br>Min %{{x}}: xG acumulado %{{y:.3f}}<extra></extra>",
        ))

    # Goles como anotaciones
    _go_m = goals[goals["match_id"] == _match_id_f]
    if not _go_m.empty and "minute" in _go_m.columns:
        for _, _g in _go_m.iterrows():
            _is_t = _g["team"] == _equipo
            _gc   = COLOR_PRIMARIO if _is_t else COLOR_SECUNDARIO
            _gmin = int(_g.get("minute", 0))
            _gmin = min(_gmin, _max_min)
            _yv   = _xg_team[_gmin] if _is_t else _xg_opp[_gmin]
            fig.add_annotation(x=_gmin, y=_yv, text="⚽", showarrow=True,
                               arrowhead=2, arrowcolor=_gc, arrowsize=0.8,
                               font=dict(size=11), bgcolor="rgba(0,0,0,0)")

    fig.add_vline(x=45, line_dash="dot", line_color=_rgba(COLOR_TEXTO, 0.3), line_width=1)
    fig.add_annotation(x=46, y=0, yref="paper", text="HT", showarrow=False,
                       font=dict(color=_rgba(COLOR_TEXTO, 0.45), size=8), yanchor="bottom")

    try:
        _hs  = int(_mr.get("home_score", 0)) if pd.notna(_mr.get("home_score")) else 0
        _as_ = int(_mr.get("away_score", 0)) if pd.notna(_mr.get("away_score")) else 0
        _ttl = f"{_mr['home_team']} {_hs}–{_as_} {_mr['away_team']}"
    except Exception:
        _ttl = ""

    fig.update_layout(
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=9),
        xaxis=dict(title="Minuto", title_font=dict(color=COLOR_TEXTO),
                   tickfont=dict(color=COLOR_TEXTO), gridcolor=_rgba(COLOR_TEXTO, 0.1),
                   range=[0, _max_min + 2]),
        yaxis=dict(title="xG acumulado", title_font=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO, 0.12), tickfont=dict(color=COLOR_TEXTO)),
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5,
                    font=dict(color=COLOR_TEXTO), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=40, b=60, l=50, r=10), height=400,
        title=dict(text=_ttl, font=dict(color=COLOR_TEXTO, size=10), x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("⚽ = gol  ·  Línea punteada = descanso  ·  Selecciona un partido en el filtro superior")


def _render_progressive_passes():
    st.markdown('<div class="section-header">🏃 Pases Progresivos</div>', unsafe_allow_html=True)
    if _passes_team.empty or "location" not in _passes_team.columns:
        st.info("Sin datos de pases."); return
    _end_col = "pass_end_location"
    if _end_col not in _passes_team.columns:
        st.info("La columna pass_end_location no está disponible."); return

    _start = _parse_loc(_passes_team["location"])
    _end   = _parse_loc(_passes_team[_end_col]).rename(columns={"x": "x2", "y": "y2"})
    _pm    = pd.concat([_start, _end], axis=1).dropna()

    _pm = _pm.assign(player=_passes_team["player"].values[:len(_pm)])

    # Pase progresivo: avanza ≥10m en x Y termina en campo ofensivo
    _prog = _pm[(_pm["x2"] - _pm["x"] >= 10) & (_pm["x2"] >= 60)].copy()
    _prog["ganancia"] = (_prog["x2"] - _prog["x"]).clip(0, 60)

    if _prog.empty:
        st.info("Sin pases progresivos con los filtros actuales."); return

    _s = _prog.sample(min(500, len(_prog)), random_state=42) if len(_prog) > 500 else _prog

    _lx, _ly = [], []
    for r in _s.itertuples():
        _lx += [r.x, r.x2, None]
        _ly += [r.y, r.y2, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=_lx, y=_ly, mode="lines",
        line=dict(color=_rgba(COLOR_PRIMARIO, 0.22), width=1),
        hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=_s["x2"], y=_s["y2"], mode="markers", showlegend=False,
        marker=dict(size=6, color=_s["ganancia"], colorscale=_CSCALE_PRI, showscale=True,
                    colorbar=dict(title=dict(text="Metros ganados", font=dict(color=COLOR_TEXTO)),
                                  tickfont=dict(color=COLOR_TEXTO), thickness=10),
                    opacity=0.8, line=dict(width=0.4, color=COLOR_FONDO)),
        customdata=np.stack([_s["player"].fillna("").values, _s["ganancia"].round(1).values], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>Ganancia: %{customdata[1]}m<extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=_s["x"], y=_s["y"], mode="markers",
        marker=dict(size=3, color=_rgba(COLOR_PRIMARIO, 0.35),
                    line=dict(width=0.2, color=COLOR_FONDO)),
        hoverinfo="skip", showlegend=False))

    fig.update_layout(**_pitch_layout(False, 440,
        f"Pases progresivos — {_equipo}  ({len(_prog):,} pases)"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Pase que avanza ≥10m y termina en campo ofensivo  ·  Color = metros ganados  ·  Punto = origen · Flecha = destino")


def _render_pass_map():
    st.markdown('<div class="section-header">➡️ Mapa de Pases</div>', unsafe_allow_html=True)
    if _passes_team.empty or "location" not in _passes_team.columns:
        st.info("Sin datos de pases."); return
    _end_col = "pass_end_location"
    if _end_col not in _passes_team.columns:
        st.info("La columna pass_end_location no está disponible."); return

    _start = _parse_loc(_passes_team["location"])
    _end   = _parse_loc(_passes_team[_end_col]).rename(columns={"x":"x2","y":"y2"})
    _pm    = pd.concat([_start, _end], axis=1).dropna()

    # Muestreo para no sobrecargar el render
    _sample = _pm.sample(min(150, len(_pm)), random_state=42) if len(_pm) > 150 else _pm

    _lx, _ly = [], []
    for r in _sample.itertuples():
        _lx += [r.x, r.x2, None]
        _ly += [r.y, r.y2, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_lx, y=_ly, mode="lines",
        line=dict(color=_rgba(COLOR_PRIMARIO, 0.3), width=1),
        hoverinfo="skip", showlegend=False,
    ))
    # Puntos de origen
    fig.add_trace(go.Scatter(
        x=_sample["x"], y=_sample["y"], mode="markers",
        marker=dict(size=4, color=COLOR_PRIMARIO, opacity=0.6),
        hovertemplate="Origen: (%{x:.0f}, %{y:.0f})<extra></extra>",
        name="Origen", showlegend=False,
    ))
    fig.update_layout(**_pitch_layout(False, 440,
        f"Mapa de pases — {_equipo}  (muestra de {len(_sample):,})"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Muestra de {len(_sample)} pases  ·  Línea = trayectoria  ·  Punto = origen")


def _render_lineup():
    import json as _j
    st.markdown('<div class="section-header">⚽ Alineación</div>', unsafe_allow_html=True)
    if _match_id_f is None:
        st.info("Selecciona un **partido específico** en el filtro para ver la alineación.")
        return
    if lineups.empty:
        st.info("Datos de alineación no disponibles.")
        return

    _POS_XY = {
        "Goalkeeper":               (6,  40),
        "Right Back":               (24, 68),
        "Right Center Back":        (20, 56),
        "Center Back":              (20, 40),
        "Left Center Back":         (20, 24),
        "Left Back":                (24, 12),
        "Right Wing Back":          (42, 72),
        "Left Wing Back":           (42,  8),
        "Right Defensive Midfield": (42, 62),
        "Defensive Midfield":       (42, 40),
        "Center Defensive Midfield":(42, 40),
        "Center Attacking Midfield":(70, 40),
        "Left Defensive Midfield":  (42, 18),
        "Right Center Midfield":    (57, 62),
        "Central Midfield":         (57, 40),
        "Left Center Midfield":     (57, 18),
        "Right Midfield":           (55, 72),
        "Left Midfield":            (55,  8),
        "Right Attacking Midfield": (72, 62),
        "Attacking Midfield":       (70, 40),
        "Left Attacking Midfield":  (72, 18),
        "Right Wing":               (82, 72),
        "Left Wing":                (82,  8),
        "Right Center Forward":     (87, 58),
        "Center Forward":           (90, 40),
        "Left Center Forward":      (87, 22),
        "Secondary Striker":        (80, 40),
    }

    def _parse_starter(val):
        try:
            if isinstance(val, str):
                val = _j.loads(val)
            if isinstance(val, list):
                for entry in val:
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("start_reason") == "Starting XI":
                        return entry.get("position", ""), True
                    _from = entry.get("from", "") or ""
                    if _from.startswith("00:00") and entry.get("from_period") == 1:
                        return entry.get("position", ""), True
        except Exception:
            pass
        return "", False

    def _build_trace(team_name: str, mirror: bool):
        """Extrae titulares de un equipo y devuelve (xs, ys, texts, hovers)."""
        _lu = lineups[(lineups["match_id"] == _match_id_f) &
                      (lineups["team_name"] == team_name)].copy()
        if _lu.empty:
            return [], [], [], []

        _pos_col = "positions" if "positions" in _lu.columns else None
        if _pos_col:
            _parsed = _lu[_pos_col].apply(_parse_starter)
            _lu["_pos"]      = _parsed.apply(lambda t: t[0])
            _lu["_is_start"] = _parsed.apply(lambda t: t[1])
        else:
            _lu["_pos"] = ""; _lu["_is_start"] = False

        _lu["_name"] = _lu["player_name"].fillna("") if "player_name" in _lu.columns else ""
        _lu["_num"]  = _lu["jersey_number"].apply(
            lambda v: str(int(v)) if pd.notna(v) and str(v) not in ("", "nan") else ""
        ) if "jersey_number" in _lu.columns else ""

        _st = _lu[_lu["_is_start"]].copy()
        if _st.empty:
            _st = (_lu.nsmallest(11, "jersey_number").copy()
                   if "jersey_number" in _lu.columns else _lu.head(11).copy())

        xs, ys, texts, hovers = [], [], [], []
        for _, row in _st.iterrows():
            _xy = _POS_XY.get(row["_pos"])
            if _xy is None:
                continue
            _x, _y = _xy
            # Cada equipo se dibuja en su propia mitad para que no se encimen
            _x = 4 + _x * 0.45
            # El rival ataca en dirección opuesta: espejo en x e y
            if mirror:
                _x = 120 - _x
                _y = 80  - _y
            _short = nombre_corto(row["_name"])
            _tp = "top center" if mirror else "bottom center"
            xs.append(_x); ys.append(_y)
            texts.append(f"{row['_num']}  {_short}")
            hovers.append((f"{row['_name']}<br><i>{row['_pos']}</i>", _tp))
        return xs, ys, texts, hovers

    _mr = matches[matches["match_id"] == _match_id_f].iloc[0]
    _opp = _mr["away_team"] if _mr["home_team"] == _equipo else _mr["home_team"]
    try:
        _hs  = int(_mr.get("home_score", 0)) if pd.notna(_mr.get("home_score")) else 0
        _as_ = int(_mr.get("away_score", 0)) if pd.notna(_mr.get("away_score")) else 0
        _ttl = f"{_mr['home_team']} {_hs}–{_as_} {_mr['away_team']}"
    except Exception:
        _ttl = f"{_equipo} vs {_opp}"

    _xs_a, _ys_a, _tx_a, _hv_a = _build_trace(_equipo, mirror=False)
    _xs_b, _ys_b, _tx_b, _hv_b = _build_trace(_opp,    mirror=True)

    if not _xs_a and not _xs_b:
        st.info("No se pudo mapear la posición de los jugadores.")
        return

    fig = go.Figure()

    # Equipo seleccionado — ataca izquierda → derecha
    if _xs_a:
        fig.add_trace(go.Scatter(
            x=_xs_a, y=_ys_a,
            mode="markers+text",
            name=_equipo,
            text=_tx_a,
            textposition="bottom center",
            textfont=dict(size=8, color=COLOR_TEXTO),
            marker=dict(size=22, color=COLOR_PRIMARIO,
                        line=dict(width=2, color=COLOR_FONDO)),
            customdata=[h[0] for h in _hv_a],
            hovertemplate="<b>%{customdata}</b><extra></extra>",
        ))

    # Rival — ataca derecha → izquierda (espejado)
    if _xs_b:
        fig.add_trace(go.Scatter(
            x=_xs_b, y=_ys_b,
            mode="markers+text",
            name=_opp,
            text=_tx_b,
            textposition="bottom center",
            textfont=dict(size=8, color=COLOR_TEXTO),
            marker=dict(size=22, color=COLOR_SECUNDARIO,
                        line=dict(width=2, color=COLOR_FONDO)),
            customdata=[h[0] for h in _hv_b],
            hovertemplate="<b>%{customdata}</b><extra></extra>",
        ))

    _layout = _pitch_layout(False, 520, _ttl)
    _layout["legend"] = dict(
        orientation="h", yanchor="top", y=-0.04, xanchor="center", x=0.5,
        font=dict(color=COLOR_TEXTO, size=10), bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(**_layout)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"<span style='color:{COLOR_PRIMARIO}'>■</span> **{_equipo}** ataca → "
        f"  <span style='color:{COLOR_SECUNDARIO}'>■</span> **{_opp}** ataca ←  "
        f"· Hover para nombre completo y posición",
        unsafe_allow_html=True,
    )


# =============================================================================
# DICCIONARIO DE RENDERIZADO
# =============================================================================

_RENDERERS = {
    "🎯 Shot Map":             _render_shot_map,
    "🔥 Heatmap de Acciones":  _render_heatmap,
    "🕸️ Red de Pases":         _render_pass_network,
    "📊 PPDA":                 _render_ppda,
    "💨 Mapa de Presión":      _render_pressure_map,
    "🛡️ Acciones Defensivas":  _render_defensive_map,
    "📈 xG por Partido":       _render_xg_per_match,
    "➡️ Mapa de Pases":        _render_pass_map,
    "📉 xG Flow por Minuto":   _render_xg_flow,
    "🏃 Pases Progresivos":    _render_progressive_passes,
    "⚽ Alineación":           _render_lineup,
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


