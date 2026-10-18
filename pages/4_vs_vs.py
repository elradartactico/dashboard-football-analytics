# =============================================================================
#  MASTER EN FOOTBALL ANALYTICS: Eventing, Tracking, Scouting e IA
#  Radar Tactico -- Formacion en habla hispana
# =============================================================================
#  Modulo 7  : Construccion del Dashboard en Streamlit
#  Leccion   : 16 -- pages/4_vs_vs.py: Comparacion Equipo vs Equipo
#  Graficos  : Radar dual · Barras comparativas · Heatmaps · Shot Maps
#              Head to Head · Top goleadores · xG por partido · PPDA
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
from plotly.subplots import make_subplots

from config import COLOR_PRIMARIO, COLOR_SECUNDARIO, NOMBRE_ANALISTA
try:
    from config import COLOR_TEXTO, COLOR_FONDO
except ImportError:
    COLOR_TEXTO = "#e8eaf0"
    COLOR_FONDO = "#0d1117"

from utils.data import load_data, get_competition_info, STAGE_ORDER
from utils.sidebar import render_sidebar


# =============================================================================
# CONFIGURACIÓN DE PÁGINA
# =============================================================================

st.set_page_config(page_title="Equipo vs Equipo", page_icon="⚔️", layout="wide")

st.markdown(f"""
<style>
    #MainMenu {{ visibility: hidden; }}
    footer    {{ visibility: hidden; }}
    .section-header {{
        font-size: 1rem; font-weight: 600; color: {COLOR_PRIMARIO};
        text-transform: uppercase; letter-spacing: 1.5px;
        border-bottom: 2px solid {COLOR_PRIMARIO}44;
        padding-bottom: 0.3rem; margin-bottom: 0.8rem;
    }}
    .kpi-card {{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px; padding: 0.9rem 1rem; text-align: center;
    }}
    .kpi-label {{ font-size: 0.72rem; color: inherit; opacity: 0.62; text-transform: uppercase; letter-spacing: 1px; font-size: clamp(0.58rem, 0.9vw, 0.75rem); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .kpi-value {{ font-size: 1.8rem; font-weight: 700; color: {COLOR_PRIMARIO}; line-height: 1.2; }}
    .kpi-sub   {{ font-size: 0.78rem; color: inherit; opacity: 0.7; }}
    .vs-header {{
        display: flex; align-items: center; justify-content: center;
        gap: 1.5rem; padding: 0.8rem 0; margin-bottom: 0.5rem;
    }}
    .vs-team-a {{ font-size: 1.6rem; font-weight: 800; color: {COLOR_PRIMARIO}; text-align: right; flex: 1; }}
    .vs-badge  {{ font-size: 1.2rem; font-weight: 700; color: inherit; opacity: 0.62; }}
    .vs-team-b {{ font-size: 1.6rem; font-weight: 800; color: {COLOR_SECUNDARIO}; text-align: left; flex: 1; }}
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

def _parse_loc(series: pd.Series) -> pd.DataFrame:
    xs, ys = [], []
    for v in series:
        try:
            if isinstance(v, str): v = _json.loads(v)
            xs.append(float(v[0])); ys.append(float(v[1]))
        except Exception:
            xs.append(np.nan); ys.append(np.nan)
    return pd.DataFrame({"x": xs, "y": ys}, index=series.index)

_XG_COL  = "shot_statsbomb_xg"
_COLOR_A = COLOR_PRIMARIO
_COLOR_B = COLOR_SECUNDARIO


# =============================================================================
# ESTADÍSTICAS POR EQUIPO (para radar y comparativas)
# =============================================================================

@st.cache_data(show_spinner=False)
def _build_team_stats(_matches, _events, _shots, _passes, _teams):
    rows = []
    for t in _teams["team"].tolist():
        mids_t   = _matches[(_matches["home_team"]==t)|(_matches["away_team"]==t)]["match_id"].unique()
        n        = len(mids_t)
        if n == 0: continue

        ev_t  = _events[(_events["team"]==t) & (_events["match_id"].isin(mids_t))]
        ev_op = _events[(_events["team"]!=t) & (_events["match_id"].isin(mids_t))]
        sh_t  = _shots[(_shots["team"]==t)   & (_shots["match_id"].isin(mids_t))]
        sh_op = _shots[(_shots["team"]!=t)   & (_shots["match_id"].isin(mids_t))]
        pa_t  = _passes[(_passes["team"]==t) & (_passes["match_id"].isin(mids_t))]
        pa_op = _passes[(_passes["team"]!=t) & (_passes["match_id"].isin(mids_t))]

        xg_for  = sh_t[_XG_COL].sum()  if _XG_COL in sh_t.columns  else 0
        xg_ag   = sh_op[_XG_COL].sum() if _XG_COL in sh_op.columns else 0
        # Goles de tiro + goles en propia puerta a favor (no son tiros en StatsBomb)
        goals_f = int((sh_t["shot_outcome"]=="Goal").sum())  + int((ev_t["type"]=="Own Goal For").sum())
        goals_a = int((sh_op["shot_outcome"]=="Goal").sum()) + int((ev_op["type"]=="Own Goal For").sum())

        pa_ok  = pa_t["pass_outcome"].isna().sum() if "pass_outcome" in pa_t.columns else len(pa_t)
        pa_ok_op = pa_op["pass_outcome"].isna().sum() if "pass_outcome" in pa_op.columns else len(pa_op)
        pass_pct = pa_ok / len(pa_t) * 100 if len(pa_t) > 0 else 0

        def_types = {"Duel","Interception","Clearance","Ball Recovery"}
        n_def_act = {"Duel","Interception","Foul Committed","Pressure"}
        ppda_def  = len(ev_t[ev_t["type"].isin(n_def_act)])
        ppda_pa   = len(pa_op)
        ppda      = float(data["ppda"].get(t, np.nan))   # definicion habitual, ver utils/data.py

        rows.append({
            "team":         t,
            "n_matches":    n,
            "goals_pm":     goals_f / n,
            "goals_ag_pm":  goals_a / n,
            "xg_pm":        xg_for  / n,
            "xga_pm":       xg_ag   / n,
            "shots_pm":     len(sh_t)  / n,
            "shots_ag_pm":  len(sh_op) / n,
            "passes_pm":    len(pa_t)  / n,
            "pass_pct":     pass_pct,
            "pressures_pm": len(ev_t[ev_t["type"]=="Pressure"]) / n,
            "defensive_pm": len(ev_t[ev_t["type"].isin(def_types)]) / n,
            "ppda":         ppda,
        })
    return pd.DataFrame(rows).set_index("team")

_team_stats = _build_team_stats(matches, events, shots, passes, teams)

_RADAR_KEYS   = ["goals_pm","xg_pm","shots_pm","passes_pm",
                 "pass_pct","pressures_pm","defensive_pm"]
_RADAR_LABELS = ["Goles/P","xG/P","Tiros/P","Pases/P",
                 "% Pases","Presiones/P","Defensiva/P"]

def _team_percentiles(team: str) -> list[float]:
    pcts = []
    for k in _RADAR_KEYS:
        if k in _team_stats.columns and team in _team_stats.index:
            dist = _team_stats[k].dropna().values
            val  = _team_stats.loc[team, k]
            pcts.append(float((np.sum(dist < val) + np.sum(dist == val) / 2) / len(dist) * 100) if len(dist) else 50.0)
        else:
            pcts.append(50.0)
    return pcts

def _tv(team: str, key: str, default=0):
    """Valor de una métrica para un equipo, con fallback."""
    try:
        return _team_stats.loc[team, key]
    except Exception:
        return default


# =============================================================================
# ENCABEZADO Y SELECCIÓN DE EQUIPOS
# =============================================================================

comp_name, season_name = get_competition_info(matches)
st.title("⚔️ Equipo vs Equipo")
st.caption(f"{comp_name} {season_name}")
st.divider()

_team_list = sorted(teams["team"].tolist())

_fc1, _fc2, _fc3 = st.columns([4, 1, 4])
with _fc1:
    _default_a = _team_list.index(equipo_sel) if equipo_sel in _team_list else 0
    _team_a = st.selectbox("Equipo A", _team_list, index=_default_a, key="ta")
with _fc2:
    st.markdown("<div style='text-align:center;padding-top:1.8rem;font-size:1.4rem;font-weight:700;opacity:0.6'>VS</div>",
                unsafe_allow_html=True)
with _fc3:
    # Por defecto, el rival del ultimo partido del equipo A (p. ej. la final)
    _default_b = (_default_a + 1) % len(_team_list)
    _pa = matches[(matches["home_team"] == _team_a) | (matches["away_team"] == _team_a)].sort_values("match_date")
    if not _pa.empty:
        _u = _pa.iloc[-1]
        _rival = _u["away_team"] if _u["home_team"] == _team_a else _u["home_team"]
        if _rival in _team_list:
            _default_b = _team_list.index(_rival)
    _team_b = st.selectbox("Equipo B", _team_list, index=_default_b, key="tb")

if _team_a == _team_b:
    st.warning("Selecciona dos equipos diferentes.")
    st.stop()

# ── Filtro de fase (solo torneos con múltiples etapas) ────────────────────────
_stage_col    = "competition_stage"
_stages_avail = []
if _stage_col in matches.columns:
    _stages_avail = sorted(
        matches[_stage_col].dropna().unique().tolist(),
        key=lambda s: STAGE_ORDER.get(s, 99)
    )
_has_stages = len(_stages_avail) > 1

if _has_stages:
    _stage_opts = {"Todas las fases": None, **{s: s for s in _stages_avail}}
    _sv = st.selectbox("🏆 Fase del torneo", list(_stage_opts.keys()),
                       label_visibility="visible", key="vs_stage")
    _stage_val = _stage_opts[_sv]
else:
    _stage_val = None

# Selector de gráficos
_ALL_CHARTS = [
    "🕸️ Radar comparativo",
    "📊 Barras comparativas",
    "🔥 Heatmaps lado a lado",
    "🎯 Shot Maps duales",
    "🏆 Head to Head",
    "⚽ Top goleadores",
    "📈 xG por partido",
    "📋 PPDA y presión",
]
_DEFAULT = _ALL_CHARTS[:4]

_gc = st.columns([8, 4])
with _gc[1]:
    _charts_sel = st.multiselect("📊 Gráficos *(máx. 6)*", _ALL_CHARTS, default=_DEFAULT,
                                 max_selections=6, label_visibility="visible")

if not _charts_sel:
    st.info("Selecciona al menos un gráfico.")
    st.stop()

st.markdown(f"""
<div class="vs-header">
    <div class="vs-team-a">{_team_a}</div>
    <div class="vs-badge">VS</div>
    <div class="vs-team-b">{_team_b}</div>
</div>""", unsafe_allow_html=True)


# =============================================================================
# DATOS FILTRADOS PARA AMBOS EQUIPOS
# =============================================================================

def _mids(team):
    _m = matches[(matches["home_team"]==team)|(matches["away_team"]==team)]
    if _stage_val is not None and _stage_col in _m.columns:
        _m = _m[_m[_stage_col] == _stage_val]
    return _m["match_id"].unique()

_mids_a, _mids_b = _mids(_team_a), _mids(_team_b)

# Eventos, tiros, pases por equipo
_ev_a  = events[(events["team"]==_team_a) & (events["match_id"].isin(_mids_a))]
_ev_b  = events[(events["team"]==_team_b) & (events["match_id"].isin(_mids_b))]
_sh_a  = shots [(shots ["team"]==_team_a) & (shots ["match_id"].isin(_mids_a))]
_sh_b  = shots [(shots ["team"]==_team_b) & (shots ["match_id"].isin(_mids_b))]
_go_a  = goals [(goals ["team"]==_team_a) & (goals ["match_id"].isin(_mids_a))]
_go_b  = goals [(goals ["team"]==_team_b) & (goals ["match_id"].isin(_mids_b))]

_na, _nb = max(len(_mids_a), 1), max(len(_mids_b), 1)


# =============================================================================
# KPIs COMPARATIVOS
# =============================================================================

def _fmt(v, decimals=2):
    return f"{v:.{decimals}f}" if pd.notna(v) else "—"

_og_a = int(((events["type"]=="Own Goal For") & (events["team"]==_team_a) & (events["match_id"].isin(_mids_a))).sum())
_og_b = int(((events["type"]=="Own Goal For") & (events["team"]==_team_b) & (events["match_id"].isin(_mids_b))).sum())

_kpi_rows = [
    ("Partidos",        _na,                              _nb),
    ("Goles",           len(_go_a) + _og_a,              len(_go_b) + _og_b),
    ("xG total",        _fmt(_sh_a[_XG_COL].sum() if _XG_COL in _sh_a.columns else 0),
                        _fmt(_sh_b[_XG_COL].sum() if _XG_COL in _sh_b.columns else 0)),
    ("xG/partido",      _fmt(_tv(_team_a,"xg_pm")),      _fmt(_tv(_team_b,"xg_pm"))),
    ("% Pases",         _fmt(_tv(_team_a,"pass_pct"),1), _fmt(_tv(_team_b,"pass_pct"),1)),
]
_k_cols = st.columns(len(_kpi_rows))
for _col, (_lbl, _va, _vb) in zip(_k_cols, _kpi_rows):
    _col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{_lbl}</div>
        <div style="display:flex;justify-content:space-around;align-items:center;margin-top:0.4rem">
            <span style="font-size:1.5rem;font-weight:700;color:{_COLOR_A}">{_va}</span>
            <span style="font-size:0.8rem;opacity:0.65">vs</span>
            <span style="font-size:1.5rem;font-weight:700;color:{_COLOR_B}">{_vb}</span>
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# FUNCIONES DE GRÁFICOS
# =============================================================================

def _render_radar_dual():
    st.markdown('<div class="section-header">🕸️ Radar Comparativo</div>', unsafe_allow_html=True)
    _labs  = _RADAR_LABELS + [_RADAR_LABELS[0]]
    _pa    = _team_percentiles(_team_a)
    _pb    = _team_percentiles(_team_b)
    _va    = _pa + [_pa[0]]
    _vb    = _pb + [_pb[0]]

    fig = go.Figure()
    for _vals, _color, _name in [(_va, _COLOR_A, _team_a), (_vb, _COLOR_B, _team_b)]:
        fig.add_trace(go.Scatterpolar(
            r=_vals, theta=_labs, fill="toself", name=_name,
            fillcolor=_rgba(_color, 0.18),
            line=dict(color=_color, width=2.5),
            hovertemplate="<b>" + _name + "</b><br>%{theta}: %{r:.0f}°<extra></extra>",
        ))
    fig.update_layout(
        polar=dict(
            bgcolor=COLOR_FONDO,
            radialaxis=dict(range=[0,100], tickvals=[25,50,75,100],
                            tickfont=dict(color=COLOR_TEXTO, size=7),
                            gridcolor=_rgba(COLOR_TEXTO,0.15)),
            angularaxis=dict(tickfont=dict(color=COLOR_TEXTO, size=10),
                             gridcolor=_rgba(COLOR_TEXTO,0.15),
                             direction="clockwise", rotation=90),
        ),
        paper_bgcolor=COLOR_FONDO, font=dict(color=COLOR_TEXTO),
        legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="center", x=0.5,
                    font=dict(color=COLOR_TEXTO, size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=30, b=70, l=70, r=70), height=460,
        title=dict(text="Percentil vs todos los equipos del torneo",
                   font=dict(color=COLOR_TEXTO, size=9), x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Color principal = {_team_a}  ·  Color secundario = {_team_b}  ·  Percentil respecto al torneo")


def _render_barras():
    st.markdown('<div class="section-header">📊 Barras Comparativas</div>', unsafe_allow_html=True)
    _metrics = {
        "Goles/P":      ("goals_pm",     False),
        "xG/P":         ("xg_pm",        False),
        "xGA/P":        ("xga_pm",       True),   # invertido: menor es mejor
        "Tiros/P":      ("shots_pm",     False),
        "Pases/P":      ("passes_pm",    False),
        "% Pases":      ("pass_pct",     False),
        "Presiones/P":  ("pressures_pm", False),
        "Defensiva/P":  ("defensive_pm", False),
    }
    _lbls = list(_metrics.keys())
    _vals_a = [round(float(_tv(_team_a, v)), 2) for v, _ in _metrics.values()]
    _vals_b = [round(float(_tv(_team_b, v)), 2) for v, _ in _metrics.values()]

    # Cada metrica se escala al mayor de los dos equipos (100 = el mas alto):
    # asi los goles (1-2) se ven igual de bien que los pases (400-600).
    _tope  = [max(abs(a), abs(b)) or 1 for a, b in zip(_vals_a, _vals_b)]
    _rel_a = [a / t * 100 for a, t in zip(_vals_a, _tope)]
    _rel_b = [b / t * 100 for b, t in zip(_vals_b, _tope)]

    fig = go.Figure()
    fig.add_trace(go.Bar(name=_team_a, x=_lbls, y=_rel_a, customdata=_vals_a,
        marker_color=_COLOR_A, opacity=0.88,
        text=[f"{v:.2f}" for v in _vals_a], textposition="outside",
        textfont=dict(color=COLOR_TEXTO, size=8),
        hovertemplate="<b>" + _team_a + "</b><br>%{x}: %{customdata:.2f}<extra></extra>"))
    fig.add_trace(go.Bar(name=_team_b, x=_lbls, y=_rel_b, customdata=_vals_b,
        marker_color=_COLOR_B, opacity=0.88,
        text=[f"{v:.2f}" for v in _vals_b], textposition="outside",
        textfont=dict(color=COLOR_TEXTO, size=8),
        hovertemplate="<b>" + _team_b + "</b><br>%{x}: %{customdata:.2f}<extra></extra>"))
    fig.update_layout(
        barmode="group", paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=9),
        xaxis=dict(tickfont=dict(color=COLOR_TEXTO), showgrid=False),
        yaxis=dict(showgrid=False, showticklabels=False, range=[0, 118]),
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5,
                    font=dict(color=COLOR_TEXTO), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20, b=60, l=40, r=10), height=440,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Barras relativas al mayor de los dos en cada métrica (la cifra es el valor real por partido)  ·  xGA = xG concedido (menor es mejor)")


def _render_heatmaps():
    st.markdown('<div class="section-header">🔥 Heatmaps lado a lado</div>', unsafe_allow_html=True)
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[_team_a, _team_b],
                        horizontal_spacing=0.06)
    for col_i, (_ev, _color) in enumerate([(_ev_a, _COLOR_A), (_ev_b, _COLOR_B)], 1):
        if _ev.empty or "location" not in _ev.columns: continue
        _loc = _parse_loc(_ev["location"]).dropna()
        fig.add_trace(go.Histogram2d(
            x=_loc["x"], y=_loc["y"], xbins=dict(start=0, end=120, size=5), ybins=dict(start=0, end=80, size=5),
            colorscale=_CSCALE_PRI, showscale=False,
            hovertemplate="x:%{x:.0f} y:%{y:.0f}<br>Acciones: %{z}<extra></extra>",
        ), row=1, col=col_i)
        # Línea de medio campo
        fig.add_shape(type="line", x0=60, y0=0, x1=60, y1=80,
                      line=dict(color=_rgba(COLOR_TEXTO,0.4), width=1, dash="dot"),
                      row=1, col=col_i)
    for ax in ["xaxis","xaxis2"]:
        fig.update_layout(**{ax: dict(showgrid=False, showticklabels=False,
                                      range=[-3,123])})
    for ax in ["yaxis","yaxis2"]:
        fig.update_layout(**{ax: dict(showgrid=False, showticklabels=False,
                                      range=[-3,83])})
    fig.update_layout(
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=10),
        margin=dict(t=40, b=10, l=10, r=10), height=400,
    )
    fig.update_annotations(font=dict(color=COLOR_TEXTO, size=11))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Densidad espacial de todas las acciones  ·  Ataque hacia la derecha")


def _render_shot_maps():
    st.markdown('<div class="section-header">🎯 Shot Maps duales</div>', unsafe_allow_html=True)
    _cfg = {
        "Goal":    ("Gol",      1.0),
        "Saved":   ("Atajado",  0.55),
        "Blocked": ("Bloqueado",0.40),
        "Off T":   ("Fuera",    0.30),
        "Post":    ("Poste",    0.35),
    }

    def _pitch_shapes_sub():
        lc = _rgba(COLOR_TEXTO, 0.4)
        kw = dict(line_color=lc, line_width=1, fillcolor="rgba(0,0,0,0)")
        return [
            dict(type="rect", x0=60, y0=0, x1=120, y1=80, **kw),
            dict(type="rect", x0=102,y0=18,x1=120, y1=62, **kw),
            dict(type="rect", x0=114,y0=30,x1=120, y1=50, **kw),
        ]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[_team_a, _team_b],
                        horizontal_spacing=0.06)
    _shown = set()
    for col_i, (_sh, _color) in enumerate([(_sh_a, _COLOR_A), (_sh_b, _COLOR_B)], 1):
        if _sh.empty or "location" not in _sh.columns: continue
        _d  = _parse_loc(_sh["location"])
        _d["outcome"] = _sh["shot_outcome"].fillna("Unknown").values
        _d["xg"]      = _sh[_XG_COL].fillna(0.05).values if _XG_COL in _sh.columns else 0.05
        _d  = _d.dropna(subset=["x","y"])
        # Se dibuja de menos a mas importante para que los goles queden encima
        for _out, (_lbl, _op) in reversed(list(_cfg.items())):
            _m = _d["outcome"] == _out
            if not _m.any(): continue
            _s = _d[_m]
            _show = _lbl not in _shown
            if _show: _shown.add(_lbl)
            fig.add_trace(go.Scatter(
                x=_s["x"], y=_s["y"], mode="markers", name=_lbl,
                legendgroup=_lbl, showlegend=_show,
                marker=dict(size=(_s["xg"]*40).clip(6,26),
                            color=_color, opacity=_op,
                            line=dict(width=0.8, color=COLOR_FONDO)),
                hovertemplate=f"<b>{_lbl}</b><br>xG: %{{customdata:.3f}}<extra></extra>",
                customdata=_s["xg"].values,
            ), row=1, col=col_i)
        for sh in _pitch_shapes_sub():
            fig.add_shape(**sh, row=1, col=col_i)

    for ax in ["xaxis","xaxis2"]:
        fig.update_layout(**{ax: dict(range=[58,123], showgrid=False, showticklabels=False)})
    for ax in ["yaxis","yaxis2"]:
        fig.update_layout(**{ax: dict(range=[-3,83],  showgrid=False, showticklabels=False)})
    fig.update_layout(
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=9),
        legend=dict(orientation="h", yanchor="top", y=-0.04, xanchor="center", x=0.5,
                    font=dict(color=COLOR_TEXTO,size=9), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=40, b=50, l=10, r=10), height=420,
    )
    fig.update_annotations(font=dict(color=COLOR_TEXTO, size=11))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Zona de ataque (medio campo derecho)  ·  Tamaño ∝ xG")


def _render_h2h():
    st.markdown('<div class="section-header">🏆 Head to Head</div>', unsafe_allow_html=True)
    # Partidos donde ambos equipos se enfrentaron
    _h2h = matches[
        (((matches["home_team"]==_team_a)&(matches["away_team"]==_team_b)) |
         ((matches["home_team"]==_team_b)&(matches["away_team"]==_team_a)))
    ].sort_values("match_date")

    if _h2h.empty:
        st.info(f"**{_team_a}** y **{_team_b}** no se enfrentaron en este torneo.")
        return

    def _safe_int(v):
        try: return int(v)
        except: return 0

    _wins_a = _wins_b = _draws = 0
    for _, r in _h2h.iterrows():
        _hs  = _safe_int(r["home_score"])
        _as_ = _safe_int(r["away_score"])
        _ht, _at = r["home_team"], r["away_team"]
        if _hs > _as_:
            if _ht == _team_a: _wins_a += 1
            else:              _wins_b += 1
        elif _as_ > _hs:
            if _at == _team_a: _wins_a += 1
            else:              _wins_b += 1
        else:
            _draws += 1

    # Tarjetas de resultado
    _rc = st.columns(len(_h2h))
    for _col, (_, r) in zip(_rc, _h2h.iterrows()):
        _ht, _at = r["home_team"], r["away_team"]
        _hs, _as_ = _safe_int(r["home_score"]), _safe_int(r["away_score"])
        _ca = _COLOR_A if _ht==_team_a else _COLOR_B
        _cb = _COLOR_B if _at==_team_b else _COLOR_A
        _stage = str(r.get("competition_stage","")).strip()
        _col.markdown(f"""
        <div class="kpi-card" style="padding:0.7rem">
            <div class="kpi-label">{str(r['match_date'])[:10]} · {_stage}</div>
            <div style="display:flex;justify-content:space-around;align-items:center;margin-top:0.3rem">
                <span style="font-size:0.9rem;font-weight:700;color:{_ca}">{_ht}</span>
                <span style="font-size:1.3rem;font-weight:800;color:{COLOR_TEXTO}">{_hs}–{_as_}</span>
                <span style="font-size:0.9rem;font-weight:700;color:{_cb}">{_at}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;margin-top:0.8rem;font-size:0.9rem;color:{COLOR_TEXTO}">
        <span style="color:{_COLOR_A};font-weight:700">{_team_a}: {_wins_a} victoria(s)</span>
        &nbsp;·&nbsp; Empate(s): {_draws}
        &nbsp;·&nbsp; <span style="color:{_COLOR_B};font-weight:700">{_team_b}: {_wins_b} victoria(s)</span>
    </div>""", unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Partidos individuales de cada equipo
    # ------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    _col_a, _col_b = st.columns(2)

    def _match_cards(col, team, color):
        _tm = matches[
            (matches["home_team"]==team) | (matches["away_team"]==team)
        ].sort_values("match_date")

        col.markdown(
            f'<div style="font-size:0.85rem;font-weight:700;color:{color};'
            f'margin-bottom:0.5rem;text-transform:uppercase;letter-spacing:1px">'
            f'Partidos de {team}</div>',
            unsafe_allow_html=True,
        )
        for _, r in _tm.iterrows():
            _hs  = _safe_int(r["home_score"])
            _as_ = _safe_int(r["away_score"])
            _ht, _at = r["home_team"], r["away_team"]
            _stage = str(r.get("competition_stage","")).strip()
            _date  = str(r.get("match_date",""))[:10]

            # Resultado desde la perspectiva del equipo
            if _ht == team:
                _won  = _hs > _as_
                _lost = _hs < _as_
            else:
                _won  = _as_ > _hs
                _lost = _as_ < _hs
            _res_color = (color if _won
                          else _rgba(COLOR_TEXTO, 0.4) if not _lost
                          else "#cc3333")
            _res_label = "V" if _won else ("E" if not _lost else "D")

            col.markdown(f"""
            <div class="kpi-card" style="padding:0.5rem 0.7rem;margin-bottom:0.4rem;
                         display:flex;align-items:center;gap:0.6rem">
                <span style="font-size:0.9rem;font-weight:800;color:{_res_color};
                             min-width:1.1rem;text-align:center">{_res_label}</span>
                <div style="flex:1;min-width:0">
                    <div style="font-size:0.7rem;opacity:0.6">{_date} · {_stage}</div>
                    <div style="font-size:0.82rem;font-weight:600;color:{COLOR_TEXTO};
                                white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                        <span style="color:{color if _ht==team else '#888'}">{_ht}</span>
                        <span style="color:{COLOR_TEXTO};margin:0 0.3rem">{_hs}–{_as_}</span>
                        <span style="color:{color if _at==team else '#888'}">{_at}</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    _match_cards(_col_a, _team_a, _COLOR_A)
    _match_cards(_col_b, _team_b, _COLOR_B)


def _render_top_goleadores():
    st.markdown('<div class="section-header">⚽ Top Goleadores</div>', unsafe_allow_html=True)
    _cl, _cr = st.columns(2)
    for _col, _go, _color, _name in [
        (_cl, _go_a, _COLOR_A, _team_a),
        (_cr, _go_b, _COLOR_B, _team_b),
    ]:
        if _go.empty or "player" not in _go.columns:
            _col.info("Sin datos.")
            continue
        _top = _go.groupby("player").size().reset_index(name="goles").nlargest(8,"goles")
        _max = _top["goles"].max()
        fig = go.Figure(go.Bar(
            y=_top["player"], x=_top["goles"], orientation="h",
            marker_color=[_color if i==0 else _rgba(_color,0.55) for i in range(len(_top))],
            text=_top["goles"], textposition="outside",
            textfont=dict(color=COLOR_TEXTO, size=9),
            hovertemplate="<b>%{y}</b><br>Goles: %{x}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
            font=dict(color=COLOR_TEXTO, size=9),
            xaxis=dict(showgrid=False, showticklabels=False,
                       range=[0, _max * 1.25]),
            yaxis=dict(showgrid=False, autorange="reversed",
                       tickfont=dict(color=COLOR_TEXTO, size=9)),
            margin=dict(t=10,b=10,l=10,r=40), height=300, showlegend=False,
            title=dict(text=_name, font=dict(color=_color, size=11), x=0.5),
        )
        _col.plotly_chart(fig, use_container_width=True)
    st.caption("Top 8 goleadores por equipo en el torneo")


def _render_xg_partido():
    st.markdown('<div class="section-header">📈 xG por Partido</div>', unsafe_allow_html=True)
    if _XG_COL not in shots.columns:
        st.info("Sin datos de xG."); return

    fig = go.Figure()
    for _sh, _mids_t, _color, _name in [
        (_sh_a, _mids_a, _COLOR_A, _team_a),
        (_sh_b, _mids_b, _COLOR_B, _team_b),
    ]:
        _xg_pm = _sh.groupby("match_id")[_XG_COL].sum().reset_index(name="xG")
        _mi = matches[["match_id","match_date","home_team","away_team",
                        "home_score","away_score"]]
        _df = _mi.merge(_xg_pm, on="match_id", how="inner")
        _df["label"] = _df.apply(
            lambda r: f"{str(r.match_date)[:10]}<br>{r.home_team} {r.home_score}–{r.away_score} {r.away_team}", axis=1)
        _df = _df.sort_values("match_date")
        fig.add_trace(go.Scatter(
            x=list(range(1, len(_df)+1)), y=_df["xG"].round(2), mode="lines+markers",
            name=_name, line=dict(color=_color, width=2.5),
            marker=dict(size=8, color=_color, line=dict(width=1.5, color=COLOR_FONDO)),
            customdata=_df["label"].values,
            hovertemplate="<b>"+_name+"</b><br>%{customdata}<br>xG: %{y:.2f}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=9),
        xaxis=dict(title="Partido #", title_font=dict(color=COLOR_TEXTO),
                   tickfont=dict(color=COLOR_TEXTO), gridcolor=_rgba(COLOR_TEXTO,0.1),
                   dtick=1),
        yaxis=dict(title="xG", title_font=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO,0.12), tickfont=dict(color=COLOR_TEXTO)),
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5,
                    font=dict(color=COLOR_TEXTO), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20, b=60, l=45, r=10), height=440,
        title=dict(text="xG generado partido a partido",
                   font=dict(color=COLOR_TEXTO, size=10), x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Hover para ver el partido correspondiente  ·  Eje X = orden cronológico")


def _render_ppda():
    st.markdown('<div class="section-header">📋 PPDA y Perfil de Presión</div>', unsafe_allow_html=True)
    _rows = [{"team": _t, "PPDA": round(float(_v), 2)} for _t, _v in data["ppda"].items()]
    if not _rows:
        st.info("Sin datos para PPDA."); return

    _df = pd.DataFrame(_rows).sort_values("PPDA")
    _colors = [_COLOR_A if t==_team_a else _COLOR_B if t==_team_b
               else _rgba(COLOR_TEXTO,0.22) for t in _df["team"]]
    fig = go.Figure(go.Bar(
        y=_df["team"], x=_df["PPDA"], orientation="h", marker_color=_colors,
        text=_df["PPDA"].apply(lambda v: f"{v:.1f}"), textposition="outside",
        textfont=dict(color=COLOR_TEXTO, size=8),
        hovertemplate="<b>%{y}</b><br>PPDA: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=9),
        xaxis=dict(title="PPDA (menor = más presionante)",
                   title_font=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO,0.1), tickfont=dict(color=COLOR_TEXTO)),
        yaxis=dict(showgrid=False, tickfont=dict(color=COLOR_TEXTO, size=8)),
        margin=dict(t=10,b=30,l=10,r=60), height=440, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Color principal = {_team_a}  ·  Color secundario = {_team_b}  ·  Gris = resto del torneo")


# =============================================================================
# DICCIONARIO Y RENDERIZADO
# =============================================================================

_RENDERERS = {
    "🕸️ Radar comparativo":   _render_radar_dual,
    "📊 Barras comparativas":  _render_barras,
    "🔥 Heatmaps lado a lado": _render_heatmaps,
    "🎯 Shot Maps duales":     _render_shot_maps,
    "🏆 Head to Head":         _render_h2h,
    "⚽ Top goleadores":       _render_top_goleadores,
    "📈 xG por partido":       _render_xg_partido,
    "📋 PPDA y presión":       _render_ppda,
}

# Gráficos que van a ancho completo
_FULL_WIDTH = {"🏆 Head to Head", "⚽ Top goleadores",
               "🔥 Heatmaps lado a lado", "🎯 Shot Maps duales"}

_pairs, _solo = [], []
for _c in _charts_sel:
    (_solo if _c in _FULL_WIDTH else _pairs).append(_c)

for i in range(0, len(_pairs), 2):
    _chunk = _pairs[i:i+2]
    if len(_chunk) == 2:
        _cl, _cr = st.columns(2)
        with _cl: _RENDERERS[_chunk[0]]()
        with _cr: _RENDERERS[_chunk[1]]()
    else:
        _cl, _ = st.columns([1, 1])
        with _cl: _RENDERERS[_chunk[0]]()
    st.markdown("<br>", unsafe_allow_html=True)

for _c in _solo:
    _RENDERERS[_c]()
    st.markdown("<br>", unsafe_allow_html=True)
