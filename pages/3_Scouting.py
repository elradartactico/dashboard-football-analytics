# =============================================================================
#  MASTER EN FOOTBALL ANALYTICS: Eventing, Tracking, Scouting e IA
#  Radar Tactico -- Formacion en habla hispana
# =============================================================================
#  Modulo 7  : Construccion del Dashboard en Streamlit
#  Leccion   : 15 -- pages/3_Scouting.py: Identificacion y analisis de talento
#  Graficos  : Scatter xG/xA · Ranking · Radar comparativo · Tabla de scouting
#              Distribucion percentil · Similitud de perfil · Top por posicion
#              Scatter personalizado
# =============================================================================

import sys
import warnings
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")

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

st.set_page_config(page_title="Scouting", page_icon="🔭", layout="wide")

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
# HELPERS
# =============================================================================

def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

_CSCALE_PRI = [[0.0, COLOR_FONDO], [0.4, _rgba(COLOR_PRIMARIO, 0.5)], [1.0, COLOR_PRIMARIO]]

_XG_COL = "shot_statsbomb_xg"

# Grupos posicionales StatsBomb → etiqueta legible
_POS_MAP = {
    "Goalkeeper":             "Portero",
    "Right Back":             "Defensa",  "Left Back":              "Defensa",
    "Center Back":            "Defensa",  "Left Center Back":       "Defensa",
    "Right Center Back":      "Defensa",  "Right Wing Back":        "Defensa",
    "Left Wing Back":         "Defensa",  "Sweeper":                "Defensa",
    "Defensive Midfield":     "Medio",    "Central Midfield":       "Medio",
    "Left Midfield":          "Medio",    "Right Midfield":         "Medio",
    "Left Defensive Midfield":"Medio",    "Right Defensive Midfield":"Medio",
    "Attacking Midfield":     "Medio",    "Left Attacking Midfield":"Medio",
    "Right Attacking Midfield":"Medio",
    "Center Defensive Midfield":"Medio", "Center Attacking Midfield":"Medio",
    "Left Center Midfield":   "Medio",    "Right Center Midfield":  "Medio",
    "Left Wing":              "Delantero","Right Wing":             "Delantero",
    "Center Forward":         "Delantero","Left Center Forward":    "Delantero",
    "Right Center Forward":   "Delantero","Secondary Striker":      "Delantero",
}
_POS_GROUPS = ["Todos", "Delantero", "Medio", "Defensa", "Portero"]

_METRIC_LABELS = {
    "goals_pm":     "Goles/90 min",
    "shots_pm":     "Tiros/90 min",
    "xg_pm":        "xG/90 min",
    "assists_pm":   "Asistencias/90 min",
    "passes_pm":    "Pases/90 min",
    "pass_pct":     "% Pases OK",
    "dribbles_pm":  "Regates/90 min",
    "pressures_pm": "Presiones/90 min",
    "defensive_pm": "Defensiva/90 min",
}
_RADAR_KEYS   = ["goals_pm","shots_pm","xg_pm","passes_pm","pass_pct",
                 "dribbles_pm","pressures_pm","defensive_pm"]
_RADAR_LABELS = ["Goles/90","Tiros/90","xG/90","Pases/90","% Pases",
                 "Regates/90","Presiones/90","Defensiva/90"]


# =============================================================================
# ESTADÍSTICAS GLOBALES POR JUGADOR
# =============================================================================

@st.cache_data(show_spinner="Calculando estadísticas de scouting...")
def _build_scout_stats(_events, _shots, _passes, _lineups):
    """Métricas por jugador sobre el torneo completo."""

    n_matches = _events.groupby("player")["match_id"].nunique().rename("n_matches")
    n_goals   = (_shots[_shots["shot_outcome"] == "Goal"]
                 .groupby("player").size().rename("goals"))
    n_shots   = _shots.groupby("player").size().rename("shots")
    xg_sum    = (_shots.groupby("player")[_XG_COL].sum().rename("xg")
                 if _XG_COL in _shots.columns else pd.Series(dtype=float, name="xg"))

    # Asistencias reales (pass_goal_assist) o key passes como proxy de xA
    if "pass_goal_assist" in _passes.columns:
        n_ast = (_passes[_passes["pass_goal_assist"] == True]
                 .groupby("player").size().rename("assists"))
    else:
        n_ast = pd.Series(dtype=float, name="assists")

    n_passes = _passes.groupby("player").size().rename("passes")
    if "pass_outcome" in _passes.columns:
        n_ok = (_passes[_passes["pass_outcome"].isna()]
                .groupby("player").size().rename("passes_ok"))
    else:
        n_ok = pd.Series(dtype=float, name="passes_ok")

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
    for s in [n_goals, n_shots, xg_sum, n_ast, n_passes, n_ok, n_drib, n_press, n_def]:
        if s is not None and not s.empty:
            df = df.join(s, how="left")
    df = df.fillna(0)
    df = df[df["n_matches"] >= 1].copy()

    # Metricas por 90 minutos jugados (no por partido): quien entra 10 minutos
    # y marca ya no aparece con "1 gol por partido".
    _min = data.get("minutos") if isinstance(data, dict) else None
    df["minutes"] = (_min.reindex(df.index) if _min is not None and len(_min) else np.nan)
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    df["minutes"] = df["minutes"].where(df["minutes"] > 0, df["n_matches"] * 90).round(0)
    nm = df["minutes"] / 90
    df["goals_pm"]     = df.get("goals",    0) / nm
    df["shots_pm"]     = df.get("shots",    0) / nm
    df["xg_pm"]        = df.get("xg",       0) / nm
    df["assists_pm"]   = df.get("assists",   0) / nm
    df["passes_pm"]    = df.get("passes",   0) / nm
    df["pass_pct"]     = np.where(df.get("passes", 0) > 0,
                                  df.get("passes_ok", 0) / df.get("passes", 1) * 100, 0)
    df["dribbles_pm"]  = df.get("dribbles",  0) / nm
    df["pressures_pm"] = df.get("pressures", 0) / nm
    df["defensive_pm"] = df.get("defensive", 0) / nm

    # Posicion y equipo desde lineups
    if not _lineups.empty and "player_name" in _lineups.columns:
        import json as _json_lu
        _lu = _lineups.copy()

        # prepare_data.py serializa 'positions' a JSON string antes de parsearla,
        # por lo que position_name queda vacío. Lo reconstruimos aquí.
        _needs_parse = (
            "position_name" not in _lu.columns
            or _lu["position_name"].fillna("").str.strip().eq("").all()
        )
        if _needs_parse and "positions" in _lu.columns:
            def _parse_pos_str(val):
                try:
                    if isinstance(val, str):
                        val = _json_lu.loads(val)
                    if isinstance(val, list) and val:
                        first = val[0]
                        return first.get("position", "") if isinstance(first, dict) else str(first)
                except Exception:
                    pass
                return ""
            _lu["position_name"] = _lu["positions"].apply(_parse_pos_str)

        _pos = _lu.drop_duplicates("player_name").set_index("player_name")
        if "position_name" in _pos.columns:
            df["position_name"] = _pos["position_name"].reindex(df.index)
            # La posicion mas frecuente del jugador en los eventos es mas fiable
            # que la de su primera alineacion (que puede ser un cambio puntual).
            if "position" in _events.columns:
                _pos_ev = (_events.dropna(subset=["player", "position"])
                           .groupby("player")["position"].agg(lambda s: s.value_counts().index[0]))
                df["position_name"] = _pos_ev.reindex(df.index).fillna(df["position_name"])
            df["pos_group"] = df["position_name"].map(_POS_MAP).fillna("Otro")
        if "team_name" in _pos.columns:
            df["team"] = _pos["team_name"].reindex(df.index)

    # Equipo desde events si no está en lineups
    if "team" not in df.columns:
        eq = _events.groupby("player")["team"].first()
        df["team"] = eq.reindex(df.index)

    return df.reset_index().rename(columns={"player": "player"})


_stats = _build_scout_stats(events, shots, passes, lineups)


def _pct_rank(values: np.ndarray, score: float) -> float:
    # Rango medio: quien empata con muchos (p. ej. 0 goles) no sale arriba
    return float((np.sum(values < score) + np.sum(values == score) / 2) / len(values) * 100) if len(values) else 50.0


# =============================================================================
# ENCABEZADO
# =============================================================================

comp_name, season_name = get_competition_info(matches)
st.title("🔭 Scouting — Identificación de Talento")
st.caption(f"{comp_name} {season_name}")
st.divider()


# =============================================================================
# FILTROS PRINCIPALES
# =============================================================================

_fc1, _fc2, _fc3, _fc4 = st.columns([2, 2, 2, 6])

with _fc1:
    st.markdown("**⚽ Posición**")
    _pos_sel = st.selectbox("Posición", _POS_GROUPS, label_visibility="collapsed")

with _fc2:
    st.markdown("**🏟️ Equipo**")
    _team_opts = ["Todos los equipos"] + sorted(teams["team"].tolist())
    _team_sel  = st.selectbox("Equipo", _team_opts, label_visibility="collapsed")

with _fc3:
    st.markdown("**⏱️ Mín. minutos**")
    _max_min     = int(max(_stats["minutes"].max(), 90)) if "minutes" in _stats.columns else 90
    _min_minutes = st.slider("Mínimo de minutos", 0, _max_min, min(180, _max_min // 2), step=30,
                             label_visibility="collapsed")

with _fc4:
    _ALL_CHARTS = [
        "🎯 Scatter xG vs xA",
        "📊 Ranking por métrica",
        "🕸️ Radar comparativo",
        "📋 Tabla de scouting",
        "📈 Distribución percentil",
        "🔍 Similitud de perfil",
        "⚡ Top por posición",
        "🎨 Scatter personalizado",
    ]
    _DEFAULT = _ALL_CHARTS[:4]
    st.markdown("**📊 Gráficos a mostrar** *(máx. 5)*")
    _charts_sel = st.multiselect("Gráficos", _ALL_CHARTS, default=_DEFAULT,
                                 max_selections=5, label_visibility="collapsed")

if not _charts_sel:
    st.info("Selecciona al menos un gráfico arriba.")
    st.stop()


# =============================================================================
# CONTROLES CONDICIONALES
# =============================================================================

_compare_players, _target_player, _x_metric, _y_metric, _dist_metric = [], None, "xg_pm", "assists_pm", "xg_pm"

_needs_compare = "🕸️ Radar comparativo" in _charts_sel
_needs_target  = "🔍 Similitud de perfil" in _charts_sel
_needs_xy      = "🎨 Scatter personalizado" in _charts_sel
_needs_rank_m  = "📊 Ranking por métrica" in _charts_sel or "📈 Distribución percentil" in _charts_sel

if any([_needs_compare, _needs_target, _needs_xy, _needs_rank_m]):
    with st.expander("⚙️ Configuración adicional de gráficos", expanded=True):
        _cx = st.columns([3, 3, 2, 2, 2])
        _all_player_names = sorted(_stats["player"].dropna().unique().tolist())

        if _needs_compare:
            with _cx[0]:
                st.markdown("**Jugadores a comparar** *(máx. 3)*")
                _compare_players = st.multiselect(
                    "Jugadores radar", _all_player_names,
                    max_selections=3, label_visibility="collapsed",
                    placeholder="Elige hasta 3 jugadores...",
                )

        if _needs_target:
            with _cx[1]:
                st.markdown("**Jugador de referencia** *(similitud)*")
                _goleador = (_stats.sort_values("goals", ascending=False)["player"].iloc[0]
                             if "goals" in _stats.columns and len(_stats) else None)
                _target_player = st.selectbox(
                    "Jugador ref", _all_player_names,
                    index=_all_player_names.index(_goleador) if _goleador in _all_player_names else 0,
                    label_visibility="collapsed")

        if _needs_rank_m:
            with _cx[2]:
                st.markdown("**Métrica de ranking**")
                _dist_metric = st.selectbox(
                    "Métrica rank", list(_METRIC_LABELS.keys()),
                    format_func=lambda k: _METRIC_LABELS[k],
                    label_visibility="collapsed",
                )

        if _needs_xy:
            with _cx[3]:
                st.markdown("**Eje X**")
                _x_metric = st.selectbox(
                    "Eje X", list(_METRIC_LABELS.keys()),
                    index=2,  # xg_pm por defecto
                    format_func=lambda k: _METRIC_LABELS[k],
                    label_visibility="collapsed",
                )
            with _cx[4]:
                st.markdown("**Eje Y**")
                _y_metric = st.selectbox(
                    "Eje Y", list(_METRIC_LABELS.keys()),
                    index=3,  # assists_pm por defecto
                    format_func=lambda k: _METRIC_LABELS[k],
                    label_visibility="collapsed",
                )


# =============================================================================
# APLICAR FILTROS AL DATAFRAME
# =============================================================================

_df = _stats.copy()

if _pos_sel != "Todos" and "pos_group" in _df.columns:
    _df = _df[_df["pos_group"] == _pos_sel]

if _team_sel != "Todos los equipos" and "team" in _df.columns:
    _df = _df[_df["team"] == _team_sel]

_df = _df[_df["minutes"] >= _min_minutes]

if _df.empty:
    st.warning("No hay jugadores con los filtros aplicados. Ajusta la posición, equipo o mínimo de minutos.")
    st.stop()


# =============================================================================
# KPIs DE UNIVERSO
# =============================================================================

_top_xg   = _df.nlargest(1, "xg_pm").iloc[0] if "xg_pm" in _df.columns and not _df.empty else None
_top_goals = _df.nlargest(1, "goals_pm").iloc[0] if "goals_pm" in _df.columns and not _df.empty else None

_kc = st.columns(5)
_kpi_data = [
    ("Jugadores", len(_df), f"{_pos_sel} · ≥{_min_minutes} min"),
    ("Mayor xG/90",
     f"{_top_xg['xg_pm']:.2f}" if _top_xg is not None else "—",
     str(_top_xg["player"]) if _top_xg is not None else "—"),
    ("Mayor goleador",
     f"{_top_goals['goals_pm']:.2f}/90" if _top_goals is not None else "—",
     str(_top_goals["player"]) if _top_goals is not None else "—"),
    ("Promedio xG/90",
     f"{_df['xg_pm'].mean():.2f}" if "xg_pm" in _df.columns else "—", "del grupo filtrado"),
    ("Equipos", _df["team"].nunique() if "team" in _df.columns else "—", "representados"),
]
for _col, (_lbl, _val, _sub) in zip(_kc, _kpi_data):
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

_COLORS_TEAM = {}
def _team_color(team: str) -> str:
    """Color estable por equipo para identificación visual consistente."""
    palette = [COLOR_PRIMARIO, COLOR_SECUNDARIO,
               "#2196f3","#4caf50","#ff9800","#9c27b0","#00bcd4","#ff5722",
               "#607d8b","#795548","#e91e63","#8bc34a","#ffc107","#3f51b5"]
    if team not in _COLORS_TEAM:
        _COLORS_TEAM[team] = palette[len(_COLORS_TEAM) % len(palette)]
    return _COLORS_TEAM[team]


def _render_scatter_xg_xa():
    st.markdown('<div class="section-header">🎯 Scatter xG vs Asistencias por 90 min</div>', unsafe_allow_html=True)
    if "xg_pm" not in _df.columns or _df.empty:
        st.info("Sin datos de xG."); return

    _x = _df["xg_pm"].values
    _y = _df["assists_pm"].values if "assists_pm" in _df.columns else np.zeros(len(_df))
    _teams = _df["team"].values if "team" in _df.columns else np.array([""] * len(_df))
    # Color por posicion (4 grupos) en vez de un color por cada equipo
    _grupo = _df["pos_group"].values if "pos_group" in _df.columns else np.array(["Jugadores"] * len(_df))
    _color_pos = {"Delantero": COLOR_PRIMARIO, "Medio": COLOR_SECUNDARIO,
                  "Defensa": "#8a8a8a", "Portero": "#c9a227"}

    fig = go.Figure()
    # Un trace por equipo para tener leyenda con colores
    for _t in [g for g in ["Delantero", "Medio", "Defensa", "Portero", "Otro", "Jugadores"] if g in set(_grupo)]:
        _m = _grupo == _t
        fig.add_trace(go.Scatter(
            x=_x[_m], y=_y[_m], mode="markers", name=str(_t),
            marker=dict(size=10, color=_color_pos.get(str(_t), "#888888"), opacity=0.82,
                        line=dict(width=0.8, color=COLOR_FONDO)),
            text=_df["player"].values[_m],
            customdata=np.stack([_df["player"].values[_m], _teams[_m],
                                 np.round(_x[_m], 3), np.round(_y[_m], 3)], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                          "xG/90: %{customdata[2]}  ·  Ast/90: %{customdata[3]}<extra></extra>",
        ))
    # Líneas de cuadrante (mediana)
    _mx, _my = np.median(_x), np.median(_y)
    for _xv in [_mx]: fig.add_vline(x=_xv, line_dash="dot", line_color=_rgba(COLOR_TEXTO, 0.25), line_width=1)
    for _yv in [_my]: fig.add_hline(y=_yv, line_dash="dot", line_color=_rgba(COLOR_TEXTO, 0.25), line_width=1)

    fig.update_layout(
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=9),
        xaxis=dict(title="xG /90 min", title_font=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO, 0.1), tickfont=dict(color=COLOR_TEXTO)),
        yaxis=dict(title="Asistencias /90 min", title_font=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO, 0.1), tickfont=dict(color=COLOR_TEXTO)),
        legend=dict(font=dict(color=COLOR_TEXTO, size=8), bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        margin=dict(t=20, b=70, l=50, r=20), height=440,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Color = posición  ·  Líneas punteadas = mediana del grupo  ·  Cuadrante sup-der = amenaza doble  ·  Pasa el cursor para ver jugador y equipo")


def _render_ranking():
    _met_lbl = _METRIC_LABELS.get(_dist_metric, _dist_metric)
    st.markdown(f'<div class="section-header">📊 Ranking — {_met_lbl}</div>', unsafe_allow_html=True)
    if _dist_metric not in _df.columns:
        st.info("Métrica no disponible."); return

    _top = _df.nlargest(20, _dist_metric)[["player","team", _dist_metric,"n_matches"]].copy()
    _top[_dist_metric] = _top[_dist_metric].round(3)
    _colors = [COLOR_PRIMARIO if i < 3 else _rgba(COLOR_TEXTO, 0.30)
               for i in range(len(_top))]

    fig = go.Figure(go.Bar(
        y=_top["player"], x=_top[_dist_metric], orientation="h",
        marker_color=_colors,
        text=_top[_dist_metric].apply(lambda v: f"{v:.2f}"),
        textposition="outside", textfont=dict(color=COLOR_TEXTO, size=8),
        customdata=_top[["player","team","n_matches"]].values,
        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                      f"{_met_lbl}: %{{x:.3f}}<br>Partidos: %{{customdata[2]}}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=9),
        xaxis=dict(title=_met_lbl, title_font=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO, 0.1), tickfont=dict(color=COLOR_TEXTO)),
        yaxis=dict(showgrid=False, autorange="reversed",
                   tickfont=dict(color=COLOR_TEXTO, size=8)),
        margin=dict(t=10, b=30, l=10, r=60), height=440, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Top 20 · {_met_lbl}  ·  Resaltado = top 3")


def _render_radar_comp():
    st.markdown('<div class="section-header">🕸️ Radar Comparativo</div>', unsafe_allow_html=True)
    if not _compare_players:
        st.info("Selecciona entre 1 y 3 jugadores en **Configuración adicional** para comparar."); return

    _colors_radar = [COLOR_PRIMARIO, COLOR_SECUNDARIO, "#2196f3"]
    _labels_r = _RADAR_LABELS + [_RADAR_LABELS[0]]
    _all_names = _stats["player"].tolist()

    fig = go.Figure()
    for i, _p in enumerate(_compare_players):
        _row = _stats[_stats["player"] == _p]
        if _row.empty: continue
        _row = _row.iloc[0]
        _pcts = []
        for _k in _RADAR_KEYS:
            if _k in _stats.columns:
                _dist = _stats[_k].dropna().values
                _pcts.append(_pct_rank(_dist, _row.get(_k, 0)))
            else:
                _pcts.append(50.0)
        _vals = _pcts + [_pcts[0]]
        _c = _colors_radar[i % len(_colors_radar)]
        fig.add_trace(go.Scatterpolar(
            r=_vals, theta=_labels_r, fill="toself", name=_p,
            fillcolor=_rgba(_c, 0.18),
            line=dict(color=_c, width=2.2),
            hovertemplate="<b>" + _p + "</b><br>%{theta}: %{r:.0f}<extra></extra>",
        ))
    fig.update_layout(
        polar=dict(
            bgcolor=COLOR_FONDO,
            radialaxis=dict(visible=True, range=[0,100], tickvals=[25,50,75,100],
                            tickfont=dict(color=COLOR_TEXTO, size=7),
                            gridcolor=_rgba(COLOR_TEXTO,0.15),
                            linecolor=_rgba(COLOR_TEXTO,0.15)),
            angularaxis=dict(tickfont=dict(color=COLOR_TEXTO, size=9),
                             gridcolor=_rgba(COLOR_TEXTO,0.15),
                             direction="clockwise", rotation=90),
        ),
        paper_bgcolor=COLOR_FONDO, font=dict(color=COLOR_TEXTO),
        legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="center", x=0.5,
                    font=dict(color=COLOR_TEXTO, size=9), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=30, b=70, l=70, r=70), height=460,
        title=dict(text="Percentil vs torneo completo",
                   font=dict(color=COLOR_TEXTO, size=9), x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Cada eje = percentil en el torneo  ·  Mayor área = mejor perfil global")


def _render_tabla():
    st.markdown('<div class="section-header">📋 Tabla de Scouting</div>', unsafe_allow_html=True)
    _cols_show = ["player","team","pos_group","n_matches","minutes",
                  "goals_pm","xg_pm","assists_pm","shots_pm",
                  "passes_pm","pass_pct","dribbles_pm","pressures_pm","defensive_pm"]
    _cols_show = [c for c in _cols_show if c in _df.columns]
    _t = _df[_cols_show].copy().sort_values("xg_pm", ascending=False).reset_index(drop=True)
    _t.index = range(1, len(_t) + 1)
    _rename = {
        "player": "Jugador", "team": "Equipo", "pos_group": "Posición",
        "n_matches": "Partidos", "minutes": "Minutos",
        "goals_pm": "Goles/90", "xg_pm": "xG/90", "assists_pm": "Ast/90",
        "shots_pm": "Tiros/90", "passes_pm": "Pases/90", "pass_pct": "% Pases",
        "dribbles_pm": "Regates/90", "pressures_pm": "Presiones/90", "defensive_pm": "Defensiva/90",
    }
    _t = _t.rename(columns=_rename)
    _num_cols = [c for c in _t.columns if c not in ("Jugador","Equipo","Posición","Partidos")]
    _t[_num_cols] = _t[_num_cols].round(2)
    st.dataframe(_t, use_container_width=True, height=420)
    _csv_scouting = _t.to_csv(index=True).encode("utf-8")
    st.download_button("⬇ Exportar CSV", data=_csv_scouting,
                       file_name="scouting.csv", mime="text/csv", key="dl_scouting")
    st.caption(f"{len(_t)} jugadores  ·  Ordenado por xG/90 min  ·  Haz clic en columna para reordenar")


def _render_dist_percentil():
    _met_lbl = _METRIC_LABELS.get(_dist_metric, _dist_metric)
    st.markdown(f'<div class="section-header">📈 Distribución — {_met_lbl}</div>', unsafe_allow_html=True)
    if _dist_metric not in _df.columns:
        st.info("Métrica no disponible."); return

    _vals = _df[_dist_metric].dropna().values
    _bins = np.linspace(_vals.min(), _vals.max(), 26)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=_vals, xbins=dict(start=_bins[0], end=_bins[-1], size=(_bins[-1]-_bins[0])/25),
        marker_color=_rgba(COLOR_PRIMARIO, 0.6),
        marker_line_color=COLOR_FONDO, marker_line_width=1,
        name="Distribución", hovertemplate="Valor: %{x:.2f}<br>Jugadores: %{y}<extra></extra>",
    ))
    # Línea de mediana y percentil 75
    for _pct, _label, _dash in [
        (np.median(_vals),         "Mediana",    "dot"),
        (np.percentile(_vals, 75), "Percentil 75","dash"),
    ]:
        fig.add_vline(x=_pct, line_dash=_dash, line_color=COLOR_SECUNDARIO, line_width=1.5)
        fig.add_annotation(x=_pct, y=1, yref="paper", text=f"{_label}<br>{_pct:.2f}",
                           showarrow=False, font=dict(color=COLOR_SECUNDARIO, size=8),
                           yanchor="top", xanchor="left")
    fig.update_layout(
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=9),
        xaxis=dict(title=_met_lbl, title_font=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO,0.1), tickfont=dict(color=COLOR_TEXTO)),
        yaxis=dict(title="Jugadores", title_font=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO,0.12), tickfont=dict(color=COLOR_TEXTO)),
        margin=dict(t=20, b=40, l=45, r=10), height=440, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Distribución de {_met_lbl} para {len(_vals)} jugadores del grupo")


def _render_similitud():
    st.markdown('<div class="section-header">🔍 Similitud de Perfil</div>', unsafe_allow_html=True)
    if not _target_player:
        st.info("Selecciona un jugador de referencia en **Configuración adicional**."); return

    _t_row = _stats[_stats["player"] == _target_player]
    if _t_row.empty:
        st.info(f"No hay datos para {_target_player}."); return
    _t_row = _t_row.iloc[0]

    # Distancia euclidiana normalizada en espacio de 8 métricas
    _met_keys = [k for k in _RADAR_KEYS if k in _df.columns]
    _t_vals   = np.array([float(_t_row.get(k, 0)) for k in _met_keys])

    _dist = []
    for _, _r in _df.iterrows():
        _rv = np.array([float(_r.get(k, 0)) for k in _met_keys])
        _std = np.array([_stats[k].std() + 1e-9 for k in _met_keys])
        _dist.append(np.sqrt(np.sum(((_rv - _t_vals) / _std) ** 2)))
    _df2 = _df.copy()
    _df2["distancia"] = _dist
    _df2["similitud"] = 100 - (_df2["distancia"] / (_df2["distancia"].max() + 1e-9) * 100)

    _x = _df2["xg_pm"].values
    _y = _df2["assists_pm"].values if "assists_pm" in _df2.columns else np.zeros(len(_df2))
    _sim = _df2["similitud"].values
    _players = _df2["player"].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=_x, y=_y, mode="markers",
        marker=dict(size=10, color=_sim, colorscale=_CSCALE_PRI, showscale=True,
                    colorbar=dict(
                        title=dict(text="Similitud %", font=dict(color=COLOR_TEXTO)),
                        tickfont=dict(color=COLOR_TEXTO),
                    ),
                    opacity=0.85, line=dict(width=0.6, color=COLOR_FONDO)),
        text=_players,
        customdata=np.stack([_players, np.round(_x,3), np.round(_y,3), np.round(_sim,1)], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>xG/90: %{customdata[1]}<br>"
                      "Ast/90: %{customdata[2]}<br>Similitud: %{customdata[3]}%<extra></extra>",
    ))
    # Marcar jugador target
    if _target_player in list(_players):
        _ti = list(_players).index(_target_player)
        fig.add_trace(go.Scatter(
            x=[_x[_ti]], y=[_y[_ti]], mode="markers+text",
            marker=dict(size=16, color=COLOR_SECUNDARIO, symbol="star",
                        line=dict(width=2, color=COLOR_FONDO)),
            text=[_target_player], textposition="top center",
            textfont=dict(color=COLOR_TEXTO, size=9),
            name=_target_player, showlegend=False,
            hovertemplate=f"<b>{_target_player}</b><br>Referencia<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=9),
        xaxis=dict(title="xG/90 min", title_font=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO,0.1), tickfont=dict(color=COLOR_TEXTO)),
        yaxis=dict(title="Asistencias/90 min", title_font=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO,0.1), tickfont=dict(color=COLOR_TEXTO)),
        margin=dict(t=20, b=40, l=50, r=10), height=440, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Estrella = {_target_player}  ·  Color más intenso = perfil más similar en 8 dimensiones")


def _render_top_pos():
    st.markdown('<div class="section-header">⚡ Top por Posición</div>', unsafe_allow_html=True)
    if "pos_group" not in _df.columns:
        st.info("Sin datos de posición."); return

    _groups = [g for g in ["Delantero","Medio","Defensa","Portero"] if g in _df["pos_group"].unique()]
    if not _groups:
        st.info("Sin grupos posicionales detectados."); return

    _n_cols = min(len(_groups), 4)
    _cols = st.columns(_n_cols)
    for _gi, (_g, _col) in enumerate(zip(_groups, _cols)):
        _sub = _df[_df["pos_group"] == _g].nlargest(6, "xg_pm")[["player","xg_pm"]].copy()
        if _sub.empty: continue
        with _col:
            st.markdown(f"**{_g}s** — xG/90")
            for _ri, _row in _sub.iterrows():
                _max_xg = _sub["xg_pm"].max()
                _pct_w = min(int(_row["xg_pm"] / _max_xg * 100), 100) if _max_xg and _max_xg > 0 else 0
                st.markdown(
                    f"<div style='margin:2px 0; font-size:0.82rem; color:{COLOR_TEXTO}'>"
                    f"<span style='font-weight:600'>{_row['player'].split(' ')[-1]}</span> "
                    f"<span style='opacity:0.6'>{_row['xg_pm']:.2f}</span>"
                    f"<div style='height:5px; width:{_pct_w}%; background:{COLOR_PRIMARIO}; "
                    f"border-radius:3px; margin-top:2px'></div></div>",
                    unsafe_allow_html=True,
                )
    st.caption("Top 6 por posición ordenados por xG/90 min")


def _render_scatter_xy():
    _xl = _METRIC_LABELS.get(_x_metric, _x_metric)
    _yl = _METRIC_LABELS.get(_y_metric, _y_metric)
    st.markdown(f'<div class="section-header">🎨 Scatter — {_xl} vs {_yl}</div>', unsafe_allow_html=True)
    if _x_metric not in _df.columns or _y_metric not in _df.columns:
        st.info("Métrica no disponible."); return

    _x = _df[_x_metric].values
    _y = _df[_y_metric].values
    _teams = _df["team"].values if "team" in _df.columns else np.array([""] * len(_df))
    # Color por posicion (4 grupos) en vez de un color por cada equipo
    _grupo = _df["pos_group"].values if "pos_group" in _df.columns else np.array(["Jugadores"] * len(_df))
    _color_pos = {"Delantero": COLOR_PRIMARIO, "Medio": COLOR_SECUNDARIO,
                  "Defensa": "#8a8a8a", "Portero": "#c9a227"}

    fig = go.Figure()
    for _t in [g for g in ["Delantero", "Medio", "Defensa", "Portero", "Otro", "Jugadores"] if g in set(_grupo)]:
        _m = _grupo == _t
        fig.add_trace(go.Scatter(
            x=_x[_m], y=_y[_m], mode="markers", name=str(_t),
            marker=dict(size=10, color=_color_pos.get(str(_t), "#888888"), opacity=0.82,
                        line=dict(width=0.8, color=COLOR_FONDO)),
            customdata=np.stack([_df["player"].values[_m], _teams[_m],
                                 np.round(_x[_m],3), np.round(_y[_m],3)], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
                          f"{_xl}: %{{customdata[2]}}  ·  {_yl}: %{{customdata[3]}}<extra></extra>",
        ))
    fig.add_vline(x=np.median(_x), line_dash="dot", line_color=_rgba(COLOR_TEXTO,0.25), line_width=1)
    fig.add_hline(y=np.median(_y), line_dash="dot", line_color=_rgba(COLOR_TEXTO,0.25), line_width=1)
    fig.update_layout(
        paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
        font=dict(color=COLOR_TEXTO, size=9),
        xaxis=dict(title=_xl, title_font=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO,0.1), tickfont=dict(color=COLOR_TEXTO)),
        yaxis=dict(title=_yl, title_font=dict(color=COLOR_TEXTO),
                   gridcolor=_rgba(COLOR_TEXTO,0.1), tickfont=dict(color=COLOR_TEXTO)),
        legend=dict(font=dict(color=COLOR_TEXTO, size=8), bgcolor="rgba(0,0,0,0)",
                    orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        margin=dict(t=20, b=70, l=50, r=20), height=440,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Ejes: {_xl} vs {_yl}  ·  Líneas = mediana  ·  Color por posición")


# =============================================================================
# DICCIONARIO Y RENDERIZADO
# =============================================================================

_RENDERERS = {
    "🎯 Scatter xG vs xA":    _render_scatter_xg_xa,
    "📊 Ranking por métrica": _render_ranking,
    "🕸️ Radar comparativo":  _render_radar_comp,
    "📋 Tabla de scouting":   _render_tabla,
    "📈 Distribución percentil": _render_dist_percentil,
    "🔍 Similitud de perfil": _render_similitud,
    "⚡ Top por posición":    _render_top_pos,
    "🎨 Scatter personalizado": _render_scatter_xy,
}

_TABLE_FULL = "📋 Tabla de scouting"
_TOP_FULL   = "⚡ Top por posición"

_pairs = []
_solo  = []
for _c in _charts_sel:
    if _c in (_TABLE_FULL, _TOP_FULL):
        _solo.append(_c)
    else:
        _pairs.append(_c)

# Gráficos en grid 2 columnas
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

# Gráficos de ancho completo
for _c in _solo:
    _RENDERERS[_c]()
    st.markdown("<br>", unsafe_allow_html=True)


# =============================================================================
# COMPARACIÓN JUGADOR VS JUGADOR (sección permanente)
# =============================================================================

st.divider()
st.markdown('<div class="section-header">👥 Comparación Jugador vs Jugador</div>', unsafe_allow_html=True)

_all_names_cmp = sorted(_stats["player"].dropna().unique().tolist())
# Por defecto se comparan los dos maximos goleadores del torneo
_top2 = (_stats.sort_values("goals", ascending=False)["player"].head(2).tolist()
         if "goals" in _stats.columns else [])
_idx_a = _all_names_cmp.index(_top2[0]) if len(_top2) > 0 and _top2[0] in _all_names_cmp else 0
_idx_b = (_all_names_cmp.index(_top2[1]) if len(_top2) > 1 and _top2[1] in _all_names_cmp
          else min(1, len(_all_names_cmp) - 1))
_cmp_c1, _cmp_c2 = st.columns(2)
with _cmp_c1:
    st.markdown("**Jugador A**")
    _player_a = st.selectbox("Jugador A", _all_names_cmp, index=_idx_a,
                              label_visibility="collapsed", key="cmp_a")
with _cmp_c2:
    st.markdown("**Jugador B**")
    _player_b = st.selectbox("Jugador B", _all_names_cmp,
                              index=_idx_b,
                              label_visibility="collapsed", key="cmp_b")


def _render_player_vs_player(p_a: str, p_b: str):
    _ra = _stats[_stats["player"] == p_a]
    _rb = _stats[_stats["player"] == p_b]
    if _ra.empty or _rb.empty:
        st.info("Sin datos para uno o ambos jugadores."); return
    _ra, _rb = _ra.iloc[0], _rb.iloc[0]

    def _get_pcts(row):
        pcts = []
        for k in _RADAR_KEYS:
            if k in _stats.columns:
                dist = _stats[k].dropna().values
                pcts.append(_pct_rank(dist, float(row.get(k, 0))))
            else:
                pcts.append(50.0)
        return pcts

    _pcts_a = _get_pcts(_ra)
    _pcts_b = _get_pcts(_rb)
    _labels_r = _RADAR_LABELS + [_RADAR_LABELS[0]]
    _vals_a   = _pcts_a + [_pcts_a[0]]
    _vals_b   = _pcts_b + [_pcts_b[0]]

    _col_radar, _col_bars = st.columns([3, 2])

    # Radar superpuesto
    with _col_radar:
        fig = go.Figure()
        for _vals, _color, _name in [(_vals_a, COLOR_PRIMARIO, p_a), (_vals_b, COLOR_SECUNDARIO, p_b)]:
            fig.add_trace(go.Scatterpolar(
                r=_vals, theta=_labels_r, fill="toself", name=_name,
                fillcolor=_rgba(_color, 0.18),
                line=dict(color=_color, width=2.2),
                hovertemplate=f"<b>{_name}</b><br>%{{theta}}: %{{r:.0f}}°<extra></extra>",
            ))
        fig.update_layout(
            polar=dict(
                bgcolor=COLOR_FONDO,
                radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100],
                                tickfont=dict(color=COLOR_TEXTO, size=7),
                                gridcolor=_rgba(COLOR_TEXTO, 0.15)),
                angularaxis=dict(tickfont=dict(color=COLOR_TEXTO, size=9),
                                 gridcolor=_rgba(COLOR_TEXTO, 0.15),
                                 direction="clockwise", rotation=90),
            ),
            paper_bgcolor=COLOR_FONDO, font=dict(color=COLOR_TEXTO),
            legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5,
                        font=dict(color=COLOR_TEXTO, size=9), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=20, b=60, l=60, r=60), height=380,
            title=dict(text="Percentil vs torneo completo",
                       font=dict(color=COLOR_TEXTO, size=9), x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Barras divergentes por métrica
    with _col_bars:
        _met_a = [round(float(_ra.get(k, 0)), 3) for k in _RADAR_KEYS]
        _met_b = [round(float(_rb.get(k, 0)), 3) for k in _RADAR_KEYS]
        # Cada metrica se escala al mayor de los dos jugadores (100 = el mas alto):
        # asi los goles (0-1) se ven igual de bien que los pases (30-70).
        _tope  = [max(abs(a), abs(b)) or 1 for a, b in zip(_met_a, _met_b)]
        _rel_a = [a / t * 100 for a, t in zip(_met_a, _tope)]
        _rel_b = [b / t * 100 for b, t in zip(_met_b, _tope)]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=_RADAR_LABELS, x=[-v for v in _rel_a], orientation="h",
            name=p_a, marker_color=COLOR_PRIMARIO, opacity=0.85,
            customdata=_met_a,
            hovertemplate=f"<b>{p_a}</b><br>%{{y}}: %{{customdata:.3f}}<extra></extra>",
        ))
        fig2.add_trace(go.Bar(
            y=_RADAR_LABELS, x=_rel_b, orientation="h",
            name=p_b, marker_color=COLOR_SECUNDARIO, opacity=0.85,
            customdata=_met_b,
            hovertemplate=f"<b>{p_b}</b><br>%{{y}}: %{{customdata:.3f}}<extra></extra>",
        ))
        fig2.update_layout(
            barmode="overlay",
            paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
            font=dict(color=COLOR_TEXTO, size=9),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=True,
                       zerolinecolor=_rgba(COLOR_TEXTO, 0.3), zerolinewidth=1.5),
            yaxis=dict(showgrid=False, tickfont=dict(color=COLOR_TEXTO, size=9)),
            legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5,
                        font=dict(color=COLOR_TEXTO, size=9), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=50, l=10, r=10), height=380,
            title=dict(text="Comparación por métrica (pasa el cursor para ver el valor)",
                       font=dict(color=COLOR_TEXTO, size=9), x=0.5),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # KPI cards comparativos
    _kpi_mets = [
        ("Goles/90",      "goals_pm"),
        ("xG/90",         "xg_pm"),
        ("Asist./90",     "assists_pm"),
        ("Pases/90",      "passes_pm"),
        ("% Pases",      "pass_pct"),
        ("Regates/90",    "dribbles_pm"),
        ("Presión/90",    "pressures_pm"),
        ("Defensiva/90",  "defensive_pm"),
    ]
    _kc = st.columns(len(_kpi_mets))
    for _col, (_lbl, _k) in zip(_kc, _kpi_mets):
        _va = float(_ra.get(_k, 0))
        _vb = float(_rb.get(_k, 0))
        _ca = COLOR_PRIMARIO   if _va >= _vb else _rgba(COLOR_PRIMARIO,   0.4)
        _cb = COLOR_SECUNDARIO if _vb >  _va else _rgba(COLOR_SECUNDARIO, 0.4)
        _col.markdown(f"""
        <div class="kpi-card" style="padding:0.45rem 0.3rem">
            <div class="kpi-label" style="font-size:0.62rem">{_lbl}</div>
            <div style="display:flex;justify-content:space-around;align-items:center;margin-top:0.3rem">
                <span style="font-size:1rem;font-weight:700;color:{_ca}">{_va:.2f}</span>
                <span style="font-size:0.65rem;opacity:0.65">vs</span>
                <span style="font-size:1rem;font-weight:700;color:{_cb}">{_vb:.2f}</span>
            </div>
        </div>""", unsafe_allow_html=True)
    st.caption(f"Color principal = {p_a}  ·  Color secundario = {p_b}  ·  Barras: cada métrica relativa al mayor de los dos")


_render_player_vs_player(_player_a, _player_b)
