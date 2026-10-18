# =============================================================================
#  MASTER EN FOOTBALL ANALYTICS: Eventing, Tracking, Scouting e IA
#  Radar Tactico -- Formacion en habla hispana
# =============================================================================
#  Modulo 7  : Construccion del Dashboard en Streamlit
#  Leccion    : 04 -- app.py: pagina de inicio / resumen del torneo
#  Ejecutar   : streamlit run app.py
#               (desde la carpeta Dashboard/)
# =============================================================================
#
#  INSTRUCCION PARA EL ALUMNO:
#    Edita config.py para personalizar el dashboard.
#    No es necesario modificar este archivo.
#
# =============================================================================

import sys
import warnings
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from matplotlib.patches import FancyBboxPatch

# -- Importaciones del proyecto -----------------------------------------------
from config import (
    EQUIPO, NOMBRE_ANALISTA, TITULO,
    COLOR_PRIMARIO, COLOR_SECUNDARIO,
)
try:
    from config import COLOR_TEXTO
except ImportError:
    COLOR_TEXTO = "#e8eaf0"
try:
    from config import COLOR_FONDO
except ImportError:
    COLOR_FONDO = "#0d1117"
from utils.data import load_data, get_competition_info, top_scorers, match_results
from utils.sidebar import render_sidebar

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Auto-sincroniza .streamlit/config.toml con los colores de config.py
# Esto previene el "flash" de colores por defecto al cambiar de página.
# Solo escribe el archivo si los valores cambiaron (evita reinicios infinitos).
# ---------------------------------------------------------------------------
def _sync_streamlit_theme():
    try:
        from config import COLOR_PRIMARIO, COLOR_FONDO
        try:    from config import COLOR_TEXTO
        except: COLOR_TEXTO   = "#0D0D0D"
        try:    from config import COLOR_WIDGET
        except: COLOR_WIDGET  = "#FFFFFF"

        _toml = Path(__file__).parent / ".streamlit" / "config.toml"
        _new  = (
            f"[theme]\n"
            f"# Sincronizado automáticamente desde config.py — no editar manualmente.\n"
            f"primaryColor             = \"{COLOR_PRIMARIO}\"\n"
            f"backgroundColor          = \"{COLOR_FONDO}\"\n"
            f"secondaryBackgroundColor = \"{COLOR_WIDGET}\"\n"
            f"textColor                = \"{COLOR_TEXTO}\"\n\n"
            f"[server]\n"
            f"maxUploadSize = 200\n\n"
            f"[runner]\n"
            f"magicEnabled = true\n"
        )
        if not _toml.exists() or _toml.read_text(encoding="utf-8") != _new:
            _toml.write_text(_new, encoding="utf-8")
    except Exception:
        pass

_sync_streamlit_theme()


# =============================================================================
# CONFIGURACION DE PAGINA (debe ser la primera llamada a Streamlit)
# =============================================================================

st.set_page_config(
    page_title=TITULO,
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CSS PERSONALIZADO
# =============================================================================

st.markdown(f"""
<style>
    /* ── Oculta el menu hamburguesa de Streamlit ─────────────────────── */
    #MainMenu {{ visibility: hidden; }}
    footer    {{ visibility: hidden; }}

    /* ── Encabezados de seccion ──────────────────────────────────────── */
    .section-header {{
        font-size: 1rem;
        font-weight: 600;
        color: {COLOR_PRIMARIO};
        text-transform: uppercase;
        letter-spacing: 1.5px;
        border-bottom: 2px solid {COLOR_PRIMARIO}44;
        padding-bottom: 0.3rem;
        margin-bottom: 0.8rem;
    }}

    /* ── Tarjetas de KPI ─────────────────────────────────────────────── */
    .kpi-card {{
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }}
    .kpi-label {{
        font-size: 0.75rem;
        color: inherit; opacity: 0.62;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: clamp(0.58rem, 0.9vw, 0.75rem);
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .kpi-value {{
        font-size: 2.2rem;
        font-weight: 700;
        color: {COLOR_PRIMARIO};
        line-height: 1.2;
    }}
    .kpi-sub {{
        font-size: 0.8rem;
        color: inherit; opacity: 0.7;
    }}

    /* ── Tabla de resultados ─────────────────────────────────────────── */
    [data-testid="stDataFrame"] td:nth-child(4) {{
        text-align: center !important;
        font-weight: 700;
    }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CARGA DE DATOS
# =============================================================================

data = load_data()

if data is None:
    st.error(
        "⚠️  No se encontró la carpeta **data/**. "
        "Ejecuta `prepare_data.py` primero para descargar los datos del torneo."
    )
    st.code("python prepare_data.py", language="bash")
    st.stop()

matches = data["matches"]
events  = data["events"]
shots   = data["shots"]
goals   = data["goals"]
teams   = data["teams"]


# =============================================================================
# SIDEBAR
# =============================================================================

equipo_sel = render_sidebar(teams["team"].tolist())


# =============================================================================
# ENCABEZADO
# =============================================================================

comp_name, season_name = get_competition_info(matches)

_col_title, _col_badge = st.columns([5, 1])
with _col_title:
    st.title(f"⚽ {comp_name} {season_name}")
    st.caption(f" Analista: **{NOMBRE_ANALISTA}**")


# =============================================================================
# KPIs — fila de metricas
# =============================================================================

st.divider()

_total_matches = len(matches)
# Los goles en propia puerta no son tiros en StatsBomb: se suman aparte
_own_goals     = int((events["type"] == "Own Goal For").sum())
_total_goals   = len(goals) + _own_goals
_total_teams   = pd.concat([matches["home_team"], matches["away_team"]]).nunique()
_avg_goals     = _total_goals / _total_matches if _total_matches else 0

_xg_col = "shot_statsbomb_xg"
_total_xg  = shots[_xg_col].sum() if _xg_col in shots.columns else 0
_avg_xg    = _total_xg / _total_matches if _total_matches else 0

_k1, _k2, _k3, _k4, _k5 = st.columns(5)

with _k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Partidos</div>
        <div class="kpi-value">{_total_matches}</div>
        <div class="kpi-sub">jugados</div>
    </div>""", unsafe_allow_html=True)

with _k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Goles Totales</div>
        <div class="kpi-value">{_total_goals}</div>
        <div class="kpi-sub">{_avg_goals:.2f} por partido</div>
    </div>""", unsafe_allow_html=True)

with _k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">xG Total</div>
        <div class="kpi-value">{_total_xg:,.1f}</div>
        <div class="kpi-sub">{_avg_xg:.2f} por partido</div>
    </div>""", unsafe_allow_html=True)

with _k4:
    _total_shots = len(shots)
    _conv_rate   = len(goals) / _total_shots * 100 if _total_shots else 0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Tiros Totales</div>
        <div class="kpi-value">{_total_shots:,}</div>
        <div class="kpi-sub">{_conv_rate:.1f}% conversion</div>
    </div>""", unsafe_allow_html=True)

with _k5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Equipos</div>
        <div class="kpi-value">{_total_teams}</div>
        <div class="kpi-sub">participantes</div>
    </div>""", unsafe_allow_html=True)


# =============================================================================
# CONTENIDO PRINCIPAL — goleadores y resultados
# =============================================================================

st.markdown("<br>", unsafe_allow_html=True)
_col_left, _col_mid, _col_right = st.columns([3, 0.3, 5])

# ── Top Goleadores ────────────────────────────────────────────────────────────
with _col_left:
    st.markdown('<div class="section-header">🥇 Top Goleadores</div>', unsafe_allow_html=True)

    _top = top_scorers(goals, n=10)
    if not _top.empty:
        st.dataframe(
            _top,
            width="stretch",
            height=360,
            column_config={
                "Goles": st.column_config.NumberColumn(format="%d"),
                "xG":    st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.download_button("⬇ Exportar CSV", data=_top.to_csv(index=True).encode("utf-8"),
                           file_name="top_goleadores.csv", mime="text/csv", key="dl_goleadores")
    else:
        st.info("Sin datos de goles disponibles.")

# ── Resultados del Torneo ─────────────────────────────────────────────────────
with _col_right:
    st.markdown('<div class="section-header">📋 Resultados</div>', unsafe_allow_html=True)

    _results = match_results(matches)
    st.dataframe(
        _results,
        width="stretch",
        height=360,
        column_config={
            "Fecha":      st.column_config.TextColumn(width="small"),
            "Etapa":      st.column_config.TextColumn(width="medium"),
            "Local":      st.column_config.TextColumn(width="medium"),
            "Resultado":  st.column_config.TextColumn(width="small"),
            "Visitante":  st.column_config.TextColumn(width="medium"),
        },
    )
    st.download_button("⬇ Exportar CSV", data=_results.to_csv(index=True).encode("utf-8"),
                       file_name="resultados.csv", mime="text/csv", key="dl_resultados")


# =============================================================================
# GRAFICOS INTERACTIVOS — Goles por fase  +  Mapa de estadios
# =============================================================================

import plotly.graph_objects as go
from utils.data import STAGE_ORDER


def _rgba(hex_color: str, alpha: float) -> str:
    """Convierte '#rrggbb' + alpha a 'rgba(r,g,b,a)' que acepta Plotly."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

st.markdown("<br>", unsafe_allow_html=True)
_gc, _mc = st.columns(2)

# ── GRAFICO 1: Goles por fase del torneo ──────────────────────────────────────
with _gc:
    st.markdown('<div class="section-header">📊 Goles por Fase del Torneo</div>', unsafe_allow_html=True)

    if "competition_stage" in matches.columns and not goals.empty:
        _stage_map = matches[["match_id", "competition_stage"]].copy()
        _gwstage   = goals.merge(_stage_map, on="match_id", how="left")

        _by_stage = (
            _gwstage.groupby("competition_stage")
            .agg(Goles=("player", "count"))
            .reset_index()
        )
        # El xG de la fase suma TODOS los tiros, no solo los que acabaron en gol
        if _xg_col in shots.columns:
            _xg_stage = (shots.merge(_stage_map, on="match_id", how="left")
                         .groupby("competition_stage")[_xg_col].sum().reset_index(name="xG"))
            _by_stage = _by_stage.merge(_xg_stage, on="competition_stage", how="left")
        else:
            _by_stage["xG"] = _by_stage["Goles"]
        _mby = matches.groupby("competition_stage").size().reset_index(name="Partidos")
        _by_stage = _by_stage.merge(_mby, on="competition_stage", how="left")
        _by_stage["Goles/Partido"] = (_by_stage["Goles"] / _by_stage["Partidos"]).round(2)
        _by_stage["xG"] = _by_stage["xG"].round(1)
        _by_stage["_sort"] = _by_stage["competition_stage"].map(STAGE_ORDER).fillna(99)
        _by_stage = _by_stage.sort_values("_sort").reset_index(drop=True)

        _fig_bar = go.Figure()
        _fig_bar.add_trace(go.Bar(
            name="Goles reales",
            x=_by_stage["competition_stage"],
            y=_by_stage["Goles"],
            marker_color=COLOR_PRIMARIO,
            marker_opacity=0.9,
            text=_by_stage["Goles"],
            textposition="outside",
            textfont=dict(color=COLOR_PRIMARIO, size=11),
            customdata=_by_stage[["xG", "Partidos", "Goles/Partido"]].values,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Goles reales: <b>%{y}</b><br>"
                "xG esperado: %{customdata[0]:.1f}<br>"
                "Partidos: %{customdata[1]}<br>"
                "Goles/Partido: %{customdata[2]:.2f}"
                "<extra></extra>"
            ),
        ))
        _fig_bar.add_trace(go.Bar(
            name="xG esperado",
            x=_by_stage["competition_stage"],
            y=_by_stage["xG"],
            marker_color=COLOR_SECUNDARIO,
            marker_opacity=0.85,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "xG esperado: <b>%{y:.1f}</b>"
                "<extra></extra>"
            ),
        ))
        _fig_bar.update_layout(
            barmode="group", bargap=0.25, bargroupgap=0.04,
            paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
            font=dict(color=COLOR_TEXTO, size=11),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(color=COLOR_TEXTO),
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis=dict(showgrid=False, tickfont=dict(color=COLOR_TEXTO)),
            yaxis=dict(gridcolor=_rgba(COLOR_TEXTO, 0.15), tickfont=dict(color=COLOR_TEXTO), title="Cantidad"),
            margin=dict(t=40, b=10, l=10, r=10),
            height=380,
        )
        st.plotly_chart(_fig_bar, use_container_width=True)
    else:
        st.info("Sin datos de fases disponibles.")

# ── GRAFICO 2: Eficiencia ofensiva — xG vs Goles por equipo ──────────────────
with _mc:
    st.markdown('<div class="section-header">🎯 Eficiencia Ofensiva por Equipo</div>', unsafe_allow_html=True)

    if _xg_col in shots.columns and not shots.empty:
        # Agregar por equipo: goles, xG, tiros
        _team_goals = (
            goals.groupby("team")
            .agg(Goles=("player", "count"))
            .reset_index()
        )
        _team_shots = (
            shots.groupby("team")
            .agg(
                xG=(_xg_col, "sum"),
                Tiros=("player", "count"),
            )
            .reset_index()
        )
        _eff = _team_shots.merge(_team_goals, on="team", how="left").fillna(0)
        _eff["xG"]        = _eff["xG"].round(2)
        _eff["Diferencia"]= (_eff["Goles"] - _eff["xG"]).round(2)
        _eff["Conversion"]= (_eff["Goles"] / _eff["Tiros"] * 100).round(1)
        _eff["Estado"]    = _eff["Diferencia"].apply(
            lambda d: "Sobre xG" if d > 0 else ("Bajo xG" if d < 0 else "Igual")
        )

        # Línea de referencia diagonal (xG = Goles)
        _ref_max = max(_eff["xG"].max(), _eff["Goles"].max()) * 1.05
        _fig_sc = go.Figure()

        # Línea diagonal
        _fig_sc.add_trace(go.Scatter(
            x=[0, _ref_max], y=[0, _ref_max],
            mode="lines",
            line=dict(color=_rgba(COLOR_TEXTO, 0.4), width=1.5, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Puntos por equipo
        _fig_sc.add_trace(go.Scatter(
            x=_eff["xG"],
            y=_eff["Goles"],
            mode="markers+text",
            # Solo se rotulan los equipos destacados; el resto, al pasar el cursor
            text=_eff["team"].where(
                (_eff["Goles"].rank(ascending=False, method="first") <= 8)
                | (_eff["Diferencia"].abs().rank(ascending=False, method="first") <= 5), ""),
            textposition="top center",
            textfont=dict(size=8, color=COLOR_TEXTO),
            marker=dict(
                size=_eff["Tiros"] / _eff["Tiros"].max() * 28 + 8,
                color=_eff["Diferencia"],
                colorscale=[[0, COLOR_SECUNDARIO], [0.5, _rgba(COLOR_TEXTO, 0.35)], [1, COLOR_PRIMARIO]],
                cmin=_eff["Diferencia"].min(),
                cmax=_eff["Diferencia"].max(),
                opacity=0.88,
                line=dict(width=0.5, color=_rgba(COLOR_TEXTO, 0.4)),
                colorbar=dict(
                    title="Goles − xG",
                    thickness=10,
                    tickfont=dict(color=COLOR_TEXTO, size=9),
                    title_font=dict(color=COLOR_TEXTO, size=9),
                ),
            ),
            customdata=_eff[["team", "Goles", "xG", "Diferencia", "Tiros", "Conversion"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Goles: <b>%{customdata[1]:.0f}</b>  |  xG: %{customdata[2]:.2f}<br>"
                "Diferencia: %{customdata[3]:+.2f}<br>"
                "Tiros: %{customdata[4]:.0f}  |  Conversión: %{customdata[5]:.1f}%"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

        _fig_sc.update_layout(
            paper_bgcolor=COLOR_FONDO,
            plot_bgcolor=COLOR_FONDO,
            font=dict(color=COLOR_TEXTO, size=10),
            xaxis=dict(
                title="xG esperado",
                gridcolor=_rgba(COLOR_TEXTO, 0.15), zeroline=False,
                tickfont=dict(color=COLOR_TEXTO),
                title_font=dict(color=COLOR_TEXTO),
            ),
            yaxis=dict(
                title="Goles anotados",
                gridcolor=_rgba(COLOR_TEXTO, 0.15), zeroline=False,
                tickfont=dict(color=COLOR_TEXTO),
                title_font=dict(color=COLOR_TEXTO),
            ),
            margin=dict(t=10, b=40, l=50, r=10),
            height=380,
        )
        st.plotly_chart(_fig_sc, use_container_width=True)
        st.caption(
            "Puntos **sobre** la diagonal = más goles que xG (clínicos) · "
            "Puntos **bajo** la diagonal = menos goles que xG (ineficientes) · "
            "Tamaño del punto ∝ tiros totales"
        )
    else:
        st.info("Sin datos de xG disponibles para este torneo.")


# =============================================================================
# GRAFICOS INTERACTIVOS — Minutos de gol  +  Top jugadores por xG
# =============================================================================

st.markdown("<br>", unsafe_allow_html=True)
_mc2, _pc = st.columns(2)

# ── GRAFICO 3: Distribución de minutos de gol ─────────────────────────────────
with _mc2:
    st.markdown('<div class="section-header">⏱️ Minutos de Gol</div>', unsafe_allow_html=True)

    if "minute" in goals.columns and not goals.empty:
        _mins = goals["minute"].dropna().astype(int).clip(upper=120)

        _bins   = [0, 10, 20, 30, 40, 45, 55, 65, 75, 85, 90, 121]
        _labels = ["1-10","11-20","21-30","31-40","41-45",
                   "46-55","56-65","66-75","76-85","86-90","90+"]
        _min_cut    = pd.cut(_mins, bins=_bins, labels=_labels, right=True)
        _min_counts = (
            _min_cut.value_counts()
            .reindex(_labels, fill_value=0)
            .reset_index()
        )
        _min_counts.columns = ["Minuto", "Goles"]

        _colors_min = [
            COLOR_SECUNDARIO if lbl in ("41-45", "86-90", "90+") else COLOR_PRIMARIO
            for lbl in _min_counts["Minuto"]
        ]

        _fig_min = go.Figure()
        _fig_min.add_trace(go.Bar(
            x=_min_counts["Minuto"],
            y=_min_counts["Goles"],
            marker_color=_colors_min,
            marker_opacity=0.88,
            text=_min_counts["Goles"],
            textposition="outside",
            textfont=dict(color=COLOR_TEXTO, size=10),
            hovertemplate="<b>%{x}</b><br>Goles: <b>%{y}</b><extra></extra>",
        ))
        _fig_min.update_layout(
            paper_bgcolor=COLOR_FONDO,
            plot_bgcolor=COLOR_FONDO,
            font=dict(color=COLOR_TEXTO, size=11),
            xaxis=dict(showgrid=False, tickfont=dict(color=COLOR_TEXTO)),
            yaxis=dict(
                gridcolor=_rgba(COLOR_TEXTO, 0.12),
                tickfont=dict(color=COLOR_TEXTO),
                title="Goles",
                title_font=dict(color=COLOR_TEXTO),
            ),
            margin=dict(t=30, b=10, l=40, r=10),
            height=380,
            showlegend=False,
        )
        st.plotly_chart(_fig_min, use_container_width=True)
        st.caption(
            f"Minutos resaltados en "
            f"<span style='color:{COLOR_SECUNDARIO}'>■</span> = tramos finales de cada tiempo",
            unsafe_allow_html=True,
        )
    else:
        st.info("Sin datos de minutos disponibles.")

# ── GRAFICO 4: Top jugadores — xG acumulado + Goles ──────────────────────────
with _pc:
    st.markdown('<div class="section-header">🎯 Top Jugadores — xG vs Goles</div>', unsafe_allow_html=True)

    if _xg_col in shots.columns and not shots.empty:
        _pl_xg = (
            shots.groupby("player")
            .agg(xG=(_xg_col, "sum"), Tiros=("player", "count"))
            .reset_index()
        )
        _pl_goals = (
            goals.groupby("player")
            .agg(Goles=("player", "count"))
            .reset_index()
        )
        _pl = _pl_xg.merge(_pl_goals, on="player", how="left").fillna(0)
        _pl["xG"]   = _pl["xG"].round(2)
        _pl["Goles"] = _pl["Goles"].astype(int)
        _pl = _pl.nlargest(15, "xG").sort_values("xG")

        _fig_pl = go.Figure()
        _fig_pl.add_trace(go.Bar(
            name="xG esperado",
            y=_pl["player"],
            x=_pl["xG"],
            orientation="h",
            marker_color=COLOR_SECUNDARIO,
            marker_opacity=0.85,
            customdata=_pl[["Tiros"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "xG: <b>%{x:.2f}</b><br>"
                "Tiros: %{customdata[0]:.0f}"
                "<extra></extra>"
            ),
        ))
        _fig_pl.add_trace(go.Bar(
            name="Goles reales",
            y=_pl["player"],
            x=_pl["Goles"],
            orientation="h",
            marker_color=COLOR_PRIMARIO,
            marker_opacity=0.9,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Goles: <b>%{x}</b>"
                "<extra></extra>"
            ),
        ))
        _fig_pl.update_layout(
            barmode="overlay",
            paper_bgcolor=COLOR_FONDO,
            plot_bgcolor=COLOR_FONDO,
            font=dict(color=COLOR_TEXTO, size=10),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(color=COLOR_TEXTO),
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis=dict(
                title="Valor", title_font=dict(color=COLOR_TEXTO),
                gridcolor=_rgba(COLOR_TEXTO, 0.12),
                tickfont=dict(color=COLOR_TEXTO),
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(color=COLOR_TEXTO, size=9),
            ),
            margin=dict(t=40, b=10, l=10, r=10),
            height=380,
        )
        st.plotly_chart(_fig_pl, use_container_width=True)
        st.caption(
            "Barra larga = xG generado · Barra corta superpuesta = goles reales · "
            "Si la barra de goles supera al xG → jugador clínico"
        )
    else:
        st.info("Sin datos de xG disponibles para este torneo.")


# =============================================================================
# CLASIFICACIÓN DEL TORNEO
# =============================================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">🏆 Clasificación del Torneo</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _build_standings(_matches):
    rows = {}
    for _, r in _matches.iterrows():
        try:
            hs  = int(r["home_score"]) if pd.notna(r.get("home_score")) else None
            as_ = int(r["away_score"]) if pd.notna(r.get("away_score")) else None
        except Exception:
            hs = as_ = None
        if hs is None or as_ is None:
            continue
        for team, gf, gc in [(r["home_team"], hs, as_), (r["away_team"], as_, hs)]:
            if team not in rows:
                rows[team] = {"Equipo": team, "PJ": 0, "PG": 0, "PE": 0, "PP": 0, "GF": 0, "GC": 0}
            rows[team]["PJ"] += 1
            rows[team]["GF"] += gf
            rows[team]["GC"] += gc
            if gf > gc:
                rows[team]["PG"] += 1
            elif gf == gc:
                rows[team]["PE"] += 1
            else:
                rows[team]["PP"] += 1
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(list(rows.values()))
    df["DG"]  = df["GF"] - df["GC"]
    df["Pts"] = df["PG"] * 3 + df["PE"]
    df = df.sort_values(["Pts", "DG", "GF"], ascending=[False, False, False]).reset_index(drop=True)
    df.index = range(1, len(df) + 1)
    return df[["Equipo", "PJ", "PG", "PE", "PP", "GF", "GC", "DG", "Pts"]]


# En torneos con eliminatorias la tabla de puntos solo tiene sentido en la fase
# de grupos: un cruce decidido por penales cuenta como empate.
_hay_eliminatorias = ("stage_sort" in matches.columns
                      and (matches["stage_sort"] > 1).any() and (matches["stage_sort"] == 1).any())
_standings = _build_standings(matches[matches["stage_sort"] == 1] if _hay_eliminatorias else matches)

if _hay_eliminatorias and not _standings.empty:
    _ult = (pd.concat([matches[["home_team", "stage_sort", "competition_stage"]].rename(columns={"home_team": "t"}),
                       matches[["away_team", "stage_sort", "competition_stage"]].rename(columns={"away_team": "t"})])
            .sort_values("stage_sort").drop_duplicates("t", keep="last").set_index("t")["competition_stage"])
    _standings["Llegó hasta"] = _standings["Equipo"].map(_ult)
    st.caption("Puntos de la **fase de grupos** (todos los grupos juntos). "
               "La columna «Llegó hasta» indica la última ronda que jugó cada equipo.")

if not _standings.empty:
    _scol_l, _scol_r = st.columns([5, 3])

    with _scol_l:
        st.dataframe(
            _standings,
            use_container_width=True,
            height=min(40 * len(_standings) + 38, 520),
            column_config={
                "Equipo": st.column_config.TextColumn(width="medium"),
                "PJ":  st.column_config.NumberColumn(format="%d", width="small"),
                "PG":  st.column_config.NumberColumn(format="%d", width="small"),
                "PE":  st.column_config.NumberColumn(format="%d", width="small"),
                "PP":  st.column_config.NumberColumn(format="%d", width="small"),
                "GF":  st.column_config.NumberColumn(format="%d", width="small"),
                "GC":  st.column_config.NumberColumn(format="%d", width="small"),
                "DG":  st.column_config.NumberColumn(format="%+d", width="small"),
                "Pts": st.column_config.NumberColumn(format="%d", width="small"),
            },
        )
        st.download_button("⬇ Exportar CSV", data=_standings.to_csv(index=True).encode("utf-8"),
                           file_name="clasificacion.csv", mime="text/csv", key="dl_standings")
        st.caption("PJ=Partidos jugados · PG=Ganados · PE=Empatados · PP=Perdidos · DG=Diferencia de gol · Pts=Puntos")

    with _scol_r:
        _top_pts = _standings.head(10).sort_values("Pts")
        _colors_pts = [
            COLOR_PRIMARIO if i == len(_top_pts) - 1
            else COLOR_SECUNDARIO if i == len(_top_pts) - 2
            else _rgba(COLOR_TEXTO, 0.25)
            for i in range(len(_top_pts))
        ]
        _fig_pts = go.Figure(go.Bar(
            y=_top_pts["Equipo"], x=_top_pts["Pts"], orientation="h",
            marker_color=_colors_pts,
            text=_top_pts["Pts"].apply(lambda v: f"{v} pts"),
            textposition="outside", textfont=dict(color=COLOR_TEXTO, size=9),
            customdata=_top_pts[["PG", "PE", "PP", "GF", "GC"]].values,
            hovertemplate="<b>%{y}</b><br>Pts: %{x}<br>PG: %{customdata[0]} PE: %{customdata[1]} PP: %{customdata[2]}<br>GF: %{customdata[3]} GC: %{customdata[4]}<extra></extra>",
        ))
        _fig_pts.update_layout(
            paper_bgcolor=COLOR_FONDO, plot_bgcolor=COLOR_FONDO,
            font=dict(color=COLOR_TEXTO, size=9),
            xaxis=dict(showgrid=False, showticklabels=False,
                       range=[0, _top_pts["Pts"].max() * 1.25]),
            yaxis=dict(showgrid=False, tickfont=dict(color=COLOR_TEXTO, size=9)),
            margin=dict(t=30, b=10, l=10, r=50), height=340, showlegend=False,
            title=dict(text="Top 10 — Puntos", font=dict(color=COLOR_TEXTO, size=10), x=0.5),
        )
        st.plotly_chart(_fig_pts, use_container_width=True)
else:
    st.info("Sin datos de resultados para calcular la clasificación.")

