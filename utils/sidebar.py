# =============================================================================
#  utils/sidebar.py — Sidebar compartido entre todas las paginas
#  No modificar. El alumno solo cambia config.py.
# =============================================================================

from pathlib import Path
import streamlit as st


def render_sidebar(teams: list[str], default_team: str | None = None) -> str:
    """Renderiza el sidebar con logo, navegación, filtro y nombre del analista."""

    # Import dentro de la función para garantizar valores frescos en cada recarga
    try:
        from config import NOMBRE_ANALISTA, LOGO_PATH, EQUIPO, COLOR_PRIMARIO, COLOR_FONDO, COLOR_SIDEBAR
        try:
            from config import COLOR_TEXTO
        except ImportError:
            COLOR_TEXTO = "#e8eaf0"
        try:
            from config import COLOR_SIDEBAR_TEXTO
        except ImportError:
            COLOR_SIDEBAR_TEXTO = "#f4c542"
        try:
            from config import COLOR_BARRA
        except ImportError:
            COLOR_BARRA = "#0d1030"
        try:
            from config import COLOR_WIDGET
        except ImportError:
            COLOR_WIDGET = "#ffffff"
    except ImportError:
        NOMBRE_ANALISTA     = "Analista"
        LOGO_PATH           = "assets/logo.png"
        EQUIPO              = "Todos"
        COLOR_PRIMARIO      = "#f4c542"
        COLOR_FONDO         = "#0d1117"
        COLOR_SIDEBAR       = "#0d1030"
        COLOR_TEXTO         = "#e8eaf0"
        COLOR_SIDEBAR_TEXTO = "#f4c542"
        COLOR_BARRA         = "#0d1030"
        COLOR_WIDGET        = "#ffffff"

    equipo_default = default_team or EQUIPO
    _dashboard_root = Path(__file__).parent.parent
    _logo = _dashboard_root / LOGO_PATH

    # CSS global en un único bloque — Streamlit solo aplica el primero si hay varios
    st.markdown(f"""
<style>
    /* ── Oculta nav automático ───────────────────────────────────────── */
    [data-testid="stSidebarNav"] {{ display: none; }}

    /* ── Barra superior ──────────────────────────────────────────────── */
    [data-testid="stHeader"] {{
        background-color: {COLOR_BARRA} !important;
    }}

    /* ── Fondos de página y sidebar ──────────────────────────────────── */
    .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {COLOR_FONDO} !important;
    }}
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
        background-color: {COLOR_SIDEBAR} !important;
    }}

    /* ── Elimina padding superior del sidebar ────────────────────────── */
    [data-testid="stSidebarContent"] > div:first-child {{
        padding-top: 0.5rem !important;
    }}
    section[data-testid="stSidebar"] > div {{
        padding-top: 0.5rem !important;
    }}

    /* ── Texto principal de la página ────────────────────────────────── */
    .stApp p, .stApp span, .stApp label,
    [data-testid="stMarkdownContainer"] p,
    h1, h2, h3, h4, h5, h6,
    [data-testid="stHeading"],
    .stDataFrame td, .stDataFrame th {{
        color: {COLOR_TEXTO} !important;
    }}

    /* ── Texto y enlaces dentro del sidebar ──────────────────────────── */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: {COLOR_SIDEBAR_TEXTO} !important;
    }}
    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] .stPageLink a,
    [data-testid="stSidebar"] .stPageLink span {{
        color: {COLOR_SIDEBAR_TEXTO} !important;
    }}
    [data-testid="stSidebar"] a:hover,
    [data-testid="stSidebar"] .stPageLink a:hover {{
        color: {COLOR_PRIMARIO} !important;
        text-decoration: none;
    }}

    /* ── Dividers del sidebar ────────────────────────────────────────── */
    [data-testid="stSidebar"] hr {{
        border-color: {COLOR_SIDEBAR_TEXTO}33 !important;
    }}

    /* ── Widgets: selectbox, multiselect, input, radio ──────────────── */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"]  > div,
    div[data-baseweb="textarea"] > div {{
        background-color: {COLOR_WIDGET} !important;
        border-color: {COLOR_PRIMARIO}55 !important;
        color: {COLOR_TEXTO} !important;
    }}
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"]  > div:focus-within,
    div[data-baseweb="textarea"] > div:focus-within {{
        border-color: {COLOR_PRIMARIO} !important;
        box-shadow: 0 0 0 2px {COLOR_PRIMARIO}33 !important;
    }}
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input,
    div[data-baseweb="input"]  input {{
        color: {COLOR_TEXTO} !important;
    }}
    /* Dropdown de opciones del selectbox / multiselect */
    [data-baseweb="popover"] ul,
    [data-baseweb="menu"] {{
        background-color: {COLOR_WIDGET} !important;
    }}
    [data-baseweb="menu"] li:hover {{
        background-color: {COLOR_PRIMARIO}22 !important;
    }}
    [data-baseweb="menu"] li[aria-selected="true"] {{
        background-color: {COLOR_PRIMARIO}33 !important;
    }}
    /* Pills del multiselect (etiquetas seleccionadas) */
    [data-baseweb="tag"] {{
        background-color: {COLOR_PRIMARIO}22 !important;
        border-color: {COLOR_PRIMARIO}55 !important;
        color: {COLOR_TEXTO} !important;
    }}
    /* Radio buttons y checkboxes */
    [data-testid="stRadio"] label,
    [data-testid="stCheckbox"] label {{
        color: {COLOR_TEXTO} !important;
    }}

    /* ── Toolbar de tablas (iconos descarga, buscar, pantalla completa) ─ */
    [data-testid="stDataFrame"] button svg,
    [data-testid="stDataFrame"] [data-testid="stBaseButton-headerNoPadding"] svg,
    [data-testid="stElementToolbarButton"] svg {{
        fill: {COLOR_TEXTO} !important;
        stroke: {COLOR_TEXTO} !important;
    }}
    [data-testid="stElementToolbar"] {{
        background-color: {COLOR_FONDO} !important;
    }}
    [data-testid="stElementToolbarButton"]:hover svg {{
        fill: {COLOR_PRIMARIO} !important;
        stroke: {COLOR_PRIMARIO} !important;
    }}
</style>
""", unsafe_allow_html=True)

    with st.sidebar:
        # -- Logo (centrado) --------------------------------------------------
        if _logo.exists():
            import base64
            _img_b64 = base64.b64encode(_logo.read_bytes()).decode()
            _ext     = _logo.suffix.lstrip(".").replace("jpg", "jpeg")
            st.markdown(
                f'<div style="text-align:center; padding: 0 0 0.2rem;">'
                f'<img src="data:image/{_ext};base64,{_img_b64}" width="160">'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.divider()

        # -- Navegación manual con emojis -------------------------------------
        st.page_link("app.py",                label="🏠  Panel General")
        st.page_link("pages/1_Equipo.py",      label="🛡️  Análisis de equipo")
        st.page_link("pages/2_Jugador.py",     label="👤  Análisis de jugador")
        st.page_link("pages/3_Scouting.py",    label="🔭  Scouting")
        st.page_link("pages/4_vs_vs.py",       label="⚔️  Equipo vs. Equipo")
        st.divider()

        equipo_sel = equipo_default

        # -- Nombre del analista al pie ---------------------------------------
        st.sidebar.markdown(
            f"""
            <div style="position:fixed; bottom:1.5rem; font-size:0.78rem; color:{COLOR_SIDEBAR_TEXTO}; line-height:1.6;">
                <span style="color:{COLOR_SIDEBAR_TEXTO}; font-weight:600">{NOMBRE_ANALISTA}</span><br>
                Powered by StatsBomb Open Data
            </div>
            """,
            unsafe_allow_html=True,
        )

    return equipo_sel
