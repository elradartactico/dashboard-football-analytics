# =============================================================================
#  MASTER EN FOOTBALL ANALYTICS: Eventing, Tracking, Scouting e IA
#  Radar Tactico -- Formacion en habla hispana
# =============================================================================
#  Modulo 7  : Construccion del Dashboard en Streamlit
#  Leccion    : 04 -- BLOQUE DE CONFIGURACION
#  Instruccion: Este es el UNICO archivo que el alumno modifica.
#               Cambia los valores de las variables de abajo para personalizar
#               tu dashboard. No toques ningun otro archivo.
# =============================================================================

# =============================================================================
# BLOQUE DE CONFIGURACION  ← SOLO CAMBIA ESTOS VALORES
# =============================================================================

# -- TU EQUIPO ----------------------------------------------------------------
#   Nombre exacto del equipo tal como aparece en StatsBomb Open Data.
#   Se usa como filtro predeterminado al abrir el dashboard.
#   Ejemplos: "Argentina", "France", "Brazil", "Germany"
EQUIPO: str = "Argentina"

# -- TU NOMBRE ----------------------------------------------------------------
#   Aparece en el sidebar y en el encabezado del dashboard.
NOMBRE_ANALISTA: str = "EL RADAR TÁCTICO"

# -- TITULO DEL DASHBOARD -----------------------------------------------------
#   Nombre del dashboard que se muestra en la pestaña del navegador.
TITULO: str = "Football Analytics Dashboard"

# -- COLORES ------------------------------------------------------------------
#   Usa cualquier codigo de color hexadecimal (ejemplo: "#f4c542").
#   COLOR_PRIMARIO   → color principal: encabezados, metricas, graficos, links.
#   COLOR_SECUNDARIO → color de acento: graficos secundarios, contrastes.
#   COLOR_FONDO         → color de fondo de la pagina principal.
#   COLOR_TEXTO         → color del texto principal (titulos, KPIs, tablas).
#                         Usa un color oscuro si COLOR_FONDO es claro, y viceversa.
#   COLOR_SIDEBAR       → color de fondo del sidebar.
#   COLOR_SIDEBAR_TEXTO → color del texto y enlaces dentro del sidebar
#                         (nombre de seccion, navegacion). Usa un color que
#                         contraste con COLOR_SIDEBAR.
#   COLOR_BARRA         → barra superior de Streamlit. Lo normal es igualarla
#                         a COLOR_FONDO.
#   COLOR_WIDGET        → fondo de los filtros (selectbox, multiselect, radio).
#
#   PAREJAS QUE DEBEN CONTRASTAR (una clara y otra oscura):
#   El dashboard usa exactamente los colores que pongas aqui; no corrige el
#   contraste por ti. Antes de guardar, revisa estas parejas:
#     · COLOR_FONDO    con COLOR_TEXTO         → todo el texto de la pagina y
#                                                la pantalla de espera.
#     · COLOR_SIDEBAR  con COLOR_SIDEBAR_TEXTO → el menu y el credito a StatsBomb.
#     · COLOR_PRIMARIO con COLOR_FONDO         → titulos de seccion, cifras de
#                                                los indicadores y graficos.
#     · COLOR_WIDGET   con COLOR_TEXTO         → el texto dentro de los filtros.
#   Tema claro: fondos claros y textos oscuros. Tema oscuro: al reves.
#   Despues de cambiar colores, deten la app (Ctrl+C) y vuelve a lanzarla:
#   el tema se aplica al arrancar.

COLOR_PRIMARIO:       str = "#55ABAB"  
COLOR_SECUNDARIO:     str = "#F3F4F8"   
COLOR_FONDO:          str = "#0C1244"  
COLOR_TEXTO:          str = "#F3F4F8"   
COLOR_SIDEBAR:        str = "#091545"   
COLOR_SIDEBAR_TEXTO:  str = "#F3F4F8"   
COLOR_BARRA:          str = "#0C1244"   
COLOR_WIDGET:         str = "#0D1D5E"   

# -- LOGO ---------------------------------------------------------------------
#   Ruta a tu imagen de logo (formato PNG o JPG, idealmente fondo transparente).
#   Coloca tu logo en la carpeta assets/ con el nombre logo.png.
#   El logo debe contrastar con DOS fondos: COLOR_SIDEBAR (donde se ve siempre)
#   y COLOR_FONDO (donde aparece en la pantalla de espera). Un logo blanco
#   necesita que ambos sean oscuros; un logo oscuro, que ambos sean claros.
#   Si no tienes logo todavia, deja este valor como esta — el dashboard
#   mostrara el titulo de texto en su lugar.
#
#   MEDIDAS RECOMENDADAS:
#     · Formato      : PNG con fondo transparente (recomendado) o JPG
#     · Tamaño       : 400 x 400 px  (logos cuadrados)
#                      600 x 200 px  (logos horizontales/rectangulares)
#     · Peso máximo  : 200 KB
#     · El dashboard lo mostrará a 160 px de ancho centrado en el sidebar.
#       Un logo de 400 px o más garantiza nitidez en pantallas de alta densidad.
LOGO_PATH: str = "assets/logo.png"
