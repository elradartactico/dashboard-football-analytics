# =============================================================================
#  MASTER EN FOOTBALL ANALYTICS: Eventing, Tracking, Scouting e IA
#  Radar Tactico -- Formacion en habla hispana
# =============================================================================
#  Modulo 7  : Construcción del Dashboard en Streamlit
#  Leccion    : 03 -- prepare_data.py: seleccionar el torneo y ejecutar la descarga una sola vez 
#  Frecuencia : UNA SOLA VEZ por torneo. Volver a ejecutar si cambias de torneo.
#  Resultado  : Carpeta data/ con archivos .parquet listos para el dashboard.
# =============================================================================


# =============================================================================
# BLOQUE 0 -- COMO USAR ESTE ARCHIVO
# =============================================================================
#
#   Este script descarga todos los datos del torneo que elijas y los guarda
#   en formato parquet — un formato comprimido, rapido y compatible con el
#   dashboard.  Solo necesitas cambiar dos numeros en el BLOQUE 2.
#
#   PASOS:
#     1. Cambia COMPETITION_ID y SEASON_ID en el BLOQUE 2.
#     2. Abre la terminal en esta carpeta.
#     3. Ejecuta:  python prepare_data.py
#     4. Espera a que termine (puede tardar 5-15 minutos segun el torneo).
#     5. Verifica que aparecio la carpeta data/ con los archivos .parquet.
#     6. No vuelvas a ejecutar este script a menos que cambies de torneo.
#
#   TORNEOS DISPONIBLES EN STATSBOMB OPEN DATA:
#   ┌─────────────────────────────────┬────────────┬───────────┬──────────┐
#   │ Torneo                          │ COMPETITION│  SEASON   │ Partidos │
#   ├─────────────────────────────────┼────────────┼───────────┼──────────┤
#   │ FIFA World Cup 2022 (Qatar)     │     43     │    106    │    64    │
#   │ FIFA World Cup 2018 (Rusia)     │     43     │      3    │    64    │
#   │ Copa America 2024               │    223     │    282    │    32    │
#   │ Copa Africa 2023 (Costa Marfil) │    1267    │    107    │    52    │
#   │ Women's World Cup 2019          │     72     │     30    │    52    │
#   │ UEFA Euro 2020                  │     55     │     43    │    51    │
#   │ La Liga 2015/16 (*)             │     11     │     27    │   380    │
#   │ La Liga 2019/20 (**)            │     11     │     42    │    33    │
#   │ Premier League 2003/04 (**)     │      2     │     44    │    38    │
#   └─────────────────────────────────┴────────────┴───────────┴──────────┘
#   (*)  Liga completa (380 partidos): usala SOLO EN LOCAL. Publicada en la
#        cuenta gratuita de Streamlit llega al limite de memoria (1 GB) y
#        tarda unos 11 minutos en descargar cada vez que la app arranca.
#   (**) Temporada parcial: StatsBomb solo publica los partidos de un equipo
#        (Barcelona en La Liga 2019/20, Arsenal en Premier League 2003/04).
#   IDs verificados contra sb.competitions() el 11 de octubre de 2026.
#   Esta tabla es una seleccion. Los 80 torneos, con cuales conviene publicar
#   y cuales usar solo en local, estan en la guia
#   "Guia_Torneos_Publicar_o_Local_StatsBomb.docx".
#   Regla rapida: hasta 70 partidos, publica; de 70 a 140, publica con espera;
#   mas de 140, solo en local.
#
# =============================================================================


# =============================================================================
# BLOQUE 1 -- IMPORTS
# =============================================================================

import sys
import warnings
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from statsbombpy import sb


# =============================================================================
# BLOQUE 2 -- ZONA DE CONFIGURACION  ← SOLO CAMBIA ESTOS VALORES
# =============================================================================

# -- TORNEO -------------------------------------------------------------------
#   Busca tu torneo en la tabla del BLOQUE 0 y copia los dos numeros aqui.
COMPETITION_ID: int = "43"    # Ejemplo: 43 = FIFA World Cup
SEASON_ID:      int = "106"  # Ejemplo: 106 = Qatar 2022

# -- OPCIONES -----------------------------------------------------------------
#   MODO_PRUEBA = True  → descarga solo los primeros 5 partidos para verificar
#                         que todo funciona antes de descargar el torneo completo.
#   MODO_PRUEBA = False → descarga el torneo completo (recomendado para el deploy).
MODO_PRUEBA: bool = False

#   Carpeta donde se guardan los archivos. No es necesario cambiarla.
CARPETA_DATOS: str = "data"

#   Limite de advertencia de tamano (MB). Si el archivo events.parquet supera
#   este valor, el script avisa pero continua.
LIMITE_MB: int = 90

#   DATOS_COMPLETOS = False → guarda solo las columnas de eventos que usa el
#                             dashboard (recomendado: archivos ~7 veces mas ligeros,
#                             necesario para ligas completas y para publicar la app).
#   DATOS_COMPLETOS = True  → guarda las ~117 columnas de StatsBomb. Usalo solo si
#                             vas a agregar tus propios graficos con otras columnas.
DATOS_COMPLETOS: bool = False


# =============================================================================
# BLOQUE 3 -- CONFIGURACION INTERNA (no modificar)
# =============================================================================

_DATA = Path(__file__).parent / CARPETA_DATOS
_DATA.mkdir(exist_ok=True)

_T0 = time.time()

def _elapsed() -> str:
    s = int(time.time() - _T0)
    return f"{s//60}m {s%60:02d}s"

def _mb(path: Path) -> float:
    return path.stat().st_size / 1_048_576 if path.exists() else 0.0

def _sep(char="=", n=60):
    print(char * n)

# Columnas de eventos que usan las paginas del dashboard. Si agregas un grafico
# que necesita otra columna de StatsBomb, anadela a esta lista y vuelve a ejecutar.
# Sube este numero si cambias las columnas: el dashboard volvera a descargar.
VERSION_DATOS: int = 2

_COLUMNAS_EVENTOS = [
    "match_id", "period", "minute", "type", "team", "player", "position",
    "location", "pass_end_location", "pass_outcome", "pass_recipient",
    "pass_goal_assist", "shot_outcome", "shot_statsbomb_xg",
    "dribble_outcome", "goalkeeper_type", "duel_type",
    "home_team", "away_team", "match_date", "home_score", "away_score",
]

def _clean_list_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas con listas/dicts a cadenas JSON para compatibilidad parquet."""
    for col in df.columns:
        sample = df[col].dropna().head(50)
        if sample.apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
            )
    return df


# =============================================================================
# BLOQUE 4 -- VALIDACION Y CARGA DE PARTIDOS
# =============================================================================

_sep()
print("  RADAR TACTICO — Preparacion de datos del dashboard")
_sep()
print(f"  Torneo  : competition_id={COMPETITION_ID}  season_id={SEASON_ID}")
print(f"  Destino : {_DATA.resolve()}")
if MODO_PRUEBA:
    print("  Modo    : PRUEBA (solo primeros 5 partidos)")
else:
    print("  Modo    : COMPLETO")
_sep()

# Verificar que el torneo existe en StatsBomb Open Data
print("\n[1/4] Verificando torneo y cargando lista de partidos...")
try:
    _matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
except Exception as e:
    print(f"\n  ERROR: No se pudo acceder al torneo.")
    print(f"  Detalle: {e}")
    print(f"\n  Verifica que COMPETITION_ID={COMPETITION_ID} y SEASON_ID={SEASON_ID}")
    print("  son correctos. Consulta la tabla del BLOQUE 0.")
    sys.exit(1)

if _matches.empty:
    print(f"\n  ERROR: El torneo ({COMPETITION_ID}/{SEASON_ID}) no tiene partidos.")
    print("  Verifica los IDs en la tabla del BLOQUE 0.")
    sys.exit(1)

# Nombre del torneo
_comp_name = str(_matches["competition"].iloc[0]) if "competition" in _matches.columns else f"competition_{COMPETITION_ID}"
_season_name = str(_matches["season"].iloc[0]) if "season" in _matches.columns else f"season_{SEASON_ID}"

print(f"  Torneo encontrado : {_comp_name}")
print(f"  Temporada         : {_season_name}")
print(f"  Partidos totales  : {len(_matches)}")

# Aplicar modo prueba
_match_ids = _matches["match_id"].tolist()
if MODO_PRUEBA:
    _match_ids = _match_ids[:5]
    print(f"  Partidos a descargar (prueba): {len(_match_ids)}")
else:
    print(f"  Partidos a descargar: {len(_match_ids)}")

# Guardar matches.parquet
_matches_out = _DATA / "matches.parquet"
_matches_clean = _clean_list_cols(_matches.copy())
# Columnas con tipos mezclados (p. ej. home_manager_id cuando un equipo tuvo dos
# entrenadores: "1001017, 226") se guardan como texto para que parquet las acepte.
for _c in _matches_clean.select_dtypes(include="object").columns:
    if _matches_clean[_c].dropna().map(type).nunique() > 1:
        _matches_clean[_c] = _matches_clean[_c].map(lambda x: x if pd.isna(x) else str(x))
_matches_clean.to_parquet(_matches_out, compression="snappy", index=False)
print(f"  matches.parquet guardado  ({_mb(_matches_out):.2f} MB)")


# =============================================================================
# BLOQUE 5 -- DESCARGA DE EVENTOS
# =============================================================================

print(f"\n[2/4] Descargando eventos de {len(_match_ids)} partidos...")
print("  (Esto puede tardar varios minutos. No cierres la terminal.)\n")

_events_list  = []
_errors_ev    = []

for _i, _mid in enumerate(_match_ids, 1):
    # Identificar el partido para mostrar progreso
    _row = _matches[_matches["match_id"] == _mid].iloc[0]
    _home = str(_row.get("home_team", "Local")).strip()
    _away = str(_row.get("away_team", "Visitante")).strip()
    _date = str(_row.get("match_date", "")).strip()
    _label = f"{_home} vs {_away}  ({_date})"

    print(f"  [{_i:>3}/{len(_match_ids)}] {_label}", end="", flush=True)

    try:
        _ev = sb.events(match_id=int(_mid), flatten_attrs=True)
        _ev["match_id"]   = int(_mid)
        _ev["home_team"]  = _home
        _ev["away_team"]  = _away
        _ev["match_date"] = _date
        # Score
        _ev["home_score"] = int(_row.get("home_score", 0))
        _ev["away_score"] = int(_row.get("away_score", 0))
        if not DATOS_COMPLETOS:
            _ev = _ev[[c for c in _COLUMNAS_EVENTOS if c in _ev.columns]]
        _events_list.append(_ev)
        print(f"  ✓  {len(_ev):,} eventos", flush=True)
    except Exception as _e:
        _errors_ev.append((_mid, str(_e)))
        print(f"  ✗  ERROR: {_e}", flush=True)

if not _events_list:
    print("\n  ERROR CRITICO: No se descargo ningun evento. Verifica tu conexion.")
    sys.exit(1)

print(f"\n  Concatenando eventos...", end=" ", flush=True)
_events_df = pd.concat(_events_list, ignore_index=True)
print(f"{len(_events_df):,} filas en total.")

# Limpiar columnas con listas/dicts para compatibilidad parquet
print("  Limpiando columnas para parquet...", end=" ", flush=True)
_events_df = _clean_list_cols(_events_df)
print("listo.")

# Guardar events.parquet
_events_out = _DATA / "events.parquet"
print("  Guardando events.parquet...", end=" ", flush=True)
_events_df.to_parquet(_events_out, compression="snappy", index=False)
_ev_mb = _mb(_events_out)
print(f"listo.  ({_ev_mb:.1f} MB)")

if _ev_mb > LIMITE_MB:
    print(f"\n  ⚠  ADVERTENCIA: events.parquet pesa {_ev_mb:.1f} MB (limite sugerido: {LIMITE_MB} MB).")
    print("     GitHub rechaza archivos > 100 MB.")
    print("     Considera usar un torneo mas corto o activa Git LFS.")


# =============================================================================
# BLOQUE 6 -- DESCARGA DE ALINEACIONES
# =============================================================================

print(f"\n[3/4] Descargando alineaciones...")

_lineups_list = []
_errors_lu    = []

for _i, _mid in enumerate(_match_ids, 1):
    try:
        _lu_dict = sb.lineups(match_id=int(_mid))
        for _team_name, _lu_df in _lu_dict.items():
            _lu_df = _lu_df.copy()
            _lu_df["match_id"]  = int(_mid)
            _lu_df["team_name"] = _team_name
            _lineups_list.append(_lu_df)
    except Exception as _e:
        _errors_lu.append((_mid, str(_e)))

if _lineups_list:
    _lineups_df = pd.concat(_lineups_list, ignore_index=True)
    _lineups_df = _clean_list_cols(_lineups_df)

    # Expandir columnas anidadas de jugadores si existen
    if "positions" in _lineups_df.columns:
        def _parse_pos(val):
            if isinstance(val, list) and val:
                return str(val[0].get("position", "")) if isinstance(val[0], dict) else str(val[0])
            return ""
        _lineups_df["position_name"] = _lineups_df["positions"].apply(_parse_pos)

    _lineups_out = _DATA / "lineups.parquet"
    _lineups_df.to_parquet(_lineups_out, compression="snappy", index=False)
    print(f"  lineups.parquet guardado  ({_mb(_lineups_out):.2f} MB)")
    print(f"  Jugadores registrados     : {_lineups_df['player_name'].nunique():,}")
    print(f"  Equipos registrados       : {_lineups_df['team_name'].nunique():,}")
else:
    print("  ⚠  No se obtuvieron alineaciones.")


# =============================================================================
# BLOQUE 7 -- GENERAR CATALOGOS (equipos y jugadores)
# =============================================================================

print(f"\n[4/4] Generando catalogos de equipos y jugadores...")

# Catalogo de equipos (del archivo matches)
_teams_df = pd.DataFrame({
    "team": pd.concat([
        _matches["home_team"].str.strip(),
        _matches["away_team"].str.strip(),
    ]).drop_duplicates().sort_values().reset_index(drop=True)
})
_teams_out = _DATA / "teams.parquet"
_teams_df.to_parquet(_teams_out, compression="snappy", index=False)
print(f"  teams.parquet guardado    ({_mb(_teams_out):.2f} MB)  "
      f"— {len(_teams_df)} equipos")

# Catalogo de jugadores (del archivo lineups si existe)
if _lineups_list:
    _players_df = (
        _lineups_df[["player_id", "player_name", "team_name", "jersey_number"]]
        .drop_duplicates(subset=["player_id"])
        .sort_values("player_name")
        .reset_index(drop=True)
    )
    if "position_name" in _lineups_df.columns:
        _players_df = _players_df.merge(
            _lineups_df[["player_id", "position_name"]].drop_duplicates("player_id"),
            on="player_id", how="left"
        )
    _players_out = _DATA / "players.parquet"
    _players_df.to_parquet(_players_out, compression="snappy", index=False)
    print(f"  players.parquet guardado  ({_mb(_players_out):.2f} MB)  "
          f"— {len(_players_df)} jugadores")


# =============================================================================
# BLOQUE 8 -- RESUMEN FINAL
# =============================================================================

# Marca de que torneo hay en data/. El dashboard la compara con el BLOQUE 2 y,
# si cambiaste de torneo, vuelve a descargar solo.
(_DATA / "torneo.json").write_text(json.dumps({
    "competition_id": COMPETITION_ID, "season_id": SEASON_ID,
    "datos_completos": DATOS_COMPLETOS, "modo_prueba": MODO_PRUEBA,
    "version": VERSION_DATOS,
}), encoding="utf-8")

_sep()
print("  RESUMEN DE LA DESCARGA")
_sep()
print(f"  Torneo   : {_comp_name} — {_season_name}")
print(f"  Partidos : {len(_match_ids)}" +
      (" (modo prueba)" if MODO_PRUEBA else ""))
print(f"  Eventos  : {len(_events_df):,} filas  |  "
      f"{_events_df['type'].nunique()} tipos de evento")
print(f"  Tiempo   : {_elapsed()}")

if _errors_ev:
    print(f"\n  ⚠  Partidos con error en eventos ({len(_errors_ev)}): "
          + ", ".join(str(m) for m, _ in _errors_ev))

print(f"\n  Archivos generados en '{_DATA.resolve()}':")
for _f in sorted(_DATA.glob("*.parquet")):
    print(f"    {_f.name:<25} {_mb(_f):>6.2f} MB")

_total_mb = sum(_mb(f) for f in _DATA.glob("*.parquet"))
print(f"    {'TOTAL':<25} {_total_mb:>6.2f} MB")

if _total_mb > 90:
    print(f"\n  ⚠  El total supera 90 MB. Revisa la seccion de TORNEOS del BLOQUE 0.")
elif _total_mb > 0:
    print(f"\n  ✓  Tamano total dentro del limite para GitHub ({_total_mb:.1f} / 90 MB).")

if MODO_PRUEBA:
    print("\n  ─────────────────────────────────────────────────────")
    print("  MODO PRUEBA completado correctamente.")
    print("  Para descargar el torneo completo, cambia:")
    print("      MODO_PRUEBA = False")
    print("  y vuelve a ejecutar: python prepare_data.py")
    print("  ─────────────────────────────────────────────────────")
else:
    print("\n  ✓  Datos listos. Ya puedes ejecutar el dashboard con:")
    print("       streamlit run app.py")

_sep()
