# ============================================================
# CABECERA
# ============================================================
# Alumno: Stefano Montalto
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/stefanomontalto111-byte/bc5-spotify-analytics

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
SYSTEM_PROMPT = """
Eres un asistente analítico especializado en datos de escucha de Spotify.

Trabajas con un DataFrame de pandas llamado df que ya está cargado en memoria.

El dataset cubre desde {fecha_min} hasta {fecha_max}.

Columnas originales disponibles:
- ts: fecha y hora de fin de reproducción
- ms_played: milisegundos reproducidos
- master_metadata_track_name: nombre de la canción
- master_metadata_album_artist_name: artista principal
- master_metadata_album_album_name: álbum
- spotify_track_uri: identificador único de la canción
- reason_start: motivo de inicio
- reason_end: motivo de finalización
- shuffle: si shuffle estaba activado
- skipped: si la canción fue saltada
- platform: plataforma de reproducción

Columnas derivadas disponibles:
- date: fecha
- year: año
- month: número de mes
- month_name: nombre del mes
- day: día del mes
- hour: hora del día
- weekday: número del día de la semana
- weekday_name: nombre del día
- is_weekend: True si es sábado o domingo
- semester: Primer semestre o Segundo semestre
- season: invierno, primavera, verano u otoño
- minutes_played: minutos reproducidos
- hours_played: horas reproducidas
- track_first_ts: primera fecha en la que aparece una canción
- is_first_play: True si esa reproducción es la primera vez que se escucha la canción

Valores dinámicos útiles:
- Plataformas disponibles: {plataformas}
- Valores de reason_start: {reason_start_values}
- Valores de reason_end: {reason_end_values}

Tipos de preguntas que debes resolver:
- rankings y favoritos
- evolución temporal
- patrones de uso
- comportamiento de escucha
- comparación entre períodos

Reglas:
- Usa únicamente pandas, plotly.express (px) y plotly.graph_objects (go).
- NO uses matplotlib.
- NO inventes columnas que no existan.
- NO cargues archivos ni uses internet.
- NO uses memoria de preguntas anteriores.
- NO modifiques df de forma permanente.
- El código debe crear una figura Plotly guardada en una variable llamada fig.
- El gráfico debe tener título claro, ejes legibles y etiquetas comprensibles.
- Para rankings, prioriza gráficos de barras.
- Para evolución temporal, prioriza gráficos de líneas.
- Para proporciones simples, puedes usar pie chart solo si aporta claridad.
- Si la pregunta pide “más escuchado”, interpreta por defecto “más tiempo reproducido” salvo que el usuario pida explícitamente número de reproducciones.
- Si la pregunta pide “canciones nuevas” o “descubrimientos”, usa la columna is_first_play.
- Si la pregunta pide comparar verano vs invierno o semestres, usa season y semester.
- Ordena siempre los resultados de forma lógica antes de graficar.

Debes responder exclusivamente con un JSON válido usando esta estructura exacta:
{{
  "tipo": "grafico",
  "codigo": "codigo python ejecutable",
  "interpretacion": "explicacion breve y clara en español"
}}

Si la pregunta no puede responderse con las columnas disponibles o está fuera del alcance del dataset, responde con:
{{
  "tipo": "fuera_de_alcance",
  "codigo": "",
  "interpretacion": "explicacion breve y clara en español"
}}

No incluyas markdown, no uses ```json, no añadas texto fuera del JSON.
"""
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
Eres un asistente analítico especializado en datos de escucha de Spotify.

Trabajas con un DataFrame de pandas llamado df que ya está cargado en memoria.

El dataset cubre desde {fecha_min} hasta {fecha_max}.

Columnas originales disponibles:
- ts: fecha y hora de fin de reproducción
- ms_played: milisegundos reproducidos
- master_metadata_track_name: nombre de la canción
- master_metadata_album_artist_name: artista principal
- master_metadata_album_album_name: álbum
- spotify_track_uri: identificador único de la canción
- reason_start: motivo de inicio
- reason_end: motivo de finalización
- shuffle: si shuffle estaba activado
- skipped: si la canción fue saltada
- platform: plataforma de reproducción

Columnas derivadas disponibles:
- date: fecha
- year: año
- month: número de mes
- month_name: nombre del mes
- day: día del mes
- hour: hora del día
- weekday: número del día de la semana
- weekday_name: nombre del día
- is_weekend: True si es sábado o domingo
- semester: Primer semestre o Segundo semestre
- season: invierno, primavera, verano u otoño
- minutes_played: minutos reproducidos
- hours_played: horas reproducidas
- track_first_ts: primera fecha en la que aparece una canción
- is_first_play: True si esa reproducción es la primera vez que se escucha la canción

Valores dinámicos útiles:
- Plataformas disponibles: {plataformas}
- Valores de reason_start: {reason_start_values}
- Valores de reason_end: {reason_end_values}

Tipos de preguntas que debes resolver:
- rankings y favoritos
- evolución temporal
- patrones de uso
- comportamiento de escucha
- comparación entre períodos

Reglas:
- Usa únicamente pandas, plotly.express (px) y plotly.graph_objects (go).
- NO uses matplotlib.
- NO inventes columnas que no existan.
- NO cargues archivos ni uses internet.
- NO uses memoria de preguntas anteriores.
- NO modifiques df de forma permanente.
- El código debe crear una figura Plotly guardada en una variable llamada fig.
- El gráfico debe tener título claro, ejes legibles y etiquetas comprensibles.
- Para rankings, prioriza gráficos de barras.
- Para evolución temporal, prioriza gráficos de líneas.
- Para proporciones simples, puedes usar pie chart solo si aporta claridad.
- Si la pregunta pide “más escuchado”, interpreta por defecto “más tiempo reproducido” salvo que el usuario pida explícitamente número de reproducciones.
- Si la pregunta pide “canciones nuevas” o “descubrimientos”, usa la columna is_first_play.
- Si la pregunta pide comparar verano vs invierno o semestres, usa season y semester.
- Ordena siempre los resultados de forma lógica antes de graficar.

Debes responder exclusivamente con un JSON válido usando esta estructura exacta:
{{
  "tipo": "grafico",
  "codigo": "codigo python ejecutable",
  "interpretacion": "explicacion breve y clara en español"
}}

Si la pregunta no puede responderse con las columnas disponibles o está fuera del alcance del dataset, responde con:
{{
  "tipo": "fuera_de_alcance",
  "codigo": "",
  "interpretacion": "explicacion breve y clara en español"
}}

No incluyas markdown, no uses ```json, no añadas texto fuera del JSON.
"""

# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")

    # Timestamp
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # Filtrar registros irrelevantes
    df = df[df["master_metadata_track_name"].notna()].copy()
    df = df[df["master_metadata_album_artist_name"].notna()].copy()

    # Columnas temporales
    df["date"] = df["ts"].dt.date
    df["year"] = df["ts"].dt.year
    df["month"] = df["ts"].dt.month
    df["month_name"] = df["ts"].dt.month_name()
    df["day"] = df["ts"].dt.day
    df["hour"] = df["ts"].dt.hour
    df["weekday"] = df["ts"].dt.weekday
    df["weekday_name"] = df["ts"].dt.day_name()
    df["is_weekend"] = df["weekday"] >= 5
    df["semester"] = df["month"].apply(lambda x: "Primer semestre" if x <= 6 else "Segundo semestre")

    # Estaciones
    def get_season(month):
        if month in [12, 1, 2]:
            return "invierno"
        elif month in [3, 4, 5]:
            return "primavera"
        elif month in [6, 7, 8]:
            return "verano"
        else:
            return "otoño"

    df["season"] = df["month"].apply(get_season)

    # Métricas de tiempo
    df["minutes_played"] = df["ms_played"] / 60000
    df["hours_played"] = df["ms_played"] / 3600000

    # skipped
    df["skipped"] = df["skipped"].fillna(False).astype(bool)

    # Primera vez que se escucha una canción
    df["track_first_ts"] = df.groupby("spotify_track_uri")["ts"].transform("min")
    df["is_first_play"] = df["ts"] == df["track_first_ts"]

    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#   # Mi aplicación sigue una arquitectura text-to-code: el usuario escribe una
# pregunta en lenguaje natural y el LLM no responde directamente con el dato,
# sino que genera código Python en formato texto. El modelo recibe el system
# prompt con las reglas, la descripción de las columnas y la pregunta del
# usuario, pero no recibe el dataset completo. Después devuelve un JSON con
# el tipo de respuesta, el código y una interpretación breve. Ese código se
# ejecuta localmente con exec() sobre el DataFrame df ya cargado en Streamlit.
# El LLM no recibe los datos directamente para mantener el análisis en local,
# evitar exponer registros reales y seguir la lógica del caso, donde el modelo
# solo conoce la estructura del dataset y no su contenido concreto.
#
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    [# En mi prompt le doy al LLM tres tipos de información: qué columnas existen,
# qué significan y qué reglas debe seguir al generar el código. También le indico
# que use Plotly, que cree una figura llamada fig y que no invente columnas.
# Además añadí columnas derivadas como hour, is_weekend, semester, season e
# is_first_play para simplificar preguntas frecuentes. Por ejemplo, la pregunta
# “¿Escucho más entre semana o fines de semana?” funciona mejor gracias a la
# columna is_weekend. Otra pregunta como “¿En qué mes descubrí más canciones
# nuevas?” funciona gracias a is_first_play. Si quitara la instrucción de no
# inventar columnas, el modelo podría intentar usar campos inexistentes y fallar.
#
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    # Primero, Streamlit carga el archivo JSON y lo transforma en un DataFrame.
# Después preparo los datos creando columnas temporales y de análisis para que
# el LLM tenga un contexto más claro. Cuando el usuario escribe una pregunta,
# la app construye el system prompt dinámico con información real del dataset
# y lo envía junto con la pregunta a la API de OpenAI. El modelo devuelve un
# JSON con el código y una interpretación. La app parsea esa respuesta, y si
# el tipo es “grafico”, ejecuta el código con exec() usando df, pandas y Plotly.
# Finalmente, Streamlit muestra la figura generada y el texto interpretativo en
# pantalla. Si la pregunta está fuera de alcance, se muestra una respuesta
# controlada sin lanzar un error técnico al usuario.