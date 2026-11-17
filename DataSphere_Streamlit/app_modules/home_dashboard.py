"""NusaInsight Home dashboard.

Uses the existing dataset loader, sentiment function, Streamlit secrets and
MongoDB collections. Missing data is shown as an empty state, never as demo data.
"""
from __future__ import annotations

import html
import os
import hashlib
from datetime import date, datetime, timedelta
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from app_modules.custom_dataset_manager import get_custom_dataset_catalog


# ---------------------------------------------------------------------------
# Theme tokens
# ---------------------------------------------------------------------------
NAVY = "#183747"
TEAL = "#0F766E"
TEAL_LIGHT = "#E6F3F1"
PAGE = "#F4F7F8"
BORDER = "#DCE6E9"
MUTED = "#71838C"


# ---------------------------------------------------------------------------
# Data access (same MongoDB database and collections as existing pages)
# ---------------------------------------------------------------------------
def _secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = os.getenv(name, default)
    return str(value or os.getenv(name, default) or "").strip()


@st.cache_resource(show_spinner=False, ttl=90)
def _connect_mongodb(uri: str):
    if not uri:
        return None, "Not configured"
    try:
        from pymongo import MongoClient

        client = MongoClient(
            uri,
            serverSelectionTimeoutMS=8000,
            connectTimeoutMS=8000,
        )
        client.admin.command("ping")
        return client, "Connected"
    except ImportError:
        return None, "PyMongo is not installed"
    except Exception:
        # Do not expose connection details, credentials, or URI in the UI.
        return None, "Not connected"



def _check_groq_api(api_key, model):
    """Test a real Groq API request and text response."""
    if not api_key:
        return False, "GROQ_API_KEY is not configured."

    try:
        from groq import Groq

        client = Groq(
            api_key=api_key,
            timeout=30.0,
            max_retries=0
        )

        completion_args = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Reply with the word OK only."}
            ],
            "temperature": 0,
            "max_completion_tokens": 256,
        }
        if "gpt-oss" in model.lower():
            completion_args["reasoning_effort"] = "low"
        response = client.chat.completions.create(**completion_args)

        if not response.choices:
            return False, "Groq returned no response choices."

        choice = response.choices[0]
        content = choice.message.content

        if content and content.strip():
            return True, f"Groq API is active. Model {model} returned a response."

        if choice.finish_reason == "length":
            return False, (
                "The token limit was reached before text was generated. "
                "Try increasing max_completion_tokens."
            )

        if choice.message.tool_calls:
            return False, "The model returned a tool call instead of text."

        return False, (
            "A response was received, but its text is empty. "
            "Check the model and reasoning settings."
        )

    except ImportError:
        return False, "The Groq library is not installed."

    except Exception as exc:
        status = getattr(exc, "status_code", None)

        if status == 401:
            return False, "The Groq API key is invalid."
        if status == 403:
            return False, "Access to this model was denied."
        if status == 404:
            return False, "The model was not found."
        if status == 429:
            return False, "The API usage limit has been reached."

        return False, f"Groq test failed: {type(exc).__name__}."

def _to_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(parsed):
            return None
        return parsed.tz_convert(None).to_pydatetime()
    except Exception:
        return None


def _read_database(sentiment_fn) -> dict[str, Any]:
    """Read current counts, activity, community sentiment and recent events."""
    data: dict[str, Any] = {
        "client": None,
        "mongo_status": "Not configured",
        "document_count": None,
        "comment_count": None,
        "comments_by_day": [0] * 7,
        "training_by_day": [0] * 7,
        "recent": [],
        "sentiment": {"Positive": 0, "Neutral": 0, "Negative": 0},
        "sentiment_sample": 0,
        "training_available": False,
        "dataset_popularity": {},
    }

    uri = _secret("mongo_uri") or _secret("MONGODB_URI")
    database_name = _secret("mongo_database") or _secret("MONGODB_DATABASE", "datasphere")
    client, status = _connect_mongodb(uri)
    data["client"], data["mongo_status"] = client, status
    if client is None:
        return data

    try:
        db = client[database_name]
        collections = [name for name in db.list_collection_names() if not name.startswith("system.")]
        data["document_count"] = sum(db[name].count_documents({}) for name in collections)

        # The existing Community page stores public posts in this collection.
        posts_collection = db["community_posts"]
        data["comment_count"] = posts_collection.count_documents({})
        posts = list(
            posts_collection.find(
                {},
                {"author": 1, "text": 1, "created_at": 1, "likes": 1, "dislikes": 1},
            ).sort("created_at", -1).limit(500)
        )

        today = date.today()
        days = [today - timedelta(days=offset) for offset in range(6, -1, -1)]
        day_index = {day: index for index, day in enumerate(days)}
        comment_counts = [0] * 7
        sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}

        for post in posts:
            created = _to_datetime(post.get("created_at"))
            if created and created.date() in day_index:
                comment_counts[day_index[created.date()]] += 1
            label = sentiment_fn(str(post.get("text") or ""))
            if label in sentiment_counts:
                sentiment_counts[label] += 1

        data["comments_by_day"] = comment_counts
        data["sentiment"] = sentiment_counts
        data["sentiment_sample"] = len(posts)

        training_logs = []
        if "training_logs" in collections:
            data["training_available"] = True
            training_logs = list(
                db["training_logs"].find(
                    {},
                    {"created_at": 1, "model_name": 1, "algorithm": 1, "dataset": 1, "status": 1},
                ).sort("created_at", -1).limit(100)
            )

        training_counts = [0] * 7
        dataset_popularity: dict[str, int] = {}
        for record in training_logs:
            created = _to_datetime(record.get("created_at"))
            if created and created.date() in day_index:
                training_counts[day_index[created.date()]] += 1

        # Popularity uses the complete training history, not only the latest
        # 100 activity records. If aggregation is unavailable, show the
        # frequency from the records already retrieved.
        if "training_logs" in collections:
            try:
                pipeline = [
                    {"$match": {"dataset": {"$exists": True, "$nin": [None, ""]}}},
                    {"$group": {"_id": "$dataset", "usage_count": {"$sum": 1}}},
                    {"$sort": {"usage_count": -1, "_id": 1}},
                ]
                for row in db["training_logs"].aggregate(pipeline):
                    name = str(row.get("_id") or "").strip()
                    if name:
                        dataset_popularity[name] = int(row.get("usage_count", 0))
            except Exception:
                for record in training_logs:
                    name = str(record.get("dataset") or "").strip()
                    if name:
                        dataset_popularity[name] = dataset_popularity.get(name, 0) + 1

        data["training_by_day"] = training_counts
        data["dataset_popularity"] = dataset_popularity

        events = []
        for post in posts[:10]:
            events.append({
                "date": _to_datetime(post.get("created_at")),
                "logo": "💬",
                "title": str(post.get("author") or "Community comments"),
                "detail": str(post.get("text") or "Comment saved"),
            })

        if "dataset_uploads" in collections:
            for record in db["dataset_uploads"].find({}).sort("created_at", -1).limit(5):
                events.append({
                    "date": _to_datetime(record.get("created_at")),
                    "logo": "📁",
                    "title": "New dataset uploaded",
                    "detail": str(record.get("filename") or record.get("name") or "Dataset saved"),
                })

        for record in training_logs[:5]:
            model = record.get("model_name") or record.get("algorithm") or "Model ML"
            dataset = record.get("dataset")
            events.append({
                "date": _to_datetime(record.get("created_at")),
                "logo": "🧠",
                "title": "Model training completed",
                "detail": str(model) + (f" · {dataset}" if dataset else ""),
            })

        events.sort(key=lambda item: item["date"] or datetime.min, reverse=True)
        data["recent"] = events[:5]

    except Exception:
        data["mongo_status"] = "Unable to read database data"
        data["document_count"] = None
        data["comment_count"] = None
        data["recent"] = []

    return data


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def _go_to(page: str):
    # Keep the existing sidebar navigation key and options intact.
    st.session_state["datasphere_navigation"] = page


def _section(title: str, subtitle: str | None = None):
    sub = f'<div class="ni-section-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="ni-section-title">{html.escape(title)}</div>{sub}',
        unsafe_allow_html=True,
    )


def _empty(title: str, detail: str):
    st.markdown(
        f'<div class="ni-empty"><b>{html.escape(title)}</b><span>{html.escape(detail)}</span></div>',
        unsafe_allow_html=True,
    )


def _summary_card(label: str, value: str, note: str, accent: str = TEAL):
    """One summary card; all three text lines stay inside the same card."""
    st.markdown(
        f"""
        <div class="ni-summary-card">
          <div class="ni-summary-label"><span class="ni-summary-dot" style="background:{accent}"></span>{html.escape(label)}</div>
          <div class="ni-summary-value">{html.escape(value)}</div>
          <div class="ni-summary-note">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _activity_chart(database_data: dict[str, Any]):
    labels = [(date.today() - timedelta(days=offset)).strftime("%a") for offset in range(6, -1, -1)]
    day_names = {"Mon": "Sen", "Tue": "Sel", "Wed": "Rab", "Thu": "Kam", "Fri": "Jum", "Sat": "Sab", "Sun": "Min"}
    labels = [day_names.get(label, label) for label in labels]
    train = database_data["training_by_day"]
    comments = database_data["comments_by_day"]

    if sum(train) + sum(comments) == 0:
        _empty(
            "No activity in the last 7 days",
            "This chart will update after a model is trained or a community comment is saved to MongoDB.",
        )
        return

    fig, ax = plt.subplots(figsize=(7.6, 3.25))
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, train, width, color="#0F766E", label="Model Training", zorder=3)
    ax.bar(x + width / 2, comments, width, color="#8ABDB5", label="Comments", zorder=3)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Activity count", fontsize=9, color=MUTED)
    ax.tick_params(axis="both", labelsize=9, colors=MUTED)
    ax.grid(axis="y", color="#E7EDF0", linewidth=.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
    fig.tight_layout(pad=1.1)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _logo_status_card(name: str, image_url: str, status: str, ok: bool):
    """Render a compact service tile with logo, name and status vertically aligned."""
    status_color = TEAL if ok else "#B7791F"
    fallback = name.replace(" AI", "").replace("MongoDB", "mongo").replace("Streamlit", "streamlit")
    st.markdown(
        f"""
        <div class="ni-status-card">
          <div class="ni-logo-box">
            <img src="{image_url}" alt="{html.escape(name)} logo"
                 loading="eager"
                 onerror="this.style.display='none';this.nextElementSibling.style.display='block';">
            <span class="ni-logo-fallback">{html.escape(fallback)}</span>
          </div>
          <div class="ni-status-name">{html.escape(name)}</div>
          <div class="ni-status-value" style="color:{status_color}">
            <span class="ni-status-dot" style="background:{status_color}"></span>
            {html.escape(status)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main Home dashboard
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False, ttl=600)
def _get_realtime_weather():
    """Resolve location from public IP providers and fetch current weather.

    Multiple geolocation providers are attempted because free IP-location
    endpoints can be rate-limited or unavailable in some networks.
    """
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": "NusaInsight-Data-AI-Studio/1.0"})
    geo = None
    errors = []

    # Provider-specific normalizers keep the rest of the dashboard independent
    # from the geolocation vendor's response schema.
    providers = (
        ("https://ipwho.is/", lambda d: (
            d.get("latitude"), d.get("longitude"),
            {"city": d.get("city"), "region": d.get("region"),
             "country_name": d.get("country")}
        ) if d.get("success", True) else (None, None, {})),
        ("https://ipapi.co/json/", lambda d: (
            d.get("latitude"), d.get("longitude"),
            {"city": d.get("city"), "region": d.get("region"),
             "country_name": d.get("country_name")}
        )),
        ("https://ip-api.com/json/?fields=status,message,city,regionName,country,lat,lon", lambda d: (
            d.get("lat"), d.get("lon"),
            {"city": d.get("city"), "region": d.get("regionName"),
             "country_name": d.get("country")}
        ) if d.get("status") == "success" else (None, None, {})),
    )

    for url, normalize in providers:
        try:
            response = session.get(url, timeout=(3.5, 6))
            response.raise_for_status()
            lat, lon, details = normalize(response.json())
            if lat is not None and lon is not None:
                geo = {**details, "latitude": float(lat), "longitude": float(lon)}
                break
            errors.append(f"{url}: location not returned")
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}")

    if geo is None:
        raise ConnectionError("Automatic IP location failed: " + "; ".join(errors))

    params = {
        "latitude": geo["latitude"],
        "longitude": geo["longitude"],
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                   "is_day,precipitation,weather_code,wind_speed_10m",
        "timezone": "auto",
    }
    weather_errors = []
    try:
        response = session.get("https://api.open-meteo.com/v1/forecast",
                               params=params, timeout=(4, 10))
        response.raise_for_status()
        payload = response.json()
        current = payload.get("current") or {}
        if current:
            return geo, current, payload.get("current_units", {})
        weather_errors.append("Open-Meteo returned no current data")
    except Exception as exc:
        weather_errors.append(f"Open-Meteo: {type(exc).__name__}")

    # Keyless secondary weather source, used only if Open-Meteo is unreachable.
    try:
        url = f"https://wttr.in/{geo['latitude']},{geo['longitude']}?format=j1"
        response = session.get(url, timeout=(4, 10))
        response.raise_for_status()
        payload = response.json()
        item = (payload.get("current_condition") or [{}])[0]
        current = {
            "temperature_2m": item.get("temp_C"),
            "apparent_temperature": item.get("FeelsLikeC"),
            "relative_humidity_2m": item.get("humidity"),
            "wind_speed_10m": item.get("windspeedKmph"),
            "precipitation": item.get("precipMM"),
            "weather_code": -1,
            "time": item.get("localObsDateTime", "now"),
        }
        if current.get("temperature_2m") is not None:
            return geo, current, {
                "temperature_2m": "°C", "apparent_temperature": "°C",
                "wind_speed_10m": "km/h", "precipitation": "mm",
            }
        weather_errors.append("wttr.in returned no current data")
    except Exception as exc:
        weather_errors.append(f"wttr.in: {type(exc).__name__}")

    raise ConnectionError("Weather providers unavailable: " + "; ".join(weather_errors))


@st.cache_data(show_spinner=False, ttl=600)
def _get_weather_for_city(city_name: str):
    """Fetch weather for a manually entered city when IP geolocation is blocked."""
    import requests

    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city_name, "count": 1, "language": "en", "format": "json"},
        timeout=(4, 10),
        headers={"User-Agent": "NusaInsight-Data-AI-Studio/1.0"},
    )
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        raise ValueError(f"City '{city_name}' was not found.")
    place = results[0]
    geo = {"city": place.get("name"), "region": place.get("admin1"),
           "country_name": place.get("country"), "latitude": place["latitude"],
           "longitude": place["longitude"]}
    params = {"latitude": geo["latitude"], "longitude": geo["longitude"],
              "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                         "is_day,precipitation,weather_code,wind_speed_10m",
              "timezone": "auto"}
    headers = {"User-Agent": "NusaInsight-Data-AI-Studio/1.0"}
    try:
        response = requests.get("https://api.open-meteo.com/v1/forecast", params=params,
                                timeout=(4, 10), headers=headers)
        response.raise_for_status()
        payload = response.json()
        current = payload.get("current") or {}
        if current:
            return geo, current, payload.get("current_units", {})
    except requests.RequestException:
        pass

    # Same secondary source as the automatic-location path.
    response = requests.get(
        f"https://wttr.in/{geo['latitude']},{geo['longitude']}?format=j1",
        timeout=(4, 10), headers=headers,
    )
    response.raise_for_status()
    item = (response.json().get("current_condition") or [{}])[0]
    current = {
        "temperature_2m": item.get("temp_C"),
        "apparent_temperature": item.get("FeelsLikeC"),
        "relative_humidity_2m": item.get("humidity"),
        "wind_speed_10m": item.get("windspeedKmph"),
        "precipitation": item.get("precipMM"),
        "weather_code": -1,
        "time": item.get("localObsDateTime", "now"),
    }
    units = {"temperature_2m": "°C", "apparent_temperature": "°C",
             "wind_speed_10m": "km/h", "precipitation": "mm"}
    return geo, current, units


def _render_realtime_weather():
    st.subheader("Realtime Weather")
    result = None
    error_message = ""
    try:
        result = _get_realtime_weather()
    except Exception as exc:
        error_message = str(exc)

    if result is None:
        st.info("Automatic location/weather lookup is unavailable on this network. "
                "You can still retrieve live weather by entering a city below.")
        with st.form("weather_manual_location", clear_on_submit=False):
            city_name = st.text_input(
                "Weather location", placeholder="Enter city, e.g. Ungaran or Semarang",
                key="weather_fallback_city",
            )
            submitted = st.form_submit_button("Get weather")
        if submitted and city_name.strip():
            try:
                result = _get_weather_for_city(city_name.strip())
                st.session_state.pop("weather_lookup_error", None)
            except Exception as exc:
                st.session_state["weather_lookup_error"] = str(exc)
        elif submitted:
            st.warning("Enter a city name to look up the weather.")
        if st.session_state.get("weather_lookup_error"):
            st.warning("Weather lookup failed. Check that this device can access the internet "
                       "and try another city.")
        if result is None:
            return

    geo, current, units = result
    if not current:
        st.warning("The weather service returned no current observations.")
        return
    weather_codes = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle",
        53: "Moderate drizzle", 55: "Dense drizzle", 61: "Slight rain",
        63: "Moderate rain", 65: "Heavy rain", 71: "Slight snow",
        73: "Moderate snow", 75: "Heavy snow", 80: "Rain showers",
        81: "Moderate rain showers", 82: "Violent rain showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with hail",
    }
    place = ", ".join(str(geo.get(k)) for k in ("city", "region", "country_name") if geo.get(k))
    st.caption(f"{place or 'Approximate location'} · Updated {current.get('time', 'now')} (local time)")
    cols = st.columns(4)
    metrics = [
        ("Temperature", f"{current.get('temperature_2m', '—')} {units.get('temperature_2m', '°C')}"),
        ("Feels like", f"{current.get('apparent_temperature', '—')} {units.get('apparent_temperature', '°C')}"),
        ("Humidity", f"{current.get('relative_humidity_2m', '—')}%"),
        ("Wind", f"{current.get('wind_speed_10m', '—')} {units.get('wind_speed_10m', 'km/h')}"),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)
    try:
        code = int(current.get("weather_code", -1))
    except (TypeError, ValueError):
        code = -1
    condition = weather_codes.get(code, "Current conditions")
    st.caption(f"Conditions: {condition} · Precipitation: "
               f"{current.get('precipitation', '—')} {units.get('precipitation', 'mm')}")
    st.caption("Weather source: Open-Meteo (with wttr.in fallback) · IP-based location is approximate.")


def render_home_dashboard(load_dataset, dataset_descriptions, dataset_tasks, analyze_sentiment):
    """Render NusaInsight Home using application catalog and current MongoDB data."""
    st.markdown(
        """
        <style>
        :root {
          --ni-navy:#183747;
          --ni-teal:#0F766E;
          --ni-teal-light:#E6F3F1;
          --ni-page:#F4F7F8;
          --ni-border:#DCE6E9;
          --ni-muted:#71838C;
        }
        .stApp { background:var(--ni-page); }
        .block-container { max-width:1500px; padding-top:1.35rem; padding-bottom:2.3rem; }

        .ni-hero {
          border-radius:16px; padding:25px 30px; margin-bottom:18px; color:#fff;
          background:linear-gradient(112deg,#183747 0%,#17545C 58%,#0F766E 100%);
          box-shadow:0 5px 18px rgba(24,55,71,.09);
        }
        .ni-hero-kicker { color:#A9E8DE; font-size:11px; font-weight:800; letter-spacing:1.5px; }
        .ni-hero-title { color:#fff; font-size:clamp(30px,3.4vw,42px); line-height:1.12; font-weight:800; margin:5px 0 7px; }
        .ni-hero-subtitle { color:#E4F1EF; font-size:14px; line-height:1.65; max-width:850px; margin:0; }

        /* Summary: label, value and note are all inside the same card. */
        .ni-summary-card {
          background:#fff; border:1px solid var(--ni-border); border-radius:13px;
          padding:15px 17px; min-height:139px; height:100%; box-sizing:border-box;
          box-shadow:0 2px 8px rgba(24,55,71,.035); display:flex; flex-direction:column;
          align-items:center; justify-content:center; text-align:center;
        }
        .ni-summary-label { color:#667C85; font-size:12px; font-weight:650; display:flex; align-items:center; justify-content:center; gap:8px; text-align:center; }
        .ni-summary-dot { width:9px; height:9px; flex:0 0 9px; border-radius:50%; }
        .ni-summary-value { color:var(--ni-navy); font-size:32px; line-height:1.15; font-weight:800; margin-top:13px; text-align:center; width:100%; }
        .ni-summary-note { color:#71838C; background:#F5F9F9; border-radius:7px; padding:6px 8px; font-size:10px; line-height:1.35; margin-top:10px; width:100%; box-sizing:border-box; text-align:center; }

        .ni-section-title { color:var(--ni-navy); font-size:17px; font-weight:800; line-height:1.3; margin:0 0 3px; }
        .ni-section-sub { color:#82929A; font-size:11px; line-height:1.45; margin-bottom:12px; }
        .ni-empty {
          background:#F8FAFB; border:1px dashed #C9D8DD; border-radius:10px;
          min-height:145px; padding:20px 15px; display:flex; flex-direction:column;
          justify-content:center; align-items:center; text-align:center; gap:6px;
          color:#71838C; font-size:12px; line-height:1.5;
        }
        .ni-empty b { color:#355363; font-size:13px; }

        /* Equal-width and equal-height panels within each row. */
        [data-testid="stHorizontalBlock"] { align-items:stretch !important; }
        [data-testid="stColumn"] { display:flex; flex-direction:column; }
        [data-testid="stColumn"] > div { width:100%; }
        [data-testid="stVerticalBlockBorderWrapper"] { box-sizing:border-box; }
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ni-equal-popular) { min-height:190px; }

        /* Keep columns aligned to the panel height. */
        [data-testid="stColumn"] {
            display:flex !important;
            flex-direction:column !important;
            align-self:stretch !important;
        }

        [data-testid="stColumn"] > div {
            width:100%;
        }

        /* Make panel containers fill the available height. */
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ni-equal-exploration),
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ni-equal-status),
        [data-testid="stVerticalBlockBorderWrapper"]:has(.ni-equal-recent) {
            flex:1 1 auto !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
          border-color:var(--ni-border) !important; border-radius:13px !important;
          background:#fff !important; box-shadow:0 2px 8px rgba(24,55,71,.025);
        }

        .ni-popular-item {
          min-height:90px; height:100%; box-sizing:border-box; border:1px solid #E1EAED;
          border-radius:11px; background:#F8FAFB; padding:12px; display:flex;
          align-items:center; gap:10px;
        }
        .ni-dataset-mark { width:38px; height:38px; flex:0 0 38px; border-radius:10px; background:#E6F3F1; display:flex; align-items:center; justify-content:center; font-size:19px; }
        .ni-dataset-name { color:var(--ni-navy); font-weight:750; font-size:12px; line-height:1.35; }
        .ni-dataset-task { color:#71838C; font-size:10px; margin-top:3px; }
        .ni-dataset-count { color:var(--ni-teal); font-size:10px; font-weight:700; margin-top:5px; }

        .ni-exploration-note { color:#71838C; font-size:11px; line-height:1.5; margin:1px 0 8px; }
        /* Each service: logo on top, service name and live status below. */
        .ni-status-card {
          display:flex; flex-direction:column; align-items:center; justify-content:center;
          min-height:88px; padding:7px 10px; margin-bottom:7px; gap:2px;
          border:1px solid #E1EAED; border-radius:11px; background:#F8FAFB;
          box-sizing:border-box; text-align:center;
        }
        .ni-status-card:last-child { margin-bottom:0; }
        .ni-logo-box {
          width:100%; height:35px; display:flex; align-items:center;
          justify-content:center; box-sizing:border-box; padding:1px 4px;
          background:transparent; border:0; border-radius:7px;
        }
        .ni-logo-box img { display:block; max-width:68px; width:auto; max-height:34px; height:auto; object-fit:contain; }
        .ni-logo-fallback {
          display:none; color:#183747; font-size:13px; line-height:1.1;
          font-weight:850; letter-spacing:.1px; text-align:center;
        }
        .ni-status-name { color:var(--ni-navy); font-weight:800; font-size:11px; line-height:1.2; margin-top:1px; }
        .ni-status-value {
          color:#71838C; font-size:9.5px; line-height:1.2; font-weight:700;
          margin-top:2px; display:flex; align-items:center; justify-content:center; gap:5px;
          text-align:center;
        }
        .ni-status-dot { width:7px; height:7px; display:inline-block; border-radius:50%; flex:0 0 7px; }

        .ni-event { display:grid; grid-template-columns:32px minmax(0,1fr); gap:8px; padding:9px 0; border-bottom:1px solid #EAF0F1; }
        .ni-event:last-child { border-bottom:0; }
        .ni-event-icon { width:32px; height:32px; border-radius:50%; background:#E6F3F1; display:flex; align-items:center; justify-content:center; font-size:15px; }
        .ni-event-title { color:var(--ni-navy); font-size:11px; font-weight:750; }
        .ni-event-detail { color:#71838C; font-size:10px; line-height:1.4; margin-top:3px; overflow-wrap:anywhere; }
        .ni-event-time { color:#91A0A6; font-size:9px; margin-top:4px; }

        [data-testid="stMetric"] { background:#fff; border:1px solid var(--ni-border); border-radius:12px; padding:13px 15px; }
        .stButton button { background:var(--ni-teal); border:1px solid var(--ni-teal); border-radius:9px; color:#fff; font-weight:650; }
        .stButton button:hover { background:#155E59; border-color:#155E59; color:#fff; }
        div[data-testid="stDataFrame"] { border:1px solid var(--ni-border); border-radius:10px; overflow:hidden; }
        @media(max-width:760px) {

            .block-container {
                padding-left:1rem;
                padding-right:1rem;
            }

            [data-testid="stVerticalBlockBorderWrapper"]:has(.ni-equal-exploration),
            [data-testid="stVerticalBlockBorderWrapper"]:has(.ni-equal-status),
            [data-testid="stVerticalBlockBorderWrapper"]:has(.ni-equal-recent) {
                height:auto !important;
                min-height:0 !important;
                max-height:none !important;
                overflow:visible !important;
            }

        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    db = _read_database(analyze_sentiment)
    dataset_tasks = dict(dataset_tasks)
    # The Home catalog shares the same custom datasets stored by Dataset Explorer.
    for item in get_custom_dataset_catalog():
        if item.get("name"):
            dataset_tasks[item["name"]] = item.get("task", "Custom")
    dataset_names = list(dataset_tasks.keys())
    algorithm_count = 9  # Existing model choices: regression, classification and K-Means.

    # ------------------------------------------------------------------
    # HERO
    # ------------------------------------------------------------------
    st.markdown(
        """
        <div class="ni-hero">
          <div class="ni-hero-kicker">NUSAINSIGHT · DATA SCIENCE PLATFORM</div>
          <div class="ni-hero-title">Explore data. Find insight.</div>
          <p class="ni-hero-subtitle">
            One workspace to explore datasets, understand patterns, build machine learning models, and share insights with the community.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # SUMMARY CARDS — notes now belong inside their own cards.
    # ------------------------------------------------------------------
    metrics = st.columns(4, gap="medium")
    with metrics[0]:
        _summary_card("Available datasets", str(len(dataset_names)), "scikit-learn and UCI catalog", TEAL)
    with metrics[1]:
        _summary_card("Model options", str(algorithm_count), "Regression · Classification · Clustering", "#5D7E9B")
    with metrics[2]:
        doc_value = f"{db['document_count']:,}" if db["document_count"] is not None else "—"
        _summary_card("MongoDB records", doc_value, "Live database count", "#47A248")
    with metrics[3]:
        comment_value = f"{db['comment_count']:,}" if db["comment_count"] is not None else "—"
        _summary_card("Community comments", comment_value, "community_posts collection", "#D08B5B")

    st.write("")
    _render_realtime_weather()
    st.write("")

    # ------------------------------------------------------------------
    # THREE ALIGNED PANELS — Exploration, System Status and Recent Activity
    # All three panels share one Streamlit row, equal column width and height.
    # ------------------------------------------------------------------
    explore_col, status_col, recent_col = st.columns(
        [1.2, 0.8, 1.0],
        gap="medium"
    )

    with explore_col:
        with st.container(border=True, height=450):
            st.markdown('<span class="ni-equal-exploration"></span>', unsafe_allow_html=True)
            _section("Data Exploration", "Choose a dataset and features to explore patterns")
            default_index = dataset_names.index("Iris") if "Iris" in dataset_names else 0
            selected_dataset = st.selectbox(
                "Select a dataset", dataset_names, index=default_index, key="ni_home_dataset",
            )

            frame = None
            features = []
            try:
                frame, features, _ = load_dataset(selected_dataset)
            except Exception as exc:
                st.error(f"The dataset could not be loaded ({type(exc).__name__}). Check the data source.")

            if frame is not None:
                numeric_features = [
                    name for name in features
                    if name in frame.columns and pd.api.types.is_numeric_dtype(frame[name])
                ]
                if len(numeric_features) >= 2:
                    x_col, y_col = st.columns(2, gap="small")
                    with x_col:
                        x_feature = st.selectbox(
                            "X-axis", numeric_features, index=0,
                            key=f"ni_home_x_{selected_dataset}",
                        )
                    with y_col:
                        y_feature = st.selectbox(
                            "Y-axis", numeric_features, index=1,
                            key=f"ni_home_y_{selected_dataset}",
                        )

                    # A compact chart sized for a one-third dashboard column.
                    fig, ax = plt.subplots(figsize=(5.1, 2.55))
                    if "class_name" in frame.columns:
                        classes = frame["class_name"].dropna().astype(str).unique().tolist()
                        palette = ["#0F766E", "#D59B55", "#6488A0", "#9A83B5", "#B06D8B"]
                        sns.scatterplot(
                            data=frame, x=x_feature, y=y_feature, hue="class_name",
                            hue_order=classes, palette=palette[:max(1, len(classes))],
                            s=24, alpha=0.78, edgecolor="white", linewidth=0.25, ax=ax,
                        )
                        ax.legend(
                            title="Kelas", frameon=False, fontsize=6.5,
                            title_fontsize=7, loc="best", markerscale=.85,
                        )
                    else:
                        ax.scatter(
                            frame[x_feature], frame[y_feature], s=24,
                            color=TEAL, alpha=0.72, edgecolors="white", linewidths=0.25,
                        )
                    ax.set_xlabel(x_feature, fontsize=7.5, color=NAVY, labelpad=4)
                    ax.set_ylabel(y_feature, fontsize=7.5, color=NAVY, labelpad=4)
                    ax.tick_params(axis="both", labelsize=6.5, colors=MUTED)
                    ax.grid(color="#E7EDF0", linewidth=0.65, alpha=0.9)
                    ax.set_axisbelow(True)
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.spines["left"].set_color(BORDER)
                    ax.spines["bottom"].set_color(BORDER)
                    fig.tight_layout(pad=0.65)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    st.caption(f"Dataset: {selected_dataset}")
                else:
                    _empty(
                        "Chart not available",
                        "The dataset must have at least two numeric features.",
                    )

    with status_col:
        with st.container(border=True, height=450):
            st.markdown('<span class="ni-equal-status"></span>', unsafe_allow_html=True)
            _section("System Status", "Current status of application services")
            groq_key = _secret("GROQ_API_KEY") or _secret("groq_api_key")
            groq_model = _secret("GROQ_MODEL", "openai/gpt-oss-120b") or "openai/gpt-oss-120b"

            key_fingerprint = hashlib.sha256(
                f"{groq_key}:{groq_model}".encode("utf-8")
            ).hexdigest() if groq_key else ""

            groq_test = st.session_state.get("ni_groq_health", {})

            if groq_test.get("key_fingerprint") != key_fingerprint:
                groq_test = {}
                st.session_state.pop("ni_groq_health", None)

            # Automatic Groq connection test
            if not groq_key:
                groq_status = "Not configured"
                groq_ok = False

            elif not groq_test:
                groq_ok, groq_message = _check_groq_api(
                    groq_key,
                    groq_model
                )

                groq_test = {
                    "key_fingerprint": key_fingerprint,
                    "ok": groq_ok,
                    "status": groq_message,
                }

                st.session_state["ni_groq_health"] = groq_test

                groq_status = "API active" if groq_ok else "Test failed"

            else:
                groq_ok = bool(groq_test.get("ok"))
                groq_status = "API active" if groq_ok else "Test failed"

            # Keep service order vertical and predictable.
            _logo_status_card(
                "MongoDB",
                "https://cdn.simpleicons.org/mongodb/47A248",
                db["mongo_status"],
                db["mongo_status"] == "Connected",
            )
            _logo_status_card(
                "Groq AI",
                "https://shorturl.at/9sd01",
                groq_status,
                groq_ok,
            )
            _logo_status_card(
                "Streamlit",
                "https://cdn.simpleicons.org/streamlit/FF4B4B",
                "Running",
                True,
            )

            if db["mongo_status"] != "Connected":
                st.caption("Check the MongoDB Atlas configuration.")

    with recent_col:
        with st.container(border=True, height=450):
            st.markdown('<span class="ni-equal-recent"></span>', unsafe_allow_html=True)
            _section("Recent Activity", "Comments, dataset uploads, and training records from MongoDB")
            if not db["recent"]:
                _empty(
                    "No saved activity yet",
                    "Comments, uploads, and training runs will appear here after they are saved to the database.",
                )
            else:
                for event in db["recent"][:5]:
                    created = event["date"]
                    time_label = (
                        created.strftime("%d %b %Y · %H:%M")
                        if created else "Time unavailable"
                    )
                    st.markdown(
                        f"""
                        <div class="ni-event">
                          <div class="ni-event-icon">{event['logo']}</div>
                          <div>
                            <div class="ni-event-title">{html.escape(str(event['title'])[:65])}</div>
                            <div class="ni-event-detail">{html.escape(str(event['detail'])[:105])}</div>
                            <div class="ni-event-time">{html.escape(time_label)}</div>
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.caption(
        "NusaInsight · This overview uses the application dataset catalog and live MongoDB data. "
        "Unavailable data is shown as empty instead of simulated values."
    )
