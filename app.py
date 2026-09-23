"""
Veritas — умный подбор event-подрядчиков (HackAlem AI).
Жёсткая фильтрация + объяснения через OpenAI API.
"""

from __future__ import annotations

import ast
import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
DATE_MIN = date(2026, 9, 23)
DATE_MAX = date(2026, 12, 31)

CITIES = ["Алматы", "Астана", "Зарубежье"]

PREFERRED_CATEGORIES = [
    "Ведущий",
    "Фотограф",
    "Банкетный зал",
    "Флорист",
    "Декоратор",
    "Видеограф",
    "Лайв-бэнд",
    "Ведущий церемонии",
    "Подарки и сувениры",
    "Фото и видеобудки",
    "Инструменталист",
]

EVENT_FORMATS = [
    "свадьба",
    "той",
    "корпоратив",
    "конференция",
    "юбилей",
    "день рождения",
]

LANGUAGES = ["русский", "казахский", "английский"]

DEMO_SCENARIOS = {
    "demo1": {
        "label": "Плотная категория (Алматы, Ведущий, 15.10.2026, Корпоратив, 1 000 000 ₸)",
        "city": "Алматы",
        "category": "Ведущий",
        "event_date": date(2026, 10, 15),
        "event_format": "корпоратив",
        "budget": 1_000_000,
        "language": None,
        "duration_hours": None,
    },
    "demo2": {
        "label": "Редкая категория (Астана, Флорист, 05.11.2026, Свадьба, 350 000 ₸)",
        "city": "Астана",
        "category": "Флорист",
        "event_date": date(2026, 11, 5),
        "event_format": "свадьба",
        "budget": 350_000,
        "language": None,
        "duration_hours": None,
    },
    "demo3": {
        "label": "Запрос без результата (Алматы, Ведущий, 25.12.2026, Той, 300 000 ₸)",
        "city": "Алматы",
        "category": "Ведущий",
        "event_date": date(2026, 12, 25),
        "event_format": "той",
        "budget": 300_000,
        "language": None,
        "duration_hours": None,
    },
}

# ---------------------------------------------------------------------------
# Парсинг списков из CSV
# ---------------------------------------------------------------------------


def parse_list_field(value: Any) -> list[str]:
    """Распарсить categories / event_formats / languages / busy_dates в список строк."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip().strip("'\"") for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            pass

    parts = re.split(r"[,;|]", text)
    result: list[str] = []
    for part in parts:
        cleaned = part.strip().strip("'\"[]")
        if cleaned:
            result.append(cleaned)
    return result


def normalize_token(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def list_contains(items: list[str], needle: str) -> bool:
    needle_n = normalize_token(needle)
    return any(normalize_token(x) == needle_n for x in items)


def format_kzt(amount: float | int) -> str:
    return f"{int(amount):,}".replace(",", " ") + " ₸"


def format_format_title(fmt: str) -> str:
    return fmt[:1].upper() + fmt[1:] if fmt else fmt


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------


def find_dataset_path() -> Path:
    candidates = [
        APP_DIR / "hackathon-dataset-anonymized.csv",
        APP_DIR / "hackathon-dataset-anonymized .csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Не найден файл hackathon-dataset-anonymized.csv в папке приложения."
    )


@st.cache_data(show_spinner=False)
def load_contractors() -> pd.DataFrame:
    path = find_dataset_path()
    df = pd.read_csv(path)

    required = [
        "id",
        "anon_name",
        "categories",
        "city",
        "price_from_kzt",
        "event_formats",
        "languages",
        "max_hours",
        "busy_dates",
        "description",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"В CSV нет колонок: {', '.join(missing)}")

    df = df.copy()
    df["categories_list"] = df["categories"].apply(parse_list_field)
    df["event_formats_list"] = df["event_formats"].apply(parse_list_field)
    df["languages_list"] = df["languages"].apply(parse_list_field)
    df["busy_dates_list"] = df["busy_dates"].apply(parse_list_field)
    df["price_from_kzt"] = pd.to_numeric(df["price_from_kzt"], errors="coerce")
    df["max_hours"] = pd.to_numeric(df["max_hours"], errors="coerce")
    df["description"] = df["description"].fillna("").astype(str)
    df["anon_name"] = df["anon_name"].fillna("Без имени").astype(str)
    df["city"] = df["city"].fillna("").astype(str)
    df["id"] = df["id"].astype(str)
    return df


def collect_categories(df: pd.DataFrame) -> list[str]:
    all_cats: set[str] = set()
    for cats in df["categories_list"]:
        all_cats.update(cats)
    preferred = [c for c in PREFERRED_CATEGORIES if c in all_cats]
    rest = sorted(c for c in all_cats if c not in preferred)
    return preferred + rest


# ---------------------------------------------------------------------------
# Фильтрация (жёсткие правила)
# ---------------------------------------------------------------------------


def reason_for_reject(
    row: pd.Series,
    *,
    event_date: date,
    event_format: str,
    budget: int,
    language: Optional[str],
    duration_hours: Optional[float],
) -> Optional[str]:
    """Вернуть код причины отклонения или None, если кандидат подходит."""
    date_str = event_date.isoformat()
    if date_str in row["busy_dates_list"]:
        return "busy"
    if pd.isna(row["price_from_kzt"]) or float(row["price_from_kzt"]) > budget:
        return "budget"
    if not list_contains(row["event_formats_list"], event_format):
        return "format"
    if language and not list_contains(row["languages_list"], language):
        return "language"
    if duration_hours is not None and not pd.isna(row["max_hours"]):
        if float(row["max_hours"]) < float(duration_hours):
            return "duration"
    return None


def filter_contractors(
    df: pd.DataFrame,
    *,
    city: str,
    category: str,
    event_date: date,
    event_format: str,
    budget: int,
    language: Optional[str] = None,
    duration_hours: Optional[float] = None,
) -> dict[str, Any]:
    city_cat = df[
        (df["city"] == city)
        & df["categories_list"].apply(lambda cats: list_contains(cats, category))
    ].copy()

    pool_size = len(city_cat)
    if pool_size == 0:
        return {
            "outcome": "B",
            "pool_size": 0,
            "matched": [],
            "reject_counts": {},
            "city": city,
            "category": category,
        }

    reject_counts: dict[str, int] = {
        "busy": 0,
        "budget": 0,
        "format": 0,
        "language": 0,
        "duration": 0,
    }
    matched_rows: list[pd.Series] = []

    for _, row in city_cat.iterrows():
        reason = reason_for_reject(
            row,
            event_date=event_date,
            event_format=event_format,
            budget=budget,
            language=language,
            duration_hours=duration_hours,
        )
        if reason is None:
            matched_rows.append(row)
        else:
            reject_counts[reason] += 1

    # Детерминированный топ-3: цена ↑, затем id
    matched_rows.sort(
        key=lambda r: (
            float(r["price_from_kzt"]) if not pd.isna(r["price_from_kzt"]) else 1e18,
            str(r["id"]),
        )
    )
    top = matched_rows[:3]

    if not top:
        return {
            "outcome": "V",
            "pool_size": pool_size,
            "matched": [],
            "reject_counts": reject_counts,
            "city": city,
            "category": category,
            "event_format": event_format,
            "event_date": event_date,
            "budget": budget,
            "language": language,
            "duration_hours": duration_hours,
        }

    return {
        "outcome": "A",
        "pool_size": pool_size,
        "matched": top,
        "matched_total": len(matched_rows),
        "reject_counts": reject_counts,
        "city": city,
        "category": category,
        "event_format": event_format,
        "event_date": event_date,
        "budget": budget,
        "language": language,
        "duration_hours": duration_hours,
    }


# ---------------------------------------------------------------------------
# Текстовые отчёты (исходы B / V / неполный топ)
# ---------------------------------------------------------------------------


def category_plural_form(category: str, n: int) -> str:
    """Простая форма для отчёта: «4 ведущих»."""
    lower = category.lower()
    mapping = {
        "ведущий": ("ведущий", "ведущих", "ведущих"),
        "фотограф": ("фотограф", "фотографа", "фотографов"),
        "флорист": ("флорист", "флориста", "флористов"),
        "декоратор": ("декоратор", "декоратора", "декораторов"),
        "видеограф": ("видеограф", "видеографа", "видеографов"),
        "лайв-бэнд": ("лайв-бэнд", "лайв-бэнда", "лайв-бэндов"),
        "ведущий церемонии": (
            "ведущий церемонии",
            "ведущих церемонии",
            "ведущих церемонии",
        ),
        "банкетный зал": ("банкетный зал", "банкетных зала", "банкетных залов"),
        "подарки и сувениры": (
            "специалист по подаркам",
            "специалиста по подаркам",
            "специалистов по подаркам",
        ),
        "фото и видеобудки": (
            "фото/видеобудка",
            "фото/видеобудки",
            "фото/видеобудок",
        ),
        "инструменталист": (
            "инструменталист",
            "инструменталиста",
            "инструменталистов",
        ),
    }
    forms = mapping.get(lower)
    if not forms:
        return f"{n} специалистов категории «{category}»"
    if n % 10 == 1 and n % 100 != 11:
        word = forms[0]
    elif 2 <= n % 10 <= 4 and not (12 <= n % 100 <= 14):
        word = forms[1]
    else:
        word = forms[2]
    return f"{n} {word}"


def build_reject_report(result: dict[str, Any]) -> str:
    city = result["city"]
    category = result["category"]
    pool = result["pool_size"]
    counts = result["reject_counts"]
    fmt_title = format_format_title(result.get("event_format", ""))

    parts: list[str] = []
    if counts.get("busy"):
        n = counts["busy"]
        parts.append(f"{n} занят{'ы' if n > 1 else ''} на эту дату")
    if counts.get("budget"):
        n = counts["budget"]
        parts.append(
            f"{n} не укладывается в бюджет"
            if n == 1
            else f"{n} не укладываются в бюджет"
        )
    if counts.get("format"):
        n = counts["format"]
        parts.append(
            f"{n} не проводит мероприятия в формате {fmt_title}"
            if n == 1
            else f"{n} не проводят мероприятия в формате {fmt_title}"
        )
    if counts.get("language"):
        n = counts["language"]
        lang = result.get("language") or "выбранный язык"
        parts.append(
            f"{n} не владеет языком «{lang}»"
            if n == 1
            else f"{n} не владеют языком «{lang}»"
        )
    if counts.get("duration"):
        n = counts["duration"]
        parts.append(
            f"{n} не покрывает нужную длительность"
            if n == 1
            else f"{n} не покрывают нужную длительность"
        )

    if not parts:
        details = "ни один не прошёл дополнительные условия"
    elif len(parts) == 1:
        details = parts[0]
    else:
        details = ", ".join(parts[:-1]) + f", {parts[-1]}"

    return (
        f"В городе {city} найдено {category_plural_form(category, pool)}, "
        f"но: {details}."
    )


def build_incomplete_note(result: dict[str, Any]) -> Optional[str]:
    matched_total = result.get("matched_total", len(result.get("matched", [])))
    if matched_total >= 3:
        return None

    counts = result.get("reject_counts", {})
    city = result["city"]
    category = result["category"]
    pool = result["pool_size"]
    fmt_title = format_format_title(result.get("event_format", ""))

    reasons: list[str] = []
    if counts.get("busy"):
        reasons.append(f"{counts['busy']} заняты на выбранную дату")
    if counts.get("budget"):
        reasons.append(f"{counts['budget']} дороже бюджета")
    if counts.get("format"):
        reasons.append(f"{counts['format']} не работают с форматом «{fmt_title}»")
    if counts.get("language"):
        reasons.append(f"{counts['language']} без нужного языка")
    if counts.get("duration"):
        reasons.append(f"{counts['duration']} с недостаточной длительностью")

    if pool < 3 and not reasons:
        return (
            f"В городе {city} всего {category_plural_form(category, pool)} — "
            f"поэтому в выдаче меньше 3 кандидатов."
        )

    reason_text = "; ".join(reasons) if reasons else "остальные не прошли фильтры"
    return (
        f"Найдено подходящих: {matched_total} из {pool} в категории «{category}» "
        f"({city}). Почему выдача неполная: {reason_text}."
    )


# ---------------------------------------------------------------------------
# Объяснения (OpenAI / заглушка на фактах)
# ---------------------------------------------------------------------------


def extract_format_fact(description: str, event_format: str) -> str:
    desc = description or ""
    fmt = event_format.lower()
    # Ищем предложение/фрагмент с упоминанием формата
    sentences = re.split(r"(?<=[.!?])\s+", desc)
    for sentence in sentences:
        if fmt in sentence.lower():
            snippet = sentence.strip()
            if len(snippet) > 180:
                snippet = snippet[:177] + "…"
            return snippet
    # fallback: первые ~160 символов описания
    snippet = re.sub(r"\s+", " ", desc).strip()
    if len(snippet) > 160:
        snippet = snippet[:157] + "…"
    return snippet or "в описании указан опыт работы с мероприятиями"


def build_fact_explanation(
    row: pd.Series,
    *,
    budget: int,
    event_format: str,
    language: Optional[str],
) -> str:
    price = int(row["price_from_kzt"]) if not pd.isna(row["price_from_kzt"]) else 0
    savings = max(0, int(budget) - price)
    langs = ", ".join(row["languages_list"]) or "не указаны"
    fmt_title = format_format_title(event_format)
    fact = extract_format_fact(str(row["description"]), event_format)

    parts = [
        f"Цена от {format_kzt(price)} — экономия бюджета {format_kzt(savings)} "
        f"относительно лимита {format_kzt(budget)}.",
        f"Формат «{fmt_title}» есть в профиле; из описания: {fact}",
    ]
    if language:
        parts.append(f"Владеет языком «{language}» (в профиле: {langs}).")
    elif row["languages_list"]:
        parts.append(f"Языки в профиле: {langs}.")
    return " ".join(parts)


def call_openai_explanation(
    row: pd.Series,
    *,
    api_key: str,
    city: str,
    category: str,
    event_date: date,
    event_format: str,
    budget: int,
    language: Optional[str],
    duration_hours: Optional[float],
) -> str:
    from openai import OpenAI

    price = int(row["price_from_kzt"]) if not pd.isna(row["price_from_kzt"]) else 0
    savings = max(0, int(budget) - price)
    langs = ", ".join(row["languages_list"]) or "не указаны"
    formats = ", ".join(row["event_formats_list"]) or "не указаны"
    max_h = (
        "не указано"
        if pd.isna(row["max_hours"])
        else str(int(row["max_hours"]))
    )

    user_prompt = f"""Запрос клиента:
- Город: {city}
- Категория: {category}
- Дата: {event_date.isoformat()}
- Формат: {event_format}
- Бюджет: {budget} KZT
- Язык: {language or "не задан"}
- Длительность (часов): {duration_hours if duration_hours is not None else "не задана"}

Кандидат:
- Имя: {row["anon_name"]}
- Цена от: {price} KZT
- Экономия относительно бюджета: {savings} KZT
- Форматы: {formats}
- Языки: {langs}
- Max hours: {max_h}
- Описание: {row["description"][:900]}

Напиши 1–2 конкретных предложения на русском, почему кандидат подходит.
ОБЯЗАТЕЛЬНО укажи точную цену и экономию бюджета, знание языков (если релевантно)
и конкретный факт опыта именно в формате «{event_format}» из описания.
СТРОГО запрещены клише: «отличный выбор», «идеально подойдёт», «рекомендуем»,
«прекрасный специалист», «лучший вариант». Без маркетинговых общих фраз.
Только факты."""

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты аналитик подбора подрядчиков. Пиши кратко, по фактам, "
                    "без клише и рекламных формулировок."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    return text or build_fact_explanation(
        row, budget=budget, event_format=event_format, language=language
    )


def explain_candidate(
    row: pd.Series,
    *,
    api_key: Optional[str],
    city: str,
    category: str,
    event_date: date,
    event_format: str,
    budget: int,
    language: Optional[str],
    duration_hours: Optional[float],
) -> tuple[str, str]:
    """Вернуть (текст, источник: openai|facts)."""
    cache_key = (
        f"{row['id']}|{city}|{category}|{event_date}|{event_format}|"
        f"{budget}|{language}|{duration_hours}|{bool(api_key)}"
    )
    cache = st.session_state.setdefault("explain_cache", {})
    if cache_key in cache:
        return cache[cache_key]

    if not api_key:
        result = (
            build_fact_explanation(
                row, budget=budget, event_format=event_format, language=language
            ),
            "facts",
        )
        cache[cache_key] = result
        return result
    try:
        text = call_openai_explanation(
            row,
            api_key=api_key,
            city=city,
            category=category,
            event_date=event_date,
            event_format=event_format,
            budget=budget,
            language=language,
            duration_hours=duration_hours,
        )
        result = (text, "openai")
    except Exception as exc:  # noqa: BLE001 — показываем факт-заглушку жюри
        fallback = build_fact_explanation(
            row, budget=budget, event_format=event_format, language=language
        )
        result = (
            f"{fallback}\n\n_OpenAI недоступен ({type(exc).__name__}) — "
            f"показано объяснение на фактах._",
            "facts",
        )
    cache[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


def init_session_defaults() -> None:
    defaults = {
        "city": "Алматы",
        "category": "Ведущий",
        "event_date": date(2026, 10, 15),
        "event_format": "корпоратив",
        "budget": 1_000_000,
        "budget_slider": 1_000_000,
        "language": "Не важно",
        "use_duration": False,
        "duration_hours": 4.0,
        "run_search": False,
        "demo_flash": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_demo(demo_key: str) -> None:
    scenario = DEMO_SCENARIOS[demo_key]
    st.session_state["city"] = scenario["city"]
    st.session_state["category"] = scenario["category"]
    st.session_state["event_date"] = scenario["event_date"]
    st.session_state["event_format"] = scenario["event_format"]
    st.session_state["budget"] = scenario["budget"]
    st.session_state["budget_slider"] = scenario["budget"]
    st.session_state["language"] = "Не важно"
    st.session_state["use_duration"] = False
    st.session_state["run_search"] = True
    st.session_state["demo_flash"] = scenario["label"]


def render_candidate_card(
    row: pd.Series,
    rank: int,
    explanation: str,
    source: str,
    budget: int,
) -> None:
    price = int(row["price_from_kzt"]) if not pd.isna(row["price_from_kzt"]) else 0
    savings = max(0, budget - price)
    cats = ", ".join(row["categories_list"])
    fmts = ", ".join(row["event_formats_list"])
    langs = ", ".join(row["languages_list"]) or "—"
    max_h = "—" if pd.isna(row["max_hours"]) else f"{int(row['max_hours'])} ч"

    with st.container(border=True):
        st.markdown(f"### #{rank} · {row['anon_name']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Цена от", format_kzt(price))
        c2.metric("Экономия", format_kzt(savings))
        c3.metric("Город", row["city"])
        c4.metric("Max hours", max_h)

        st.caption(f"ID: `{row['id']}` · Категории: {cats}")
        st.write(f"**Форматы:** {fmts}")
        st.write(f"**Языки:** {langs}")

        badge = "OpenAI · gpt-4o-mini" if source == "openai" else "Факты из профиля"
        st.info(f"**Почему подходит** ({badge})\n\n{explanation}")

        with st.expander("Описание подрядчика"):
            st.write(row["description"] or "—")


def main() -> None:
    st.set_page_config(
        page_title="Veritas · Подбор event-подрядчиков",
        page_icon="🎯",
        layout="wide",
    )
    init_session_defaults()

    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; }
        div[data-testid="stMetricValue"] { font-size: 1.15rem; }
        .veritas-hero {
            background: linear-gradient(120deg, #0f2b24 0%, #1a4d3e 45%, #c4a35a 140%);
            color: #f7f3ea;
            padding: 1.25rem 1.5rem;
            border-radius: 14px;
            margin-bottom: 1rem;
        }
        .veritas-hero h1 { margin: 0; font-size: 1.7rem; letter-spacing: 0.02em; }
        .veritas-hero p { margin: 0.35rem 0 0; opacity: 0.9; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="veritas-hero">
          <h1>VERITAS · Умный подбор event-подрядчиков</h1>
          <p>HackAlem AI · жёсткая фильтрация + объяснения на фактах (OpenAI)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Демо-кнопки для жюри ---
    st.markdown("##### Демо для жюри")
    d1, d2, d3 = st.columns(3)
    with d1:
        if st.button(DEMO_SCENARIOS["demo1"]["label"], use_container_width=True, type="primary"):
            apply_demo("demo1")
            st.rerun()
    with d2:
        if st.button(DEMO_SCENARIOS["demo2"]["label"], use_container_width=True):
            apply_demo("demo2")
            st.rerun()
    with d3:
        if st.button(DEMO_SCENARIOS["demo3"]["label"], use_container_width=True):
            apply_demo("demo3")
            st.rerun()

    if st.session_state.get("demo_flash"):
        st.success(f"Загружен сценарий: {st.session_state['demo_flash']}")
        st.session_state["demo_flash"] = None

    try:
        df = load_contractors()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Не удалось загрузить датасет: {exc}")
        st.stop()

    categories = collect_categories(df)
    if st.session_state["category"] not in categories and categories:
        st.session_state["category"] = categories[0]

    # --- Sidebar ---
    with st.sidebar:
        st.header("Параметры поиска")
        st.caption(f"В базе: {len(df)} подрядчиков")

        env_key = os.getenv("OPENAI_API_KEY", "")
        api_key_input = st.text_input(
            "OpenAI API Key",
            value=env_key,
            type="password",
            help="Берётся из os.getenv('OPENAI_API_KEY'), можно ввести вручную.",
        )
        api_key = (api_key_input or env_key or "").strip() or None
        if api_key:
            st.caption("Ключ задан · объяснения через gpt-4o-mini (temperature=0)")
        else:
            st.caption("Ключа нет · будут фактические заглушки без клише")

        st.divider()
        city = st.selectbox("Город", CITIES, key="city")
        category = st.selectbox("Категория", categories, key="category")
        event_date = st.date_input(
            "Дата мероприятия",
            min_value=DATE_MIN,
            max_value=DATE_MAX,
            key="event_date",
        )
        event_format = st.selectbox(
            "Формат мероприятия", EVENT_FORMATS, key="event_format"
        )
        def _sync_budget_from_slider() -> None:
            st.session_state["budget"] = int(st.session_state["budget_slider"])

        def _sync_budget_from_input() -> None:
            val = int(st.session_state["budget"])
            st.session_state["budget_slider"] = min(max(val, 50_000), 6_000_000)

        st.slider(
            "Бюджет, ₸",
            min_value=50_000,
            max_value=6_000_000,
            step=50_000,
            key="budget_slider",
            format="%d",
            on_change=_sync_budget_from_slider,
        )
        st.number_input(
            "Бюджет (точное значение), ₸",
            min_value=10_000,
            max_value=20_000_000,
            step=10_000,
            key="budget",
            on_change=_sync_budget_from_input,
        )
        st.caption(f"Выбрано: {format_kzt(int(st.session_state['budget']))}")

        language_opt = st.selectbox(
            "Язык (опционально)",
            ["Не важно"] + LANGUAGES,
            key="language",
        )
        use_duration = st.checkbox("Указать длительность", key="use_duration")
        duration_hours: Optional[float] = None
        if use_duration:
            duration_hours = float(
                st.number_input(
                    "Длительность, часов",
                    min_value=1.0,
                    max_value=24.0,
                    step=1.0,
                    key="duration_hours",
                )
            )

        search_clicked = st.button(
            "Найти подрядчиков", type="primary", use_container_width=True
        )
        if search_clicked:
            st.session_state["run_search"] = True

    language = None if language_opt == "Не важно" else language_opt
    budget = int(st.session_state["budget"])

    # Сводка запроса
    with st.expander("Текущий запрос", expanded=True):
        cols = st.columns(6)
        cols[0].write(f"**Город**\n\n{city}")
        cols[1].write(f"**Категория**\n\n{category}")
        cols[2].write(f"**Дата**\n\n{event_date.strftime('%d.%m.%Y')}")
        cols[3].write(f"**Формат**\n\n{format_format_title(event_format)}")
        cols[4].write(f"**Бюджет**\n\n{format_kzt(budget)}")
        cols[5].write(
            f"**Язык / часы**\n\n{language or '—'} / "
            f"{int(duration_hours) if duration_hours else '—'}"
        )

    if not st.session_state.get("run_search"):
        st.info("Выберите параметры слева или нажмите одну из демо-кнопок сверху.")
        st.stop()

    result = filter_contractors(
        df,
        city=city,
        category=category,
        event_date=event_date if isinstance(event_date, date) else date.fromisoformat(str(event_date)),
        event_format=event_format,
        budget=budget,
        language=language,
        duration_hours=duration_hours,
    )

    outcome = result["outcome"]

    if outcome == "B":
        st.warning(
            f"В городе {result['city']} нет специалистов в категории {result['category']}"
        )
        st.caption("Исход Б · категория отсутствует в выбранном городе")
        return

    if outcome == "V":
        st.error("Подходящих подрядчиков не найдено")
        st.markdown(build_reject_report(result))
        st.caption("Исход В · в городе есть специалисты, но все отсеялись фильтрами")
        with st.expander("Детализация отсева"):
            st.json(result["reject_counts"])
        return

    # Исход А
    matched = result["matched"]
    st.success(
        f"Найдено подходящих: {result['matched_total']} · показываем топ-{len(matched)}"
    )
    incomplete = build_incomplete_note(result)
    if incomplete:
        st.warning(incomplete)

    for i, row in enumerate(matched, start=1):
        explanation, source = explain_candidate(
            row,
            api_key=api_key,
            city=city,
            category=category,
            event_date=event_date if isinstance(event_date, date) else date.fromisoformat(str(event_date)),
            event_format=event_format,
            budget=budget,
            language=language,
            duration_hours=duration_hours,
        )
        render_candidate_card(row, i, explanation, source, budget)


if __name__ == "__main__":
    main()
