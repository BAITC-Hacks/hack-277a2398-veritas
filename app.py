"""Veritas: умный подбор event-подрядчиков для HackAlem AI."""
from __future__ import annotations

import ast
import csv
import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Optional

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATE_MIN, DATE_MAX = date(2026, 9, 23), date(2026, 12, 31)
CITIES = ["Алматы", "Астана", "Зарубежье"]
LANGUAGES = ["Казахский", "Русский", "Английский"]
EVENT_FORMATS = ["свадьба", "той", "корпоратив", "конференция", "юбилей", "день рождения"]
CATEGORIES = ["Ведущий", "Фотограф", "Банкетный зал", "Флорист", "Декоратор", "Видеограф", "Лайв-бэнд", "Ведущий церемонии", "Подарки и сувениры", "Фото и видеобудки", "Инструменталист"]
DEMOS = {
    "d1": ("Плотная категория (Алматы, Ведущий, 15.10.2026, Корпоратив, 1 000 000 ₸)", "Алматы", "Ведущий", date(2026, 10, 15), "корпоратив", 1_000_000),
    "d2": ("Редкая категория (Астана, Флорист, 05.11.2026, Свадьба, 350 000 ₸)", "Астана", "Флорист", date(2026, 11, 5), "свадьба", 350_000),
    "d3": ("Запрос без результата (Алматы, Ведущий, 25.12.2026, Той, 300 000 ₸)", "Алматы", "Ведущий", date(2026, 12, 25), "той", 300_000),
}


def parse_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    return [p.strip().strip("'\"[]") for p in re.split(r"[,;|]", text) if p.strip().strip("'\"[]")]


def parse_number(value: Any) -> Optional[float]:
    try:
        text = str(value or "").strip()
        return float(text) if text else None
    except (TypeError, ValueError):
        return None


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def contains(values: list[str], target: str) -> bool:
    return any(norm(v) == norm(target) for v in values)


def find_dataset_path() -> Path:
    for candidate in (APP_DIR / "hackathon-dataset-anonymized.csv", APP_DIR / "hackathon-dataset-anonymized .csv"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Не найден hackathon-dataset-anonymized.csv рядом с app.py")


@st.cache_data(show_spinner=False)
def load_contractors() -> list[dict[str, Any]]:
    required = {"id", "anon_name", "categories", "city", "price_from_kzt", "event_formats", "languages", "max_hours", "busy_dates", "description"}
    with find_dataset_path().open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("В CSV отсутствуют колонки: " + ", ".join(sorted(missing)))
        rows = list(reader)
    for row in rows:
        row["categories_list"] = parse_list_field(row.get("categories"))
        row["event_formats_list"] = parse_list_field(row.get("event_formats"))
        row["languages_list"] = parse_list_field(row.get("languages"))
        row["busy_dates_list"] = parse_list_field(row.get("busy_dates"))
        row["price_from_kzt"] = parse_number(row.get("price_from_kzt"))
        row["max_hours"] = parse_number(row.get("max_hours"))
        row["description"] = (row.get("description") or "").strip()
        for key in ("id", "anon_name", "city"):
            row[key] = (row.get(key) or "").strip()
    return rows


def format_kzt(amount: float | int) -> str:
    return f"{int(amount):,}".replace(",", " ") + " ₸"


def candidate_score(row: dict[str, Any], *, event_format: str, budget: int, language: Optional[str]) -> float:
    """Фиксированные веса, без случайности: язык 45%, релевантность 35%, цена 20%."""
    langs = row["languages_list"]
    lang_score = 1.0 if language and contains(langs, language) else (0.5 if langs else 0.0)
    description = norm(row["description"])
    keywords = {
        "свадьба": ("свадьб", "свадеб", "церемон", "молодож", "торжеств"),
        "той": ("той", "традиц", "казах", "бата", "домбра"),
        "корпоратив": ("корпоратив", "команд", "компан", "бренд", "партнер", "партнёр"),
        "конференция": ("конференц", "форум", "делов", "спикер", "панел"),
        "юбилей": ("юбиле", "торжеств", "семейн", "поздравлен"),
        "день рождения": ("день рожден", "именин", "праздник", "шоу"),
    }.get(norm(event_format), ())
    rel_score = min(1.0, sum(1 for word in keywords if word in description) / max(1, min(3, len(keywords))))
    price = row["price_from_kzt"]
    price_score = max(0.0, 1.0 - abs(budget - price) / max(1, budget)) if price is not None else 0.0
    return (0.45 * lang_score if language else 0.0) + 0.35 * rel_score + 0.20 * price_score


def reject_reason(row: dict[str, Any], event_date: date, event_format: str, budget: int, language: Optional[str], duration: Optional[float]) -> Optional[str]:
    if event_date.isoformat() in row["busy_dates_list"]:
        return "busy"
    if row["price_from_kzt"] is None or row["price_from_kzt"] > budget:
        return "budget"
    if not contains(row["event_formats_list"], event_format):
        return "format"
    if language and not contains(row["languages_list"], language):
        return "language"
    if duration is not None and row["max_hours"] is not None and row["max_hours"] < duration:
        return "duration"
    return None


def filter_contractors(rows: list[dict[str, Any]], *, city: str, category: str, event_date: date, event_format: str, budget: int, language: Optional[str] = None, duration: Optional[float] = None) -> dict[str, Any]:
    pool = [r for r in rows if norm(r["city"]) == norm(city) and contains(r["categories_list"], category)]
    counts = {k: 0 for k in ("busy", "budget", "format", "language", "duration")}
    passed = []
    for row in pool:
        reason = reject_reason(row, event_date, event_format, budget, language, duration)
        if reason:
            counts[reason] += 1
        else:
            passed.append(row)
    passed.sort(key=lambda r: (-candidate_score(r, event_format=event_format, budget=budget, language=language), str(r["id"]), norm(r["anon_name"])))
    outcome = "B" if not pool else ("A" if passed else "V")
    return {"outcome": outcome, "pool": pool, "pool_size": len(pool), "matched": passed[:3], "matched_total": len(passed), "reject_counts": counts, "city": city, "category": category, "event_date": event_date, "event_format": event_format, "budget": budget, "language": language, "duration": duration}


def rejection_report(result: dict[str, Any]) -> str:
    names = {"busy": "заняты на эту дату", "budget": "не укладываются в бюджет", "format": f"не проводят мероприятия в формате {result['event_format'].capitalize()}", "language": f"не знают язык {result['language']}", "duration": "не подходят по длительности"}
    details = [f"{n} {names[key]}" for key, n in result["reject_counts"].items() if n]
    return f"В городе найдено {result['pool_size']} специалистов категории «{result['category']}», но: " + ("; ".join(details) if details else "никто не прошёл условия") + "."


def explanation_facts(row: dict[str, Any], budget: int, event_format: str, language: Optional[str]) -> str:
    price = int(row["price_from_kzt"] or 0)
    saving = max(0, budget - price)
    desc = re.sub(r"\s+", " ", row["description"]).strip()
    # Выбираем строку описания с конкретным совпадением по формату; иначе даём содержательный фрагмент.
    sentences = re.split(r"(?<=[.!?])\s+", desc)
    tokens = {"той": ("той", "традиц", "домбр", "бата"), "свадьба": ("свадьб", "церемон", "молодож"), "корпоратив": ("корпоратив", "команд", "компан", "квн", "телевиз", "тв"), "конференция": ("конференц", "форум", "спикер", "делов"), "юбилей": ("юбиле", "семейн", "торжеств"), "день рождения": ("именин", "день рожден", "праздник")}.get(norm(event_format), ())
    fact = next((s.strip() for s in sentences if any(t in norm(s) for t in tokens)), desc[:280])
    parts = [f"Цена — {format_kzt(price)}, экономия относительно бюджета — {format_kzt(saving)}.", f"Релевантный факт из профиля для формата «{event_format}»: {fact or 'в профиле нет подробного описания'}"]
    parts.append(f"В профиле указаны языки: {', '.join(row['languages_list']) or 'не указаны'}." if not language else f"Владеет требуемым языком «{language}».")
    return " ".join(parts)


def call_openai(row: dict[str, Any], *, api_key: str, city: str, category: str, event_date: date, event_format: str, budget: int, language: Optional[str], duration: Optional[float]) -> str:
    from openai import OpenAI
    price = int(row["price_from_kzt"] or 0)
    prompt = f"""Запрос заказчика: город {city}; категория {category}; дата {event_date.isoformat()}; формат {event_format}; бюджет {budget} KZT; язык {language or 'не задан'}; длительность {duration or 'не задана'} ч.
Кандидат: {row['anon_name']}; цена от {price} KZT; экономия {max(0, budget-price)} KZT; языки: {', '.join(row['languages_list']) or 'не указаны'}; форматы: {', '.join(row['event_formats_list'])}; описание: {row['description'][:1800]}

Напиши 2–3 ёмких предложения по-русски, персонально и с глубоким контекстом. Вытащи наиболее уникальную конкретную зацепку именно из этого description (например награду, эфир/ТВ, лигу КВН, традиции тоя, проекты или необычную специализацию) и объясни, почему она ценна именно для данного формата события. Другим кандидатам при том же запросе нужны принципиально иные аргументы и интонация: не используй шаблонную структуру или общие фразы. Обязательно назови точную цену, экономию относительно бюджета и язык, если он был запрошен. Не выдумывай факты и достижения; если уникальных деталей нет, честно опирайся на конкретные имеющиеся сведения. Запрещены клише вроде «отличный выбор». Только факты из профиля."""
    response = OpenAI(api_key=api_key).chat.completions.create(model="gpt-4o", temperature=0, messages=[{"role": "system", "content": "Ты внимательный персональный консультант по event-подрядчикам. Анализируй только предоставленные факты; для каждого профиля находи его собственную сильную сторону."}, {"role": "user", "content": prompt}])
    return (response.choices[0].message.content or "").strip() or explanation_facts(row, budget, event_format, language)


def explain(row: dict[str, Any], *, api_key: Optional[str], **context: Any) -> tuple[str, str]:
    key = f"{row['id']}|{context['city']}|{context['event_date']}|{context['event_format']}|{context['budget']}|{context['language']}|{bool(api_key)}"
    cache = st.session_state.setdefault("explain_cache", {})
    if key in cache:
        return cache[key]
    if api_key:
        try:
            result = (call_openai(row, api_key=api_key, **context), "OpenAI gpt-4o")
        except Exception as exc:
            result = (explanation_facts(row, context["budget"], context["event_format"], context["language"]) + f"\n\n_OpenAI недоступен ({type(exc).__name__}); показано объяснение по фактам._", "Объяснение по фактам")
    else:
        result = (explanation_facts(row, context["budget"], context["event_format"], context["language"]), "Объяснение по фактам")
    cache[key] = result
    return result


def set_demo(name: str) -> None:
    _, city, category, event_date, fmt, budget = DEMOS[name]
    st.session_state.update(city=city, category=category, event_date=event_date, event_format=fmt, budget=budget, language="Не важно", duration_enabled=False, run_search=True)


def main() -> None:
    st.set_page_config(page_title="Veritas — подбор event-подрядчиков", page_icon="✨", layout="wide")
    st.title("VERITAS · Умный подбор event-подрядчиков")
    st.caption("HackAlem AI · Детерминированная фильтрация и контекстные объяснения")
    cols = st.columns(3)
    for col, name in zip(cols, DEMOS):
        if col.button(DEMOS[name][0], use_container_width=True, type="primary" if name == "d1" else "secondary"):
            set_demo(name)
            st.rerun()
    try:
        rows = load_contractors()
    except Exception as exc:
        st.error(f"Не удалось загрузить датасет: {exc}")
        st.stop()
    available = sorted({c for row in rows for c in row["categories_list"]})
    categories = [c for c in CATEGORIES if c in available] + sorted(c for c in available if c not in CATEGORIES)
    with st.sidebar:
        st.header("Параметры поиска")
        api_env = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API Key", value=api_env, type="password", help="Можно также задать переменную окружения OPENAI_API_KEY.").strip() or None
        st.caption("Объяснения через gpt-4o" if api_key else "Без ключа будут показаны объяснения по фактам профиля")
        city = st.selectbox("Город", CITIES, key="city")
        category = st.selectbox("Категория", categories, key="category")
        event_date = st.date_input("Дата мероприятия", min_value=DATE_MIN, max_value=DATE_MAX, key="event_date")
        event_format = st.selectbox("Формат мероприятия", EVENT_FORMATS, key="event_format")
        budget = int(st.number_input("Бюджет, ₸", min_value=10_000, max_value=20_000_000, step=10_000, key="budget"))
        lang_choice = st.selectbox("Желаемый язык (необязательно)", ["Не важно"] + LANGUAGES, key="language")
        language = None if lang_choice == "Не важно" else lang_choice
        duration_enabled = st.checkbox("Указать длительность", key="duration_enabled")
        duration = float(st.number_input("Длительность, часов", 1.0, 24.0, 4.0, 1.0)) if duration_enabled else None
        if st.button("Найти подрядчиков", type="primary", use_container_width=True):
            st.session_state["run_search"] = True
    st.info(f"**Запрос:** {city} · {category} · {event_date:%d.%m.%Y} · {event_format.capitalize()} · {format_kzt(budget)} · {language or 'язык любой'} · {str(duration)+' ч' if duration else 'длительность любая'}")
    if not st.session_state.get("run_search"):
        st.info("Выберите параметры и нажмите «Найти подрядчиков» или запустите один из демо-сценариев выше.")
        return
    result = filter_contractors(rows, city=city, category=category, event_date=event_date, event_format=event_format, budget=budget, language=language, duration=duration)
    if result["outcome"] == "B":
        st.warning(f"В городе {city} нет специалистов в категории {category}")
        return
    if result["outcome"] == "V":
        st.error("Подходящих подрядчиков не найдено")
        st.write(rejection_report(result))
        return
    st.success(f"Подходящих кандидатов: {result['matched_total']} · показан детерминированный топ-{len(result['matched'])}")
    if result["matched_total"] < 3:
        st.warning(f"Выдача неполная: найдено {result['matched_total']} из 3 возможных. " + (rejection_report(result) if result["pool_size"] > result["matched_total"] else f"В городе найдено только {result['pool_size']} специалист(а) этой категории."))
    for i, row in enumerate(result["matched"], 1):
        with st.container(border=True):
            st.subheader(f"{i}. {row['anon_name']}")
            st.write(f"**Цена от:** {format_kzt(row['price_from_kzt'] or 0)} · **Экономия:** {format_kzt(max(0, budget-int(row['price_from_kzt'] or 0)))}")
            st.caption(f"ID: {row['id']} · Категории: {', '.join(row['categories_list'])} · Форматы: {', '.join(row['event_formats_list'])} · Языки: {', '.join(row['languages_list']) or 'не указаны'}")
            explanation, source = explain(row, api_key=api_key, city=city, category=category, event_date=event_date, event_format=event_format, budget=budget, language=language, duration=duration)
            st.info(f"**Почему подходит · {source}**\n\n{explanation}")
            with st.expander("Описание профиля"):
                st.write(row["description"] or "Описание отсутствует")


if __name__ == "__main__":
    main()
