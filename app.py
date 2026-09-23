"""Veritas — подбор event-подрядчиков. CSV parsing uses only Python stdlib."""
from __future__ import annotations

import ast
import csv
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Optional
from urllib.request import Request, urlopen

import streamlit as st

ROOT = Path(__file__).resolve().parent
DATE_MIN, DATE_MAX = date(2026, 9, 23), date(2026, 12, 31)
CITIES = ["Алматы", "Астана", "Зарубежье"]
LANGUAGES = ["Казахский", "Русский", "Английский"]
FORMATS = ["свадьба", "той", "корпоратив", "конференция", "юбилей", "день рождения"]
PREFERRED = ["Ведущий", "Фотограф", "Банкетный зал", "Флорист", "Декоратор", "Видеограф", "Лайв-бэнд", "Ведущий церемонии", "Подарки и сувениры", "Фото и видеобудки", "Инструменталист"]
DEMOS = {
    "demo1": ("Плотная категория (Алматы, Ведущий, 15.10.2026, Корпоратив, 1 000 000 ₸)", "Алматы", "Ведущий", date(2026, 10, 15), "корпоратив", 1_000_000),
    "demo2": ("Редкая категория (Астана, Флорист, 05.11.2026, Свадьба, 350 000 ₸)", "Астана", "Флорист", date(2026, 11, 5), "свадьба", 350_000),
    "demo3": ("Запрос без результата (Алматы, Ведущий, 25.12.2026, Той, 300 000 ₸)", "Алматы", "Ведущий", date(2026, 12, 25), "той", 300_000),
}
MODES = ["🟢 Демо-ключ Veritas (OpenAI gpt-4o)", "🔑 Ввести свой OpenAI API Key", "⚡ Автономный режим (Без API / Offline Fallback)"]


def parse_list(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text or text.casefold() in {"none", "null", "nan"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            result = ast.literal_eval(text)
            if isinstance(result, (list, tuple)):
                return [str(x).strip() for x in result if str(x).strip()]
        except (ValueError, SyntaxError):
            pass
    return [x.strip().strip("'\"[]") for x in re.split(r"[,;|]", text) if x.strip().strip("'\"[]")]


def number(value: Any) -> Optional[float]:
    try:
        return float(value) if str(value or "").strip() else None
    except (TypeError, ValueError):
        return None


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().casefold())


def has(values: list[str], value: str) -> bool:
    return any(norm(v) == norm(value) for v in values)


def money(value: int | float) -> str:
    return f"{int(value):,}".replace(",", " ") + " ₸"


def team_api_key() -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("OPENAI_API_KEY", "")).strip() or None
    except Exception:
        # Missing secrets.toml is expected for local/offline installations.
        return None


def dataset_path() -> Path:
    for p in (ROOT / "hackathon-dataset-anonymized.csv", ROOT / "hackathon-dataset-anonymized .csv"):
        if p.is_file():
            return p
    raise FileNotFoundError("Рядом с app.py не найден файл датасета CSV")


@st.cache_data(show_spinner=False)
def load_data() -> list[dict[str, Any]]:
    needed = {"id", "anon_name", "categories", "city", "price_from_kzt", "event_formats", "languages", "max_hours", "busy_dates", "description"}
    with dataset_path().open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError("В CSV отсутствуют колонки: " + ", ".join(sorted(missing)))
        rows = list(reader)
    for r in rows:
        for col in ("categories", "event_formats", "languages", "busy_dates"):
            r[col + "_list"] = parse_list(r.get(col))
        r["price_from_kzt"], r["max_hours"] = number(r.get("price_from_kzt")), number(r.get("max_hours"))
        for col in ("id", "anon_name", "city", "description"):
            r[col] = (r.get(col) or "").strip()
    return rows


def score(r: dict[str, Any], fmt: str, budget: int, language: Optional[str]) -> float:
    # Fixed weighted score: language fit/breadth 45%, format evidence 35%, price proximity 20%.
    langs = r["languages_list"]
    lang_fit = min(1.0, len(langs) / 3) if langs else 0.0
    description = norm(r["description"])
    keywords = {
        "свадьба": ("свадьб", "церемон", "молодож", "торжеств"), "той": ("той", "традиц", "казах", "бата", "домбра"),
        "корпоратив": ("корпоратив", "команд", "компан", "бренд", "партнер", "партнёр", "квн", "телевиз", "тв"),
        "конференция": ("конференц", "форум", "делов", "спикер", "панел"), "юбилей": ("юбиле", "торжеств", "семейн"),
        "день рождения": ("день рожден", "именин", "праздник", "шоу"),
    }.get(norm(fmt), ())
    relevance = min(1.0, sum(word in description for word in keywords) / max(1, min(3, len(keywords))))
    price = r["price_from_kzt"]
    proximity = max(0.0, 1 - abs(budget - price) / max(1, budget)) if price is not None else 0.0
    return 0.45 * lang_fit + 0.35 * relevance + 0.20 * proximity


def search(rows: list[dict[str, Any]], city: str, category: str, event_date: date, fmt: str, budget: int, language: Optional[str], duration: Optional[float]) -> dict[str, Any]:
    pool = [r for r in rows if norm(r["city"]) == norm(city) and has(r["categories_list"], category)]
    counts = {k: 0 for k in ("busy", "budget", "format", "language", "duration")}
    passed = []
    for r in pool:
        reason = None
        if event_date.isoformat() in r["busy_dates_list"]: reason = "busy"
        elif r["price_from_kzt"] is None or r["price_from_kzt"] > budget: reason = "budget"
        elif not has(r["event_formats_list"], fmt): reason = "format"
        elif language and not has(r["languages_list"], language): reason = "language"
        elif duration is not None and r["max_hours"] is not None and r["max_hours"] < duration: reason = "duration"
        if reason: counts[reason] += 1
        else: passed.append(r)
    passed.sort(key=lambda r: (-score(r, fmt, budget, language), str(r["id"]), norm(r["anon_name"])))
    return {"pool": pool, "counts": counts, "passed": passed, "outcome": "B" if not pool else "A" if passed else "V"}


def offline_explanation(r: dict[str, Any], budget: int, fmt: str, requested_language: Optional[str]) -> str:
    """Pure local deterministic explanation; performs no external calls."""
    price = int(r["price_from_kzt"] or 0)
    savings = budget - price
    desc = re.sub(r"\s+", " ", r["description"]).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", desc) if s.strip()]
    evidence_terms = {
        "той": ("той", "дәстүр", "традиц", "бата", "домбр", "казах"),
        "свадьба": ("свадьб", "церемон", "молодож", "венчан"),
        "корпоратив": ("корпоратив", "команд", "компан", "квн", "телевиз", "эфир", "бренд"),
        "конференция": ("конференц", "форум", "спикер", "панел", "делов"),
        "юбилей": ("юбиле", "семейн", "торжеств"),
        "день рождения": ("день рожден", "именин", "праздник", "шоу"),
    }.get(norm(fmt), ())
    fact = next((s for s in sentences if any(term in norm(s) for term in evidence_terms)), "")
    if not fact:
        fact = sentences[0] if sentences else "В описании профиля нет подробностей."
    if len(fact) > 300:
        fact = fact[:297].rsplit(" ", 1)[0] + "…"
    langs = r["languages_list"]
    formats = r["event_formats_list"]
    matched_langs = [x for x in langs if not requested_language or norm(x) == norm(requested_language)]
    matched_formats = [x for x in formats if norm(x) == norm(fmt)]
    lang_text = ", ".join(matched_langs) if matched_langs else ", ".join(langs) or "не указаны"
    fmt_text = ", ".join(matched_formats) if matched_formats else ", ".join(formats) or "не указаны"
    return (f"Цена от {money(price)}; экономия относительно бюджета — {money(savings)}. "
            f"Совпавший формат: {fmt_text}; языки профиля: {lang_text}. "
            f"Связь с вашим событием «{fmt}»: {fact}")


def openai_explanation(r: dict[str, Any], key: str, city: str, category: str, event_date: date, fmt: str, budget: int, language: Optional[str], duration: Optional[float]) -> str:
    price = int(r["price_from_kzt"] or 0)
    prompt = f"""Запрос: {city}, категория {category}, дата {event_date.isoformat()}, формат {fmt}, бюджет {budget} KZT, язык {language or 'любой'}, длительность {duration or 'не задана'} ч.
Кандидат: {r['anon_name']}; цена {price} KZT; экономия {budget-price} KZT; языки {', '.join(r['languages_list'])}; форматы {', '.join(r['event_formats_list'])}; description: {r['description'][:1800]}
Сформулируй 2–3 персональных предложения на русском. Найди уникальную конкретную зацепку только из description (награда, ТВ, КВН, традиции тоя, проекты или специализация) и объясни её ценность именно для заданного формата. Для разных кандидатов аргументы и стиль должны заметно отличаться. Укажи точную цену, экономию и требуемый язык, если задан. Не выдумывай факты; при отсутствии деталей опирайся на то, что реально указано. Не используй клише «отличный выбор»."""
    body = json.dumps({
        "model": "gpt-4o", "temperature": 0,
        "messages": [{"role": "system", "content": "Ты консультант по event-подрядчикам. Используй только подтверждённые факты профиля."}, {"role": "user", "content": prompt}],
    }).encode("utf-8")
    request = Request("https://api.openai.com/v1/chat/completions", data=body, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=18) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload["choices"][0]["message"].get("content") or "").strip()


def explain(r: dict[str, Any], mode: str, key: Optional[str], **ctx: Any) -> tuple[str, str]:
    cache = st.session_state.setdefault("explanation_cache", {})
    cache_id = "|".join((r["id"], mode, str(ctx["event_date"]), ctx["event_format"], str(ctx["budget"]), str(ctx["language"]), str(bool(key))))
    if cache_id in cache:
        return cache[cache_id]
    if mode == MODES[2] or not key:
        result = (offline_explanation(r, ctx["budget"], ctx["event_format"], ctx["language"]), "Детерминированный Fallback (Offline)")
    else:
        try:
            content = openai_explanation(r, key, ctx["city"], ctx["category"], ctx["event_date"], ctx["event_format"], ctx["budget"], ctx["language"], ctx["duration"])
            result = (content, "OpenAI gpt-4o") if content else (offline_explanation(r, ctx["budget"], ctx["event_format"], ctx["language"]), "Детерминированный Fallback (Offline)")
        except Exception:
            # Covers network errors, quota/rate limits, SDK errors; fallback is local and deterministic.
            result = (offline_explanation(r, ctx["budget"], ctx["event_format"], ctx["language"]), "Детерминированный Fallback (Offline)")
    cache[cache_id] = result
    return result


def main() -> None:
    st.set_page_config(page_title="Veritas — подбор event-подрядчиков", page_icon="✨", layout="wide")
    st.title("VERITAS · Умный подбор event-подрядчиков")
    st.caption("HackAlem AI · Детерминированный подбор и персональные объяснения")
    cols = st.columns(3)
    for col, demo_id in zip(cols, DEMOS):
        label, city, cat, dt, fmt, budget = DEMOS[demo_id]
        if col.button(label, use_container_width=True):
            st.session_state.update(city=city, category=cat, event_date=dt, event_format=fmt, budget=budget, language_choice="Не важно", duration_enabled=False, run_search=True)
            st.rerun()
    try:
        rows = load_data()
    except Exception as e:
        st.error(f"Не удалось загрузить датасет: {e}")
        st.stop()
    all_categories = sorted({c for r in rows for c in r["categories_list"]})
    categories = [c for c in PREFERRED if c in all_categories] + [c for c in all_categories if c not in PREFERRED]
    with st.sidebar:
        st.header("Параметры поиска")
        st.markdown("**Режим генерации объяснений:**")
        mode = st.radio("Режим генерации объяснений:", MODES, index=0, label_visibility="collapsed", key="generation_mode")
        key: Optional[str] = None
        if mode == MODES[0]:
            key = team_api_key()
            if key:
                st.success("Командный ключ подключён · генерация объяснений GPT-4o")
            else:
                st.warning("Ключ команды не настроен. Используется автономный генератор. Добавьте OPENAI_API_KEY в окружение или Streamlit Secrets для режима GPT-4o.")
        elif mode == MODES[1]:
            key = st.text_input("Ваш OpenAI API Key", type="password", help="Ключ используется только для запросов из этой сессии.").strip() or None
            if not key:
                st.info("Введите ключ или переключитесь в автономный режим.")
        else:
            st.info("Работает локально: сетевых запросов и API-ключей нет.")
        st.divider()
        city = st.selectbox("Город", CITIES, key="city")
        category = st.selectbox("Категория", categories, key="category")
        event_date = st.date_input("Дата мероприятия", min_value=DATE_MIN, max_value=DATE_MAX, key="event_date")
        fmt = st.selectbox("Формат мероприятия", FORMATS, key="event_format")
        budget = int(st.number_input("Бюджет, ₸", min_value=10_000, max_value=20_000_000, step=10_000, key="budget"))
        lang_choice = st.selectbox("Желаемый язык (необязательно)", ["Не важно"] + LANGUAGES, key="language_choice")
        language = None if lang_choice == "Не важно" else lang_choice
        duration_enabled = st.checkbox("Указать длительность", key="duration_enabled")
        duration = float(st.number_input("Длительность, часов", min_value=1.0, max_value=24.0, value=4.0, step=1.0)) if duration_enabled else None
        if st.button("Найти подрядчиков", type="primary", use_container_width=True):
            st.session_state["run_search"] = True
    st.info(f"**Запрос:** {city} · {category} · {event_date:%d.%m.%Y} · {fmt.capitalize()} · {money(budget)} · {language or 'любой язык'} · {str(duration)+' ч' if duration else 'любая длительность'}")
    if not st.session_state.get("run_search"):
        st.info("Задайте параметры или выберите демо-сценарий, затем нажмите «Найти подрядчиков».")
        return
    result = search(rows, city, category, event_date, fmt, budget, language, duration)
    if result["outcome"] == "B":
        st.warning(f"В городе {city} нет специалистов в категории {category}")
        return
    counts = result["counts"]
    reason_labels = {"busy": "заняты на эту дату", "budget": "не укладываются в бюджет", "format": f"не проводят мероприятия в формате {fmt}", "language": f"не знают язык {language}", "duration": "не подходят по длительности"}
    details = "; ".join(f"{n} {reason_labels[k]}" for k, n in counts.items() if n) or "условиям поиска"
    if result["outcome"] == "V":
        st.error("Подходящих подрядчиков не найдено")
        st.write(f"В городе найдено {len(result['pool'])} специалистов категории «{category}», но: {details}.")
        return
    passed = result["passed"]
    st.success(f"Подходящих кандидатов: {len(passed)} · показан детерминированный топ-{min(3, len(passed))}")
    if len(passed) < 3:
        st.warning(f"Выдача неполная: найдено {len(passed)} из 3. " + (f"В городе найдено {len(result['pool'])} специалистов; {details}." if len(result['pool']) > len(passed) else f"В городе только {len(result['pool'])} специалист(а) этой категории."))
    ctx = dict(city=city, category=category, event_date=event_date, event_format=fmt, budget=budget, language=language, duration=duration)
    for i, r in enumerate(passed[:3], 1):
        with st.container(border=True):
            st.subheader(f"{i}. {r['anon_name']}")
            st.write(f"**Цена от:** {money(r['price_from_kzt'] or 0)} · **Экономия:** {money(max(0, budget-int(r['price_from_kzt'] or 0)))}")
            st.caption(f"ID: {r['id']} · Категории: {', '.join(r['categories_list'])} · Форматы: {', '.join(r['event_formats_list'])} · Языки: {', '.join(r['languages_list']) or 'не указаны'}")
            text, badge = explain(r, mode, key, **ctx)
            st.info(f"**Объяснение: {badge}**\n\n{text}")
            with st.expander("Описание профиля"):
                st.write(r["description"] or "Описание отсутствует")


if __name__ == "__main__":
    main()
