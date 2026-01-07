# -*- coding: utf-8 -*-
"""
main.py

Прототип веб-ориентированной информационной системы для сбора, анализа и визуализации
открытых данных ТОЛЬКО из World Bank Indicators API (v2).

Что умеет:
1) Сбор данных: автоматическая загрузка временных рядов по индикаторам (World Bank API).
2) Обработка и анализ: очистка, сортировка, расчёт YoY, YoY%, CAGR, базовые выводы.
3) Представление: простой веб-дашборд (HTML + Plotly JS) с интерактивным графиком и таблицей.
4) Экспорт отчётов: CSV и PDF (ReportLab) по выбранным параметрам.

Запуск:
    pip install fastapi uvicorn httpx pandas matplotlib reportlab
    python "main.py"

Открыть в браузере:
    http://127.0.0.1:8000

Примечание:
- World Bank API: https://api.worldbank.org/v2/  (формат JSON через ?format=json)
- Для простоты список индикаторов/стран в интерфейсе — "пресеты".
  В API-эндпойнтах можно передавать любые корректные коды country (ISO3) и indicator.

Автор: Roman P. - RomanDevelops
"""

from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# --- PDF export (ReportLab) ---
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

PDF_FONT_NAME = "Helvetica"


def _try_register_pdf_font() -> str:
    """Регистрация шрифта с поддержкой кириллицы для PDF. Возвращает имя доступного шрифта."""
    candidates = [
        ("DejaVuSans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ("DejaVuSansCondensed", "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"),
    ]
    for name, path in candidates:
        try:
            pdfmetrics.registerFont(TTFont(name, path))
            return name
        except Exception:
            continue
    return "Helvetica"  # fallback


PDF_FONT_NAME = _try_register_pdf_font()


def _try_register_pdf_font_bold() -> str:
    candidates = [
        ("DejaVuSans-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("DejaVuSansCondensed-Bold", "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"),
    ]
    for name, path in candidates:
        try:
            pdfmetrics.registerFont(TTFont(name, path))
            return name
        except Exception:
            continue
    return "Helvetica-Bold"


PDF_FONT_NAME_BOLD = _try_register_pdf_font_bold()

# --- Chart image for PDF (matplotlib) ---
import matplotlib.pyplot as plt


WB_BASE_URL = "https://api.worldbank.org/v2"
#WB_BASE_URL = "https://search.worldbank.org/v3"
DEFAULT_TIMEOUT_S = 20.0


# ---------------------------
# Presets (можно расширять)
# ---------------------------
PRESET_COUNTRIES: List[Dict[str, str]] = [
    {"id": "POL", "name": "Poland"},
    {"id": "RUS", "name": "Russian Federation"},
    {"id": "DEU", "name": "Germany"},
    {"id": "USA", "name": "United States"},
    {"id": "CHN", "name": "China"},
]

PRESET_INDICATORS: List[Dict[str, str]] = [
    {"id": "NY.GDP.MKTP.CD", "name": "GDP (current US$)"},
    {"id": "NY.GDP.PCAP.CD", "name": "GDP per capita (current US$)"},
    {"id": "SP.POP.TOTL", "name": "Population, total"},
    {"id": "FP.CPI.TOTL.ZG", "name": "Inflation, consumer prices (annual %)"},
    {"id": "SL.UEM.TOTL.ZS", "name": "Unemployment, total (% of labor force)"},
    {"id": "IT.NET.USER.ZS", "name": "Individuals using the Internet (% of population)"},
]


# ---------------------------
# Errors
# ---------------------------
class WorldBankAPIError(RuntimeError):
    pass


# ---------------------------
# Simple in-memory TTL cache
# ---------------------------
@dataclass
class CacheItem:
    expires_at: float
    value: Any


class TTLCache:
    def __init__(self, ttl_seconds: int = 600):
        self.ttl_seconds = ttl_seconds
        self._store: Dict[str, CacheItem] = {}

    def get(self, key: str) -> Any:
        item = self._store.get(key)
        if not item:
            return None
        if time.time() >= item.expires_at:
            self._store.pop(key, None)
            return None
        return item.value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = CacheItem(expires_at=time.time() + self.ttl_seconds, value=value)


cache = TTLCache(ttl_seconds=10 * 60)


# ---------------------------
# World Bank client
# ---------------------------
class WorldBankClient:
    def __init__(self, base_url: str = WB_BASE_URL, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    async def _get_json(self, path: str, params: Dict[str, Any]) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        # World Bank API: обязательно format=json
        params = dict(params)
        params["format"] = "json"
        params.setdefault("per_page", 200)

        cache_key = f"GET:{url}:{sorted(params.items())}"
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.get(url, params=params)
                r.raise_for_status()
                payload = r.json()
        except httpx.RequestError as e:
            raise WorldBankAPIError(f"Нет связи с источником данных (World Bank API): {e}") from e
        except httpx.HTTPStatusError as e:
            raise WorldBankAPIError(f"Ошибка HTTP при запросе к World Bank API: {e}") from e
        except ValueError as e:
            raise WorldBankAPIError(f"Неверный формат ответа (ожидался JSON): {e}") from e

        # В случае ошибок WB может вернуть dict с message[]
        if isinstance(payload, dict) and "message" in payload:
            raise WorldBankAPIError(f"World Bank API error: {payload.get('message')}")

        cache.set(cache_key, payload)
        return payload

    async def fetch_timeseries(
        self,
        country_iso3: str,
        indicator_id: str,
        start_year: int,
        end_year: int,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Возвращает (meta, rows) по /country/{country}/indicator/{indicator}.
        """
        if start_year > end_year:
            raise WorldBankAPIError("start_year не может быть больше end_year")

        # параметр date поддерживает диапазон YYYY:YYYY
        date_range = f"{start_year}:{end_year}"
        payload = await self._get_json(
            path=f"country/{country_iso3}/indicator/{indicator_id}",
            params={"date": date_range, "per_page": 500},
        )

        if not isinstance(payload, list) or len(payload) < 2:
            raise WorldBankAPIError("Неожиданная структура ответа World Bank API")

        meta = payload[0] or {}
        rows = payload[1] or []
        if not isinstance(rows, list):
            raise WorldBankAPIError("Неожиданная структура данных (rows)")

        return meta, rows


wb = WorldBankClient()


# ---------------------------
# Processing / analysis
# ---------------------------
def rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    data = []
    for r in rows:
        year = r.get("date")
        value = r.get("value")
        if year is None:
            continue
        try:
            year_i = int(year)
        except Exception:
            continue

        data.append(
            {
                "country_iso3": r.get("countryiso3code"),
                "country": (r.get("country") or {}).get("value"),
                "indicator_id": (r.get("indicator") or {}).get("id"),
                "indicator": (r.get("indicator") or {}).get("value"),
                "year": year_i,
                "value": value,
            }
        )

    df = pd.DataFrame(data)
    if df.empty:
        return df

    df = df.dropna(subset=["year"]).copy()
    # Значения могут быть None
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df = df.sort_values("year").reset_index(drop=True)

    # Расчёты
    df["yoy_change"] = df["value"].diff()
    df["yoy_change_pct"] = df["value"].pct_change() * 100.0

    return df


def compute_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or df.shape[0] < 2:
        return {"ok": False, "message": "Недостаточно данных для анализа."}

    first = df.iloc[0]
    last = df.iloc[-1]
    years = int(last["year"]) - int(first["year"])
    if years <= 0:
        cagr = None
    else:
        try:
            cagr = (float(last["value"]) / float(first["value"])) ** (1.0 / years) - 1.0
        except Exception:
            cagr = None

    delta = float(last["value"]) - float(first["value"])
    delta_pct = (delta / float(first["value"]) * 100.0) if float(first["value"]) != 0 else None

    # тренд по последним 3 точкам (просто знак наклона)
    tail = df.tail(3)
    trend = "stable"
    if tail.shape[0] >= 2:
        x0 = float(tail.iloc[0]["year"])
        y0 = float(tail.iloc[0]["value"])
        x1 = float(tail.iloc[-1]["year"])
        y1 = float(tail.iloc[-1]["value"])
        slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
        if slope > 0:
            trend = "up"
        elif slope < 0:
            trend = "down"

    return {
        "ok": True,
        "period": f"{int(first['year'])}-{int(last['year'])}",
        "first_year": int(first["year"]),
        "first_value": float(first["value"]),
        "last_year": int(last["year"]),
        "last_value": float(last["value"]),
        "delta": delta,
        "delta_pct": delta_pct,
        "cagr": cagr,
        "min_value": float(df["value"].min()),
        "max_value": float(df["value"].max()),
        "trend": trend,
    }


def format_number(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    if abs(x) >= 1000 and abs(x) < 1e15:
        return f"{x:,.0f}".replace(",", " ")
    return f"{x:.4g}"


def build_conclusions(summary: Dict[str, Any]) -> List[str]:
    if not summary.get("ok"):
        return [summary.get("message", "Нет данных.")]

    trend_map = {"up": "восходящий", "down": "нисходящий", "stable": "стабильный"}
    trend_ru = trend_map.get(summary.get("trend"), "стабильный")

    bullets = [
        f"Период анализа: {summary['period']}.",
        f"Последнее значение ({summary['last_year']}): {format_number(summary['last_value'])}.",
        f"Изменение за период: {format_number(summary['delta'])} ({format_number(summary['delta_pct'])}%).",
        f"Минимум/максимум: {format_number(summary['min_value'])} / {format_number(summary['max_value'])}.",
        f"Оценка тренда по последним наблюдениям: {trend_ru}.",
    ]
    if summary.get("cagr") is not None:
        bullets.append(f"Среднегодовой темп (CAGR): {format_number(summary['cagr'] * 100.0)}% в год.")
    return bullets


# ---------------------------
# CSV & PDF export
# ---------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def build_chart_png(df: pd.DataFrame, title: str) -> bytes:
    """
    Генерирует PNG-график (matplotlib) в памяти для вставки в PDF.
    """
    fig = plt.figure(figsize=(8, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(df["year"], df["value"], marker="o")
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)

    out = io.BytesIO()
    fig.tight_layout()
    fig.savefig(out, format="png", dpi=160)
    plt.close(fig)
    return out.getvalue()


def df_to_pdf_bytes(df: pd.DataFrame, title: str, conclusions: List[str]) -> bytes:
    """
    Простой PDF-отчёт: заголовок, график, таблица и ключевые выводы.
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    left = 2.0 * cm
    top = height - 2.0 * cm

    # Header
    c.setFont(PDF_FONT_NAME_BOLD, 14)
    c.drawString(left, top, "Analytic report (World Bank Open Data)")
    c.setFont(PDF_FONT_NAME, 11)
    c.drawString(left, top - 0.7 * cm, title)

    # Chart
    chart_png = build_chart_png(df, title=title)
    img = ImageReader(io.BytesIO(chart_png))
    c.drawImage(img, left, top - 7.2 * cm, width=width - 4.0 * cm, height=4.6 * cm, preserveAspectRatio=True, anchor="nw")

    y = top - 8.2 * cm

    # Conclusions
    c.setFont(PDF_FONT_NAME_BOLD, 12)
    c.drawString(left, y, "Key findings")
    y -= 0.6 * cm
    c.setFont(PDF_FONT_NAME, 10)
    for b in conclusions:
        c.drawString(left, y, f"- {b}")
        y -= 0.5 * cm
        if y < 3.5 * cm:
            c.showPage()
            y = top

    # Table (last 12 rows)
    df_tail = df.tail(12).copy()
    c.setFont(PDF_FONT_NAME_BOLD, 12)
    c.drawString(left, y, "Data fragment (last 12 observations)")
    y -= 0.7 * cm

    c.setFont(PDF_FONT_NAME_BOLD, 10)
    c.drawString(left, y, "Year")
    c.drawString(left + 3.0 * cm, y, "Value")
    c.setFont(PDF_FONT_NAME, 10)
    y -= 0.5 * cm

    for _, r in df_tail.iterrows():
        c.drawString(left, y, str(int(r["year"])))
        c.drawString(left + 3.0 * cm, y, format_number(float(r["value"])))
        y -= 0.45 * cm
        if y < 2.5 * cm:
            c.showPage()
            y = top

    c.showPage()
    c.save()
    return buf.getvalue()


# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(
    title="Open Data Dashboard (World Bank API)",
    description="Прототип: сбор, анализ и визуализация открытых данных (World Bank).",
    version="0.1.0",
)


@app.get("/health", response_class=JSONResponse)
async def health() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/api/presets", response_class=JSONResponse)
async def presets() -> Dict[str, Any]:
    return {"countries": PRESET_COUNTRIES, "indicators": PRESET_INDICATORS}


@app.get("/api/data", response_class=JSONResponse)
async def api_data(
    country: str = Query(..., min_length=3, max_length=3, description="ISO3 country code (например POL)"),
    indicator: str = Query(..., min_length=3, max_length=40, description="Indicator id (например SP.POP.TOTL)"),
    start_year: int = Query(2010, ge=1960, le=2100),
    end_year: int = Query(2024, ge=1960, le=2100),
) -> Dict[str, Any]:
    try:
        meta, rows = await wb.fetch_timeseries(country, indicator, start_year, end_year)
        df = rows_to_df(rows)
        if df.empty:
            raise WorldBankAPIError("Данные не найдены (возможно, неверный код страны/индикатора или нет данных в диапазоне).")
        summary = compute_summary(df)
        conclusions = build_conclusions(summary)
        return {
            "meta": meta,
            "summary": summary,
            "conclusions": conclusions,
            "data": df.to_dict(orient="records"),
        }
    except WorldBankAPIError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка обработки: {e}") from e


@app.get("/export/csv")
async def export_csv(
    country: str = Query(...),
    indicator: str = Query(...),
    start_year: int = Query(2010),
    end_year: int = Query(2024),
):
    try:
        _, rows = await wb.fetch_timeseries(country, indicator, start_year, end_year)
        df = rows_to_df(rows)
        if df.empty:
            raise WorldBankAPIError("Нет данных для экспорта.")
        content = df_to_csv_bytes(df)
        filename = f"dashboard_{country}_{indicator}_{start_year}_{end_year}.csv"
        return StreamingResponse(
            io.BytesIO(content),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except WorldBankAPIError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@app.get("/export/pdf")
async def export_pdf(
    country: str = Query(...),
    indicator: str = Query(...),
    start_year: int = Query(2010),
    end_year: int = Query(2024),
):
    try:
        _, rows = await wb.fetch_timeseries(country, indicator, start_year, end_year)
        df = rows_to_df(rows)
        if df.empty:
            raise WorldBankAPIError("Нет данных для экспорта.")
        summary = compute_summary(df)
        conclusions = build_conclusions(summary)

        # Заголовок для отчёта (берём из данных, если доступно)
        country_name = str(df.iloc[-1].get("country") or country)
        indicator_name = str(df.iloc[-1].get("indicator") or indicator)
        title = f"{country_name} — {indicator_name} ({start_year}–{end_year})"

        pdf_bytes = df_to_pdf_bytes(df, title=title, conclusions=conclusions)
        filename = f"report_{country}_{indicator}_{start_year}_{end_year}.pdf"
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except WorldBankAPIError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


# ---------------------------
# Web dashboard (single HTML page)
# ---------------------------
DASHBOARD_HTML = r"""
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Open Data Dashboard — World Bank</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 0; background: #f6f7fb; }
    header { padding: 18px 22px; background: white; border-bottom: 1px solid #e6e7ee; }
    h1 { font-size: 18px; margin: 0; }
    .wrap { max-width: 1100px; margin: 18px auto; padding: 0 14px; }
    .card { background: white; border: 1px solid #e6e7ee; border-radius: 14px; padding: 14px; box-shadow: 0 8px 22px rgba(0,0,0,0.04); }
    .grid { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 10px; }
    label { display: block; font-size: 12px; color: #444; margin-bottom: 6px; }
    select, input { width: 100%; padding: 10px; border-radius: 10px; border: 1px solid #cfd3e6; background: #fff; }
    button { padding: 10px 12px; border-radius: 10px; border: 0; background: #2b59ff; color: white; font-weight: 600; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    .muted { color: #666; font-size: 12px; }
    .error { color: #b00020; }
    #chart { height: 380px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid #eee; text-align: left; padding: 8px; }
    th { font-weight: 700; }
    ul { margin: 8px 0 0 18px; }
    .actions a { text-decoration: none; color: #2b59ff; font-weight: 600; }
  </style>
</head>
<body>
  <header>
    <h1>Open Data Dashboard — World Bank API</h1>
    <div class="muted">Сбор → анализ → визуализация → экспорт (CSV/PDF)</div>
  </header>

  <div class="wrap">
    <div class="card">
      <div class="grid">
        <div>
          <label>Страна (ISO3)</label>
          <select id="country"></select>
        </div>
        <div>
          <label>Индикатор</label>
          <select id="indicator"></select>
        </div>
        <div>
          <label>Начальный год</label>
          <input id="start_year" type="number" value="2010" min="1960" max="2100" />
        </div>
        <div>
          <label>Конечный год</label>
          <input id="end_year" type="number" value="2024" min="1960" max="2100" />
        </div>
      </div>

      <div class="row" style="margin-top: 12px;">
        <button id="loadBtn">Загрузить</button>
        <div class="actions muted">
          <span id="links"></span>
        </div>
        <div id="status" class="muted"></div>
      </div>
      <div id="err" class="error" style="margin-top: 8px;"></div>
    </div>

    <div class="card" style="margin-top: 14px;">
      <div id="chart"></div>
    </div>

    <div class="card" style="margin-top: 14px;">
      <div class="row" style="justify-content: space-between;">
        <div>
          <strong>Ключевые выводы</strong>
          <div class="muted" id="metaLine"></div>
        </div>
      </div>
      <div id="bullets"></div>
    </div>

    <div class="card" style="margin-top: 14px;">
      <strong>Данные</strong>
      <div class="muted">Показаны последние 15 строк</div>
      <div style="overflow:auto; margin-top: 8px;">
        <table id="tbl">
          <thead><tr><th>Year</th><th>Value</th><th>YoY</th><th>YoY%</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>

    <div class="muted" style="margin: 14px 2px;">
      Источник данных: World Bank Indicators API (v2).
    </div>
  </div>

<script>
  const el = (id) => document.getElementById(id);

  function fmtNum(x) {
    if (x === null || x === undefined || Number.isNaN(x)) return "—";
    // group by thousands
    const abs = Math.abs(x);
    if (abs >= 1000 && abs < 1e15) return Math.round(x).toLocaleString("ru-RU");
    return (Math.round(x * 10000) / 10000).toString();
  }

  async function loadPresets() {
    const r = await fetch("/api/presets");
    const p = await r.json();
    for (const c of p.countries) {
      const opt = document.createElement("option");
      opt.value = c.id;
      opt.textContent = `${c.name} (${c.id})`;
      el("country").appendChild(opt);
    }
    for (const i of p.indicators) {
      const opt = document.createElement("option");
      opt.value = i.id;
      opt.textContent = `${i.name} (${i.id})`;
      el("indicator").appendChild(opt);
    }
    // defaults
    el("country").value = "POL";
    el("indicator").value = "SP.POP.TOTL";
  }

  function buildLinks(country, indicator, start_year, end_year) {
    const q = new URLSearchParams({country, indicator, start_year, end_year}).toString();
    el("links").innerHTML = `
      <a href="/export/csv?${q}">Скачать CSV</a>
      &nbsp;·&nbsp;
      <a href="/export/pdf?${q}">Скачать PDF</a>
    `;
  }

  async function loadData() {
    el("err").textContent = "";
    el("status").textContent = "Загрузка...";
    el("loadBtn").disabled = true;

    const country = el("country").value;
    const indicator = el("indicator").value;
    const start_year = parseInt(el("start_year").value, 10);
    const end_year = parseInt(el("end_year").value, 10);
    buildLinks(country, indicator, start_year, end_year);

    const q = new URLSearchParams({country, indicator, start_year, end_year}).toString();
    try {
      const r = await fetch(`/api/data?${q}`);
      const payload = await r.json();
      if (!r.ok) throw new Error(payload.detail || "Ошибка запроса");

      const data = payload.data || [];
      const years = data.map(d => d.year);
      const vals = data.map(d => d.value);

      Plotly.newPlot("chart", [
        {x: years, y: vals, mode: "lines+markers", name: indicator}
      ], {
        margin: {t: 30, l: 50, r: 20, b: 40},
        title: `${country} / ${indicator}`
      }, {responsive: true});

      // bullets
      const bullets = payload.conclusions || [];
      el("bullets").innerHTML = "<ul>" + bullets.map(b => `<li>${b}</li>`).join("") + "</ul>";

      // meta line
      const meta = payload.meta || {};
      el("metaLine").textContent = meta.lastupdated ? `lastupdated: ${meta.lastupdated}` : "";

      // table: last 15
      const tbody = el("tbl").querySelector("tbody");
      tbody.innerHTML = "";
      const tail = data.slice(-15);
      for (const r of tail) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${r.year}</td>
          <td>${fmtNum(r.value)}</td>
          <td>${fmtNum(r.yoy_change)}</td>
          <td>${fmtNum(r.yoy_change_pct)}</td>
        `;
        tbody.appendChild(tr);
      }

      el("status").textContent = `ОК: получено ${data.length} наблюдений.`;
    } catch (e) {
      el("err").textContent = e.message || String(e);
      el("status").textContent = "Ошибка.";
    } finally {
      el("loadBtn").disabled = false;
    }
  }

  el("loadBtn").addEventListener("click", loadData);

  loadPresets().then(loadData);
</script>
<p>&copy Roman P. - RomanDevelops </p>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def dashboard(_: Request) -> HTMLResponse:
    return HTMLResponse(content=DASHBOARD_HTML)

# uvicorn main:app --reload
# Для запуска напрямую: python "main.py"
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
