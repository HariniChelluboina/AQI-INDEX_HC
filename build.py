"""
build.py - Air Quality and Pollution Data Analysis dashboard generator

What this program does (top to bottom):
  1. Creates sample hourly data for 5 monitoring stations (or reads your CSV).
  2. Calculates the AQI for PM2.5, PM10, NO2 and CO (US EPA method).
  3. Draws charts and a map with Matplotlib (saved as PNG images).
  4. Writes docs/index.html, a dashboard with navigation, location selector,
     timeframe tabs, a clickable map, tables and alert settings.

Libraries needed: pandas, numpy, matplotlib (nothing else).

To use real data, put a file at data/stations.csv with the columns
    station, time, pm25, pm10, no2, co
where `station` is one of the ids in STATIONS below, pm25/pm10 are in ug/m3,
no2 is in ppb and co is in ppm. If the file is missing, sample data is used.
"""

import json
import math
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # draw charts without opening a window
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# =============================================================== 1. SETTINGS
SITE_NAME = "City AirWatch"
TZ_OFFSET = pd.Timedelta(hours=5, minutes=30)  # GitHub's computer uses UTC
TZ_LABEL = "IST"
CONTACT_URL = "https://github.com/YOUR-USERNAME/air-quality-analysis"
DATA_CSV = Path("data/stations.csv")

OUT = Path("docs")
IMG = OUT / "img"
IMG.mkdir(parents=True, exist_ok=True)

# Colour-blind safe palette (Okabe-Ito) used for every chart
BLUE, ORANGE, GREEN, PINK = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
INK, MUTED = "#1f2933", "#52606d"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,        # less "ink" = clearer charts
    "axes.spines.right": False,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "hatch.linewidth": 1.2,
})

# ====================================================== 2. AQI RULES (EPA)
# Official palette. 'fg' is the text colour that stays readable on 'bg'.
# 'icon' is a second signal besides colour, for colour-blind users.
CATEGORIES = [
    dict(name="Good", lo=0, hi=50, bg="#00E400", fg="#000000", icon="✔",
         advice="Air quality is satisfactory. Enjoy your usual outdoor activities."),
    dict(name="Moderate", lo=51, hi=100, bg="#FFFF00", fg="#000000", icon="▲",
         advice="Air quality is acceptable; however, sensitive individuals may experience minor irritation."),
    dict(name="Unhealthy for Sensitive Groups", lo=101, hi=150, bg="#FF7E00", fg="#000000", icon="◆",
         advice="Sensitive groups should reduce prolonged outdoor exertion."),
    dict(name="Unhealthy", lo=151, hi=200, bg="#FF0000", fg="#FFFFFF", icon="■",
         advice="Everyone should limit prolonged outdoor exertion. Consider a mask outdoors."),
    dict(name="Very Unhealthy", lo=201, hi=300, bg="#8F3F97", fg="#FFFFFF", icon="✚",
         advice="Avoid outdoor activity and keep windows closed."),
    dict(name="Hazardous", lo=301, hi=500, bg="#7E0023", fg="#FFFFFF", icon="✖",
         advice="Stay indoors and use an air purifier if you have one."),
]

# (concentration low, concentration high, AQI low, AQI high)
# Please double-check against the latest US EPA technical document.
BREAKPOINTS = {
    "pm25": [(0.0, 9.0, 0, 50), (9.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
             (55.5, 125.4, 151, 200), (125.5, 225.4, 201, 300), (225.5, 325.4, 301, 500)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
             (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)],
    "no2":  [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
             (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 2049, 301, 500)],
    "co":   [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
             (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 50.4, 301, 500)],
}

# label, units, WHO 24-hour guideline, factor to convert our unit to the WHO unit
POLLUTANTS = {
    "pm25": dict(label="PM2.5", html="PM<sub>2.5</sub>", unit="µg/m³", who=15, who_unit="µg/m³", conv=1.0,
                 color=BLUE, hatch=""),
    "pm10": dict(label="PM10", html="PM<sub>10</sub>", unit="µg/m³", who=45, who_unit="µg/m³", conv=1.0,
                 color=ORANGE, hatch="//"),
    "no2":  dict(label="NO₂", html="NO<sub>2</sub>", unit="ppb", who=25, who_unit="µg/m³", conv=1.88,
                 color=GREEN, hatch=".."),
    "co":   dict(label="CO", html="CO", unit="ppm", who=4, who_unit="mg/m³", conv=1.145,
                 color=PINK, hatch="xx"),
}


def sub_aqi(pollutant, conc):
    """Turn one concentration into an AQI number (straight line between breakpoints)."""
    for c_lo, c_hi, i_lo, i_hi in BREAKPOINTS[pollutant]:
        if conc <= c_hi:
            conc = max(conc, c_lo)
            return int(round((i_hi - i_lo) / (c_hi - c_lo) * (conc - c_lo) + i_lo))
    return 500


def cat_index(aqi):
    for i, c in enumerate(CATEGORIES):
        if aqi <= c["hi"]:
            return i
    return len(CATEGORIES) - 1


# ============================================================ 3. STATIONS
# x, y place each station on the schematic map (0-100). Change names freely.
STATIONS = [
    dict(id="central", name="City Centre", kind="Urban background", x=50, y=50,
         base=1.0, traffic=1.0, industry=0.0),
    dict(id="industrial", name="Industrial Area", kind="Industrial", x=80, y=32,
         base=1.1, traffic=0.5, industry=1.0),
    dict(id="highway", name="Highway Road", kind="Roadside", x=26, y=64,
         base=1.0, traffic=1.5, industry=0.1),
    dict(id="residential", name="Green Park", kind="Residential", x=22, y=26,
         base=0.75, traffic=0.5, industry=0.0),
    dict(id="market", name="Old Market", kind="Commercial", x=68, y=74,
         base=0.95, traffic=1.2, industry=0.0),
]
DEFAULT_STATION = "central"


# ================================================================ 4. DATA
def generate_sample_data():
    """Pretend data: traffic peaks at 9 AM and 7 PM, factory shifts at midday,
    quieter weekends and a different 'weather' level each day."""
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None)) + TZ_OFFSET
    hours = pd.date_range(end=now.floor("h"), periods=7 * 24, freq="h")
    rng = np.random.default_rng(2026)
    codes, uniques = pd.factorize(hours.normalize())
    weather = rng.uniform(0.8, 1.35, len(uniques))[codes]
    h = hours.hour.values
    weekend = hours.dayofweek.values >= 5
    rush = np.exp(-((h - 9) ** 2) / 6) + 1.1 * np.exp(-((h - 19) ** 2) / 6)
    shift = ((h >= 10) & (h <= 16)).astype(float)
    night = 0.35 * np.exp(-((h - 3) ** 2) / 18)

    data = {}
    for st in STATIONS:
        spikes = (rng.random(len(h)) < 0.04) * rng.uniform(0.8, 1.8, len(h))
        traffic = st["traffic"] * rush * np.where(weekend, 0.6, 1.0)
        industry = st["industry"] * (shift + spikes)
        level = st["base"] * weather * (1 + 0.9 * traffic + 0.7 * industry + night)

        def noise(sd):
            return rng.normal(0, sd, len(h))

        data[st["id"]] = pd.DataFrame({
            "pm25": np.clip(26 * level + noise(3), 3, None),
            "pm10": np.clip(48 * level + noise(6), 5, None),
            "no2": np.clip(12 * st["base"] * (1 + 1.4 * traffic + 0.5 * industry) + noise(1.5), 1, None),
            "co": np.clip(0.6 * st["base"] * (1 + 1.2 * traffic + 0.3 * industry) + noise(0.08), 0.1, None),
        }, index=hours)
    return data


def load_csv_data(path):
    raw = pd.read_csv(path, parse_dates=["time"])
    data = {}
    for st in STATIONS:
        d = raw[raw["station"] == st["id"]].sort_values("time").set_index("time").tail(7 * 24)
        data[st["id"]] = d[["pm25", "pm10", "no2", "co"]].copy()
    return data


def add_aqi(df):
    for p in POLLUTANTS:
        df["aqi_" + p] = df[p].apply(lambda v, p=p: sub_aqi(p, v))
    cols = ["aqi_" + p for p in POLLUTANTS]
    df["aqi"] = df[cols].max(axis=1).astype(int)          # overall AQI = worst pollutant
    df["dominant"] = df[cols].idxmax(axis=1).str.replace("aqi_", "", regex=False)
    return df


raw_data = load_csv_data(DATA_CSV) if DATA_CSV.exists() else generate_sample_data()
USING_SAMPLE = not DATA_CSV.exists()


def summarize(st, df):
    """Collect every number the page needs for one station."""
    last = df.iloc[-1]
    aqi = int(last["aqi"])
    ci = cat_index(aqi)
    cat = CATEGORIES[ci]
    last24 = df.tail(24)
    frac = min(0.999, max(0.0, (aqi - cat["lo"]) / (cat["hi"] - cat["lo"] + 1)))
    wk = df.loc[df.index.dayofweek < 5, "aqi"]
    we = df.loc[df.index.dayofweek >= 5, "aqi"]
    by_hour = df.groupby(df.index.hour)["aqi"].mean()
    subs = {p: int(last["aqi_" + p]) for p in POLLUTANTS}
    total = sum(subs.values())
    daily = df.groupby(df.index.date)["aqi"].mean().round().astype(int)
    who = {p: last24[p].mean() * POLLUTANTS[p]["conv"] / POLLUTANTS[p]["who"] for p in POLLUTANTS}
    return dict(
        st=st, id=st["id"], name=st["name"], kind=st["kind"], x=st["x"], y=st["y"],
        df=df, last=last, aqi=aqi, ci=ci, cat=cat, dominant=last["dominant"],
        scale_pos=(ci + frac) / len(CATEGORIES) * 100,
        updated=last.name.strftime("%d %b %Y, %I:%M %p") + " " + TZ_LABEL,
        dist=round(math.hypot(st["x"] - 50, st["y"] - 50) * 0.12, 1),
        peak_t=last24["aqi"].idxmax(), peak=int(last24["aqi"].max()),
        low_t=last24["aqi"].idxmin(), low=int(last24["aqi"].min()),
        weekday=None if wk.empty else round(wk.mean()),
        weekend=None if we.empty else round(we.mean()),
        worst_hour=int(by_hour.idxmax()), worst_hour_aqi=round(by_hour.max()),
        subs=subs, shares={p: subs[p] / total * 100 for p in subs},
        daily=[dict(l=pd.Timestamp(d).strftime("%a %d"), a=int(v), c=cat_index(int(v)))
               for d, v in daily.items()],
        who=who,
    )


SUMMARIES = []
for station in STATIONS:
    frame = add_aqi(raw_data[station["id"]])
    SUMMARIES.append(summarize(station, frame))
BY_ID = {s["id"]: s for s in SUMMARIES}


# ============================================================== 5. CHARTS
def save(fig, name, tight=False):
    if tight:
        fig.savefig(IMG / name, dpi=110, bbox_inches="tight")
    else:
        fig.tight_layout()
        fig.savefig(IMG / name, dpi=110)
    plt.close(fig)


def chart_trend24(s):
    d = s["df"].tail(24)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    hour = d.index.hour
    rush = ((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 20))
    ax.fill_between(d.index, 0, 1, where=rush, transform=ax.get_xaxis_transform(),
                    color=ORANGE, alpha=0.18, step="mid", linewidth=0, label="Usual traffic hours")
    ax.plot(d.index, d["aqi"], color=BLUE, linewidth=2.4, label="AQI")
    ax.scatter([s["peak_t"]], [s["peak"]], color=BLUE, s=55, zorder=5)
    ax.annotate(f"Peak {s['peak']} at {s['peak_t']:%H:%M}", (s["peak_t"], s["peak"]),
                textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=INK)
    top = max(s["peak"] * 1.2, 60)
    ax.set_ylim(0, top)
    for t in (50, 100, 150, 200):
        if t < top:
            ax.axhline(t, color="#9aa5b1", linewidth=0.8, linestyle=":")
            ax.text(d.index[-1], t, f" {t}", va="center", fontsize=8, color=MUTED)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylabel("AQI")
    ax.legend(frameon=False, loc="lower right", fontsize=9, ncol=2)
    save(fig, f"{s['id']}_trend24.png")


def chart_trend7(s):
    d = s["df"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for day in pd.unique(d.index.normalize()):
        if pd.Timestamp(day).dayofweek >= 5:
            ax.axvspan(day, day + pd.Timedelta(days=1), color="#cbd2d9", alpha=0.45, linewidth=0)
    ax.plot(d.index, d["aqi"], color="#7b8794", linewidth=1, label="Hourly AQI")
    ax.plot(d.index, d["aqi"].rolling(24, min_periods=1).mean(), color=BLUE, linewidth=2.6,
            label="24-hour average")
    ax.set_ylim(0, max(d["aqi"].max() * 1.15, 60))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    ax.set_ylabel("AQI")
    ax.legend(frameon=False, loc="upper left", fontsize=9, ncol=2,
              title="Grey bands = weekends", title_fontsize=8)
    save(fig, f"{s['id']}_trend7.png")


def chart_heatmap(s):
    d = s["df"].assign(date=s["df"].index.date, hour=s["df"].index.hour)
    grid = d.pivot_table(index="date", columns="hour", values="aqi")
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    im = ax.imshow(grid.values, aspect="auto", cmap="viridis")  # colour-blind friendly
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels([pd.Timestamp(x).strftime("%a %d") for x in grid.index])
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("AQI")
    cb.outline.set_visible(False)
    save(fig, f"{s['id']}_heatmap.png")


def chart_donut(s):
    keys = list(POLLUTANTS)
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    wedges, _ = ax.pie([s["shares"][k] for k in keys], colors=[POLLUTANTS[k]["color"] for k in keys],
                       startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2))
    for w, k in zip(wedges, keys):
        w.set_hatch(POLLUTANTS[k]["hatch"])          # pattern as well as colour
    ax.text(0, 0, f"AQI\n{s['aqi']}", ha="center", va="center", fontsize=15, fontweight="bold", color=INK)
    labels = [f"{POLLUTANTS[k]['label']}  {s['shares'][k]:.0f}%" for k in keys]
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=10)
    save(fig, f"{s['id']}_donut.png", tight=True)


def chart_who(s):
    keys = list(POLLUTANTS)
    ratios = [s["who"][k] for k in keys]
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    bars = ax.barh([POLLUTANTS[k]["label"] for k in keys], ratios,
                   color=[POLLUTANTS[k]["color"] for k in keys], edgecolor="white")
    for b, k in zip(bars, keys):
        b.set_hatch(POLLUTANTS[k]["hatch"])
    ax.axvline(1, color=INK, linestyle="--", linewidth=1.2)
    ax.text(1, len(keys) - 0.45, " WHO guideline", fontsize=8, color=INK, va="bottom")
    for b, r in zip(bars, ratios):
        ax.text(r, b.get_y() + b.get_height() / 2, f" {r:.1f}×", va="center", fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(ratios) * 1.25, 1.5))
    ax.set_xlabel("Last 24-hour average ÷ WHO 24-hour guideline")
    save(fig, f"{s['id']}_who.png")


AQI_CMAP = LinearSegmentedColormap.from_list("aqi", [
    (0.0, "#00E400"), (25 / 300, "#00E400"), (75 / 300, "#FFFF00"), (125 / 300, "#FF7E00"),
    (175 / 300, "#FF0000"), (250 / 300, "#8F3F97"), (1.0, "#7E0023")])


def make_map():
    """Smooth pollution surface between stations (inverse-distance weighting)."""
    gx, gy = np.meshgrid(np.linspace(0, 100, 220), np.linspace(0, 100, 220))
    num, den = np.zeros_like(gx), np.zeros_like(gx)
    for s in SUMMARIES:
        w = 1.0 / ((gx - s["x"]) ** 2 + (gy - s["y"]) ** 2 + 60.0)
        num += w * s["aqi"]
        den += w
    field = num / den

    fig = plt.figure(figsize=(8.6, 5.4))
    ax = fig.add_axes([0, 0, 1, 1])                 # axes fill the whole picture, so
    ax.imshow(field, extent=(0, 100, 0, 100), origin="lower", cmap=AQI_CMAP,
              vmin=0, vmax=300, alpha=0.9, interpolation="bilinear", aspect="auto")
    ax.set_xlim(0, 100)                             # x,y here = percent positions
    ax.set_ylim(0, 100)                             # used for the HTML pins
    ax.set_axis_off()
    for v in range(10, 100, 20):                    # faint street grid
        ax.plot([v, v], [0, 100], color="white", linewidth=0.8, alpha=0.35)
        ax.plot([0, 100], [v, v], color="white", linewidth=0.8, alpha=0.35)

    levels = [lv for lv in (50, 100, 150, 200) if field.min() < lv < field.max()]
    if levels:
        cs = ax.contour(gx, gy, field, levels=levels, colors=INK, linewidths=1.1, linestyles="--")
        ax.clabel(cs, fmt="AQI %d", fontsize=8)

    hx, hy = [0, 30, 60, 100], [58, 66, 78, 84]     # highway
    ax.plot(hx, hy, color=INK, linewidth=7, solid_capstyle="round", zorder=3)
    ax.plot(hx, hy, color="white", linewidth=1.5, linestyle=(0, (4, 4)), zorder=4)
    box = dict(boxstyle="round,pad=0.25", fc="white", ec=INK, alpha=0.92)
    ax.text(88, 90, "Highway", fontsize=8, ha="center", bbox=box, zorder=6)

    fx, fy = [86, 92, 74], [40, 32, 20]             # factories
    ax.scatter(fx, fy, marker="^", s=140, color=INK, edgecolor="white", linewidth=1.5, zorder=5)
    ax.text(90, 47, "Factories", fontsize=8, ha="center", bbox=box, zorder=6)
    ax.text(1.5, 2.5, "Schematic map, not to scale", fontsize=8, color=INK, bbox=box, zorder=6)
    fig.savefig(IMG / "map.png", dpi=110)           # no tight bbox: keeps pin positions exact
    plt.close(fig)


for summary in SUMMARIES:
    chart_trend24(summary)
    chart_trend7(summary)
    chart_heatmap(summary)
    chart_donut(summary)
    chart_who(summary)
make_map()


# =========================================================== 6. HTML PIECES
def fmt(p, v):
    return f"{v:.2f}" if p == "co" else f"{v:.1f}"


def hidden_if(sid):
    return "" if sid == DEFAULT_STATION else " hidden"


def pill(ci):
    c = CATEGORIES[ci]
    return f'<span class="pill" style="background:{c["bg"]};color:{c["fg"]}">{c["icon"]} {escape(c["name"])}</span>'


def scale_html(s):
    segs = "".join(
        f'<span style="background:{c["bg"]};color:{c["fg"]}" title="{escape(c["name"])} {c["lo"]}-{c["hi"]}">{c["icon"]}</span>'
        for c in CATEGORIES)
    return (f'<div class="scale" role="img" aria-label="Scale from Good to Hazardous. '
            f'{escape(s["name"])} is in the {escape(s["cat"]["name"])} range.">'
            f'<span class="marker" style="left:{s["scale_pos"]:.1f}%">▼</span>{segs}</div>')


def dashboard_html():
    out = []
    for s in SUMMARIES:
        c = s["cat"]
        dom = POLLUTANTS[s["dominant"]]
        cards = []
        for p, meta in POLLUTANTS.items():
            sub = s["subs"][p]
            pc = CATEGORIES[cat_index(sub)]
            lead = " leading" if p == s["dominant"] else ""
            cards.append(
                f'<div class="poll{lead}"><h3>{meta["html"]}</h3>'
                f'<p class="poll-val">{fmt(p, s["last"][p])} <span>{meta["unit"]}</span></p>'
                f'<p class="poll-sub"><span class="dot" style="background:{pc["bg"]};color:{pc["fg"]}">{pc["icon"]}</span> '
                f'Index {sub}{" (leading)" if lead else ""}</p></div>')
        out.append(f'''
<div class="sv" data-station="{s["id"]}"{hidden_if(s["id"])}>
  <section class="hero" aria-label="Current air quality in {escape(s["name"])}">
    <div>
      <h2>Is the air safe to breathe right now?</h2>
      <p class="where">{escape(s["name"])}, {escape(s["kind"].lower())} station</p>
      <div class="aqi-box" style="background:{c["bg"]};color:{c["fg"]}">
        <span class="aqi-num">{s["aqi"]}</span>
        <span class="aqi-lbl">AQI<br>{c["icon"]} {escape(c["name"])}</span>
      </div>
    </div>
    <div>
      <p class="level">Air quality is {escape(c["name"])}.</p>
      <p class="advice">{escape(c["advice"])}</p>
      {scale_html(s)}
      <p class="small">Leading pollutant: {dom["html"]}. Last updated {s["updated"]}.</p>
    </div>
  </section>
  <div class="poll-grid">{"".join(cards)}</div>
</div>''')
    return "\n".join(out)


def pins_html():
    out = []
    for s in SUMMARIES:
        c = s["cat"]
        out.append(
            f'<button class="pin" data-pick="{s["id"]}" style="left:{s["x"]}%;top:{100 - s["y"]}%;'
            f'background:{c["bg"]};color:{c["fg"]}" aria-label="{escape(s["name"])}, AQI {s["aqi"]}, {escape(c["name"])}">'
            f'{s["aqi"]}<span class="pin-label">{escape(s["name"])}</span></button>')
    return "".join(out)


def legend_html():
    return "".join(
        f'<span class="chip" style="background:{c["bg"]};color:{c["fg"]}">{c["icon"]} {escape(c["name"])} {c["lo"]}-{c["hi"]}</span>'
        for c in CATEGORIES)


def trends_html():
    out = []
    for s in SUMMARIES:
        rows = ""
        for t, r in s["df"].tail(6)[::-1].iterrows():
            rows += (f'<tr><td>{t:%H:%M}</td><td>{pill(cat_index(int(r["aqi"])))} {int(r["aqi"])}</td>'
                     + "".join(f'<td>{fmt(p, r[p])}</td>' for p in POLLUTANTS) + '</tr>')
        heads = "".join(f'<th scope="col">{m["html"]}<br><span class="unit">{m["unit"]}</span></th>'
                        for m in POLLUTANTS.values())
        if s["weekday"] is not None and s["weekend"] is not None:
            diff = f"Weekdays average AQI {s['weekday']}; weekends average {s['weekend']}."
        else:
            diff = "Not enough weekend data yet to compare weekdays with weekends."
        out.append(f'''
<div class="sv" data-station="{s["id"]}"{hidden_if(s["id"])}>
  <div class="pane" data-pane="realtime" hidden>
    <h3>Last 6 hours</h3>
    <div class="table-scroll"><table>
      <thead><tr><th scope="col">Time</th><th scope="col">AQI</th>{heads}</tr></thead>
      <tbody>{rows}</tbody></table></div>
  </div>
  <div class="pane" data-pane="24h">
    <h3>Hourly AQI, last 24 hours</h3>
    <img src="img/{s["id"]}_trend24.png" alt="Line chart of hourly AQI at {escape(s["name"])} over the last 24 hours. Peak {s["peak"]} at {s["peak_t"]:%H:%M}, lowest {s["low"]} at {s["low_t"]:%H:%M}. Shaded bands mark usual traffic hours.">
    <p class="insight">Highest: AQI {s["peak"]} at {s["peak_t"]:%H:%M}. Lowest: AQI {s["low"]} at {s["low_t"]:%H:%M}.</p>
  </div>
  <div class="pane" data-pane="7d" hidden>
    <h3>Last 7 days</h3>
    <img src="img/{s["id"]}_trend7.png" alt="Line chart of hourly AQI at {escape(s["name"])} over 7 days with a 24-hour average line. Weekends are shaded grey.">
    <p class="insight">{diff}</p>
    <h3>When is it worst? Day against hour</h3>
    <img src="img/{s["id"]}_heatmap.png" alt="Heatmap of AQI at {escape(s["name"])} by day and hour. Brighter yellow cells mean higher AQI. On average the highest hour is {s["worst_hour"]:02d}:00.">
    <p class="insight">On average, {s["worst_hour"]:02d}:00 is the most polluted hour (AQI {s["worst_hour_aqi"]}).</p>
  </div>
</div>''')
    return "\n".join(out)


def breakdown_html():
    out = []
    for s in SUMMARIES:
        dom = POLLUTANTS[s["dominant"]]
        worst = max(s["who"], key=s["who"].get)
        shares = ", ".join(f"{POLLUTANTS[k]['label']} {s['shares'][k]:.0f} percent" for k in POLLUTANTS)
        out.append(f'''
<div class="sv" data-station="{s["id"]}"{hidden_if(s["id"])}>
  <h3>Share of each pollutant in the index</h3>
  <img class="donut" src="img/{s["id"]}_donut.png" alt="Donut chart of pollutant shares at {escape(s["name"])}: {shares}.">
  <p class="insight">{dom["html"]} contributes the most right now ({s["shares"][s["dominant"]]:.0f}%).</p>
  <h3>Compared with WHO guidelines</h3>
  <img src="img/{s["id"]}_who.png" alt="Bar chart of last 24-hour average pollutant levels as multiples of the WHO guideline. {POLLUTANTS[worst]['label']} is highest at {s['who'][worst]:.1f} times.">
  <p class="insight">{POLLUTANTS[worst]["html"]} is {s["who"][worst]:.1f}× the WHO 24-hour guideline.</p>
</div>''')
    return "\n".join(out)


def ranking_html():
    rows = []
    for n, s in enumerate(sorted(SUMMARIES, key=lambda x: x["aqi"]), 1):
        c = s["cat"]
        width = min(100, s["aqi"] / 300 * 100)
        rows.append(
            f'<li class="rank-row"><span class="rank-n">{n}</span>'
            f'<button class="linkbtn" data-pick="{s["id"]}">{escape(s["name"])}</button>'
            f'<span class="rank-bar"><span style="width:{width:.0f}%;background:{c["bg"]}"></span></span>'
            f'<span class="rank-val">{c["icon"]} {s["aqi"]}</span></li>')
    return "\n".join(rows)


def table_html():
    rows = []
    for s in SUMMARIES:
        rows.append(
            f'<tr data-row="{s["id"]}">'
            f'<td data-v="{escape(s["name"])}"><button class="linkbtn" data-pick="{s["id"]}">{escape(s["name"])}</button></td>'
            f'<td data-v="{s["dist"]}">{s["dist"]} km</td>'
            f'<td data-v="{s["aqi"]}">{pill(s["ci"])} {s["aqi"]}</td>'
            f'<td data-v="Operational">Operational</td>'
            f'<td data-v="{s["df"].index[-1]:%Y%m%d%H%M}">{s["updated"]}</td></tr>')
    return "\n".join(rows)


# ================================================================ 7. PAGE
CSS = r"""
:root{--bg:#eef3f1;--ink:#1f2933;--muted:#52606d;--line:#cfd9d4;--accent:#1b5e5a;--soft:#dcebe7}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:1rem}
body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,"Segoe UI",Roboto,system-ui,-apple-system,sans-serif;line-height:1.55}
h1,h2,h3{line-height:1.25;margin:0 0 .5rem}
h2{font-size:1.35rem;font-weight:700}
h3{font-size:1rem;font-weight:700;margin-top:1.25rem}
a{color:#0b5cad}
[hidden]{display:none!important}
.wrap{max-width:1120px;margin:0 auto;padding:0 1rem}
.sr-only,.skip{position:absolute;left:-9999px}
.skip:focus{left:1rem;top:1rem;background:#fff;padding:.6rem 1rem;z-index:20;border-radius:8px}
:focus-visible{outline:3px solid #0b5cad;outline-offset:2px}
header.site{background:#fff;border-bottom:1px solid var(--line)}
.top{display:flex;flex-wrap:wrap;gap:1rem 2rem;align-items:center;justify-content:space-between;padding:1rem}
.brand{font-size:1.5rem;font-weight:800;letter-spacing:-.02em;margin:0}
.brand small{display:block;font-size:.85rem;font-weight:400;letter-spacing:0;color:var(--muted)}
nav ul{list-style:none;display:flex;flex-wrap:wrap;gap:.25rem;margin:0;padding:0}
nav a{display:block;padding:.5rem .85rem;border-radius:999px;text-decoration:none;color:var(--ink);font-weight:600;font-size:.95rem}
nav a:hover{background:var(--soft)}
.loc-pick label{display:block;font-size:.85rem;color:var(--muted);font-weight:600}
select{font:inherit;padding:.5rem .7rem;border:1.5px solid #7b8794;border-radius:10px;background:#fff;color:var(--ink);min-height:44px}
main{padding-bottom:2rem}
section{margin-top:1.5rem}
.card{background:#fff;border:1px solid var(--line);border-radius:14px;padding:1.25rem}
.small{font-size:.9rem;color:var(--muted)}
.alert{margin-top:1rem;padding:.8rem 1rem;border-radius:10px;background:#fff4e5;border:2px solid #b35900;font-weight:600}
.hero{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:1.5rem 2.5rem;align-items:center;background:#fff;border-radius:22px;padding:1.75rem;border:1px solid var(--line)}
.where{margin:0 0 1rem;color:var(--muted)}
.aqi-box{display:inline-flex;align-items:center;gap:1rem;padding:.6rem 1.4rem .6rem 1.2rem;border-radius:20px;border:2px solid var(--ink)}
.aqi-num{font-size:4.6rem;font-weight:800;line-height:1;letter-spacing:-.03em}
.aqi-lbl{font-weight:700;font-size:1rem;line-height:1.3}
.level{font-size:1.5rem;font-weight:700;margin:0 0 .25rem}
.advice{font-size:1.05rem;margin:0 0 1.5rem;max-width:34rem}
.scale{position:relative;display:flex;margin:1.3rem 0 .9rem;border-radius:8px;overflow:visible;border:1.5px solid var(--ink)}
.scale>span:not(.marker){flex:1;text-align:center;font-size:.8rem;padding:.2rem 0}
.marker{position:absolute;top:-1.25rem;transform:translateX(-50%);font-size:1rem;line-height:1;color:var(--ink)}
.poll-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:.75rem;margin-top:.75rem}
.poll{background:#fff;border:1px solid var(--line);border-radius:12px;padding:.8rem 1rem}
.poll.leading{border:2px solid var(--ink)}
.poll h3{margin:0;color:var(--muted);font-size:.9rem}
.poll-val{font-size:1.7rem;font-weight:700;margin:.1rem 0;letter-spacing:-.01em}
.poll-val span{font-size:.85rem;font-weight:400;color:var(--muted)}
.poll-sub{margin:0;font-size:.9rem;color:var(--muted)}
.dot,.pill,.chip{display:inline-block;border-radius:999px;font-weight:600;border:1px solid rgba(0,0,0,.35)}
.dot{width:1.4rem;height:1.4rem;text-align:center;font-size:.75rem;line-height:1.25rem}
.pill{padding:0 .6rem;font-size:.82rem}
.chip{padding:.15rem .7rem;font-size:.85rem}
.map-wrap{position:relative;max-width:860px;margin:1rem auto 0;border-radius:12px;border:1px solid var(--line)}
.map-wrap img{display:block;width:100%;height:auto;border-radius:12px}
.pin{position:absolute;transform:translate(-50%,-50%);width:48px;height:48px;border-radius:50%;border:3px solid var(--ink);font:inherit;font-weight:800;font-size:1rem;cursor:pointer;padding:0}
.pin.selected{outline:4px solid #0b5cad;outline-offset:3px}
.pin-label{position:absolute;top:calc(100% + 4px);left:50%;transform:translateX(-50%);white-space:nowrap;background:#fff;color:var(--ink);padding:1px 7px;border-radius:6px;border:1px solid var(--ink);font-size:.78rem;font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:1rem;justify-content:center}
.split{display:grid;grid-template-columns:3fr 2fr;gap:1.25rem;align-items:start}
.tabs{display:flex;flex-wrap:wrap;gap:.4rem;margin:.5rem 0 1rem}
.tab{min-height:44px;padding:.4rem 1.1rem;border:1.5px solid #7b8794;border-radius:999px;background:#fff;color:var(--ink);font:inherit;font-weight:600;cursor:pointer}
.tab[aria-selected="true"]{background:var(--accent);border-color:var(--accent);color:#fff}
img{max-width:100%;height:auto}
.donut{max-width:380px;display:block}
.insight{margin:.35rem 0 0;font-weight:600}
.table-scroll{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.95rem}
th,td{text-align:left;padding:.55rem .6rem;border-bottom:1px solid var(--line);vertical-align:middle}
th{font-weight:700;color:var(--muted);white-space:nowrap}
.unit{font-weight:400;font-size:.8rem}
th button{font:inherit;font-weight:700;color:var(--muted);background:none;border:0;padding:0;cursor:pointer}
tr.sel td{background:var(--soft)}
.linkbtn{font:inherit;font-weight:600;color:#0b5cad;background:none;border:0;padding:.25rem 0;cursor:pointer;text-decoration:underline;text-align:left}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1.25rem}
.rank{list-style:none;margin:.5rem 0 0;padding:0;display:grid;gap:.6rem}
.rank-row{display:grid;grid-template-columns:1.5rem 9rem 1fr 4.5rem;align-items:center;gap:.6rem}
.rank-n{color:var(--muted);font-weight:700}
.rank-bar{height:1rem;background:#e4ebe7;border-radius:999px;overflow:hidden}
.rank-bar span{display:block;height:100%;border-right:2px solid var(--ink)}
.rank-val{font-weight:700}
.custom-svg{width:100%;max-width:620px;height:auto}
.range-pick{display:flex;flex-wrap:wrap;gap:1rem;align-items:end;margin-bottom:.75rem}
.range-pick label{display:block;font-size:.85rem;color:var(--muted);font-weight:600}
.settings{display:grid;gap:1rem;max-width:36rem}
details{background:#fff;border:1px solid var(--line);border-radius:12px;padding:.7rem 1rem;margin-top:.6rem}
summary{cursor:pointer;font-weight:700}
footer{border-top:1px solid var(--line);margin-top:2rem;padding:1.25rem 0 2rem}
@media (max-width:820px){.split{grid-template-columns:1fr}.rank-row{grid-template-columns:1.5rem 7rem 1fr 4rem}.aqi-num{font-size:3.6rem}}
"""

JS = r"""
(function () {
  "use strict";
  var DATA = %%DATA%%;
  var state = { station: DATA.default, pane: "24h" };
  function q(s, r) { return (r || document).querySelector(s); }
  function qa(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
  function catIndex(v) {
    for (var i = 0; i < DATA.categories.length; i++) { if (v <= DATA.categories[i].hi) { return i; } }
    return DATA.categories.length - 1;
  }

  function render() {
    qa(".sv").forEach(function (el) { el.hidden = el.getAttribute("data-station") !== state.station; });
    qa(".pane").forEach(function (el) { el.hidden = el.getAttribute("data-pane") !== state.pane; });
    qa(".tab").forEach(function (b) {
      b.setAttribute("aria-selected", b.getAttribute("data-pane") === state.pane ? "true" : "false");
    });
    qa(".pin").forEach(function (p) { p.classList.toggle("selected", p.getAttribute("data-pick") === state.station); });
    qa("tr[data-row]").forEach(function (r) { r.classList.toggle("sel", r.getAttribute("data-row") === state.station); });
    q("#station-select").value = state.station;
    var st = DATA.stations[state.station];
    q("#map-selected").textContent = "Selected: " + st.name + " (" + st.kind + "). AQI " + st.aqi + ", " + DATA.categories[st.ci].name + ".";
    if (state.pane === "custom") { drawCustom(); }
  }

  function drawCustom() {
    var st = DATA.stations[state.station];
    var a = parseInt(q("#range-start").value, 10), b = parseInt(q("#range-end").value, 10);
    if (a > b) { var t = a; a = b; b = t; }
    var rows = st.daily.slice(a, b + 1);
    if (!rows.length) { return; }
    var W = 600, H = 250, pad = 38, bw = (W - pad * 2) / rows.length;
    var max = Math.max.apply(null, rows.map(function (r) { return r.a; }).concat([100])) * 1.15;
    var total = 0;
    var svg = '<svg class="custom-svg" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Daily average AQI at ' + st.name + '">';
    rows.forEach(function (r, i) {
      total += r.a;
      var h = (H - pad * 2) * r.a / max, x = pad + i * bw + bw * 0.15, w = bw * 0.7, y = H - pad - h;
      var c = DATA.categories[r.c];
      svg += '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" fill="' + c.bg + '" stroke="#1f2933" stroke-width="1.5"/>';
      svg += '<text x="' + (x + w / 2) + '" y="' + (y - 6) + '" text-anchor="middle" font-size="13" font-weight="700" fill="#1f2933">' + c.icon + " " + r.a + "</text>";
      svg += '<text x="' + (x + w / 2) + '" y="' + (H - 14) + '" text-anchor="middle" font-size="12" fill="#52606d">' + r.l + "</text>";
    });
    svg += '<line x1="' + pad + '" y1="' + (H - pad) + '" x2="' + (W - pad) + '" y2="' + (H - pad) + '" stroke="#52606d"/></svg>';
    q("#custom-chart").innerHTML = svg;
    var avg = Math.round(total / rows.length);
    q("#custom-summary").textContent = "Average AQI from " + rows[0].l + " to " + rows[rows.length - 1].l + ": " + avg + " (" + DATA.categories[catIndex(avg)].name + ").";
  }

  qa("[data-pick]").forEach(function (el) {
    el.addEventListener("click", function () { state.station = el.getAttribute("data-pick"); render(); });
  });
  q("#station-select").addEventListener("change", function () { state.station = this.value; render(); });
  qa(".tab").forEach(function (b) {
    b.addEventListener("click", function () { state.pane = b.getAttribute("data-pane"); render(); });
  });
  q("#range-start").addEventListener("change", drawCustom);
  q("#range-end").addEventListener("change", drawCustom);

  // sortable station table
  qa("th[data-sort]").forEach(function (th) {
    var asc = true;
    th.querySelector("button").addEventListener("click", function () {
      var idx = th.cellIndex, numeric = th.getAttribute("data-sort") === "n";
      var tb = th.closest("table").tBodies[0];
      var rows = Array.prototype.slice.call(tb.rows);
      rows.sort(function (r1, r2) {
        var x = r1.cells[idx].getAttribute("data-v"), y = r2.cells[idx].getAttribute("data-v");
        if (numeric) { return asc ? parseFloat(x) - parseFloat(y) : parseFloat(y) - parseFloat(x); }
        return asc ? x.localeCompare(y) : y.localeCompare(x);
      });
      rows.forEach(function (r) { tb.appendChild(r); });
      th.setAttribute("aria-sort", asc ? "ascending" : "descending");
      asc = !asc;
    });
  });

  // alert settings (saved only in this browser)
  var settings = { on: true, min: 101 };
  try { var saved = JSON.parse(localStorage.getItem("airwatch-settings")); if (saved) { settings = saved; } } catch (e) {}
  function renderAlerts() {
    var hits = [];
    for (var id in DATA.stations) { if (DATA.stations[id].aqi >= settings.min) { hits.push(DATA.stations[id]); } }
    var banner = q("#alert-banner");
    banner.hidden = !(settings.on && hits.length);
    banner.textContent = "⚠ Alert: " + hits.map(function (h) { return h.name + " (AQI " + h.aqi + ")"; }).join(", ") + " at or above your alert level.";
    q("#alert-list").textContent = hits.length
      ? "Stations at or above your alert level: " + hits.map(function (h) { return h.name; }).join(", ") + "."
      : "No station is at or above your alert level right now.";
  }
  q("#alert-level").value = String(settings.min);
  q("#alert-on").checked = settings.on;
  function saveSettings() {
    settings = { on: q("#alert-on").checked, min: parseInt(q("#alert-level").value, 10) };
    try { localStorage.setItem("airwatch-settings", JSON.stringify(settings)); } catch (e) {}
    renderAlerts();
  }
  q("#alert-level").addEventListener("change", saveSettings);
  q("#alert-on").addEventListener("change", saveSettings);

  render();
  renderAlerts();
})();
"""

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Air quality and pollution data analysis: AQI, PM2.5, PM10, NO2 and CO.">
<title>%%SITE%%: Air Quality and Pollution Data Analysis</title>
<style>%%CSS%%</style>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="site"><div class="wrap top">
  <h1 class="brand">%%SITE%%<small>Air Quality and Pollution Data Analysis</small></h1>
  <nav aria-label="Main"><ul>
    <li><a href="#dashboard">Dashboard</a></li>
    <li><a href="#trends">Historic Trends</a></li>
    <li><a href="#map">Geospatial Map</a></li>
    <li><a href="#breakdown">Pollutant Breakdown</a></li>
    <li><a href="#alerts">Alerts &amp; Settings</a></li>
  </ul></nav>
  <div class="loc-pick"><label for="station-select">Location</label>
    <select id="station-select">%%OPTIONS%%</select></div>
</div></header>

<main id="main" class="wrap">
  <div id="alert-banner" class="alert" role="status" hidden></div>

  <section id="dashboard" aria-label="Dashboard">%%DASHBOARD%%</section>

  <section id="map" class="card" aria-labelledby="h-map">
    <h2 id="h-map">Where is the air cleaner?</h2>
    <p class="small">Colours show estimated AQI between stations. Dashed lines mark AQI 50, 100, 150 and 200. Select a pin to update the whole page.</p>
    <div class="map-wrap">
      <img src="img/map.png" alt="Schematic map of the city with five monitoring stations, a highway and factories. Colours blend from green to red as air quality gets worse. %%MAPALT%%">
      %%PINS%%
    </div>
    <div class="legend">%%LEGEND%%</div>
    <p id="map-selected" class="small" aria-live="polite"></p>
  </section>

  <section id="trends" class="split">
    <div class="card" id="panel-a">
      <h2>Trends over time</h2>
      <div class="tabs" role="group" aria-label="Timeframe">
        <button class="tab" data-pane="realtime" aria-selected="false">Real-time</button>
        <button class="tab" data-pane="24h" aria-selected="true">Last 24 Hours</button>
        <button class="tab" data-pane="7d" aria-selected="false">Last 7 Days</button>
        <button class="tab" data-pane="custom" aria-selected="false">Custom Range</button>
      </div>
      %%TRENDS%%
      <div class="pane" data-pane="custom" hidden>
        <div class="range-pick">
          <div><label for="range-start">From</label><select id="range-start">%%RANGE_START%%</select></div>
          <div><label for="range-end">To</label><select id="range-end">%%RANGE_END%%</select></div>
        </div>
        <div id="custom-chart"></div>
        <p id="custom-summary" class="insight" aria-live="polite"></p>
        <p class="small">Daily averages by calendar day. The first day may be a partial day.</p>
      </div>
    </div>
    <div class="card" id="breakdown">
      <h2>What is in the air?</h2>
      %%BREAKDOWN%%
      <p class="small">WHO guidelines are the long-term health reference. Staying below 1× is the goal.</p>
    </div>
  </section>

  <section id="locations" class="two" aria-label="Location comparison">
    <div class="card">
      <h2>Cleanest to most polluted</h2>
      <ol class="rank">%%RANKING%%</ol>
    </div>
    <div class="card">
      <h2>Monitoring stations</h2>
      <div class="table-scroll"><table class="sortable">
        <thead><tr>
          <th scope="col" data-sort="s"><button>Station</button></th>
          <th scope="col" data-sort="n"><button>Distance</button></th>
          <th scope="col" data-sort="n"><button>AQI</button></th>
          <th scope="col" data-sort="s"><button>Status</button></th>
          <th scope="col" data-sort="s"><button>Last updated</button></th>
        </tr></thead>
        <tbody>%%TABLE%%</tbody>
      </table></div>
      <p class="small">Select a column heading to sort.</p>
    </div>
  </section>

  <section id="alerts" class="card" aria-labelledby="h-alerts">
    <h2 id="h-alerts">Alerts and settings</h2>
    <div class="settings">
      <div><label for="alert-level">Alert me when a station reaches</label><br>
        <select id="alert-level">
          <option value="51">Moderate (51 and above)</option>
          <option value="101">Unhealthy for Sensitive Groups (101 and above)</option>
          <option value="151">Unhealthy (151 and above)</option>
          <option value="201">Very Unhealthy (201 and above)</option>
        </select></div>
      <label><input type="checkbox" id="alert-on"> Show an alert banner at the top of this page</label>
      <p id="alert-list" class="insight" aria-live="polite"></p>
      <p class="small">Settings are saved only in this browser. Alerts appear while this page is open.</p>
    </div>
  </section>
</main>

<footer><div class="wrap">
  <p><strong>Data sources and disclaimer.</strong> %%SOURCE%% Estimates may vary from official readings. AQI follows the US EPA method; guideline comparisons use WHO 2021 values.</p>
  <details><summary>Chart accessibility notes</summary>
    <ul>
      <li>Colour is never the only signal: every AQI colour has an icon, a name and a number.</li>
      <li>Chart lines and bars use the Okabe-Ito palette; pollutant bars and slices also use hatch patterns.</li>
      <li>The heatmap uses viridis, which stays readable for most colour-vision types.</li>
      <li>Charts have no boxes, grid walls or 3D effects, so the ink goes to the data. Key values are written in text below each chart.</li>
      <li>Text meets WCAG AA contrast (4.5 to 1). Graphics meet 3 to 1. Every chart has alt text.</li>
    </ul></details>
  <details><summary>About this project</summary>
    <p>A Python project using Pandas, NumPy and Matplotlib. The program builds this page automatically on GitHub. Source code: <a href="%%CONTACT%%">project repository</a>.</p></details>
  <details><summary>Privacy and terms</summary>
    <p>This page collects no personal data. Alert settings stay in your browser. Information here is for learning and is not medical advice.</p></details>
</div></footer>
<script>%%JS%%</script>
</body>
</html>
"""

# ------------------------------------------------------------- 8. WRITE FILE
page_data = {
    "default": DEFAULT_STATION,
    "categories": [dict(name=c["name"], hi=c["hi"], bg=c["bg"], icon=c["icon"]) for c in CATEGORIES],
    "stations": {s["id"]: dict(name=s["name"], kind=s["kind"], aqi=s["aqi"], ci=s["ci"], daily=s["daily"])
                 for s in SUMMARIES},
}
data_json = json.dumps(page_data).replace("</", "<\\/")

n_days = len(SUMMARIES[0]["daily"])
labels = [d["l"] for d in SUMMARIES[0]["daily"]]
opt_start = "".join(f'<option value="{i}">{l}</option>' for i, l in enumerate(labels))
opt_end = "".join(f'<option value="{i}"{" selected" if i == n_days - 1 else ""}>{l}</option>'
                  for i, l in enumerate(labels))
options = "".join(f'<option value="{s["id"]}"{" selected" if s["id"] == DEFAULT_STATION else ""}>{escape(s["name"])}</option>'
                  for s in SUMMARIES)
worst = max(SUMMARIES, key=lambda x: x["aqi"])
best = min(SUMMARIES, key=lambda x: x["aqi"])
map_alt = f"Highest reading: {worst['name']}, AQI {worst['aqi']}. Lowest: {best['name']}, AQI {best['aqi']}."
if USING_SAMPLE:
    source = ("Demo data generated by Python (NumPy) for learning. It does not describe a real city. "
              "Add data/stations.csv to use real readings.")
else:
    source = "Data from data/stations.csv (add your organisation name and source link here)."

js = JS.replace("%%DATA%%", data_json)
html = (PAGE
        .replace("%%CSS%%", CSS)
        .replace("%%JS%%", js)
        .replace("%%SITE%%", SITE_NAME)
        .replace("%%OPTIONS%%", options)
        .replace("%%DASHBOARD%%", dashboard_html())
        .replace("%%MAPALT%%", escape(map_alt))
        .replace("%%PINS%%", pins_html())
        .replace("%%LEGEND%%", legend_html())
        .replace("%%TRENDS%%", trends_html())
        .replace("%%RANGE_START%%", opt_start)
        .replace("%%RANGE_END%%", opt_end)
        .replace("%%BREAKDOWN%%", breakdown_html())
        .replace("%%RANKING%%", ranking_html())
        .replace("%%TABLE%%", table_html())
        .replace("%%SOURCE%%", source)
        .replace("%%CONTACT%%", CONTACT_URL))
(OUT / "index.html").write_text(html, encoding="utf-8")
print("Done! Open docs/index.html in your browser.")
