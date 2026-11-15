# build.py: reads air quality data, makes charts, writes a website

import numpy as np                 # numbers and maths
import pandas as pd                # tables of data (like Excel in Python)
import matplotlib
matplotlib.use("Agg")              # draw charts without opening a window
import matplotlib.pyplot as plt    # charts
from pathlib import Path           # file paths
from string import Template        # fills blanks in our HTML page

# ---------- Output folder ----------
OUT = Path("docs")                 # GitHub Pages will serve this folder
OUT.mkdir(exist_ok=True)

# ---------- 1. DATA ----------
# For now we create sample data so the project runs immediately.
# Later, replace this block with:
#   df = pd.read_csv("data/air_quality.csv", parse_dates=["time"])
# Your CSV needs columns: time, pm25, pm10, no2
rng = np.random.default_rng(42)                       # random generator (seed = repeatable)
hours = pd.date_range("2026-11-08", periods=7 * 24, freq="h")
h = hours.hour.values
rush = np.exp(-((h - 9) ** 2) / 8) + np.exp(-((h - 19) ** 2) / 8)  # peaks at 9am and 7pm

df = pd.DataFrame({
    "time": hours,
    "pm25": np.clip(40 + 60 * rush + rng.normal(0, 6, len(h)), 5, None),
    "pm10": np.clip(80 + 80 * rush + rng.normal(0, 10, len(h)), 10, None),
    "no2":  np.clip(25 + 30 * rush + rng.normal(0, 4, len(h)), 3, None),
})

# ---------- 2. AQI CALCULATION (from PM2.5) ----------
# (conc_low, conc_high, aqi_low, aqi_high) following the US EPA method.
# Check the latest EPA table if you need exact official numbers.
BREAKPOINTS = [
    (0.0,   9.0,   0,   50),
    (9.1,   35.4,  51,  100),
    (35.5,  55.4,  101, 150),
    (55.5,  125.4, 151, 200),
    (125.5, 225.4, 201, 300),
    (225.5, 500.4, 301, 500),
]

def pm25_to_aqi(conc):
    conc = round(conc, 1)
    for c_lo, c_hi, i_lo, i_hi in BREAKPOINTS:
        if c_lo <= conc <= c_hi:
            # straight-line formula between two points
            return round((i_hi - i_lo) / (c_hi - c_lo) * (conc - c_lo) + i_lo)
    return 500

df["aqi"] = df["pm25"].apply(pm25_to_aqi)

# ---------- 3. CATEGORIES (your colour palette) ----------
# (max AQI, name, background colour, text colour, icon, advice)
CATEGORIES = [
    (50,  "Good",                           "#00E400", "#000000", "✔", "Air quality is satisfactory. Enjoy outdoor activities."),
    (100, "Moderate",                       "#FFFF00", "#000000", "▲", "Acceptable; however, sensitive individuals may experience minor irritation."),
    (150, "Unhealthy for Sensitive Groups", "#FF7E00", "#000000", "◆", "Sensitive groups should reduce prolonged outdoor exertion."),
    (200, "Unhealthy",                      "#FF0000", "#FFFFFF", "■", "Everyone should limit prolonged outdoor exertion."),
    (300, "Very Unhealthy",                 "#8F3F97", "#FFFFFF", "●", "Avoid outdoor activity; keep windows closed."),
    (10_000, "Hazardous",                   "#7E0023", "#FFFFFF", "✖", "Stay indoors and use an air purifier if you have one."),
]

def get_category(aqi):
    for top, name, bg, fg, icon, advice in CATEGORIES:
        if aqi <= top:
            return dict(name=name, bg=bg, fg=fg, icon=icon, advice=advice)

latest = df.iloc[-1]                       # the most recent row
aqi_now = int(latest["aqi"])
cat = get_category(aqi_now)

# ---------- 4. CHARTS ----------
BLUE = "#0072B2"   # colourblind-safe blue (Okabe-Ito palette)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)

# 4a. last 24 hours
last24 = df.tail(24)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(last24["time"], last24["aqi"], color=BLUE, linewidth=2)
ax.set_title("AQI over the last 24 hours")
ax.set_ylabel("AQI")
ax.spines[["top", "right"]].set_visible(False)   # remove clutter
ax.tick_params(axis="x", rotation=30)
save(fig, "trend24.png")

# 4b. last 7 days
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(df["time"], df["aqi"], color=BLUE, linewidth=1.5)
ax.set_title("AQI over the last 7 days")
ax.set_ylabel("AQI")
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", rotation=30)
save(fig, "trend7.png")

# 4c. heatmap: day (rows) vs hour (columns)
df["date"] = df["time"].dt.date
df["hour"] = df["time"].dt.hour
grid = df.pivot_table(index="date", columns="hour", values="aqi")
fig, ax = plt.subplots(figsize=(8, 3.5))
im = ax.imshow(grid.values, aspect="auto", cmap="viridis")  # viridis is colourblind-friendly
ax.set_yticks(range(len(grid.index)))
ax.set_yticklabels([d.strftime("%a %d") for d in grid.index])
ax.set_xlabel("Hour of day")
ax.set_title("When is pollution highest? (Day vs Hour)")
fig.colorbar(im, ax=ax, label="AQI")
save(fig, "heatmap.png")

# 4d. pollutants compared with WHO 24-hour guidelines (µg/m³)
WHO = {"PM2.5": 15, "PM10": 45, "NO₂": 25}
now_vals = {"PM2.5": latest["pm25"], "PM10": latest["pm10"], "NO₂": latest["no2"]}
ratios = [now_vals[k] / WHO[k] for k in WHO]
fig, ax = plt.subplots(figsize=(8, 3))
ax.barh(list(WHO), ratios, color=BLUE, hatch="//", edgecolor="white")  # hatch = pattern, not just colour
ax.axvline(1, color="black", linestyle="--")
ax.set_xlabel("Times the WHO guideline (dashed line = 1×)")
ax.set_title("Pollutants vs WHO guidelines")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "who.png")

# ---------- 5. THE WEB PAGE ----------
PAGE = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>City AirWatch</title>
<style>
  body { font-family: Inter, system-ui, sans-serif; margin: 0; background: #f6f8f7; color: #1f2933; line-height: 1.5; }
  header, main, footer { max-width: 960px; margin: auto; padding: 1rem; }
  .hero { background: #fff; border-radius: 16px; padding: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,.1); }
  .badge { display: inline-block; padding: .4rem 1rem; border-radius: 999px; font-weight: 700;
           background: $bg; color: $fg; }
  .aqi { font-size: 4rem; font-weight: 800; margin: .5rem 0 0; }
  .small { color: #52606d; font-size: .9rem; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin-top: 1rem; }
  .card { background: #fff; border-radius: 16px; padding: 1rem; box-shadow: 0 1px 4px rgba(0,0,0,.1); }
  img { max-width: 100%; height: auto; }
</style>
</head>
<body>
<header><h1>City AirWatch</h1><p class="small">Air Quality and Pollution Data Analysis</p></header>
<main>
  <section class="hero">
    <p class="small">Is the air safe to breathe right now?</p>
    <div class="aqi">$aqi</div>
    <span class="badge">$icon $name</span>
    <p><strong>Health advice:</strong> $advice</p>
    <p class="small">Dominant pollutant: PM2.5 &middot; Last updated: $updated</p>
  </section>

  <div class="grid">
    <div class="card"><img src="trend24.png" alt="Line chart of AQI over the last 24 hours, with peaks near 9 AM and 7 PM"></div>
    <div class="card"><img src="trend7.png" alt="Line chart of AQI over the last 7 days showing a repeating daily pattern"></div>
    <div class="card"><img src="heatmap.png" alt="Heatmap of AQI by day and hour; darker or brighter cells mean higher pollution"></div>
    <div class="card"><img src="who.png" alt="Bar chart comparing PM2.5, PM10 and NO2 with WHO guideline levels"></div>
  </div>
</main>
<footer class="small">
  Data: sample data for learning purposes. Replace with a real source (e.g., CPCB or OpenAQ). Estimates may vary.
</footer>
</body>
</html>""")

html = PAGE.substitute(
    aqi=aqi_now, name=cat["name"], icon=cat["icon"], advice=cat["advice"],
    bg=cat["bg"], fg=cat["fg"],
    updated=latest["time"].strftime("%d %b %Y, %I:%M %p"),
)
(OUT / "index.html").write_text(html, encoding="utf-8")
print("Done! Open docs/index.html in your browser.")
