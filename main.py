"""GTA VI week-one trophy rarity forecaster (make.fun gta6-trophy-rarity-forecast-01), Python 3.13, stdlib only.

Model, in one paragraph
-----------------------
An Exophase earn rate is (players holding the trophy) / (players tracked). On launch night the page tracked a few
hundred accounts (review / early-access copies) that already held almost every trophy; those accounts are still in
the counts, so every rare trophy sits on a "dev floor" of a handful of players that is diluted as players tracked
grow. Everything above that floor is real progress. If each player's progress is play-speed x time and speed is
log-logistic across players, the real share of a milestone follows logit(r) = beta * (ln t - ln tau): a straight line
in logit rate against log hours since launch. So, per trophy:

  1. find the dev floor (launch-night count, capped by the latest count) and subtract it;
  2. fit the slope of logit(real rate) against ln(hours since launch) over the recent distinct refreshes;
  3. shrink that slope toward a pooled slope-vs-rarity line fitted across the well-measured trophies;
  4. extrapolate to the target, damped, and add back the floor diluted by the forecast players tracked;
  5. spread: a logit-scale band that grows with the horizon and the slope's uncertainty, skewed upward for
     trophies still sitting on the floor (they can only take off, and 0.01% is the bottom of the logit clamp).

Run: MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6449, -0.6745, 0.0, 0.6745, 1.6449]

# Hours are counted from here (the house curve baseline uses the same origin).
LAUNCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
# A read at or below this many players tracked is the launch-night page of review / early-access accounts.
LAUNCH_NIGHT_MAX_PLAYERS = 2000
# Real holders below this are too few to fit a trend on (Exophase prints 2 decimals: 0.01% of 50k is 5 players).
MIN_FIT_COUNT = 15
# Fit window, in hours before the last refresh (recency half-weight at half of it).
WINDOW_HOURS = 30.0
# Extrapolated log-time trends overshoot over a long horizon; keep this share of the fitted slope.
DAMP = 0.85
# Players-tracked growth: share of the fitted log-log slope kept, and its bounds.
N_DAMP = 0.8
N_SLOPE_DEFAULT = 0.9


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def logit_raw(pct: float) -> float:
    """Logit of a percent without the score's 0.01 clamp (for internal shares far below 0.01%)."""
    p = min(1.0 - 1e-9, max(1e-9, pct / 100.0))
    return math.log(p / (1.0 - p))


def expit_pct(v: float) -> float:
    if v >= 0:
        return 100.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return 100.0 * e / (1.0 + e)


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(s: str) -> float:
    return (parse_time(s) - LAUNCH).total_seconds() / 3600.0


def weighted_slope(xs, ys, ws):
    """Least-squares slope of y on x with weights; None when x has no spread."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-9:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    return sxy / sxx, mx, my, sxx


def distinct_refreshes(history):
    """Exophase refreshes on its own clock; keep each distinct page once, at the first read that showed it."""
    out = []
    prev = None
    for read in history:
        key = (read.get("players_tracked"), read.get("trophies_earned"), tuple(read.get("rates") or []))
        if key != prev:
            out.append(read)
            prev = key
    return out


def forecast_case(case: dict) -> dict:
    trophies = case["trophies"]
    k = len(trophies)
    history = [r for r in case.get("history") or [] if r.get("rates") and r.get("players_tracked")]
    target_h = hours_since_launch(case["target"])

    if not history:
        # Nothing to go on: a wide band around 1%.
        return {"trophies": [{"id": t["id"], "q": [0.01, 0.1, 1.0, 5.0, 30.0]} for t in trophies]}

    reads = distinct_refreshes(history)
    last = reads[-1]
    n_last = float(last["players_tracked"])
    h_last = max(1.0, hours_since_launch(last["at"]))
    h_target = max(h_last + 0.5, target_h)
    dlog = math.log(h_target / h_last)

    # Only the launch-night page so far (a few hundred review / early-access accounts holding nearly everything):
    # Exophase's first refresh adds the launch's real players, nearly all holding only the first trophy, so every
    # rate below 100% is diluted. Forecast the launch-night count over a grown player base, skewed up for the real
    # players who earn the early trophies.
    if n_last <= LAUNCH_NIGHT_MAX_PLAYERS:
        return launch_night_forecast(trophies, last)

    def rate_at(read, i):
        rates = read["rates"]
        if i >= len(rates):
            return None
        return rates[i]

    # 1. Dev floor: holders on the launch-night page, capped by what the latest page shows.
    first = history[0]
    floor = [0.0] * k
    if first["players_tracked"] <= LAUNCH_NIGHT_MAX_PLAYERS and n_last > first["players_tracked"]:
        n0 = float(first["players_tracked"])
        for i in range(k):
            r0 = rate_at(first, i)
            if r0 is None:
                continue
            floor[i] = r0 * n0 / 100.0
    last_count = []
    for i in range(k):
        r = rate_at(last, i)
        last_count.append((r if r is not None else 0.0) * n_last / 100.0)
    floor = [min(floor[i], last_count[i]) for i in range(k)]

    # Refreshes from the post-launch-night page on. Exophase may refresh only a few times a day, so keep every one,
    # weighted toward the recent ones (half weight WINDOW_HOURS before the last refresh).
    # The first hours after launch are a ramp (everyone just started) whose log-time slope is far steeper than the
    # trend that carries on, and ln(hours) gives those early points huge leverage: fit the last WINDOW_HOURS, from
    # hour 8 on, and widen only when Exophase refreshed too rarely to leave three distinct pages there.
    post = [r for r in reads if r["players_tracked"] > LAUNCH_NIGHT_MAX_PLAYERS and hours_since_launch(r["at"]) >= 2.0]
    recent = [r for r in post if hours_since_launch(r["at"]) >= max(8.0, h_last - WINDOW_HOURS)]
    if len(recent) < 3:
        recent = [r for r in post if hours_since_launch(r["at"]) >= 6.0] or post

    def recency(r):
        return 0.5 ** ((h_last - hours_since_launch(r["at"])) / (WINDOW_HOURS / 2.0))

    # 2. Players tracked at the target: log-log trend of the recent refreshes.
    n_slope = N_SLOPE_DEFAULT
    if len(recent) >= 2:
        xs = [math.log(hours_since_launch(r["at"])) for r in recent]
        ys = [math.log(r["players_tracked"]) for r in recent]
        fit = weighted_slope(xs, ys, [recency(r) for r in recent])
        if fit is not None and (max(xs) - min(xs)) > 0.08:
            n_slope = min(2.0, max(0.1, fit[0] * N_DAMP))
    n_target = n_last * math.exp(n_slope * dlog)

    # 3. Per-trophy slope of logit(real rate) on ln(hours).
    fits = {}
    for i in range(k):
        pts = []
        for r in recent:
            rate = rate_at(r, i)
            if rate is None:
                continue
            n = float(r["players_tracked"])
            real = rate * n / 100.0 - floor[i]
            if real < MIN_FIT_COUNT or rate >= 97.0:
                continue
            pts.append((math.log(hours_since_launch(r["at"])), logit_raw(100.0 * real / n), real, recency(r)))
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            if max(xs) - min(xs) < 0.05:
                continue
            ys = [p[1] for p in pts]
            ws = [p[3] for p in pts]
            fit = weighted_slope(xs, ys, ws)
            if fit is None:
                continue
            slope, mx, my, sxx = fit
            if len(pts) >= 3:
                resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
                rss = sum(w * e * e for w, e in zip(ws, resid))
                dof = max(1.0, sum(ws) - 2.0)
                se = math.sqrt(rss / dof / sxx) if sxx > 0 else 1.0
                se = max(se, 0.1)
            else:
                se = 0.5
            fits[i] = (slope, se, last_count[i] - floor[i])

    # 4. Pooled slope as a function of rarity (rarer trophies at the leading edge climb faster in logit terms).
    pool_a, pool_b = 1.0, 0.0
    good = [(i, f) for i, f in fits.items() if f[2] >= 60]
    if len(good) >= 3:
        xs, ys, ws = [], [], []
        for i, (slope, se, cnt) in good:
            r_real = 100.0 * max(cnt, 1.0) / n_last
            xs.append(logit_raw(r_real))
            ys.append(slope)
            ws.append(math.sqrt(cnt) / (1.0 + se))
        fit = weighted_slope(xs, ys, ws)
        sw = sum(ws)
        my = sum(w * y for y, w in zip(ys, ws)) / sw
        if fit is not None and len(good) >= 5:
            b = min(0.1, max(-0.6, fit[0]))
            pool_a, pool_b = fit[2] - b * fit[1], b
        else:
            pool_a, pool_b = my, 0.0
    pool_a = min(4.0, max(0.0, pool_a))

    def pooled_slope(level_logit):
        s = pool_a + pool_b * level_logit
        return min(3.5, max(0.0, s))

    rows = []
    for i, t in enumerate(trophies):
        tid = t["id"]
        rate_last = rate_at(last, i)
        if rate_last is None:
            rate_last = 0.0
        real_cnt = max(0.0, last_count[i] - floor[i])

        # A trophy every tracked player holds (the first one): it stays at 100.
        if rate_last >= 99.95:
            rows.append({"id": tid, "q": [99.9, 99.98, 100, 100, 100]})
            continue

        # Nobody holds it yet, not even the launch-night accounts. The platinum and 100% completion need every
        # other trophy: not in week one. The score clamps at 0.01%, so any quantile at or below it costs nothing
        # when the page still prints 0.00% or 0.01%.
        if rate_last <= 0.0:
            text = (t.get("name", "") + " " + t.get("description", "")).lower()
            if t.get("tier") == "platinum" or "100%" in text:
                rows.append({"id": tid, "q": [0.0, 0.0, 0.01, 0.01, 0.015]})
            else:
                rows.append({"id": tid, "q": [0.0, 0.01, 0.01, 0.02, 0.08]})
            continue

        floor_t = floor[i] / n_target * 100.0  # the dev floor diluted at the target

        if real_cnt >= MIN_FIT_COUNT:
            real_pct = 100.0 * real_cnt / n_last
            lvl = logit_raw(real_pct)
            sp = pooled_slope(lvl)
            if i in fits:
                slope, se, _ = fits[i]
                w = real_cnt / (real_cnt + 150.0) * (1.0 / (1.0 + se))
                s = w * slope + (1.0 - w) * sp
            else:
                se = 0.8
                s = sp
            s = min(3.5, max(-0.5, s)) * DAMP
            m_real = lvl + s * dlog
            med = expit_pct(m_real) + floor_t
            m = logit_raw(med)
            # Spread: slope uncertainty over the horizon, a model-error floor, plus count noise for thin trophies.
            slope_sd = 0.35 + min(1.0, se) * 0.5 + 0.6 / math.sqrt(1.0 + real_cnt / 40.0)
            sd = math.sqrt(0.22 ** 2 + (slope_sd * dlog) ** 2)
            sd_lo, sd_hi = sd, sd * 1.1
        else:
            # Still on (or near) the dev floor: real holders too few to fit. Grow a tentative real share with the
            # pooled slope from a few-player start; the band is wide and skewed up.
            start = max(real_cnt, 2.0)
            lvl = logit_raw(100.0 * start / n_last)
            s = min(2.0, pooled_slope(lvl)) * DAMP
            m_real = lvl + s * dlog
            med = expit_pct(m_real) + floor_t
            m = logit_raw(med)
            sd_lo = 0.5
            sd_hi = 0.8 + 0.8 * dlog

        q = []
        for z in Z:
            v = m + z * (sd_hi if z > 0 else sd_lo)
            q.append(expit_pct(v))
        # Exophase prints 2 decimals and the score clamps to 0.01..99.99.
        q = [min(100.0, max(0.0, round(x, 4))) for x in q]
        for j in range(1, 5):
            if q[j] < q[j - 1]:
                q[j] = q[j - 1]
        rows.append({"id": tid, "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception as exc:  # never leave a case without a valid forecast
            print(f"{path.name}: model failed ({exc!r}); falling back to a wide persistence band", file=sys.stderr)
            forecast = fallback(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


# Launch-night dilution: median growth of players tracked at the first refresh, and its logit-scale spread.
LAUNCH_GROWTH = 5.5
LAUNCH_SD_LO = 0.35
LAUNCH_SD_HI = 0.55


def launch_night_forecast(trophies, last) -> dict:
    rows = []
    for i, t in enumerate(trophies):
        rates = last["rates"]
        r = rates[i] if i < len(rates) and rates[i] is not None else 0.0
        if r >= 99.95:
            rows.append({"id": t["id"], "q": [99.9, 99.98, 100, 100, 100]})
            continue
        if r <= 0.0:
            rows.append({"id": t["id"], "q": [0.0, 0.0, 0.01, 0.01, 0.03]})
            continue
        m = logit_raw(r / LAUNCH_GROWTH)
        q = [round(expit_pct(m + z * (LAUNCH_SD_HI if z > 0 else LAUNCH_SD_LO)), 4) for z in Z]
        for j in range(1, 5):
            q[j] = max(q[j], q[j - 1])
        rows.append({"id": t["id"], "q": q})
    return {"trophies": rows}


def fallback(case: dict) -> dict:
    hist = case.get("history") or []
    last = hist[-1]["rates"] if hist else []
    rows = []
    for i, t in enumerate(case["trophies"]):
        r = last[i] if i < len(last) and last[i] is not None else 0.0
        lg = logit(r)
        q = [round(expit_pct(lg + z * 1.2), 4) for z in Z]
        rows.append({"id": t["id"], "q": q})
    return {"trophies": rows}


if __name__ == "__main__":
    main()
