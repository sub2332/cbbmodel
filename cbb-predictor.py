#!/usr/bin/env python3
"""
College Basketball (NCAA men's Division I) Game Winner / Win Probability Predictor
================================================================================
Requires:  pip install pandas numpy scikit-learn pyarrow

Built to mirror the NFL predictor: same walk-forward testing, factor-accuracy tables, tuned Elo,
offense / defense unit rankings and a shareable HTML report (no parlays tab).

Pipeline
--------
 1. Download (and cache) ESPN schedules + team box scores from the free hoopR-mbb-data repository
 2. Per team per game: possessions, offensive / defensive efficiency, tempo and the Four Factors
 3. Strictly PRE-GAME features (everything uses games played BEFORE the one being predicted)
 4. Elo with margin of victory, tuned K / home court / season carryover (toward the conference average)
 5. Adjusted efficiency ratings (KenPom-style offense / defense / tempo ratings that update after every game)
 6. Four Factors unit rankings: offense and defense each rated 0-100 from eFG%, turnovers, rebounding,
    free throws, 2P% / 3P%, steals and blocks, opponent-adjusted, with last season blended in early
 7. Conference games (closer, smaller home edge), early-season shrink, home court; recent form and rest are
    tested every run too (they did not help on top of the ratings, so they are off by default)
 8. Home-minus-Away matchup features -> Logistic Regression (no intercept, so neutral sites are symmetric)
 9. Walk-forward test (every season predicted by a model trained only on earlier seasons) + factor tests
10. Predict the next day of games and write an HTML report

How to run:  python3 cbb_predictor.py
  First run downloads ~30 MB of data into cbb_data_cache/ next to this file and takes a few minutes;
  later runs re-download only the current season. The report (cbb_predictions.html) is written next to this file.
  Pick a different day with PREDICT_DATE below (or CBB_PREDICT_DATE=2026-11-28 python3 cbb_predictor.py).
  Backtest a past date with AS_OF_DATE / CURRENT_SEASON (e.g. CBB_SEASON=2025 CBB_AS_OF=2025-01-21).

Leakage rules: every rolling / cumulative / rating feature is computed on each team's chronological
game sequence and uses only games BEFORE the one being predicted.

Season labels follow ESPN / hoopR: the 2026-27 season is "2027".
"""

import base64
import html
import json
import os
import re
import time
import urllib.request
import warnings
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
FIRST_SEASON = 2014                    # earliest season loaded (2014 is only a warm-up year for the ratings)
TRAIN_START_SEASON = 2015              # first season used to train / tune anything
EXCLUDE_FROM_TRAINING = [2021]         # COVID season (no fans, many cancellations): kept for rating history only
CURRENT_SEASON = None                  # None = automatic (the season that is under way: Nov 2026 -> 2027)
WF_START_SEASON = 2019                 # walk-forward test: every season from here on is predicted by a model
WF_MIN_TRAIN_GAMES = 2000              #   trained only on the seasons before it
TUNE_SEASONS = None                    # seasons used to tune Elo / ratings / unit settings (None = before WF_START_SEASON)
LIVE_MODEL_USES_ALL_GAMES = True       # the model that predicts upcoming games is fit on every completed game

PREDICT_DATE = None                    # None = automatic: the first day (from today on) that has unplayed games.
                                       # Or a date string such as "2026-11-28".
PREDICT_DAYS = 1                       # how many days of games to show starting from PREDICT_DATE
AS_OF_DATE = None                      # TEST ONLY: pretend today is this date ("2025-01-20"); later results are hidden
TIMEZONE = "America/Chicago"           # game days and tip times are shown in this time zone

_e = os.environ.get
if _e("CBB_PREDICT_DATE"):
    PREDICT_DATE = _e("CBB_PREDICT_DATE").strip()
if _e("CBB_AS_OF"):
    AS_OF_DATE = _e("CBB_AS_OF").strip()
if _e("CBB_SEASON"):
    CURRENT_SEASON = int(_e("CBB_SEASON"))

# ---- data source (free, updated daily): https://github.com/sportsdataverse/hoopR-mbb-data ----
DATA_URL = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/{folder}/parquet/{prefix}_{season}.parquet"
DATA_FILES = {"box": ("team_box", "team_box"), "sched": ("schedules", "mbb_schedule"), "player": ("player_box", "player_box")}
USE_PLAYER_DATA = True                 # player box scores (~4 MB a season) for the roster-continuity factor
CACHE_DIR = Path(_e("CBB_CACHE_DIR") or (Path(__file__).resolve().parent / "cbb_data_cache"))
CACHE_HOURS = 3                        # the current season is downloaded again when the cached copy is older than this
OFFLINE = bool(_e("CBB_OFFLINE"))      # True = never download, use cached files only

# ---- who counts as Division I ----
D1_MIN_GAMES = 12                      # a team with at least this many games in a season's schedule is treated as D1
                                       # (non-D1 opponents appear only a few times). Only D1-vs-D1 games train the model.

# ---- Elo settings (tuned automatically on the tuning seasons when ELO_AUTO_TUNE is True) ----
ELO_START = 1500.0
ELO_NEW_TEAM = 1350.0                  # a team that first shows up after the first season (new to D1) starts here
ELO_NON_D1 = 1150.0                    # fixed rating for non-D1 opponents (not updated)
ELO_K = 42.5                           # (the values here are what the auto-tune picked on 2015-2018)
ELO_HOME_ADV = 90.0                    # Elo points for playing at home (0 at neutral sites)
ELO_USE_MOV = True                     # scale each update by the margin of victory (538 basketball formula)
ELO_CARRYOVER = 0.65                   # share of last season's edge kept into the new season
ELO_REGRESS_TO = "conf"                # each new season ratings move toward: "conf" = the conference average, "league" = D1 average
ELO_EARLY_BOOST = 1.5                  # K multiplier for a team's first ELO_EARLY_GAMES games of a season
ELO_EARLY_GAMES = 8
ELO_AUTO_TUNE = True
ELO_K_CANDIDATES = [20.0, 35.0, 50.0]
ELO_HFA_CANDIDATES = [70.0, 100.0]
ELO_CARRY_CANDIDATES = [0.5, 0.7, 0.85]
ELO_REGRESS_CANDIDATES = ["league", "conf"]
ELO_MOV_CANDIDATES = [False, True]
ELO_EARLY_CANDIDATES = [1.0, 1.5, 2.0]

# ---- Adjusted efficiency ratings (offense / defense points per 100 possessions above average, plus tempo) ----
EFF_AUTO_TUNE = True
EFF_LR = 0.09                          # how fast the ratings move after each game
EFF_CARRY = 0.97                       # share of last season's rating kept into the new season
EFF_HFA = 4.5                          # home edge, points per 100 possessions (about 2.8 points a game)
EFF_TEMPO_LR = 0.08
EFF_LR_CANDIDATES = [0.06, 0.09, 0.12]
EFF_CARRY_CANDIDATES = [0.6, 0.75, 0.9, 0.97]
EFF_HFA_CANDIDATES = [3.0, 4.5, 6.0]
EFF_NON_D1 = 12.0                      # a non-D1 team starts this many points per 100 worse on offense AND defense
LEAGUE_ORTG = 104.0                    # starting league average (points per 100 possessions); then learned from the games
LEAGUE_TEMPO = 68.0                    # starting league average possessions per 40 minutes

# ---- Four Factors unit rankings (offense / defense 0-100 ratings; carry REAL model weight) ----
#   eff    points per 100 possessions (scored / allowed)
#   efg    effective FG% ((FGM + 0.5 * 3PM) / FGA)
#   tov    turnovers per possession (offense: lower is better; defense: forced, higher is better)
#   orb    offensive rebound rate (offense: own boards; defense: opponent's, lower is better)
#   ftr    free-throw rate (FTA / FGA)
#   two    2-point FG%
#   three  3-point FG% (noisy, so it gets a small weight)
#   stl    steals (offense: steals allowed per possession; defense: steals per possession)
#   blk    blocks (offense: share of own 2-pt tries blocked; defense: share of opponent 2-pt tries blocked)
# Weights below follow the factor-accuracy test on 2015-2025 data (see the report's Record & accuracy tab).
# (Tested: this mix, an efficiency-only rating and a pure Four Factors rating were all within noise; this one was best.)
RANK_STAT_WEIGHTS = {"eff": 0.40, "efg": 0.15, "tov": 0.10, "orb": 0.10, "ftr": 0.04,
                     "two": 0.10, "three": 0.03, "stl": 0.04, "blk": 0.04}
RANK_DIRECTION = {"eff": (True, False), "efg": (True, False), "tov": (False, True), "orb": (True, False),
                  "ftr": (True, False), "two": (True, False), "three": (True, False),
                  "stl": (False, True), "blk": (False, True)}
UNIT_SOURCES = {k: (f"off_{k}", f"def_{k}") for k in RANK_DIRECTION}
RANK_STAT_NAMES = {"eff": "efficiency (pts / 100 poss)", "efg": "effective FG%", "tov": "turnover rate",
                   "orb": "offensive rebound rate", "ftr": "free-throw rate", "two": "2-point FG%",
                   "three": "3-point FG%", "stl": "steal rate", "blk": "block rate"}
# display: key -> (offense label, defense label, number format, display scale)
RANK_STAT_META = {
    "eff": ("Pts / 100", "Pts allowed / 100", "{:.1f}", 1),
    "efg": ("eFG%", "Opp eFG%", "{:.1f}", 100),
    "tov": ("TOV%", "Forced TOV%", "{:.1f}", 100),
    "orb": ("OREB%", "Opp OREB%", "{:.1f}", 100),
    "ftr": ("FT rate", "Opp FT rate", "{:.1f}", 100),
    "two": ("2P%", "Opp 2P%", "{:.1f}", 100),
    "three": ("3P%", "Opp 3P%", "{:.1f}", 100),
    "stl": ("Stolen %", "Steal %", "{:.1f}", 100),
    "blk": ("Blocked %", "Block %", "{:.1f}", 100),
}
UNIT_AUTO_TUNE = True                  # TEST the settings below on the tuning seasons and use the best
UNIT_MODE = "season"                   # "season" = this season's earlier games (+ last season blended in early)
                                       # "ewma"   = all earlier games, recent ones counting more
UNIT_PRIOR_GAMES = 4                   # "season" mode: last season's average counts like this many games (fades out)
UNIT_HALFLIFE = 10                     # "ewma" mode: a game this many games old counts half
OPP_ADJUST = 1.0                       # opponent adjustment: 0 = off, 1 = fully remove opponent quality
UNIT_MODE_CANDIDATES = ["season", "ewma"]
UNIT_PRIOR_CANDIDATES = [0, 4, 8, 12]
UNIT_HALFLIFE_CANDIDATES = [6, 12]
UNIT_ADJ_CANDIDATES = [0.0, 0.5, 1.0]
UNIT_WINDOW_TEXT = ""

# ---- other factors ----
# ---- Injuries / missing players (no free college injury feed exists, so absences are read from the box scores) ----
INJURY_DETECT = True                   # a rotation player who missed the team's LAST game is presumed out (until he plays again
                                       # or falls out of the last-5-games rotation). Add known injuries by hand below.
ROT_MPG = 10.0                         # "rotation player" = averaged at least this many minutes over the team's last 5 games
VAL_PRIOR_MIN = 150                    # a player's value is shrunk toward last season's (or a bench-level guess) by this many minutes
BENCH_TOP = 7                          # the team's backups = everyone outside its top 7 in minutes (their value replaces the player)
# Players you KNOW are out (or doubtful) for the games being predicted. Team = name as on the report cards.
MANUAL_INJURIES = {
    # "Duke": [("Player Name", "Out")],              <- example; status "Out" or "Doubtful"
}
# Players the model presumes out (missed the last game) who you know WILL play:
MANUAL_AVAILABLE = {
    # "Kansas": ["Player Name"],
}
# Players coming BACK after missing games. The team's ratings were built partly without him, so his team gets a boost:
# (his minutes / 40) x (his value - the bench's value) x (share of the team's last 10 games he missed).
# Works for any length of absence (also covers MANUAL_AVAILABLE, so list a returning player here only).
MANUAL_RETURNING = {
    # "Kentucky": ["Player Name"],
}
INJ_STATUS = {"Out": 1.0, "Doubtful": 0.75}
CONT_GAMES = 15                        # roster continuity matters early: its effect fades out over a team's first 15 games
CONT_MEAN = 0.55                       # typical share of a team's minutes played by returning players (the neutral value)
CALIBRATE = True                       # confidence correction: shrink (or stretch) the probabilities so that, on past
                                       # seasons, "75%" picks really won about 75% of the time (fit on earlier seasons only)
FORM_GAMES = 5                         # recent form = how much a team beat its expected margin over its last N games
EARLY_GAMES = 10                       # "Early season" factor: fades from a team's 1st game to its 10th
REST_CAP = 7                           # days of rest are capped at this (a season opener counts as fully rested)
MARGIN_SD = None                       # None = learned (points of margin per unit of probit); about 11 in college basketball
TOTAL_RIDGE_ALPHA = 20.0
MODEL_C = 0.5                          # logistic regularization (smaller = steadier)

# Which factor groups the model uses (None = all). Chosen from the factor tests on 2015-2024 (see the report):
ACTIVE_FEATURE_GROUPS = ["Elo", "Efficiency ratings", "Four Factors units", "Conference game", "Early season",
                         "Roster continuity", "Injuries", "Home court"]
# Tested, no help on top of these (the report keeps testing them, so you can see if that changes): "Recent form", "Rest"
GROUP_WEIGHT_MULTIPLIERS = {           # 1.0 = learned weight; above 1 = lean on it more; 0 = ignore
    "Elo": 1.0,
    "Efficiency ratings": 1.0,
    "Four Factors units": 1.0,
    "Recent form": 1.0,
    "Rest": 1.0,
    "Conference game": 1.0,
    "Early season": 1.0,
    "Roster continuity": 1.0,
    "Injuries": 1.0,
    "Home court": 1.0,
}

# ---- report ----
REPORT_FILENAME = "cbb_predictions.html"
OPEN_REPORT_IN_BROWSER = not os.environ.get("CI")
LOGO_MODE = "embed"                    # "embed" (logos baked into the file), "link" (needs internet to view) or "none"
LOGO_PX = 64
FETCH_LIVE_SCORES = True               # fill in final scores from ESPN's live scoreboard for yesterday and today (the data
                                       # source only updates during the day, so last night's results would otherwise be missing)
FETCH_ODDS = True                      # look up today's betting lines from ESPN's public scoreboard (shown, not used)
TRAP_TOP_N = 64                        # the Trapezoid and efficiency-landscape charts show the top N teams by efficiency margin
MARKET_FLAG_PTS = 10.0                 # flag a game when the model differs from the moneyline by this many points
COMPARE_CONFIGS = True                 # run the walk-forward set-up comparison (simpler models vs the shipped one)


def read_parquet_file(path):
    """Read one parquet file (kept as a function so it can be swapped out)."""
    return pd.read_parquet(path)


# ----------------------------------------------------------------------------
# SMALL HELPERS
# ----------------------------------------------------------------------------
def today_local():
    if AS_OF_DATE:
        return pd.Timestamp(AS_OF_DATE).normalize()
    return pd.Timestamp.now(tz=TIMEZONE).tz_localize(None).normalize()


def season_of(day):
    return day.year + 1 if day.month >= 8 else day.year


def logloss_vec(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def to_day(s):
    """A date column -> naive midnight timestamps (handles dates, strings, or days-since-1970 numbers)."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="D", errors="coerce")
    d = pd.to_datetime(s.astype(str), errors="coerce", utc=True)
    return d.dt.tz_localize(None).dt.normalize()


# ----------------------------------------------------------------------------
# 1. DATA LOADING
# ----------------------------------------------------------------------------
def fetch_file(kind, season, current):
    """Path to the cached parquet for (kind, season), downloading it if needed. None if unavailable."""
    folder, prefix = DATA_FILES[kind]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{prefix}_{season}.parquet"
    fresh = path.exists() and (not current or time.time() - path.stat().st_mtime < CACHE_HOURS * 3600)
    if fresh or OFFLINE:
        return path if path.exists() else None
    url = DATA_URL.format(folder=folder, prefix=prefix, season=season)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cbb-predictor/1.0 (personal project)"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        if len(data) < 1000 or data[:4] != b"PAR1":
            raise ValueError("not a parquet file")
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
    except Exception as ex:
        if path.exists():
            print(f"  ! could not refresh {path.name} ({ex}); using the cached copy")
        else:
            print(f"  ! {prefix} {season} unavailable ({ex})")
            return None
    return path


SCHED_COLS = ["game_id", "season", "season_type", "start_date", "date", "game_date", "neutral_site",
              "conference_competition", "status_type_completed", "status_type_name", "status_period",
              "home_id", "away_id", "home_score", "away_score", "home_conference_id", "away_conference_id",
              "notes_headline", "venue_full_name", "venue_address_city", "venue_address_state", "tournament_id",
              "home_short_display_name", "away_short_display_name", "home_display_name", "away_display_name",
              "home_abbreviation", "away_abbreviation", "home_logo", "away_logo", "home_location", "away_location"]
BOX_COLS = ["game_id", "season", "team_id", "opponent_team_id", "team_score", "field_goals_made",
            "field_goals_attempted", "three_point_field_goals_made", "three_point_field_goals_attempted",
            "free_throws_made", "free_throws_attempted", "offensive_rebounds", "defensive_rebounds",
            "total_turnovers", "turnovers", "steals", "blocks", "assists", "fouls",
            "team_short_display_name", "team_display_name", "team_abbreviation", "team_logo", "team_location"]


PLAYER_COLS = ["game_id", "season", "team_id", "athlete_id", "minutes", "athlete_display_name", "points",
               "field_goals_made", "field_goals_attempted", "free_throws_made", "free_throws_attempted",
               "offensive_rebounds", "defensive_rebounds", "assists", "steals", "blocks", "turnovers", "fouls"]


def load_season_files(seasons, cur):
    sched, box, players = [], [], []
    for s in seasons:
        ps, pb = fetch_file("sched", s, s == cur), fetch_file("box", s, s == cur)
        if ps is None:
            continue
        try:
            d = read_parquet_file(ps)
            sched.append(d[[c for c in SCHED_COLS if c in d.columns]].copy())
        except Exception as ex:
            print(f"  ! schedule {s} could not be read ({ex}); skipped")
            continue
        if pb is not None:
            try:
                b = read_parquet_file(pb)
                box.append(b[[c for c in BOX_COLS if c in b.columns]].copy())
            except Exception as ex:
                print(f"  ! box scores {s} could not be read ({ex})")
        if USE_PLAYER_DATA:
            pp = fetch_file("player", s, s == cur)
            if pp is not None:
                try:
                    d = read_parquet_file(pp)
                    d = d[[c for c in PLAYER_COLS if c in d.columns]].copy()
                    for c in PLAYER_COLS:
                        if c == "athlete_display_name":
                            d[c] = d[c].astype(str) if c in d.columns else ""
                        else:
                            d[c] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan
                    players.append(d[d["minutes"] > 0].dropna(subset=["game_id", "team_id", "athlete_id"]))
                except Exception as ex:
                    print(f"  ! player box scores {s} could not be read ({ex})")
    if not sched:
        raise SystemExit("No schedule data could be loaded. Check the internet connection (or the cache folder).")
    return (pd.concat(sched, ignore_index=True), (pd.concat(box, ignore_index=True) if box else pd.DataFrame(columns=BOX_COLS)),
            (pd.concat(players, ignore_index=True) if players else pd.DataFrame(columns=PLAYER_COLS)))


def build_games(sched, finals=None):
    """One row per game: ids, local game day and tip time, site, scores, completion.
    finals: {game_id: (home_id, home_score, away_id, away_score)} from the live scoreboard, for games the data
    source has not caught up on yet."""
    g = sched.copy()
    for c in SCHED_COLS:
        if c not in g.columns:
            g[c] = np.nan
    g = g.dropna(subset=["game_id", "home_id", "away_id"])
    g["game_id"] = g["game_id"].astype("int64")
    g["home_id"] = pd.to_numeric(g["home_id"], errors="coerce").astype("int64")
    g["away_id"] = pd.to_numeric(g["away_id"], errors="coerce").astype("int64")
    g["season"] = pd.to_numeric(g["season"], errors="coerce").astype(int)
    src = g["start_date"].where(g["start_date"].notna(), g["date"])
    tip = pd.to_datetime(src, utc=True, errors="coerce")
    tip_local = tip.dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    fallback = to_day(g["game_date"]) if g["game_date"].notna().any() else pd.Series(pd.NaT, index=g.index)
    g["tip"] = tip_local.where(tip_local.notna(), fallback + pd.Timedelta(hours=19))
    g["day"] = g["tip"].dt.normalize()
    g = g.dropna(subset=["day"])
    g["neutral"] = g["neutral_site"].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(int)
    g["conf_game"] = g["conference_competition"].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(int)
    for c in ("home_score", "away_score", "status_period"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    if finals:
        for i, gid, hid in zip(g.index, g["game_id"].values, g["home_id"].values):
            f_ = finals.get(int(gid))
            if f_ is None:
                continue
            hs, as_ = (f_[1], f_[3]) if f_[0] == hid else (f_[3], f_[1])
            if not (pd.notna(g.at[i, "home_score"]) and g.at[i, "home_score"] + g.at[i, "away_score"] > 0
                    and str(g.at[i, "status_type_completed"]).lower() in ("true", "1", "1.0")):
                g.at[i, "home_score"], g.at[i, "away_score"] = hs, as_
                g.at[i, "status_type_completed"] = True
    done = g["status_type_completed"].astype(str).str.lower().isin(["true", "1", "1.0"])
    done &= g["home_score"].notna() & g["away_score"].notna() & ((g["home_score"] + g["away_score"]) > 0)
    if AS_OF_DATE:                                        # test mode: hide everything from that day on
        done &= g["day"] < today_local()
    g["completed"] = done
    g.loc[~done, ["home_score", "away_score"]] = np.nan
    st = g["status_type_name"].astype(str).str.upper()
    dead = st.str.contains("CANCEL|POSTPON|FORFEIT|SUSPEND|DELAY", regex=True) & ~done
    stale = ~done & (g["day"] < today_local())            # never played and already in the past
    g = g[~(dead | stale)]
    g["result"] = g["home_score"] - g["away_score"]
    g["periods"] = g["status_period"].where(g["status_period"] >= 2, 2).fillna(2)
    g = g.sort_values(["tip", "game_id"]).drop_duplicates("game_id", keep="last").reset_index(drop=True)
    return g


def team_directory(sched, box):
    """team_id -> {short, name, abbr, logo} (latest season wins)."""
    rows = []
    for side in ("home", "away"):
        d = sched[["season", f"{side}_id", f"{side}_short_display_name", f"{side}_display_name",
                   f"{side}_abbreviation", f"{side}_logo", f"{side}_location"]].copy()
        d.columns = ["season", "id", "short", "name", "abbr", "logo", "loc"]
        rows.append(d)
    if len(box) and "team_id" in box.columns:
        d = box[["season", "team_id"] + [c for c in ("team_short_display_name", "team_display_name", "team_abbreviation",
                                                      "team_logo", "team_location") if c in box.columns]].copy()
        d.columns = ["season", "id"] + [{"team_short_display_name": "short", "team_display_name": "name",
                                          "team_abbreviation": "abbr", "team_logo": "logo",
                                          "team_location": "loc"}[c] for c in d.columns[2:]]
        rows.append(d)
    d = pd.concat(rows, ignore_index=True).dropna(subset=["id"])
    d["id"] = pd.to_numeric(d["id"], errors="coerce")
    d = d.dropna(subset=["id"]).sort_values("season")
    out = {}
    for r in d.itertuples():
        t = int(r.id)
        cur = out.setdefault(t, {})
        for k in ("short", "name", "abbr", "logo", "loc"):
            v = getattr(r, k, None)
            if isinstance(v, str) and v and v.lower() != "nan":
                cur[k] = v
    for t, v in out.items():
        v.setdefault("short", v.get("loc") or v.get("name") or str(t))
        v.setdefault("abbr", v["short"][:5].upper())
    return out


# ----------------------------------------------------------------------------
# 2. PER-GAME TEAM STATS (possessions, efficiency, Four Factors)
# ----------------------------------------------------------------------------
def build_game_stats(box, games):
    """One row per (game_id, team): possessions, tempo and offense / defense rate stats."""
    if box.empty:
        return pd.DataFrame(columns=["game_id", "team"])
    b = box.copy()
    b["game_id"] = pd.to_numeric(b["game_id"], errors="coerce")
    b["team"] = pd.to_numeric(b["team_id"], errors="coerce")
    b = b.dropna(subset=["game_id", "team"])
    b["game_id"], b["team"] = b["game_id"].astype("int64"), b["team"].astype("int64")
    num = {"fgm": "field_goals_made", "fga": "field_goals_attempted", "tpm": "three_point_field_goals_made",
           "tpa": "three_point_field_goals_attempted", "ftm": "free_throws_made", "fta": "free_throws_attempted",
           "orb": "offensive_rebounds", "drb": "defensive_rebounds", "stl": "steals", "blk": "blocks"}
    for k, c in num.items():
        b[k] = pd.to_numeric(b[c], errors="coerce") if c in b.columns else np.nan
    tov = pd.to_numeric(b["total_turnovers"], errors="coerce") if "total_turnovers" in b.columns else np.nan
    if "turnovers" in b.columns:
        tov = tov.where(pd.notna(tov), pd.to_numeric(b["turnovers"], errors="coerce")) if isinstance(tov, pd.Series) \
            else pd.to_numeric(b["turnovers"], errors="coerce")
    b["to"] = tov
    b = b.drop_duplicates(["game_id", "team"])
    keep = ["game_id", "team"] + list(num) + ["to"]
    b = b[keep]
    o = b.rename(columns={c: "o_" + c for c in keep[2:]}).rename(columns={"team": "opp"})
    m = b.merge(games[["game_id", "home_id", "away_id", "periods"]], on="game_id", how="inner")
    m["opp"] = np.where(m["team"] == m["home_id"], m["away_id"], m["home_id"])
    m = m.merge(o, on=["game_id", "opp"], how="inner")
    poss_t = m["fga"] - m["orb"] + m["to"] + 0.475 * m["fta"]
    poss_o = m["o_fga"] - m["o_orb"] + m["o_to"] + 0.475 * m["o_fta"]
    m["poss"] = (poss_t + poss_o) / 2
    bad = (m["poss"] < 40) | (m["fga"] <= 0) | (m["o_fga"] <= 0)
    minutes = 40 + 5 * (m["periods"].clip(lower=2) - 2)
    m["tempo"] = m["poss"] * 40 / minutes
    out = pd.DataFrame({"game_id": m["game_id"], "team": m["team"], "poss": m["poss"], "tempo": m["tempo"]})
    pts_t = 2 * (m["fgm"] - m["tpm"]) + 3 * m["tpm"] + m["ftm"]
    pts_o = 2 * (m["o_fgm"] - m["o_tpm"]) + 3 * m["o_tpm"] + m["o_ftm"]
    out["off_eff"] = 100 * pts_t / m["poss"]
    out["def_eff"] = 100 * pts_o / m["poss"]
    out["off_efg"] = (m["fgm"] + 0.5 * m["tpm"]) / m["fga"]
    out["def_efg"] = (m["o_fgm"] + 0.5 * m["o_tpm"]) / m["o_fga"]
    out["off_tov"] = m["to"] / m["poss"]
    out["def_tov"] = m["o_to"] / m["poss"]
    out["off_orb"] = m["orb"] / (m["orb"] + m["o_drb"])
    out["def_orb"] = m["o_orb"] / (m["o_orb"] + m["drb"])
    out["off_ftr"] = m["fta"] / m["fga"]
    out["def_ftr"] = m["o_fta"] / m["o_fga"]
    out["off_two"] = (m["fgm"] - m["tpm"]) / (m["fga"] - m["tpa"])
    out["def_two"] = (m["o_fgm"] - m["o_tpm"]) / (m["o_fga"] - m["o_tpa"])
    out["off_three"] = m["tpm"] / m["tpa"]
    out["def_three"] = m["o_tpm"] / m["o_tpa"]
    out["off_3par"] = m["tpa"] / m["fga"]
    out["def_3par"] = m["o_tpa"] / m["o_fga"]
    out["off_ftp"] = m["ftm"] / m["fta"]
    out["off_stl"] = m["o_stl"] / m["poss"]               # steals the opponent got (lower is better for this offense)
    out["def_stl"] = m["stl"] / m["poss"]
    out["off_blk"] = m["o_blk"] / (m["fga"] - m["tpa"])   # share of own 2-pt tries blocked
    out["def_blk"] = m["blk"] / (m["o_fga"] - m["o_tpa"])
    stat_cols = [c for c in out.columns if c not in ("game_id", "team")]
    out.loc[bad.values, stat_cols] = np.nan
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def build_team_games(games, game_stats):
    """Long format: one row per team per game (played or scheduled), in time order."""
    def side(team, opp, is_home, pf, pa, conf_t):
        return pd.DataFrame({"game_id": games["game_id"], "season": games["season"], "day": games["day"],
                             "tip": games["tip"], "team": games[team], "opp": games[opp], "is_home": is_home,
                             "neutral": games["neutral"], "conf_game": games["conf_game"],
                             "conf_id": pd.to_numeric(games[conf_t], errors="coerce"),
                             "pts_for": games[pf], "pts_against": games[pa], "completed": games["completed"]})
    tg = pd.concat([side("home_id", "away_id", 1, "home_score", "away_score", "home_conference_id"),
                    side("away_id", "home_id", 0, "away_score", "home_score", "away_conference_id")], ignore_index=True)
    n = tg.groupby(["team", "season"])["game_id"].transform("size")
    # D1 teams play ~30 games, non-D1 opponents only a handful. If a season's schedule is still short (data that only
    # lists games played so far), the bar drops with it: 40% of a typical (75th percentile) team's count, 3 to 12 games.
    q75 = tg.groupby(["team", "season"]).size().groupby(level=1).quantile(0.75)
    bar = tg["season"].map((0.4 * q75).clip(3, D1_MIN_GAMES))
    tg["d1"] = (n >= bar).astype(int)
    opp_d1 = tg[["game_id", "team", "d1"]].rename(columns={"team": "opp", "d1": "opp_d1"})
    tg = tg.merge(opp_d1, on=["game_id", "opp"], how="left")
    tg = tg.sort_values(["team", "tip", "game_id"]).reset_index(drop=True)
    tg = tg.merge(game_stats, on=["game_id", "team"], how="left")
    tg = tg.sort_values(["team", "tip", "game_id"]).reset_index(drop=True)
    tg["sgi"] = tg.groupby(["team", "season"]).cumcount()                  # 0 = team's first game of the season
    tg["gp"] = tg.groupby(["team", "season"])["completed"].transform(lambda x: x.astype(int).cumsum().shift(1)).fillna(0)
    return tg


def team_conferences(tg):
    """(team, season) -> the team's conference id that season (most common)."""
    d = tg.dropna(subset=["conf_id"])
    return d.groupby(["team", "season"])["conf_id"].agg(lambda x: x.mode().iat[0] if len(x.mode()) else np.nan).to_dict()


# ----------------------------------------------------------------------------
# 3. PRE-GAME FEATURES: rest days, games played
# ----------------------------------------------------------------------------
def add_continuity(tg, players):
    """
    Roster continuity, from games BEFORE this one this season:
      rot_cont  = share of the team's minutes played by players who also played for THIS team last season
      exp_share = share played by players who played D1 minutes ANYWHERE last season (returners + transfers)
    NaN before a team's first game (nothing known yet) and when last season's player data is missing.
    """
    tg["rot_cont"], tg["exp_share"] = np.nan, np.nan
    if players is None or players.empty:
        return tg
    p = players.copy()
    p["game_id"], p["team_id"], p["athlete_id"] = (p["game_id"].astype("int64"), p["team_id"].astype("int64"),
                                                   p["athlete_id"].astype("int64"))
    p["season"] = p["season"].astype(int)
    tot = p.groupby(["season", "team_id", "athlete_id"])["minutes"].sum().reset_index()
    prev_team = tot[["season", "team_id", "athlete_id"]].assign(season=tot["season"] + 1, ret=1.0)
    prev_any = tot[["season", "athlete_id"]].drop_duplicates().assign(season=lambda d: d["season"] + 1, exp=1.0)
    p = p.merge(prev_team, on=["season", "team_id", "athlete_id"], how="left").merge(prev_any, on=["season", "athlete_id"], how="left")
    p["m_ret"] = p["minutes"] * p["ret"].fillna(0.0)
    p["m_exp"] = p["minutes"] * p["exp"].fillna(0.0)
    have_prev = set(tot["season"] + 1)
    g = p.groupby(["game_id", "team_id"])[["minutes", "m_ret", "m_exp"]].sum().reset_index().rename(columns={"team_id": "team"})
    t = tg[["game_id", "team", "season"]].merge(g, on=["game_id", "team"], how="left")
    key = [tg["team"], tg["season"]]
    out = {}
    for c in ("minutes", "m_ret", "m_exp"):
        v = pd.Series(t[c].fillna(0.0).values, index=tg.index)
        out[c] = v.groupby(key).cumsum() - v                     # this season's EARLIER games only
    ok = (out["minutes"] > 0) & tg["season"].isin(have_prev)
    tg["rot_cont"] = (out["m_ret"] / out["minutes"]).where(ok)
    tg["exp_share"] = (out["m_exp"] / out["minutes"]).where(ok)
    return tg


INJ_NOW = {}           # {(game_id, team): {"rows": [...], "loss": float}} for the games being shown


def _norm(x):
    return re.sub(r"[^a-z ]", "", str(x).lower()).strip()


def add_injuries(tg, players, games, show_days=(), directory=None):
    """
    inj = estimated value a team is missing, in Game Score points per game:
        sum over missing rotation players of (his minutes per game / 40) x (his value per 40 - his team's bench value per 40)
    Player value per 40 = Hollinger Game Score per 40 minutes this season (earlier games only), shrunk toward last season's
    rate (or a bench-level rate for newcomers) by VAL_PRIOR_MIN minutes. Bench value = minutes-weighted value of the players
    outside the team's top BENCH_TOP in minutes (the backups who absorb the missing minutes).
    Missing = a rotation player (ROT_MPG+ over the last 5 games) who did not play in the team's previous game, plus
    MANUAL_INJURIES for the games being shown. Uses only information from before the game.
    """
    tg["inj"], tg["inj_n"] = 0.0, 0
    INJ_NOW.clear()
    if not INJURY_DETECT or players is None or players.empty or "points" not in players.columns:
        return tg
    p = players.dropna(subset=["minutes"]).copy()
    p = p[p["minutes"] > 0]
    f = lambda c: p[c].fillna(0.0)
    p["gs"] = (f("points") + 0.4 * f("field_goals_made") - 0.7 * f("field_goals_attempted")
               - 0.4 * (f("free_throws_attempted") - f("free_throws_made")) + 0.7 * f("offensive_rebounds")
               + 0.3 * f("defensive_rebounds") + f("steals") + 0.7 * f("assists") + 0.7 * f("blocks")
               - 0.4 * f("fouls") - f("turnovers"))
    for c in ("game_id", "team_id", "athlete_id"):
        p[c] = p[c].astype("int64")
    p["season"] = p["season"].astype(int)
    lr = float(p["gs"].sum() / p["minutes"].sum())                   # league Game Score per minute
    bench_rate = 0.75 * lr
    tot = p.groupby(["season", "athlete_id"])[["gs", "minutes"]].sum()
    prior = ((tot["gs"] + bench_rate * 200) / (tot["minutes"] + 200)).rename("r").reset_index()
    prior["season"] += 1
    prior_rate = {(s_, a_): r_ for s_, a_, r_ in zip(prior["season"], prior["athlete_id"], prior["r"])}
    gpl = p.groupby(["season", "athlete_id"]).size()
    prior_mpg = {(s_ + 1, a_): m_ / n_ for (s_, a_), m_, n_ in zip(tot.index, tot["minutes"].values, gpl.reindex(tot.index).values)}
    names = dict(zip(p["athlete_id"], p["athlete_display_name"]))
    box = {}
    for gid, tm, a, mn, gs_ in zip(p["game_id"].values, p["team_id"].values, p["athlete_id"].values, p["minutes"].values, p["gs"].values):
        box.setdefault((gid, tm), []).append((a, mn, gs_))
    show = set(pd.to_datetime(list(show_days))) if show_days else set()
    directory = directory or {}
    man_out, man_ok, man_ret = {}, {}, {}
    for dct, out_ in ((MANUAL_INJURIES, man_out), (MANUAL_AVAILABLE, man_ok), (MANUAL_RETURNING, man_ret)):
        for team_name, lst in dct.items():
            tids = [t for t, v in directory.items() if _norm(team_name) in (_norm(v.get("short")), _norm(v.get("name")), _norm(v.get("abbr")), _norm(v.get("loc")))]
            for t in tids:
                for item in lst:
                    nm, stt = (item, "Out") if isinstance(item, str) else (item[0], item[1])
                    out_.setdefault(t, {})[_norm(nm)] = str(stt).title()
            if not tids:
                print(f"  ! manual injury list: team '{team_name}' not found (use the name shown on the report cards)")
    inj = np.zeros(len(tg))
    inj_n = np.zeros(len(tg), dtype=int)
    order = tg.sort_values(["team", "tip", "game_id"])
    for team, d in order.groupby("team", sort=False):
        cur_s, recent, cum, last_played, last_ok, hist10, ngp = None, [], {}, set(), False, [], {}
        for idx, gid, s_, done, day in zip(d.index.values, d["game_id"].values, d["season"].values, d["completed"].values, d["day"].values):
            if gid < 0:
                continue
            if s_ != cur_s:
                cur_s, recent, cum, last_played, last_ok, hist10, ngp = s_, [], {}, set(), False, [], {}
            if recent:
                mp = {}
                for g_ in recent:
                    for a_, m_ in g_.items():
                        mp[a_] = mp.get(a_, 0.0) + m_
                mp = {a_: v_ / len(recent) for a_, v_ in mp.items()}
            else:
                mp = {}

            def val40(a_):
                gsum, msum = cum.get(a_, (0.0, 0.0))
                r0 = prior_rate.get((s_, a_), bench_rate)
                return 40 * (gsum + r0 * VAL_PRIOR_MIN) / (msum + VAL_PRIOR_MIN)
            ranked = sorted(mp.items(), key=lambda kv: -kv[1])
            bench = ranked[BENCH_TOP:]
            bm = sum(v_ for _, v_ in bench)
            bval = (sum(val40(a_) * v_ for a_, v_ in bench) / bm) if bm > 1 else 40 * bench_rate
            is_show = pd.Timestamp(day) in show
            missing = {}
            if last_ok:
                for a_, v_ in ranked:
                    if v_ >= ROT_MPG and a_ not in last_played:
                        missing[a_] = ("Missed last game", 1.0)
            returning = {}
            if is_show:
                mo, mk, mr = man_out.get(team, {}), man_ok.get(team, {}), man_ret.get(team, {})
                for a_, v_ in ranked:
                    nm = _norm(names.get(a_, ""))
                    if nm in mk or nm in mr:
                        missing.pop(a_, None)
                    elif nm in mo:
                        missing[a_] = (f"{mo[nm]} (added by hand)", INJ_STATUS.get(mo[nm], 1.0))
                if mr:
                    # anyone who has played for this team this season (or, failing that, last season anywhere)
                    pool = set(cum) | {a_ for a_, n_ in names.items() if _norm(n_) in mr and (s_, a_) in prior_rate}
                    for a_ in pool:
                        if _norm(names.get(a_, "")) not in mr:
                            continue
                        gp_ = ngp.get(a_, 0)
                        mpg_ = (cum[a_][1] / gp_) if gp_ else prior_mpg.get((s_, a_), 0.0)
                        miss_ = (sum(1 for g_ in hist10 if a_ not in g_) / len(hist10)) if hist10 else 0.0
                        gain = (mpg_ / 40) * max(val40(a_) - bval, 0.0) * miss_
                        returning[a_] = (mpg_, gain, miss_)
            loss = sum(sev * (mp[a_] / 40) * max(val40(a_) - bval, 0.0) for a_, (_, sev) in missing.items())
            loss -= sum(gn for _, gn, _ in returning.values())
            inj[tg.index.get_loc(idx)] = loss
            inj_n[tg.index.get_loc(idx)] = len(missing)
            if is_show:
                rows = []
                for a_, (mpg_, gn, miss_) in returning.items():
                    rows.append(dict(name=names.get(a_, str(a_)), mpg=mpg_, val=val40(a_), bench=bval,
                                     status=f"Returning (missed {miss_:.0%} of the last {len(hist10)} games)", loss=-gn))
                for a_, v_ in ranked[:10]:
                    if a_ in returning:
                        continue
                    st_, sev = missing.get(a_, ("Available", 0.0))
                    rows.append(dict(name=names.get(a_, str(a_)), mpg=v_, val=val40(a_), bench=bval, status=st_,
                                     loss=sev * (v_ / 40) * max(val40(a_) - bval, 0.0) if a_ in missing else 0.0))
                if is_show and (team in man_out or team in man_ret):
                    have = {_norm(r_["name"]) for r_ in rows}
                    for nm in list(man_out.get(team, {})) + list(man_ret.get(team, {})):
                        if nm not in have:
                            print(f"  ! manual injury: '{nm}' is not in {directory.get(team, {}).get('short', team)}'s recent rotation; ignored")
                INJ_NOW[(int(gid), int(team))] = {"rows": rows, "loss": loss, "bench": bval}
            if done and (gid, team) in box:
                rows_ = box[(gid, team)]
                recent = (recent + [{a_: m_ for a_, m_, _ in rows_}])[-5:]
                for a_, m_, g_ in rows_:
                    gsum, msum = cum.get(a_, (0.0, 0.0))
                    cum[a_] = (gsum + g_, msum + m_)
                    ngp[a_] = ngp.get(a_, 0) + 1
                hist10 = (hist10 + [{a_ for a_, _, _ in rows_}])[-10:]
                last_played, last_ok = {a_ for a_, _, _ in rows_}, True
            elif done:
                last_ok = False
    tg["inj"], tg["inj_n"] = inj, inj_n
    return tg


def add_rest(tg):
    prev = tg.groupby(["team", "season"])["day"].shift(1)
    rest = (tg["day"] - prev).dt.days
    tg["rest"] = rest.fillna(REST_CAP).clip(0, REST_CAP)
    return tg


# ----------------------------------------------------------------------------
# 4. FOUR FACTORS UNIT RATINGS (pre-game, opponent-adjusted, league percentiles)
# ----------------------------------------------------------------------------
OFF_UNIT_COLS = [f"ur_{k}_off_pre" for k in RANK_STAT_WEIGHTS]
DEF_UNIT_COLS = [f"ur_{k}_def_pre" for k in RANK_STAT_WEIGHTS]


def set_unit_config(mode, prior, adj, halflife=None):
    global UNIT_MODE, UNIT_PRIOR_GAMES, OPP_ADJUST, UNIT_HALFLIFE, UNIT_WINDOW_TEXT
    UNIT_MODE, UNIT_PRIOR_GAMES, OPP_ADJUST = mode, prior, adj
    if halflife is not None:
        UNIT_HALFLIFE = halflife
    if mode == "ewma":
        t = f"all earlier games, recent ones counting more (a game {UNIT_HALFLIFE:g} games old counts half)"
    else:
        t = "this season's earlier games"
        if prior > 0:
            t += f", blended early in the year with last season's averages (counted as {prior:g} games)"
    if adj > 0:
        t += f", adjusted for opponent strength ({adj:.0%})"
    UNIT_WINDOW_TEXT = t


set_unit_config(UNIT_MODE, UNIT_PRIOR_GAMES, OPP_ADJUST, UNIT_HALFLIFE)


def _fallback(v, tg):
    """Last season's league average of a stat (the neutral value for a team with no history)."""
    lg = v.groupby(tg["season"]).mean()
    prev = lg.shift(1).fillna(v.mean())
    return tg["season"].map(prev).astype(float)


def _pre_values(v, tg, mode, prior_games, halflife, fallback):
    """Pre-game value of a per-game stat using only EARLIER games."""
    v = v.astype(float)
    if mode == "ewma":
        r = v.groupby(tg["team"]).transform(
            lambda x: x.shift(1).ewm(halflife=halflife, min_periods=1, ignore_na=True).mean())
        return r.fillna(fallback)
    key = [tg["team"], tg["season"]]
    ok = v.notna().astype(float)
    one = v.fillna(0.0)
    s = one.groupby(key).cumsum() - one                   # sum of this season's EARLIER games
    n = ok.groupby(key).cumsum() - ok
    mean = s / n.where(n > 0)
    if prior_games > 0:
        tsm = v.groupby(key).mean()
        prev = tsm.groupby(level=0).shift(1)               # the team's average last season
        pr = pd.Series(prev.reindex(pd.MultiIndex.from_arrays(key)).values, index=v.index)
        blended = (n * mean.fillna(0.0) + prior_games * pr) / (n + prior_games)
        mean = blended.where(pr.notna(), mean)
    return mean.fillna(fallback)


def add_unit_inputs(tg, mode=None, prior_games=None, adj=None, halflife=None):
    """ur_<stat>_off_pre / ur_<stat>_def_pre: pre-game offense / defense values of every unit stat."""
    mode = UNIT_MODE if mode is None else mode
    prior_games = UNIT_PRIOR_GAMES if prior_games is None else prior_games
    adj = OPP_ADJUST if adj is None else adj
    hl = UNIT_HALFLIFE if halflife is None else halflife
    tg = tg.copy()
    key = pd.MultiIndex.from_arrays([tg["game_id"], tg["team"]])
    opp_key = pd.MultiIndex.from_arrays([tg["game_id"], tg["opp"]])
    dayk = [tg["season"], tg["day"]]
    d1 = tg["d1"] == 1

    def opp_of(x):
        s = pd.Series(x.values, index=key)
        s = s[~s.index.duplicated()]
        return pd.Series(s.reindex(opp_key).values, index=tg.index)

    for k in RANK_STAT_WEIGHTS:
        co, cd = UNIT_SOURCES[k]
        raw_o, raw_d = tg[co].astype(float), tg[cd].astype(float)
        fb_o, fb_d = _fallback(raw_o, tg), _fallback(raw_d, tg)
        po = _pre_values(raw_o, tg, mode, prior_games, hl, fb_o)
        pdf = _pre_values(raw_d, tg, mode, prior_games, hl, fb_d)
        if adj > 0:
            mo = po.where(d1).groupby(dayk).transform("mean").fillna(fb_o)
            md = pdf.where(d1).groupby(dayk).transform("mean").fillna(fb_d)
            d_off = (opp_of(pdf) - md).fillna(0.0)          # how much this opponent's defense usually gives up vs average
            d_def = (opp_of(po) - mo).fillna(0.0)           # how good this opponent's offense usually is vs average
            po = _pre_values(raw_o - adj * d_off, tg, mode, prior_games, hl, fb_o)
            pdf = _pre_values(raw_d - adj * d_def, tg, mode, prior_games, hl, fb_d)
        tg[f"ur_{k}_off_pre"], tg[f"ur_{k}_def_pre"] = po.values, pdf.values
    return tg


def add_unit_ranks(tg):
    """
    off_unit / def_unit: 0-1 rating = weighted average of the team's D1 percentile on each stat in RANK_STAT_WEIGHTS
    (1 = best). The reference group is every D1 team's pre-game value at the start of that week of the season, so
    nothing from the game itself (or later) is used.
    """
    tg = tg.copy()
    start = pd.to_datetime((tg["season"] - 1).astype(str) + "-10-01")
    tg["wk"] = ((tg["day"] - start).dt.days // 7).clip(lower=0)
    cols = OFF_UNIT_COLS + DEF_UNIT_COLS
    ref = tg[tg["d1"] == 1].groupby(["season", "wk", "team"])[cols].first().reset_index()
    teams = ref[["season", "team"]].drop_duplicates()
    weeks = ref[["season", "wk"]].drop_duplicates()
    grid = teams.merge(weeks, on="season").merge(ref, on=["season", "wk", "team"], how="left")
    grid = grid.sort_values(["team", "season", "wk"])
    grid[cols] = grid.groupby(["team", "season"])[cols].ffill()
    grid[cols] = grid.groupby(["team", "season"])[cols].bfill()
    ref_sorted = {kk: {c: np.sort(g_[c].dropna().values) for c in cols} for kk, g_ in grid.groupby(["season", "wk"])}
    tot = sum(RANK_STAT_WEIGHTS.values())
    off = np.zeros(len(tg))
    dfn = np.zeros(len(tg))
    for (s_, w_), idx in tg.groupby(["season", "wk"]).groups.items():
        rs = ref_sorted.get((s_, w_))
        if rs is None:                                     # a week with no D1 reference: use the closest earlier one
            cands = [kk for kk in ref_sorted if kk[0] == s_ and kk[1] <= w_] or [kk for kk in ref_sorted if kk[0] == s_]
            if not cands:
                off[tg.index.get_indexer(idx)] = 0.5
                dfn[tg.index.get_indexer(idx)] = 0.5
                continue
            rs = ref_sorted[max(cands)]
        pos = tg.index.get_indexer(idx)
        for k, w in RANK_STAT_WEIGHTS.items():
            for side, arr, hi in (("off", off, RANK_DIRECTION[k][0]), ("def", dfn, RANK_DIRECTION[k][1])):
                ref_v = rs[f"ur_{k}_{side}_pre"]
                x = tg[f"ur_{k}_{side}_pre"].values[pos]
                if len(ref_v) == 0:
                    p = np.full(len(x), 0.5)
                elif hi:
                    p = np.searchsorted(ref_v, x, side="right") / len(ref_v)
                else:
                    p = (len(ref_v) - np.searchsorted(ref_v, x, side="left")) / len(ref_v)
                p = np.where(np.isnan(x), 0.5, p)
                arr[pos] += w / tot * p
    tg["off_unit"], tg["def_unit"] = off, dfn
    return tg


def _unit_cv_score(tg):
    """Leave-one-season-out log-loss (tuning seasons) of a model that uses only the unit-rating matchups."""
    t = add_unit_ranks(tg)
    t = t[t["completed"] & (t["d1"] == 1) & (t["opp_d1"] == 1)]
    h = t[t["is_home"] == 1].set_index("game_id")
    a = t[t["is_home"] == 0].set_index("game_id")
    d = pd.DataFrame({"season": h["season"], "x1": h["off_unit"] - a["def_unit"], "x2": a["off_unit"] - h["def_unit"],
                      "hf": 1 - h["neutral"], "m": h["pts_for"] - h["pts_against"]}).dropna()
    d = d[(d["m"] != 0) & d["season"].isin(tune_seasons())]
    y = (d["m"] > 0).astype(int).values
    X = d[["x1", "x2", "hf"]].values
    seas = d["season"].values
    loss, hit = [], []
    for sn in np.unique(seas):
        te = seas == sn
        m = LogisticRegression(C=1.0, fit_intercept=False, max_iter=500).fit(X[~te], y[~te])
        p = m.predict_proba(X[te])[:, 1]
        loss.append(logloss_vec(y[te], p))
        hit.append((p >= 0.5) == y[te])
    return float(np.concatenate(loss).mean()), float(np.concatenate(hit).mean())


def choose_unit_config(tg):
    """Try the unit-stat settings on the TUNING seasons and keep the best. Returns the table for the report."""
    if not UNIT_AUTO_TUNE:
        set_unit_config(UNIT_MODE, UNIT_PRIOR_GAMES, OPP_ADJUST, UNIT_HALFLIFE)
        print(f"\nUnit-stat settings (fixed): {UNIT_WINDOW_TEXT}")
        return []
    print("\n" + "=" * 78)
    print(f"UNIT-STAT SETTINGS TEST (Four Factors ratings alone, tuning seasons {tune_seasons()[0]}-{tune_seasons()[-1]})")
    print("=" * 78)
    print(f"  {'window':<8s} {'half-life':>9s} {'last-season blend':>17s} {'opp. adjust':>11s} {'log-loss':>9s} {'accuracy':>9s}")
    res = []
    for mode in UNIT_MODE_CANDIDATES:
        for hl in (UNIT_HALFLIFE_CANDIDATES if mode == "ewma" else [None]):
            for pg in (UNIT_PRIOR_CANDIDATES if mode == "season" else [0]):
                for ad in UNIT_ADJ_CANDIDATES:
                    try:
                        ll, acc = _unit_cv_score(add_unit_inputs(tg, mode, pg, ad, hl or UNIT_HALFLIFE))
                    except Exception as ex:
                        print(f"  ! {mode}/{pg}/{ad}/{hl} skipped ({ex})")
                        continue
                    res.append((ll, acc, mode, pg, ad, hl))
                    print(f"  {mode:<8s} {('-' if hl is None else f'{hl:g}'):>9s} {pg:>17g} {ad:>11.1f} {ll:9.4f} {acc:9.3f}")
    if not res:
        return []
    ll, acc, mode, pg, ad, hl = min(res, key=lambda r: r[0])
    set_unit_config(mode, pg, ad, hl)
    print(f"  Using: {UNIT_WINDOW_TEXT}  (log-loss {ll:.4f}).")
    return res


# ----------------------------------------------------------------------------
# 5. ELO RATINGS
# ----------------------------------------------------------------------------
def tune_seasons():
    if TUNE_SEASONS:
        return list(TUNE_SEASONS)
    return [s for s in range(TRAIN_START_SEASON, WF_START_SEASON) if s not in EXCLUDE_FROM_TRAINING]


ELO_PRIOR_DEV = {}


def compute_elo(games, d1set, conf_map, carry=None, k=None, hfa=None, mov=None, early=None, regress=None):
    """
    Walk through every game in time order. Each game first RECORDS both teams' pre-game ratings (what the model
    sees), then updates them if the game has been played:
        p_home = 1 / (1 + 10 ** (-(home + HFA - away) / 400))        (no HFA at neutral sites)
        change = K x MOV multiplier x (result - p_home)
    MOV multiplier (538 basketball) = (|margin| + 3) ** 0.8 / (7.5 + 0.006 x winner's Elo edge).
    Each new season every rating moves toward its conference average (or the D1 average) by (1 - carryover).
    Non-D1 opponents have a fixed rating. Returns (per-game frame, current ratings).
    """
    carry = ELO_CARRYOVER if carry is None else carry
    k_ = ELO_K if k is None else k
    hfa_ = ELO_HOME_ADV if hfa is None else hfa
    mov_ = ELO_USE_MOV if mov is None else mov
    early_ = ELO_EARLY_BOOST if early is None else early
    reg = ELO_REGRESS_TO if regress is None else regress
    first = int(games["season"].min())
    ratings, gcount, rows, cur = {}, {}, [], None
    cols = zip(games["game_id"].values, games["season"].values, games["home_id"].values, games["away_id"].values,
               games["home_score"].values, games["away_score"].values, games["neutral"].values, games["completed"].values)
    for gid, s, h, a, hs, as_, neu, done in cols:
        if s != cur:
            if cur is not None and ratings:
                d1_now = [t for t in ratings if (t, s) in d1set]
                lg = np.mean([ratings[t] for t in d1_now]) if d1_now else ELO_START
                tgt = {}
                if reg == "conf":
                    by = {}
                    for t in d1_now:
                        by.setdefault(conf_map.get((t, s)), []).append(ratings[t])
                    tgt = {c: np.mean(v) for c, v in by.items() if c is not None and len(v) >= 4}
                for t in d1_now:
                    m_ = tgt.get(conf_map.get((t, s)), lg)
                    ELO_PRIOR_DEV[(t, s)] = ratings[t] - m_          # last season's edge over the regression target
                    ratings[t] = m_ + carry * (ratings[t] - m_)
            gcount, cur = {}, s
        hd, ad = (h, s) in d1set, (a, s) in d1set
        if hd and h not in ratings:
            ratings[h] = ELO_START if s == first else ELO_NEW_TEAM
        if ad and a not in ratings:
            ratings[a] = ELO_START if s == first else ELO_NEW_TEAM
        rh = ratings[h] if hd else ELO_NON_D1
        ra = ratings[a] if ad else ELO_NON_D1
        gap = rh + (0.0 if neu else hfa_) - ra
        p = 1.0 / (1.0 + 10 ** (-gap / 400.0))
        rows.append((gid, rh, ra, p))
        if done:
            margin = hs - as_
            actual = 1.0 if margin > 0 else 0.0
            mult = 1.0
            if mov_ and margin != 0:
                wgap = gap if margin > 0 else -gap
                mult = (abs(margin) + 3) ** 0.8 / max(7.5 + 0.006 * wgap, 1.0)
            kk = k_ * (early_ if min(gcount.get(h, 0), gcount.get(a, 0)) < ELO_EARLY_GAMES else 1.0)
            delta = kk * mult * (actual - p)
            if hd:
                ratings[h] = rh + delta
            if ad:
                ratings[a] = ra - delta
            gcount[h] = gcount.get(h, 0) + 1
            gcount[a] = gcount.get(a, 0) + 1
    out = pd.DataFrame(rows, columns=["game_id", "h_elo_pre", "a_elo_pre", "elo_home_prob"])
    out["elo_diff"] = out["h_elo_pre"] - out["a_elo_pre"]
    return out, ratings


def _eval_mask(games):
    """Completed D1-vs-D1 games of the tuning seasons."""
    return games["completed"].values & games["both_d1"].values & games["season"].isin(tune_seasons()).values


def choose_elo_params(games, d1set, conf_map):
    """Coarse grid, then one-setting-at-a-time refinement, scored by Elo-only log-loss on the tuning seasons."""
    base = dict(carry=ELO_CARRYOVER, k=ELO_K, hfa=ELO_HOME_ADV, mov=ELO_USE_MOV, early=ELO_EARLY_BOOST, regress=ELO_REGRESS_TO)
    if not ELO_AUTO_TUNE:
        return base, []
    mask = _eval_mask(games)
    y = (games["result"].values[mask] > 0).astype(float)
    cache = {}

    def score(p):
        key = tuple(sorted(p.items()))
        if key not in cache:
            el, _ = compute_elo(games, d1set, conf_map, **p)
            pr = el["elo_home_prob"].values[mask]
            cache[key] = (float(logloss_vec(y, pr).mean()), float(((pr >= 0.5) == y).mean()))
        return cache[key]

    print("\n" + "=" * 78)
    print("ELO TUNING TEST (Elo alone, tuning seasons only)")
    print("=" * 78)
    t0 = time.time()
    default = score(base)
    grid = []
    for c in ELO_CARRY_CANDIDATES:
        for rg in ELO_REGRESS_CANDIDATES:
            for k in ELO_K_CANDIDATES:
                for h in ELO_HFA_CANDIDATES:
                    for mv in ELO_MOV_CANDIDATES:
                        p = {**base, "carry": c, "regress": rg, "k": k, "hfa": h, "mov": mv, "early": 1.0}
                        grid.append(score(p) + (p,))
    grid.sort(key=lambda r: r[0])
    best_ll, _, best = grid[0]
    best = dict(best)
    steps = [("k", lambda v: [v * f for f in (0.7, 0.85, 1.15, 1.3)]),
             ("hfa", lambda v: [v + d for d in (-20, -10, 10, 20)]),
             ("carry", lambda v: [min(max(v + d, 0.0), 1.0) for d in (-0.1, -0.05, 0.05, 0.1)]),
             ("early", lambda v: ELO_EARLY_CANDIDATES)]
    for _ in range(3):
        improved = False
        for name, cands in steps:
            for v in cands(best[name]):
                t = {**best, name: v}
                ll = score(t)[0]
                if ll < best_ll - 1e-5:
                    best, best_ll, improved = t, ll, True
        if not improved:
            break
    acc = score(best)[1]
    print(f"  {'log-loss':>9s} {'accuracy':>9s}  settings")
    print(f"  {default[0]:9.4f} {default[1]:9.3f}  starting values in the CONFIG")
    print(f"  {grid[0][0]:9.4f} {grid[0][1]:9.3f}  after the coarse grid")
    print(f"  {best_ll:9.4f} {acc:9.3f}  final")
    print(f"  Chosen: K {best['k']:.1f}, home court {best['hfa']:.0f} Elo pts, margin of victory {best['mov']}, "
          f"carryover {best['carry']:.2f} toward the {'conference' if best['regress'] == 'conf' else 'D1'} average, "
          f"early-season K x{best['early']:g}   ({time.time() - t0:.0f}s, {len(cache)} settings tried)")
    info = [("Starting values", default[0], default[1]), ("Tuned", best_ll, acc)]
    return best, info


# ----------------------------------------------------------------------------
# 6. ADJUSTED EFFICIENCY RATINGS (offense / defense per 100 possessions + tempo)
# ----------------------------------------------------------------------------
def compute_eff_ratings(games, d1set, lr=None, carry=None, hfa=None, tlr=None):
    """
    Each team has o (points per 100 possessions it scores above average), d (points per 100 its defense ALLOWS above
    average; lower is better) and t (possessions above average). Before a game:
        home pts/100 = L + o_home + d_away + hfa/2      away pts/100 = L + o_away + d_home - hfa/2
        possessions  = T + t_home + t_away
    After the game each rating moves by lr x (actual - expected). Ratings shrink toward 0 each new season.
    Returns (per-game frame, current {team: (AdjO, AdjD, AdjT)}).
    """
    lr = EFF_LR if lr is None else lr
    carry = EFF_CARRY if carry is None else carry
    hfa = EFF_HFA if hfa is None else hfa
    tlr = EFF_TEMPO_LR if tlr is None else tlr
    first = int(games["season"].min())
    o, d, t = {}, {}, {}
    L, T = LEAGUE_ORTG, LEAGUE_TEMPO
    rows, cur = [], None
    # The predictions only use differences (o_home + d_away - o_away - d_home), so "every offense +c, every defense -c"
    # changes nothing. The DISPLAYED AdjO / AdjD are therefore centered on this season's D1 average.
    d1_by_season = {}
    for tm_, s_ in d1set:
        d1_by_season.setdefault(s_, set()).add(tm_)
    members, so, sd, st_, nm = set(), 0.0, 0.0, 0.0, 1
    cols = zip(games["game_id"].values, games["season"].values, games["home_id"].values, games["away_id"].values,
               games["home_score"].values, games["away_score"].values, games["neutral"].values,
               games["completed"].values, games["poss"].values, games["tempo"].values)
    for gid, s, h, a, hs, as_, neu, done, poss, tempo in cols:
        if s != cur:
            if cur is not None:
                for dct, c_ in ((o, carry), (d, carry), (t, max(carry, 0.7))):
                    for k_ in dct:
                        dct[k_] *= c_
            cur = s
            members = d1_by_season.get(s, set())
            for tm in members:
                if tm not in o:
                    v = 0.0 if s == first else 4.0
                    o[tm], d[tm], t[tm] = -v, v, 0.0
            nm = max(len(members), 1)
            so, sd, st_ = sum(o[x] for x in members), sum(d[x] for x in members), sum(t[x] for x in members)
        for tm in (h, a):
            if tm not in o:
                if (tm, s) in d1set:
                    v = 0.0 if s == first else 4.0
                else:
                    v = EFF_NON_D1
                o[tm], d[tm], t[tm] = -v, v, 0.0
        hh = 0.0 if neu else hfa / 2
        eh = L + o[h] + d[a] + hh
        ea = L + o[a] + d[h] - hh
        ep = T + t[h] + t[a]
        m100 = (o[h] + d[a]) - (o[a] + d[h])
        mo, md, mt = so / nm, sd / nm, st_ / nm
        rows.append((gid, m100, m100 * ep / 100, (eh + ea) * ep / 100, ep,
                     L + o[h] - mo, L + d[h] - md, T + t[h] - mt, L + o[a] - mo, L + d[a] - md, T + t[a] - mt, np.nan))
        if done:
            pa_ = poss if poss == poss and poss > 0 else ep
            ah, aa = 100 * hs / pa_, 100 * as_ / pa_
            e1, e2 = ah - eh, aa - ea
            o[h] += lr * e1
            d[a] += lr * e1
            o[a] += lr * e2
            d[h] += lr * e2
            hm, am = h in members, a in members
            so += lr * (e1 * hm + e2 * am)
            sd += lr * (e1 * am + e2 * hm)
            if tempo == tempo and tempo > 0:
                te = tempo - ep
                t[h] += tlr * te
                t[a] += tlr * te
                st_ += tlr * te * (hm + am)
            res = (ah - aa) - (eh - ea)
            rows[-1] = rows[-1][:-1] + (res,)
            if (h, s) in d1set and (a, s) in d1set:
                L += 0.002 * ((ah + aa) / 2 - L)
                if tempo == tempo and tempo > 0:
                    T += 0.002 * (tempo - T)
    out = pd.DataFrame(rows, columns=["game_id", "er_m100", "er_pm", "er_tot", "er_poss",
                                      "h_adjo", "h_adjd", "h_adjt", "a_adjo", "a_adjd", "a_adjt", "er_res"])
    mo, md, mt = so / nm, sd / nm, st_ / nm
    current = {tm: (L + o[tm] - mo, L + d[tm] - md, T + t[tm] - mt) for tm in o}
    return out, current


def choose_eff_params(games, d1set):
    """Pick speed / carryover / home edge by margin error on the TUNING seasons."""
    if not EFF_AUTO_TUNE:
        return dict(lr=EFF_LR, carry=EFF_CARRY, hfa=EFF_HFA), None
    mask = _eval_mask(games)
    act = games["result"].values[mask]
    best = None
    t0 = time.time()
    for lr in EFF_LR_CANDIDATES:
        for c in EFF_CARRY_CANDIDATES:
            for h in EFF_HFA_CANDIDATES:
                er, _ = compute_eff_ratings(games, d1set, lr, c, h)
                hf = (1 - games["neutral"].values[mask]) * h * er["er_poss"].values[mask] / 100
                pred = er["er_pm"].values[mask] + hf
                err = float(np.nanmean((act - pred) ** 2))
                if best is None or err < best[0]:
                    best = (err, lr, c, h)
    err, lr, c, h = best
    print("\n" + "=" * 78)
    print("ADJUSTED EFFICIENCY RATINGS (tuned on the tuning seasons)")
    print("=" * 78)
    print(f"  Chosen: rating speed {lr:g}, season carryover {c:g}, home edge {h:g} pts per 100 possessions "
          f"(margin error {np.sqrt(err):.2f} pts RMS, {time.time() - t0:.0f}s)")
    return dict(lr=lr, carry=c, hfa=h), float(np.sqrt(err))


# ----------------------------------------------------------------------------
# 7. MATCHUP FEATURES
# ----------------------------------------------------------------------------
def add_form(tg, eff):
    """Recent form: average of how much the team beat its EXPECTED margin (per 100 possessions) in its last
    FORM_GAMES games this season (prior games only)."""
    r = eff[["game_id", "er_res"]]
    tg = tg.merge(r, on="game_id", how="left")
    tg["res"] = np.where(tg["is_home"] == 1, tg["er_res"], -tg["er_res"])
    tg["form"] = (tg.groupby(["team", "season"])["res"]
                    .transform(lambda x: x.shift(1).rolling(FORM_GAMES, min_periods=1).mean()).fillna(0.0))
    return tg.drop(columns=["er_res"])


TEAM_FEATS = ["off_unit", "def_unit", "rest", "form", "gp", "d1", "sgi", "rot_cont", "exp_share", "elo_dev", "inj", "inj_n"] + OFF_UNIT_COLS + DEF_UNIT_COLS


def build_matchups(games, tg, elo, eff):
    h = tg[tg["is_home"] == 1][["game_id"] + TEAM_FEATS].rename(columns={c: "h_" + c for c in TEAM_FEATS})
    a = tg[tg["is_home"] == 0][["game_id"] + TEAM_FEATS].rename(columns={c: "a_" + c for c in TEAM_FEATS})
    base = ["game_id", "season", "day", "tip", "home_id", "away_id", "home_score", "away_score", "result", "neutral",
            "conf_game", "completed", "both_d1", "season_type", "notes_headline", "venue_full_name",
            "venue_address_city", "venue_address_state", "poss"]
    g = games[[c for c in base if c in games.columns]].merge(h, on="game_id").merge(a, on="game_id")
    g = g.merge(elo, on="game_id", how="left").merge(eff.drop(columns=["er_res"]), on="game_id", how="left")
    g["home_field"] = 1 - g["neutral"]
    g["rk_home_off_vs_away_def"] = g["h_off_unit"] - g["a_def_unit"]
    g["rk_away_off_vs_home_def"] = g["a_off_unit"] - g["h_def_unit"]
    g["unit_edge"] = (g["rk_home_off_vs_away_def"] - g["rk_away_off_vs_home_def"]) * 100
    g["form_diff"] = g["h_form"] - g["a_form"]
    g["rest_diff"] = g["h_rest"] - g["a_rest"]
    g["b2b_diff"] = (g["a_rest"] <= 1).astype(float) - (g["h_rest"] <= 1).astype(float)
    g["conf_x_elo"] = g["conf_game"] * g["elo_diff"]
    g["conf_home"] = g["conf_game"] * g["home_field"]
    early_w = np.clip(1.0 - np.minimum(g["h_gp"], g["a_gp"]) / EARLY_GAMES, 0.0, 1.0)   # 1 at the season opener, 0 after
    g["early_elo"] = early_w * g["elo_diff"]                                            # EARLY_GAMES games
    g["early_er"] = early_w * g["er_pm"]
    # roster continuity: a team that kept its players should keep more of last season's edge than the ratings assume
    cw = np.clip(1.0 - np.minimum(g["h_gp"], g["a_gp"]) / CONT_GAMES, 0.0, 1.0)
    hc, ac = g["h_rot_cont"].fillna(CONT_MEAN), g["a_rot_cont"].fillna(CONT_MEAN)
    he, ae = g["h_exp_share"].fillna(0.75), g["a_exp_share"].fillna(0.75)
    g["cont_x_dev"] = cw * ((hc - CONT_MEAN) * g["h_elo_dev"].fillna(0) - (ac - CONT_MEAN) * g["a_elo_dev"].fillna(0)) / 100
    g["cont_diff"] = cw * (hc - ac)
    g["inj_diff"] = g["a_inj"] - g["h_inj"]                     # missing value, away minus home (positive favors home)
    g["exp_diff"] = cw * (he - ae)
    g["tot_eff_off"] = g["h_ur_eff_off_pre"] + g["a_ur_eff_off_pre"]
    g["tot_eff_def"] = g["h_ur_eff_def_pre"] + g["a_ur_eff_def_pre"]
    g["home_win"] = np.where(g["completed"] & (g["result"] != 0), (g["result"] > 0).astype(float), np.nan)
    return g


FEATURE_GROUPS = {
    "Elo": ["elo_diff"],
    "Efficiency ratings": ["er_pm"],
    "Four Factors units": ["rk_home_off_vs_away_def", "rk_away_off_vs_home_def"],
    "Recent form": ["form_diff"],
    "Rest": ["rest_diff", "b2b_diff"],
    "Conference game": ["conf_x_elo", "conf_home"],
    "Early season": ["early_elo", "early_er"],
    "Roster continuity": ["cont_x_dev", "cont_diff", "exp_diff"],
    "Injuries": ["inj_diff"],
    "Home court": ["home_field"],
}
FEATURE_LABELS = {
    "elo_diff": "Elo rating (home - away)",
    "er_pm": "Adjusted efficiency margin, projected points (home - away)",
    "rk_home_off_vs_away_def": "Home offense rating vs away defense rating",
    "rk_away_off_vs_home_def": "Away offense rating vs home defense rating (negative expected)",
    "form_diff": "Recent form vs expectation (home - away)",
    "rest_diff": "Days of rest (home - away)",
    "b2b_diff": "Back-to-back days (away - home)",
    "conf_x_elo": "Conference game x Elo gap",
    "conf_home": "Home court in a conference game (extra)",
    "early_elo": "Elo gap x early-season weight (lets ratings count less early)",
    "early_er": "Efficiency gap x early-season weight",
    "cont_x_dev": "Returning minutes x last season's edge (early season)",
    "cont_diff": "Returning minutes share (home - away, early season)",
    "exp_diff": "Minutes by players with D1 experience last season (home - away, early season)",
    "inj_diff": "Missing players' value over their backups (away - home)",
    "home_field": "Home court (0 at neutral sites)",
}
TOTAL_FEATS = ["er_tot", "er_poss", "tot_eff_off", "tot_eff_def"]


def all_model_features():
    return [f for grp in FEATURE_GROUPS.values() for f in grp]


def model_features():
    if ACTIVE_FEATURE_GROUPS is None:
        return all_model_features()
    return [f for g_ in ACTIVE_FEATURE_GROUPS for f in FEATURE_GROUPS[g_]]


# ----------------------------------------------------------------------------
# 8. MODEL
# ----------------------------------------------------------------------------
def make_model(C=None):
    """No intercept and no centering: every feature is home-minus-away (or 0 at a neutral site), so flipping the two
    teams at a neutral site gives exactly 1 - p."""
    return Pipeline([("scaler", StandardScaler(with_mean=False)),
                     ("clf", LogisticRegression(C=MODEL_C if C is None else C, fit_intercept=False, max_iter=3000))])


def make_total_model():
    return Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=TOTAL_RIDGE_ALPHA))])


def feature_multipliers(feats):
    mult = {f: 1.0 for f in feats}
    for grp, m in GROUP_WEIGHT_MULTIPLIERS.items():
        for f in FEATURE_GROUPS.get(grp, []):
            if f in mult:
                mult[f] = float(m)
    return mult


class AdjustedLogit:
    """Logistic regression whose learned weights are scaled by per-group multipliers, then re-calibrated
    (one slope, no intercept) so the probabilities stay sensible."""
    def __init__(self, feats, multipliers=None, C=None):
        self.feats = list(feats)
        multipliers = multipliers or {}
        self.mult = np.array([multipliers.get(f, 1.0) for f in self.feats])
        self.base = make_model(C)

    def fit(self, X, y):
        self.base.fit(X[self.feats], y)
        self.coef_raw = self.base.named_steps["clf"].coef_[0]
        z = self._score(X).reshape(-1, 1)
        if np.allclose(self.mult, 1.0):
            self.slope = 1.0
        else:
            self.slope = float(LogisticRegression(C=1e6, fit_intercept=False, max_iter=1000).fit(z, y).coef_[0][0])
        return self

    def _score(self, X):
        Xs = self.base.named_steps["scaler"].transform(X[self.feats])
        return Xs @ (self.coef_raw * self.mult)

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.slope * self._score(X)))
        return np.column_stack([1.0 - p, p])

    @property
    def effective_coefs(self):
        return pd.Series(self.coef_raw * self.mult * self.slope, index=self.feats)


def _early_w(X):
    gp = np.minimum(X["h_gp"].values, X["a_gp"].values) if "h_gp" in X else np.full(len(X), 99.0)
    return np.clip(1.0 - gp / 10.0, 0.0, 1.0)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_calibration(p, X, y):
    """Confidence correction: new log-odds = (a + b x early-season weight) x model log-odds. a < 1 shrinks
    overconfident probabilities toward 50%. Fit on OUT-OF-SAMPLE predictions only."""
    z = _logit(p)
    ew = _early_w(X)
    m = LogisticRegression(C=1e4, fit_intercept=False, max_iter=1000).fit(np.c_[z, z * ew], y)
    return float(m.coef_[0][0]), float(m.coef_[0][1])


class Calibrated:
    """A model plus the confidence correction (a, b)."""
    def __init__(self, model, a=1.0, b=0.0):
        self.model, self.a, self.b = model, a, b

    def __getattr__(self, k):
        return getattr(self.model, k)

    def scale(self, X):
        return self.a + self.b * _early_w(X)

    def predict_proba(self, X):
        z = _logit(self.model.predict_proba(X)[:, 1]) * self.scale(X)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p, p])


def trainable(games):
    """Completed D1-vs-D1 games usable for training and testing."""
    return games[games["home_win"].notna() & games["both_d1"] & (games["season"] >= TRAIN_START_SEASON)
                 & ~games["season"].isin(EXCLUDE_FROM_TRAINING)]


def _fit_eval(feats, train, test):
    y = test["home_win"].values.astype(float)
    if feats:
        p = make_model().fit(train[feats], train["home_win"]).predict_proba(test[feats])[:, 1]
    else:
        p = np.full(len(test), 0.5)
    return logloss_vec(y, p), ((p >= 0.5) == y).astype(float)


def _by_season(feats, done, seasons):
    loss, hit = [], []
    for sn in seasons:
        l, h = _fit_eval(feats, done[done["season"] != sn], done[done["season"] == sn])
        loss.append(l)
        hit.append(h)
    return np.concatenate(loss), np.concatenate(hit)


def feature_ablation(games):
    """
    Factor-accuracy test on every completed season (each season predicted by a model trained on the OTHER seasons,
    results pooled). Groups: each ALONE (plus home court, which every model needs) and the full model WITHOUT it.
    Stats: each unit-rating stat alone (its offense-vs-defense edge for both teams, plus home court).
    Lower log-loss = better. 'noise' = one standard error of the difference.
    """
    done = trainable(games).copy()
    if PREDICT_SEASON in done["season"].values:
        done = done[done["season"] != PREDICT_SEASON]
    seasons = sorted(done["season"].unique())
    if len(seasons) < 2:
        return None
    n = len(done)
    feats = model_features()
    hf = ["home_field"]
    base_loss, base_hit = _by_season(hf, done, seasons)            # "home court only" is the baseline
    full_loss, full_hit = _by_season(feats, done, seasons)
    rows = []
    for grp, cols in FEATURE_GROUPS.items():
        if grp == "Home court":
            continue
        a_loss, a_hit = _by_season(list(dict.fromkeys(cols + hf)), done, seasons)
        w_feats = [c for c in feats if c not in cols]
        in_model = any(c in feats for c in cols)
        w_loss = _by_season(w_feats, done, seasons)[0] if in_model else _by_season(feats + cols, done, seasons)[0]
        d_alone = a_loss - base_loss
        d_w = (w_loss - full_loss) if in_model else (full_loss - w_loss)
        rows.append({"name": grp, "in_model": in_model, "alone_ll": a_loss.mean(), "alone_acc": float(a_hit.mean()),
                     "alone_vs_none": d_alone.mean(), "noise": d_alone.std(ddof=1) / np.sqrt(n),
                     "cost_of_removing": d_w.mean(), "cost_noise": d_w.std(ddof=1) / np.sqrt(n)})
    stat_rows = []
    for k in RANK_STAT_WEIGHTS:
        so = 1.0 if RANK_DIRECTION[k][0] else -1.0
        sd_ = 1.0 if RANK_DIRECTION[k][1] else -1.0
        done[f"edge_{k}"] = (so * (done[f"h_ur_{k}_off_pre"] - done[f"a_ur_{k}_off_pre"])
                             + sd_ * (done[f"h_ur_{k}_def_pre"] - done[f"a_ur_{k}_def_pre"])).fillna(0.0)
        l, h_ = _by_season([f"edge_{k}"] + hf, done, seasons)
        d = l - base_loss
        stat_rows.append({"name": RANK_STAT_NAMES[k], "key": k, "alone_ll": l.mean(), "alone_acc": float(h_.mean()),
                          "alone_vs_none": d.mean(), "noise": d.std(ddof=1) / np.sqrt(n)})
    return {"n_test": n, "seasons": f"{seasons[0]}-{seasons[-1]}", "none_ll": base_loss.mean(),
            "none_acc": float(base_hit.mean()), "full_ll": full_loss.mean(), "full_acc": float(full_hit.mean()),
            "groups": rows, "stats": stat_rows}


def print_ablation(ab):
    if not ab:
        return
    print("\n" + "=" * 78)
    print(f"WHICH FACTORS PREDICT WINNERS?  ({ab['n_test']:,} games, seasons {ab['seasons']})")
    print("Each season is predicted by a model trained on the other seasons; learned weights only.")
    print("=" * 78)
    print(f"Home court only        : log-loss {ab['none_ll']:.4f}, accuracy {ab['none_acc']:.3f}")
    print(f"All active factors     : log-loss {ab['full_ll']:.4f}, accuracy {ab['full_acc']:.3f}")
    print(f"\n  {'group':<20s} {'in model':>8s} {'alone LL':>9s} {'alone acc':>9s} {'vs none':>9s} {'noise':>7s}  {'value in full model':>19s}")
    for r in sorted(ab["groups"], key=lambda r: r["alone_vs_none"]):
        print(f"  {r['name']:<20s} {('yes' if r['in_model'] else 'no'):>8s} {r['alone_ll']:9.4f} {r['alone_acc']:9.3f} "
              f"{r['alone_vs_none']:+9.4f} {r['noise']:7.4f}  {r['cost_of_removing']:+12.4f} (±{r['cost_noise']:.4f})")
    print("  (vs none: negative = better than home court alone. value in full model: positive = it helps; for a group")
    print("   that is NOT in the model it is the gain from ADDING it)")
    print(f"\n  {'unit stat (alone)':<30s} {'LL':>9s} {'acc':>7s} {'vs none':>9s} {'noise':>7s}")
    for r in sorted(ab["stats"], key=lambda r: r["alone_vs_none"]):
        print(f"  {r['name']:<30s} {r['alone_ll']:9.4f} {r['alone_acc']:7.3f} {r['alone_vs_none']:+9.4f} {r['noise']:7.4f}")


def _walk_forward(done, feats, use_mult=True, C=None):
    """Pooled walk-forward predictions: each season >= WF_START_SEASON by a model trained on earlier seasons only."""
    ps, idx = [], []
    for s_ in sorted(done["season"].unique()):
        if s_ < WF_START_SEASON:
            continue
        tr, te = done[done["season"] < s_], done[done["season"] == s_]
        if len(tr) < WF_MIN_TRAIN_GAMES or te.empty:
            continue
        m_ = AdjustedLogit(feats, feature_multipliers(feats) if use_mult else None, C).fit(tr, tr["home_win"])
        ps.append(m_.predict_proba(te)[:, 1])
        idx.append(te.index.values)
    if not ps:
        return None
    return pd.Series(np.concatenate(ps), index=np.concatenate(idx))


def compare_configs(games):
    """Walk-forward comparison of simpler / different set-ups on the SAME games."""
    done = trainable(games)
    done = done[done["season"] < PREDICT_SEASON]
    active = list(ACTIVE_FEATURE_GROUPS) if ACTIVE_FEATURE_GROUPS else list(FEATURE_GROUPS)
    cfgs = [("Current model (as shipped)", active, True, None)]
    for g_ in active:
        if g_ != "Home court":
            cfgs.append((f"Without {g_}", [x for x in active if x != g_], True, None))
    for g_ in FEATURE_GROUPS:
        if g_ not in active:
            cfgs.append((f"Adding {g_}", active + [g_], True, None))
    cfgs += [("Same factors, weaker regularization (C = 5)", active, False, 5.0),
             ("Same factors, stronger regularization (C = 0.05)", active, False, 0.05),
             ("Elo + Efficiency ratings", ["Elo", "Efficiency ratings", "Home court"], False, None),
             ("Efficiency ratings only", ["Efficiency ratings", "Home court"], False, None),
             ("Four Factors units only", ["Four Factors units", "Home court"], False, None),
             ("Elo only", ["Elo", "Home court"], False, None)]
    out, ref = [], None
    for name, groups, use_mult, C in cfgs:
        feats = [f for g_ in groups for f in FEATURE_GROUPS[g_]]
        p = _walk_forward(done, feats, use_mult, C)
        if p is None:
            continue
        y = done.loc[p.index, "home_win"].values
        ll = logloss_vec(y, p.values)
        if ref is None:
            ref = ll
        d = ll - ref
        out.append({"name": name, "n_feats": len(feats), "acc": float(((p.values >= 0.5) == y).mean()), "ll": float(ll.mean()),
                    "vs_cur": float(d.mean()), "noise": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0})
    return {"rows": out, "n": int(len(ref)) if ref is not None else 0}


def print_compare(cp):
    if not cp or not cp["rows"]:
        return
    print("\n" + "=" * 78)
    print(f"MODEL SET-UP COMPARISON (walk-forward, {cp['n']:,} games, same games for every row)")
    print("=" * 78)
    print(f"  {'set-up':<50s} {'acc':>6s} {'log-loss':>9s} {'vs current':>11s} {'noise':>7s}")
    for r in cp["rows"]:
        print(f"  {r['name']:<50s} {r['acc'] * 100:5.1f}% {r['ll']:9.4f} {r['vs_cur']:+11.4f} {r['noise']:7.4f}")
    print("  vs current: negative = better than the shipped model; gaps under ~2x noise are not reliable.")


def train_and_evaluate(games):
    global MARGIN_SD
    feats = model_features()
    done_all = trainable(games)
    done = done_all[done_all["season"] < PREDICT_SEASON]          # the season being predicted stays out of the test
    mults = feature_multipliers(feats)
    report = {"feat_mult": mults, "group_mult": {g: m for g, m in GROUP_WEIGHT_MULTIPLIERS.items()
                                                 if m != 1.0 and (ACTIVE_FEATURE_GROUPS is None or g in ACTIVE_FEATURE_GROUPS)}}
    print("\n" + "=" * 78)
    print("MODEL VALIDATION")
    print("=" * 78)
    p = _walk_forward(done, feats, True)
    test = done.loc[p.index].copy() if p is not None else done.iloc[0:0].copy()
    cal = (1.0, 0.0)
    if len(test):
        test["wf_p_uncal"] = p.values
        if CALIBRATE:
            # each season's correction is fit on the walk-forward predictions of EARLIER seasons only
            pc = test["wf_p_uncal"].values.copy()
            seas = sorted(test["season"].unique())
            for s_ in seas[1:]:
                prev = test["season"] < s_
                a_, b_ = fit_calibration(test.loc[prev, "wf_p_uncal"].values, test[prev], test.loc[prev, "home_win"].values)
                cur_ = (test["season"] == s_).values
                pc[cur_] = 1 / (1 + np.exp(-_logit(pc[cur_]) * (a_ + b_ * _early_w(test[cur_]))))
            test["wf_p"] = pc
            m1 = test["season"] > seas[0]
            yy = test.loc[m1, "home_win"].values
            report["cal_test"] = (int(m1.sum()), float(logloss_vec(yy, test.loc[m1, "wf_p_uncal"].values).mean()),
                                  float(logloss_vec(yy, test.loc[m1, "wf_p"].values).mean()),
                                  float(((test.loc[m1, "wf_p_uncal"] >= 0.5) == yy).mean()),
                                  float(((test.loc[m1, "wf_p"] >= 0.5) == yy).mean()))
            cal = fit_calibration(test["wf_p_uncal"].values, test, test["home_win"].values)
            report["cal"] = cal
            print(f"Confidence correction: log-odds x {cal[0]:.3f} (x {cal[0] + cal[1]:.3f} at a team's first game). "
                  f"On {report['cal_test'][0]:,} later walk-forward games: log-loss {report['cal_test'][1]:.4f} -> {report['cal_test'][2]:.4f}")
        else:
            test["wf_p"] = test["wf_p_uncal"]
        test["wf_p_raw"] = _walk_forward(done, feats, False).reindex(test.index).values
        y = test["home_win"].values
        report.update(n_test=len(test), wf_first=int(test["season"].min()), wf_last=int(test["season"].max()))
        report["acc"] = accuracy_score(y, (test["wf_p"] >= 0.5).astype(int))
        report["ll"] = log_loss(y, test["wf_p"], labels=[0, 1])
        report["acc_raw"] = accuracy_score(y, (test["wf_p_raw"] >= 0.5).astype(int))
        report["ll_raw"] = log_loss(y, test["wf_p_raw"], labels=[0, 1])
        hw = test["home_field"] == 1
        report["base_acc"] = float((test.loc[hw, "home_win"] == 1).mean()) if hw.any() else 0.5
        pe = test["elo_home_prob"].clip(1e-6, 1 - 1e-6)
        report["elo_acc"] = accuracy_score(y, (pe >= 0.5).astype(int))
        report["elo_ll"] = log_loss(y, pe, labels=[0, 1])
        z = np.array([NormalDist().inv_cdf(min(max(x, 0.005), 0.995)) for x in test["wf_p"]])
        sd_fit = float(np.sum(z * test["result"]) / np.sum(z * z))
        if MARGIN_SD is None:
            MARGIN_SD = sd_fit
        pm = MARGIN_SD * z
        report["mae_margin"] = float(np.mean(np.abs(pm - test["result"])))
        report["mae_margin_base"] = float(np.mean(np.abs(test["result"] - test["result"].mean())))
        # walk-forward totals
        tot_err = []
        for s_ in sorted(test["season"].unique()):
            tr = done[done["season"] < s_]
            te = test[test["season"] == s_]
            tm_ = make_total_model().fit(tr[TOTAL_FEATS], tr["home_score"] + tr["away_score"])
            tot_err.append(np.abs(tm_.predict(te[TOTAL_FEATS]) - (te["home_score"] + te["away_score"])))
        report["mae_total"] = float(np.mean(np.concatenate(tot_err)))
        tt = test["home_score"] + test["away_score"]
        report["mae_total_base"] = float(np.mean(np.abs(tt - tt.mean())))
        print(f"Walk-forward test: each season {report['wf_first']}-{report['wf_last']} predicted by a model trained only on "
              f"earlier seasons ({len(test):,} D1 games)")
        print(f"Accuracy : {report['acc']:.3f}   Log-loss : {report['ll']:.4f}  (coin flip = 0.6931)")
        print(f"Learned weights only (no multipliers): accuracy {report['acc_raw']:.3f}, log-loss {report['ll_raw']:.4f}")
        print(f"Elo only : accuracy {report['elo_acc']:.3f}, log-loss {report['elo_ll']:.4f}")
        print(f"Always pick the home team (true home games): {report['base_acc']:.3f}")
        print(f"Margin from win probability: {MARGIN_SD:.2f} pts per probit unit (fit {sd_fit:.2f}).  "
              f"Avg miss: margin {report['mae_margin']:.1f} (guessing the average {report['mae_margin_base']:.1f}), "
              f"total {report['mae_total']:.1f} (guessing the average {report['mae_total_base']:.1f})")
        wf_rows = []
        print(f"\n  {'season':>6s} {'games':>6s} {'model acc':>9s} {'model LL':>9s} {'Elo acc':>8s} {'Elo LL':>8s}")
        for s_, d_ in test.groupby("season"):
            y_ = d_["home_win"].values
            a_, l_ = float(((d_["wf_p"] >= 0.5) == y_).mean()), float(logloss_vec(y_, d_["wf_p"].values).mean())
            ea_, el_ = float(((d_["elo_home_prob"] >= 0.5) == y_).mean()), float(logloss_vec(y_, d_["elo_home_prob"].values).mean())
            wf_rows.append((int(s_), len(d_), a_, l_, ea_, el_))
            print(f"  {int(s_):>6d} {len(d_):>6d} {a_:9.3f} {l_:9.4f} {ea_:8.3f} {el_:8.4f}")
        report["wf_table"] = wf_rows
        pk = np.maximum(test["wf_p"].values, 1 - test["wf_p"].values)
        ok = (test["wf_p"].values >= 0.5) == (y == 1)
        hist, calib = [], []
        for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
            m_ = (pk >= lo) & (pk < hi)
            if m_.sum() >= 30:
                hist.append((lo, hi, int(m_.sum()), float(ok[m_].mean()), float(pk[m_].mean())))
        report["conf_hist"] = hist
        print("\n  Calibration (picked team's win probability vs how often it won):")
        for lo, hi, n_, acc_, avg_ in hist:
            print(f"    {lo * 100:3.0f}-{min(hi, 1) * 100:3.0f}%: {n_:6,d} games, predicted {avg_ * 100:5.1f}%, won {acc_ * 100:5.1f}%")
        # early vs late season
        early = (test["h_gp"] < 8) | (test["a_gp"] < 8)
        report["early_late"] = [("First 8 games of a team's season", int(early.sum()), float(ok[early.values].mean())),
                                ("Rest of the season", int((~early).sum()), float(ok[~early.values].mean()))]
    else:
        print("No completed test games available.")
        if MARGIN_SD is None:
            MARGIN_SD = 11.0

    live_train = done_all if LIVE_MODEL_USES_ALL_GAMES else done
    base_model = AdjustedLogit(feats, mults).fit(live_train, live_train["home_win"])
    model = Calibrated(base_model, *cal)
    report["record_model"] = (Calibrated(AdjustedLogit(feats, mults).fit(done, done["home_win"]), *cal)
                              if len(done) >= WF_MIN_TRAIN_GAMES else model)
    report["total_model"] = make_total_model().fit(live_train[TOTAL_FEATS], live_train["home_score"] + live_train["away_score"])
    report["n_train"] = len(live_train)
    coefs = model.effective_coefs
    print("\nStandardized coefficients (positive = favors HOME team; |size| = reliance)")
    print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).round(4).to_string())
    tot = coefs.abs().sum() or 1.0
    shares = {grp: coefs.reindex([c for c in cols if c in coefs.index]).abs().sum() / tot
              for grp, cols in FEATURE_GROUPS.items() if any(c in coefs.index for c in cols)}
    tw = sum(RANK_STAT_WEIGHTS.values())
    unit_parts = {RANK_STAT_NAMES[k]: shares.get("Four Factors units", 0.0) * w / tw for k, w in RANK_STAT_WEIGHTS.items() if w > 0}
    print("\nShare of total |coefficient| by factor")
    for name, v in sorted(shares.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<22s} {v:6.1%}")
    report.update(coefs=coefs, shares=shares, unit_parts=unit_parts)
    try:
        report["ablation"] = feature_ablation(games)
        print_ablation(report["ablation"])
    except Exception as ex:
        report["ablation"] = None
        print(f"\n  ! factor accuracy test skipped ({ex})")
    report["compare"] = None
    if COMPARE_CONFIGS:
        try:
            report["compare"] = compare_configs(games)
            print_compare(report["compare"])
        except Exception as ex:
            print(f"\n  ! model set-up comparison skipped ({ex})")
    return model, report


def season_record(model, games):
    """What the model's picks would have scored so far this season (model trained on earlier seasons only)."""
    feats = model_features()
    d = games[(games["season"] == PREDICT_SEASON) & games["home_win"].notna() & games["both_d1"]].sort_values("tip").copy()
    if d.empty:
        return None
    d["p_home"] = model.predict_proba(d)[:, 1]
    d["correct"] = (d["p_home"] >= 0.5) == (d["home_win"] == 1)
    d["conf"] = np.maximum(d["p_home"], 1 - d["p_home"])
    d["elo_ok"] = (d["elo_home_prob"] >= 0.5) == (d["home_win"] == 1)
    hm = d["home_field"] == 1
    wk = d["day"] - pd.to_timedelta(d["day"].dt.weekday, unit="D")
    weeks = [(f"{k:%b %d}", int(v.sum()), len(v)) for k, v in d.groupby(wk)["correct"]]
    bands = []
    for lo, hi, label in ((0.5, 0.6, "50-60%"), (0.6, 0.7, "60-70%"), (0.7, 0.8, "70-80%"), (0.8, 0.9, "80-90%"), (0.9, 1.01, "90%+")):
        m = d[(d["conf"] >= lo) & (d["conf"] < hi)]
        if len(m):
            bands.append({"label": label, "w": int(m["correct"].sum()), "n": len(m)})
    ll = float(logloss_vec(d["home_win"].values, d["p_home"].values).mean())
    rows = [{"day": r.day, "away": r.away_id, "home": r.home_id, "pick": r.home_id if r.p_home >= 0.5 else r.away_id,
             "conf": float(r.conf), "score": f"{int(r.away_score)}-{int(r.home_score)}", "ok": bool(r.correct)}
            for r in d.itertuples()][-300:]
    return {"n": len(d), "w": int(d["correct"].sum()), "ll": ll, "elo_w": int(d["elo_ok"].sum()),
            "home_w": int((d.loc[hm, "home_win"] == 1).sum()), "home_n": int(hm.sum()),
            "weeks": weeks, "bands": bands, "rows": rows}


def print_season_record(rec):
    print("\n" + "=" * 78)
    print(f"MODEL RECORD SO FAR IN {PREDICT_SEASON} (D1 vs D1)")
    print("=" * 78)
    if not rec:
        print("No completed games yet.")
        return
    n, w = rec["n"], rec["w"]
    print(f"Model picks : {w}-{n - w}  ({100 * w / n:.1f}%), log-loss {rec['ll']:.4f}")
    print(f"Elo only    : {rec['elo_w']}-{n - rec['elo_w']}  ({100 * rec['elo_w'] / n:.1f}%)")
    if rec["home_n"]:
        print(f"Home team   : {rec['home_w']}-{rec['home_n'] - rec['home_w']} in true home games ({100 * rec['home_w'] / rec['home_n']:.1f}%)")
    print("By confidence: " + "  ".join(f"{b['label']} {b['w']}-{b['n'] - b['w']}" for b in rec["bands"]))


# ----------------------------------------------------------------------------
# 9. UPCOMING GAMES
# ----------------------------------------------------------------------------
def pick_days(games):
    """The days to predict (list of timestamps) or [] if nothing is left to play."""
    if PREDICT_DATE:
        start = pd.Timestamp(PREDICT_DATE).normalize()
    else:
        up = games[~games["completed"] & (games["day"] >= today_local())]
        if up.empty:
            return []
        start = up["day"].min()
    return [start + pd.Timedelta(days=i) for i in range(max(1, PREDICT_DAYS))]


def group_contributions(model, X):
    """Per-game log-odds each factor group adds for the HOME team versus an even matchup (positive favors home)."""
    Xs = model.base.named_steps["scaler"].transform(X[model.feats])
    c = pd.DataFrame(Xs * (model.coef_raw * model.mult * model.slope), columns=model.feats, index=X.index)
    if hasattr(model, "scale"):
        c = c.mul(model.scale(X), axis=0)
    out = {}
    for grp, cols in FEATURE_GROUPS.items():
        cc = [x for x in cols if x in c.columns]
        if cc:
            out[grp] = c[cc].sum(axis=1)
    return pd.DataFrame(out)


def confidence_words(p):
    if p >= 0.85:
        return "Very confident"
    if p >= 0.72:
        return "Confident"
    if p >= 0.62:
        return "Moderate lean"
    if p >= 0.55:
        return "Slight lean"
    return "Toss-up"


def build_why(r, gc, hist, names):
    """Plain-language reasons for the pick and concerns about it."""
    home_pick = r.p_home >= 0.5
    P, O = (r.home_id, r.away_id) if home_pick else (r.away_id, r.home_id)
    Pn, On = names(P), names(O)
    sg = 1.0 if home_pick else -1.0
    pp = max(r.p_home, 1 - r.p_home)
    v = {g: sg * float(x) for g, x in gc.items()}
    pre, opp = ("h", "a") if home_pick else ("a", "h")
    g = lambda side, name: getattr(r, f"{side}_{name}")
    rk = lambda x: "" if x is None or pd.isna(x) else f"#{int(x)}"
    reasons, cs = [], []
    if v.get("Elo", 0) >= 0.08:
        reasons.append((v["Elo"], f"{Pn} is the stronger team on overall rating (Elo {g(pre, 'elo_pre'):.0f} vs {g(opp, 'elo_pre'):.0f})."))
    if v.get("Efficiency ratings", 0) >= 0.08:
        em_p, em_o = g(pre, "adjo") - g(pre, "adjd"), g(opp, "adjo") - g(opp, "adjd")
        reasons.append((v["Efficiency ratings"], f"Better adjusted efficiency: {Pn} {em_p:+.1f} vs {On} {em_o:+.1f} points per 100 "
                                                 f"possessions (projected {abs(r.er_pm):.1f}-point edge before home court)."))
    if v.get("Four Factors units", 0) >= 0.08:
        e1 = g(pre, "off_unit") - g(opp, "def_unit")
        e2 = g(opp, "off_unit") - g(pre, "def_unit")
        if e1 >= -e2:
            reasons.append((v["Four Factors units"], f"Favorable matchup: {Pn}'s offense ({rk(g(pre, 'off_rank'))}) "
                                                     f"against {On}'s defense ({rk(g(opp, 'def_rank'))})."))
        else:
            reasons.append((v["Four Factors units"], f"{Pn}'s defense ({rk(g(pre, 'def_rank'))}) matches up well with "
                                                     f"{On}'s offense ({rk(g(opp, 'off_rank'))})."))
    if v.get("Recent form", 0) >= 0.06:
        reasons.append((v["Recent form"], f"{Pn} has been playing above its ratings lately "
                                          f"({g(pre, 'form'):+.1f} per 100 possessions over the last {FORM_GAMES} games, vs {g(opp, 'form'):+.1f})."))
    if v.get("Rest", 0) >= 0.05:
        reasons.append((v["Rest"], f"{Pn} is better rested ({int(g(pre, 'rest'))} days vs {int(g(opp, 'rest'))})."))
    if v.get("Home court", 0) >= 0.08:
        reasons.append((v["Home court"] * 0.8, f"{Pn} is at home."))

    def outs(t):
        info = INJ_NOW.get((int(r.game_id), int(t)), {})
        rr = sorted([x for x in info.get("rows", []) if x["loss"] > 0], key=lambda x: -x["loss"])
        return ", ".join(f'{x["name"]} ({x["mpg"]:.0f} min/g)' for x in rr[:3]), info.get("loss", 0.0)
    o_txt, o_loss = outs(O)
    p_txt, p_loss = outs(P)

    def backs(t):
        info = INJ_NOW.get((int(r.game_id), int(t)), {})
        return ", ".join(x["name"] for x in info.get("rows", []) if x["loss"] < 0)
    if backs(P):
        reasons.append((0.25, f"{Pn} gets {backs(P)} back, and its ratings were built partly without him."))
    if backs(O):
        cs.append((0.45, f"{On} gets {backs(O)} back, so it may be better than its ratings."))
    if o_txt and v.get("Injuries", 0) >= 0.03:
        reasons.append((0.2 + v["Injuries"], f"{On} is likely without {o_txt}; their backups are a step down."))
    reasons.sort(key=lambda t: -t[0])
    reasons = [t for _, t in reasons[:4]] or [f"No single factor stands out; {Pn} gets a small edge from several factors added together."]
    if pp < 0.56:
        cs.append((1.0, f"Very close call: {Pn} is only {pp * 100:.0f}% to win, barely better than a coin flip."))
    if v.get("Elo", 0) <= -0.08:
        cs.append((-v["Elo"], f"{On} actually rates higher on Elo ({g(opp, 'elo_pre'):.0f} vs {g(pre, 'elo_pre'):.0f})."))
    if v.get("Efficiency ratings", 0) <= -0.08:
        cs.append((-v["Efficiency ratings"], f"{On} has the better adjusted efficiency margin."))
    if v.get("Four Factors units", 0) <= -0.08:
        cs.append((-v["Four Factors units"], f"Matchup risk: the Four Factors ratings lean toward {On}."))
    if v.get("Recent form", 0) <= -0.06:
        cs.append((-v["Recent form"], f"{On} has been playing above its ratings lately ({g(opp, 'form'):+.1f} per 100)."))
    if v.get("Home court", 0) <= -0.08:
        cs.append((0.5, f"{Pn} is on the road; the pick has to overcome {On}'s home court."))
    if g(opp, "rest") <= 1 or g(pre, "rest") <= 1:
        who = Pn if g(pre, "rest") <= 1 else On
        cs.append((0.4, f"{who} is playing on back-to-back days."))
    if p_txt:
        cs.append((0.5 + max(-v.get("Injuries", 0), 0), f"{Pn} is likely without {p_txt} (see the Injuries tab)."))
    if min(g("h", "gp"), g("a", "gp")) < 5:
        cs.append((0.45, "Early in the season: the ratings still lean on last season, and rosters change a lot from year to year."))
    if r.neutral:
        cs.append((0.2, "Neutral site: no home court for either team."))
    vp = getattr(r, "v_prob", np.nan)
    if vp == vp and vp is not None:
        mk = vp if home_pick else 1 - vp
        if mk < 0.5:
            cs.append((0.9, f"Vegas disagrees: the betting market has {On} favored ({(1 - mk) * 100:.0f}%), the model gives {Pn} {pp * 100:.0f}%."))
        elif (pp - mk) * 100 >= MARKET_FLAG_PTS:
            cs.append((0.7, f"The model likes {Pn} much more than the market does ({pp * 100:.0f}% vs {mk * 100:.0f}%)."))
        elif (mk - pp) * 100 >= MARKET_FLAG_PTS:
            cs.append((0.6, f"The market is even more confident in {Pn} ({mk * 100:.0f}% vs the model's {pp * 100:.0f}%)."))
    cs.sort(key=lambda t: -t[0])
    concerns = [t for _, t in cs[:5]] or ["Nothing stands out: the factors agree with the pick."]
    hist_txt = ""
    for lo, hi, n_, acc_, _ in (hist or []):
        if lo <= pp < hi:
            hist_txt = f"Picks in this confidence range ({lo * 100:.0f}-{min(hi, 1) * 100:.0f}%) won {acc_ * 100:.0f}% of the time in past seasons ({n_:,} games)."
    short = {"Elo": "Elo", "Efficiency ratings": "Efficiency", "Four Factors units": "Four Factors", "Recent form": "Form",
             "Rest": "Rest", "Conference game": "Conf game", "Early season": "Early season",
             "Roster continuity": "Continuity", "Injuries": "Injuries", "Home court": "Home court"}
    shifts = sorted(((short.get(k, k), 100.0 * pp * (1 - pp) * x) for k, x in v.items()), key=lambda z: -abs(z[1]))
    return dict(word=confidence_words(pp), pick=P, other=O, pp=pp, reasons=reasons, concerns=concerns, hist=hist_txt, shifts=shifts)


def _num(x):
    try:
        if isinstance(x, str):
            x = x.strip().upper()
            if x in ("EVEN", "EV", "PK"):
                return 100.0
            x = x.replace("+", "")
        v = float(x)
        return v if v == v else np.nan
    except Exception:
        return np.nan


_SCOREBOARD = {}


def fetch_scoreboard(d):
    """ESPN's public scoreboard JSON for one day (D1 games), cached per run. None if it can't be reached."""
    key = f"{d:%Y%m%d}"
    if key not in _SCOREBOARD:
        url = ("https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
               f"?dates={key}&groups=50&limit=500")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                _SCOREBOARD[key] = json.loads(resp.read().decode("utf-8"))
        except Exception as ex:
            print(f"  ! ESPN scoreboard for {d:%b %d} unavailable ({ex})")
            _SCOREBOARD[key] = None
    return _SCOREBOARD[key]


def fetch_live_finals(days):
    """{game_id: (home_id, home_score, away_id, away_score)} for games ESPN lists as final on these days."""
    out = {}
    if not FETCH_LIVE_SCORES or AS_OF_DATE:
        return out
    for d in days:
        data = fetch_scoreboard(d)
        for ev in (data or {}).get("events", []):
            try:
                st = (ev.get("status") or {}).get("type") or {}
                if not st.get("completed"):
                    continue
                comp = ev["competitions"][0]
                h = next(c for c in comp["competitors"] if c.get("homeAway") == "home")
                a = next(c for c in comp["competitors"] if c.get("homeAway") == "away")
                hs, as_ = float(h.get("score")), float(a.get("score"))
                if hs + as_ <= 0:
                    continue
                out[int(ev["id"])] = (int(h.get("id") or h["team"]["id"]), hs, int(a.get("id") or a["team"]["id"]), as_)
            except Exception:
                continue
    if out:
        print(f"  Live scoreboard: {len(out)} final(s) for {', '.join(f'{d:%b %d}' for d in days)}")
    return out


def fetch_odds(days):
    """{game_id: {spread (positive = home favored), total, ml_home, ml_away, prob}} from ESPN's public scoreboard.
    Shown on the cards for context only; the model never uses it. Anything that fails is skipped."""
    out = {}
    if not FETCH_ODDS:
        return out
    for d in days:
        data = fetch_scoreboard(d)
        if not data:
            continue
        for ev in data.get("events", []):
            try:
                comp = ev["competitions"][0]
                odds = (comp.get("odds") or [None])[0]
                if not odds:
                    continue
                home = next(c for c in comp["competitors"] if c.get("homeAway") == "home")
                ho, ao = odds.get("homeTeamOdds") or {}, odds.get("awayTeamOdds") or {}
                mlh, mla = _num(ho.get("moneyLine")), _num(ao.get("moneyLine"))
                ml = odds.get("moneyline") or {}
                if mlh != mlh:
                    mlh = _num(((ml.get("home") or {}).get("close") or {}).get("odds"))
                if mla != mla:
                    mla = _num(((ml.get("away") or {}).get("close") or {}).get("odds"))
                sp = abs(_num(odds.get("spread")))
                det = str(odds.get("details") or "")
                if sp != sp:
                    m_ = re.search(r"([+-]?\d+(\.\d+)?)\s*$", det)
                    sp = abs(float(m_.group(1))) if m_ else np.nan
                home_fav = None
                if ho.get("favorite") is True or ao.get("favorite") is False:
                    home_fav = True
                elif ao.get("favorite") is True or ho.get("favorite") is False:
                    home_fav = False
                elif det:
                    ab = str((home.get("team") or {}).get("abbreviation") or "")
                    home_fav = bool(ab) and det.upper().startswith(ab.upper() + " ")
                if det.upper().startswith("EVEN") or det.upper() == "PK":
                    sp = 0.0
                spread = np.nan if sp != sp or home_fav is None else (sp if home_fav else -sp)
                prob = np.nan
                if mlh == mlh and mla == mla:
                    ph = -mlh / (-mlh + 100) if mlh < 0 else 100 / (mlh + 100)
                    pa = -mla / (-mla + 100) if mla < 0 else 100 / (mla + 100)
                    prob = ph / (ph + pa)
                    if spread == spread and spread != 0 and (spread > 0) != (prob > 0.5):
                        spread = -spread                   # trust the moneylines for the sign
                out[int(ev["id"])] = dict(spread=spread, total=_num(odds.get("overUnder")), ml_home=mlh, ml_away=mla, prob=prob)
            except Exception:
                continue
    print(f"  Betting lines found for {len(out)} game(s)")
    return out


def predict_upcoming(model, games, report, names, st=None):
    feats = model_features()
    days = pick_days(games)
    if not days:
        print(f"\nNo unplayed {PREDICT_SEASON} games found.")
        return None
    wk = games[games["day"].isin(days)].sort_values(["tip", "game_id"]).copy()
    if wk.empty:
        print(f"\nNo games on {days[0]:%b %d, %Y}.")
        return None
    odds = fetch_odds(days)
    wk["v_spread"] = wk["game_id"].map(lambda x: odds.get(x, {}).get("spread", np.nan))
    wk["v_total"] = wk["game_id"].map(lambda x: odds.get(x, {}).get("total", np.nan))
    wk["v_ml_home"] = wk["game_id"].map(lambda x: odds.get(x, {}).get("ml_home", np.nan))
    wk["v_ml_away"] = wk["game_id"].map(lambda x: odds.get(x, {}).get("ml_away", np.nan))
    wk["v_prob"] = wk["game_id"].map(lambda x: odds.get(x, {}).get("prob", np.nan))
    for side, col in (("h", "home_id"), ("a", "away_id")):
        for u in ("off", "def"):
            wk[f"{side}_{u}_rank"] = (wk[col].map(st[f"{u}_unit_rk"]) if st is not None and f"{u}_unit_rk" in st.columns
                                      else np.nan)
    wk["p_home"] = model.predict_proba(wk)[:, 1]
    wk["winner"] = np.where(wk["p_home"] >= 0.5, wk["home_id"], wk["away_id"])
    wk["pred_margin"] = [MARGIN_SD * NormalDist().inv_cdf(min(max(x, 0.005), 0.995)) for x in wk["p_home"]]
    wk["pred_total"] = np.clip(report["total_model"].predict(wk[TOTAL_FEATS]), 100, 190)
    hs = np.rint(np.maximum((wk["pred_total"] + wk["pred_margin"]) / 2, 0)).astype(int)
    as_ = np.rint(np.maximum((wk["pred_total"] - wk["pred_margin"]) / 2, 0)).astype(int)
    tie = hs == as_
    wk["score_home"] = np.where(tie & (wk["p_home"] >= 0.5), hs + 1, hs)
    wk["score_away"] = np.where(tie & (wk["p_home"] < 0.5), as_ + 1, as_)
    try:
        gc = group_contributions(model, wk)
    except Exception as ex:
        print(f"  ! per-game explanations unavailable ({ex})")
        gc = None
    label = f"{days[0]:%a %b %d, %Y}" + (f" to {days[-1]:%a %b %d}" if len(days) > 1 else "")
    print("\n" + "=" * 78)
    print(f"PREDICTIONS FOR {label.upper()}  ({len(wk)} games, {int(wk['completed'].sum())} already final)")
    print("=" * 78)
    for r in wk.itertuples():
        conf = max(r.p_home, 1 - r.p_home) * 100
        res = ""
        if r.completed:
            res = f"  FINAL {int(r.away_score)}-{int(r.home_score)} " + ("RIGHT" if (r.p_home >= 0.5) == (r.result > 0) else "WRONG")
        print(f"  {names(r.away_id):>22s} {'vs' if r.neutral else '@'} {names(r.home_id):<22s} pick {names(r.winner):<20s} "
              f"{conf:5.1f}%  {r.score_away:>3d}-{r.score_home:<3d}{res}")
    rows = []
    for r in wk.itertuples():
        d = {c: getattr(r, c) for c in wk.columns if not c.startswith("_")}
        d["why"] = build_why(r, gc.loc[r.Index].to_dict(), report.get("conf_hist"), names) if gc is not None else None
        rows.append(d)
    return {"days": days, "label": label, "rows": rows}


# ----------------------------------------------------------------------------
# SEASON TEAM STATS & RANKINGS (display)
# ----------------------------------------------------------------------------
def build_season_team_stats(tg, games, elo_ratings, eff_now, season, phantom):
    """Season-to-date numbers for every D1 team, with D1 ranks (1 = best). Uses completed games of `season`."""
    t = tg[(tg["season"] == season) & tg["completed"] & (tg["d1"] == 1)].copy()
    if t.empty:
        return None
    t["w"] = (t["pts_for"] > t["pts_against"]).astype(int)
    t["l"] = 1 - t["w"]

    def rec(d):
        return f"{int(d['w'].sum())}-{int(d['l'].sum())}"
    st = pd.DataFrame(index=sorted(t["team"].unique()))
    grp = t.groupby("team")
    st["games"] = grp.size()
    st["record"] = grp.apply(rec)
    for name, m in (("home_record", (t["is_home"] == 1) & (t["neutral"] == 0)), ("away_record", (t["is_home"] == 0) & (t["neutral"] == 0)),
                    ("neutral_record", t["neutral"] == 1), ("conf_record", t["conf_game"] == 1)):
        sub = t[m].groupby("team")
        st[name] = sub.apply(rec).reindex(st.index).fillna("0-0")
    st["ppg"] = grp["pts_for"].mean()
    st["oppg"] = grp["pts_against"].mean()
    st["pd_pg"] = st["ppg"] - st["oppg"]
    sums = grp[["poss"]].sum()
    for c in ["off_eff", "def_eff", "off_efg", "def_efg", "off_tov", "def_tov", "off_orb", "def_orb", "off_ftr", "def_ftr",
              "off_two", "def_two", "off_three", "def_three", "off_3par", "def_3par", "off_ftp", "off_stl", "def_stl",
              "off_blk", "def_blk", "tempo"]:
        st[c] = grp[c].mean()
    st["elo"] = [elo_ratings.get(x, np.nan) for x in st.index]
    st["adj_o"] = [eff_now.get(x, (np.nan,) * 3)[0] for x in st.index]
    st["adj_d"] = [eff_now.get(x, (np.nan,) * 3)[1] for x in st.index]
    st["adj_t"] = [eff_now.get(x, (np.nan,) * 3)[2] for x in st.index]
    st["adj_em"] = st["adj_o"] - st["adj_d"]
    # schedule strength: average adjusted margin of the opponents faced (D1 opponents)
    opp_em = {x: st.at[x, "adj_em"] for x in st.index}
    t["opp_em"] = t["opp"].map(opp_em)
    st["sos"] = t.groupby("team")["opp_em"].mean()
    last5 = {}
    for x, d in t.sort_values("tip").groupby("team"):
        last5[x] = [dict(day=r.day, opp=r.opp, home=r.is_home, neutral=r.neutral, pf=int(r.pts_for), pa=int(r.pts_against))
                    for r in d.tail(5).itertuples()]
    st["last5"] = pd.Series(last5)
    if phantom is not None and len(phantom):
        ph = phantom.set_index("team")
        for c in ["off_unit", "def_unit"] + OFF_UNIT_COLS + DEF_UNIT_COLS:
            st[c] = ph[c].reindex(st.index)
        st["ovr_unit"] = (st["off_unit"] + st["def_unit"]) / 2
    higher = {"ppg": True, "oppg": False, "pd_pg": True, "off_eff": True, "def_eff": False, "off_efg": True, "def_efg": False,
              "off_tov": False, "def_tov": True, "off_orb": True, "def_orb": False, "off_ftr": True, "def_ftr": False,
              "off_two": True, "def_two": False, "off_three": True, "def_three": False, "off_3par": True, "def_3par": False,
              "off_ftp": True, "off_stl": False, "def_stl": True, "off_blk": False, "def_blk": True, "tempo": True,
              "elo": True, "adj_o": True, "adj_d": False, "adj_t": True, "adj_em": True, "sos": True,
              "off_unit": True, "def_unit": True, "ovr_unit": True}
    for c, hi in higher.items():
        if c in st.columns:
            st[c + "_rk"] = st[c].rank(ascending=not hi, method="min")
    return st


# ----------------------------------------------------------------------------
# 9b. CHARTS (Trapezoid of Excellence, efficiency landscape, Elo race, luck)
# ----------------------------------------------------------------------------
def find_champions(games):
    """{season: champion team_id} from the national championship game of each season in the data."""
    nh = games["notes_headline"].astype(str)
    m = nh.str.contains("national championship", case=False) & nh.str.contains("men", case=False) & games["completed"]
    out = {}
    for r in games[m].sort_values("tip").itertuples():
        out[int(r.season)] = int(r.home_id if r.home_score > r.away_score else r.away_id)
    return out


def build_chart_data(st, games, eff, elo, mg, report, names, today_ids):
    """Everything the Charts tab draws, as plain numbers (drawn in the browser, works offline)."""
    if st is None or st.empty:
        return None
    cur = PREDICT_SEASON
    # league-average tempo by season (D1 games): the eff ratings are centered, so a team's tempo is compared to this
    e = eff.merge(games[["game_id", "season", "both_d1", "day", "home_id", "away_id", "completed"]], on="game_id")
    e = e[e["both_d1"]]
    t_avg = ((e["h_adjt"] + e["a_adjt"]) / 2).groupby(e["season"]).mean().to_dict()
    # past champions: their ratings going into the title game, tempo relative to that season's average
    champs = []
    for s_, tid in sorted(find_champions(games).items()):
        if s_ >= cur:
            continue
        g_ = e[(e["season"] == s_) & ((e["home_id"] == tid) | (e["away_id"] == tid))].sort_values("day")
        if g_.empty:
            continue
        r = g_.iloc[-1]
        h = r["home_id"] == tid
        em = (r["h_adjo"] - r["h_adjd"]) if h else (r["a_adjo"] - r["a_adjd"])
        tt = (r["h_adjt"] if h else r["a_adjt"]) - t_avg.get(s_, np.nan)
        champs.append({"s": int(s_), "n": names(tid), "em": round(float(em), 2), "rt": round(float(tt), 2)})
    t_cur = float(st["adj_t"].mean())
    trap = None
    if len(champs) >= 4:
        ems = np.array([c["em"] for c in champs])
        rts = np.array([c["rt"] for c in champs])
        tc = float(np.median(rts))
        y0 = float(ems.min()) - 2.0
        w0 = 1.5
        k = float(max(0.05, np.max((np.abs(rts - tc) - w0) / np.maximum(ems - y0, 0.5)))) * 1.1
        y1 = float(max(st["adj_em"].max(), ems.max())) + 4.0
        trap = {"y0": round(y0, 2), "y1": round(y1, 2), "xc": round(t_cur + tc, 2), "w0": w0, "k": round(k, 4)}

    def inside(t_, em_):
        if trap is None or em_ < trap["y0"]:
            return False
        return abs(t_ - trap["xc"]) <= trap["w0"] + trap["k"] * (em_ - trap["y0"])
    teams = []
    for tid, r in st.iterrows():
        if pd.isna(r.get("adj_em")):
            continue
        teams.append({"id": int(tid), "n": names(tid), "rec": str(r["record"]), "em": round(float(r["adj_em"]), 2),
                      "o": round(float(r["adj_o"]), 2), "d": round(float(r["adj_d"]), 2), "t": round(float(r["adj_t"]), 2),
                      "elo": round(float(r["elo"]), 1) if r["elo"] == r["elo"] else None,
                      "rk": int(r["adj_em_rk"]) if r.get("adj_em_rk") == r.get("adj_em_rk") else None,
                      "in": bool(inside(float(r["adj_t"]), float(r["adj_em"]))), "today": int(tid) in today_ids})
    # Elo through the season (D1 teams of the current season)
    el = elo.merge(games[["game_id", "season", "day", "home_id", "away_id", "completed"]], on="game_id")
    el = el[(el["season"] == cur) & el["completed"]]
    hist = {}
    for r in el.sort_values("day").itertuples():
        for tid, v in ((r.home_id, r.h_elo_pre), (r.away_id, r.a_elo_pre)):
            if tid in st.index:
                hist.setdefault(int(tid), []).append([r.day.strftime("%Y-%m-%d"), round(float(v), 1)])
    for tid in hist:
        cur_elo = st.at[tid, "elo"] if tid in st.index else None
        if cur_elo == cur_elo and cur_elo is not None:
            hist[tid].append([pd.Timestamp(report.get("as_of_day", today_local())).strftime("%Y-%m-%d"), round(float(cur_elo), 1)])
    # luck: wins minus the wins the model expected before each game (D1 vs D1, this season)
    luck = []
    rm = report.get("record_model")
    d = mg[(mg["season"] == cur) & mg["home_win"].notna() & mg["both_d1"]]
    if rm is not None and len(d):
        ph = rm.predict_proba(d)[:, 1]
        exp_w, act_w, n_g = {}, {}, {}
        for hid, aid, p_, hw in zip(d["home_id"], d["away_id"], ph, d["home_win"]):
            for tid, pp, ww in ((hid, p_, hw), (aid, 1 - p_, 1 - hw)):
                exp_w[tid] = exp_w.get(tid, 0.0) + pp
                act_w[tid] = act_w.get(tid, 0.0) + ww
                n_g[tid] = n_g.get(tid, 0) + 1
        for tid in exp_w:
            if tid in st.index:
                luck.append({"id": int(tid), "n": names(tid), "v": round(act_w[tid] - exp_w[tid], 2),
                             "w": int(act_w[tid]), "x": round(exp_w[tid], 1), "g": n_g[tid], "today": int(tid) in today_ids})
    return {"season": cur, "t_avg": round(t_cur, 2), "teams": teams, "top_n": TRAP_TOP_N, "champs": champs, "trap": trap,
            "elo": {str(k): v for k, v in hist.items()}, "luck": luck}


# ----------------------------------------------------------------------------
# 10. SHAREABLE HTML REPORT
# ----------------------------------------------------------------------------
REPORT_CSS = """
:root{color-scheme:light dark;--goodbg:rgba(10,125,60,.13);--badbg:rgba(180,35,24,.11);--bg:#f6f7f9;--card:#fff;--text:#161a1f;--muted:#667085;
--line:#e4e7ec;--accent:#c2410c;--pos:#0a7d3c;--neg:#b42318;--bar:#fde3d3;--barfav:#c2410c;--winbg:rgba(194,65,12,.09)}
@media (prefers-color-scheme:dark){:root{--goodbg:rgba(74,222,128,.15);--badbg:rgba(248,113,113,.14);--bg:#0f1318;--card:#171c23;--text:#e8ebef;
--muted:#98a2b3;--line:#2a313b;--accent:#fb923c;--pos:#4ade80;--neg:#f87171;--bar:#3a2a20;--barfav:#fb923c;--winbg:rgba(251,146,60,.13)}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1440px;margin:0 auto;padding:28px 16px 56px}
h1{font-size:26px;margin:0 0 4px}
h4{margin:0 0 6px;font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.sub{color:var(--muted);margin:0 0 6px}
.teams{font-weight:700;font-size:17px}.at{color:var(--muted);margin:0 6px;font-weight:400}
.rec{display:block;color:var(--muted);font-size:12px;font-weight:400}
.k{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.who{font-size:19px;font-weight:800;color:var(--accent)}
.score{white-space:normal;font-weight:600;min-width:0}.score .n{font-size:19px;font-weight:700;color:var(--muted)}
.score .n.w{color:var(--accent);font-weight:800}
.pct{font-size:20px;font-weight:800}.tier{color:var(--muted);font-size:12px;margin-left:8px}
.bar{display:block;height:8px;background:var(--bar);border-radius:4px;overflow:hidden;margin-top:5px}
.fill{display:block;height:100%;background:var(--barfav)}
.fin{display:block;white-space:normal;font-size:12px;font-weight:700;margin-top:3px}.fin.ok{color:#2e9e5b}.fin.no{color:#c0392b}
.vegas{text-align:right;min-width:0}.vegas .vr{display:block;white-space:nowrap;font-size:14px;color:var(--muted)}
.vegas .vr.pk{color:#2e9e5b;font-weight:800;font-size:15px}.vegas .sub{display:block;color:var(--muted);font-size:12px;margin-top:3px}
.vegas .flag{display:inline-block;white-space:nowrap;margin-top:4px;padding:1px 7px;border-radius:999px;background:#d98a1f22;color:#d98a1f;font-size:11.5px;font-weight:700}
.vegas .ag{display:block;font-size:12px;font-weight:700;margin-top:2px}.ag.yes{color:#2e9e5b}.ag.no{color:#d98a1f}
.whybox{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:0 0 12px}
.whybox ul{margin:6px 0 0;padding-left:18px}.whybox li{margin:5px 0;font-size:13.5px;line-height:1.4}
.shifts{display:flex;flex-wrap:wrap;gap:5px}.shift{font-size:12px;padding:2px 8px;border-radius:999px;background:var(--bar);color:var(--muted)}
.shift.up{background:rgba(46,158,91,.16);color:#2e9e5b}.shift.dn{background:rgba(192,57,43,.14);color:#c0392b}
.whyconf{margin:0 0 4px;font-size:14px}.whyhist{margin:8px 0 0;font-size:12px;color:var(--muted)}
@media (max-width:760px){.whybox{grid-template-columns:1fr}}
.edges{margin:0 0 14px;padding:10px 14px;border:1px solid var(--line);border-radius:10px;background:var(--card)}
.edges h3{margin:0 0 6px;font-size:14px}.edges div{font-size:13px;color:var(--muted);margin:2px 0}
.tl{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.logo{display:inline-block;flex:none;width:22px;height:22px;background-size:contain;background-repeat:no-repeat;background-position:center}
.logo.lg{width:30px;height:30px}
th.tm .tl{justify-content:center}
.probline{color:var(--muted);font-size:13px;margin-bottom:12px}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.panel{border:1px solid var(--line);border-radius:10px;padding:10px 12px}
table{border-collapse:collapse;width:100%}
th,td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:left}
th{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);font-weight:600}
tr:last-child td{border-bottom:0}
table.cmp td.v{text-align:center;width:34%;font-variant-numeric:tabular-nums}
table.cmp td.v.good{font-weight:700;background:var(--goodbg);color:var(--pos)}
table.cmp td.v.bad{background:var(--badbg);color:var(--neg)}
table.cmp td.lbl{text-align:center;color:var(--muted);font-size:12px}
table.cmp th.tm{text-align:center;font-size:13px;color:var(--text)}
.rk{font-size:11px;margin-left:5px;font-weight:600}
.rk.good{color:var(--pos)}.rk.bad{color:var(--neg)}.rk.mid{color:var(--muted)}
.l5 summary{cursor:pointer}.l5 div{font-size:12.5px;line-height:1.5;text-align:left}.l5 .gw{color:var(--pos)}.l5 .gl{color:var(--neg)}
.scroll{overflow-x:auto}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
details.section{background:var(--card);border:1px solid var(--line);border-radius:14px;overflow:hidden;margin-top:14px}
details.section>summary{cursor:pointer;list-style:none;padding:12px 16px;font-weight:700}
details.section .inner{padding:0 16px 14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.stat{background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.stat .v{font-size:22px;font-weight:700}.stat .l{color:var(--muted);font-size:12px}
.cbar{display:inline-block;height:8px;border-radius:4px;vertical-align:middle}.cbar.p{background:var(--pos)}.cbar.n{background:var(--neg)}
.note{color:var(--muted);font-size:13px;margin-top:10px}
.good-t{color:var(--pos);font-weight:700}.meh-t{color:var(--muted)}
.subtabs .cnt{display:inline-block;margin-left:4px;padding:0 6px;border-radius:999px;background:rgba(192,57,43,.16);color:#c0392b;font-size:11px;font-weight:700}
.st.out{color:#c0392b;font-weight:700}.pr.full{color:#2e9e5b}
.imp.high{color:#c0392b;font-weight:800}.imp.medium{color:#d98a1f;font-weight:700}.imp.low,.imp.none{color:var(--muted)}.imp.back{color:#2e9e5b;font-weight:800}
.toptabs{display:flex;gap:6px;margin:14px 0 14px;border-bottom:1px solid var(--line);overflow-x:auto;scrollbar-width:none}.toptabs::-webkit-scrollbar{display:none}
.toptab{flex:0 0 auto;white-space:nowrap;background:none;border:0;border-bottom:3px solid transparent;color:var(--muted);font:inherit;font-weight:700;padding:8px 14px;cursor:pointer}
.toptab.on{color:var(--accent);border-bottom-color:var(--accent)}
.toppane[hidden],.gdetail[hidden],.tp[hidden]{display:none}
.app{display:grid;grid-template-columns:300px minmax(0,1fr);gap:16px;align-items:start}
.gside{position:sticky;top:10px}
.gsearch{width:100%;padding:8px 10px;margin-bottom:8px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--text);font:inherit}
.glist{max-height:calc(100vh - 70px);overflow-y:auto;display:flex;flex-direction:column;gap:6px;padding-right:2px}
.gitem{display:grid;grid-template-columns:1fr auto;gap:2px 8px;align-items:center;text-align:left;background:var(--card);color:var(--text);
border:1px solid var(--line);border-left:4px solid transparent;border-radius:10px;padding:9px 11px;font:inherit;cursor:pointer}
.gitem:hover{border-color:var(--accent)}.gitem.on{border-color:var(--accent);border-left-color:var(--accent);background:var(--winbg)}
.gi-teams{font-weight:700;font-size:13.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.gi-teams .at{margin:0 5px}
.gi-pick{grid-column:1;font-size:12.5px;color:var(--muted)}.gi-pick b{color:var(--text)}
.chip{grid-column:2;grid-row:1/3;font-size:11px;font-weight:700;padding:2px 8px;border-radius:999px;background:var(--bar);color:var(--muted);white-space:nowrap}
.chip.ok{background:rgba(46,158,91,.18);color:#2e9e5b}.chip.no{background:rgba(192,57,43,.16);color:#c0392b}.chip.flag{background:#d98a1f22;color:#d98a1f}
.gview{background:var(--card);border:1px solid var(--line);border-radius:14px;min-width:0}
.ghead{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px 22px;align-items:start;padding:16px;border-bottom:1px solid var(--line)}
.ghead .teams{grid-column:1/-1}
.subtabs{display:flex;gap:4px;padding:8px 12px 0;border-bottom:1px solid var(--line);flex-wrap:wrap}
.subtabs button{background:none;border:0;border-bottom:3px solid transparent;color:var(--muted);font:inherit;font-weight:600;padding:7px 12px;cursor:pointer}
.subtabs button.on{color:var(--accent);border-bottom-color:var(--accent)}
.tp{padding:14px 16px 16px}
table.mp td{vertical-align:middle}.mp-row{cursor:pointer}.mp-row:hover td{background:var(--winbg)}
.mp-pick .tl{font-weight:800;color:var(--accent)}.mp-conf{min-width:150px}.mp-conf .bar{margin-top:3px;max-width:140px}
.mp-v{font-size:13px;color:var(--muted);white-space:nowrap}.mp-v.far{color:#d98a1f;font-weight:700}
.mp-res{font-weight:700;white-space:nowrap}.mp-res.ok{color:#2e9e5b}.mp-res.no{color:#c0392b}
.secgroup>details.section{border:0;background:none;margin:0}.secgroup>details.section>summary{display:none}.secgroup>details.section>.inner{padding:0}
.secnav{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px}
.secnav button{background:var(--card);color:var(--muted);border:1px solid var(--line);border-radius:999px;font:inherit;font-size:13px;font-weight:600;padding:5px 13px;cursor:pointer}
.secnav button.on{background:var(--winbg);color:var(--accent);border-color:var(--accent)}
table.sortable th{cursor:pointer;white-space:nowrap}table.sortable th.sorted{color:var(--accent)}
.tfilter{padding:7px 10px;margin:0 0 10px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--text);font:inherit;width:260px;max-width:100%}
@media (max-width:860px){.app{grid-template-columns:minmax(0,1fr)}.gside{position:static}.glist{flex-direction:row;overflow-x:auto;max-height:none;padding-bottom:6px}
.gitem{flex:0 0 auto;width:max-content;min-width:176px;grid-template-columns:auto;padding-right:14px}
.gitem .chip{grid-column:1;grid-row:auto;justify-self:start;margin-top:3px}}
.chbox{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin-bottom:16px}
.chbox h3{margin:0 0 4px;font-size:17px}.chbox .desc{color:var(--muted);font-size:13px;margin:0 0 8px}
.chart{width:100%;height:auto;display:block;font-family:inherit}
.chart .grid{stroke:var(--line);stroke-width:1}.chart .zero{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 4;opacity:.7}
.chart .tick{fill:var(--muted);font-size:11px}.chart .axl{fill:var(--muted);font-size:12px;font-weight:600}
.chart .pt{fill:var(--muted);opacity:.45;cursor:pointer}.chart .pt.in{fill:var(--pos);opacity:.85}
.chart .pt.today{stroke:var(--accent);stroke-width:2;opacity:1}.chart .pt.hl{fill:var(--accent);opacity:1;stroke:var(--text);stroke-width:2}
.chart .pt.dim{opacity:.12}.chart .lgi{cursor:pointer}.chart .lgi.out{opacity:.45}.chart .lgi.dim{opacity:.1}
.chart .hit{fill:transparent;cursor:pointer}.chart .hlring{fill:var(--winbg);stroke:var(--accent);stroke-width:2.5}.chart .lab{font-size:11px;fill:var(--text);pointer-events:none;paint-order:stroke;stroke:var(--card);stroke-width:3px}
.chart .lab.hl{font-weight:800;fill:var(--accent)}.chart .trap{fill:rgba(46,158,91,.10);stroke:var(--pos);stroke-width:1.5;stroke-dasharray:6 4}
.chart .champ{fill:#d4a017;stroke:var(--card);stroke-width:1;cursor:pointer}.chart .clab{font-size:10px;fill:#b8860b;pointer-events:none}
.chart .tier{stroke:var(--muted);stroke-width:1;stroke-dasharray:2 5;opacity:.8}.chart .tierl{fill:var(--muted);font-size:10.5px}
.chart .eline{fill:none;stroke-width:2.2;opacity:.9}.chart .eline.hl{stroke-width:4}
.chart .bar.pos{fill:var(--pos)}.chart .bar.neg{fill:var(--neg)}.chart .bar.hl{stroke:var(--text);stroke-width:2}
.chctl{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:0 0 14px}
.chctl input[type=text]{padding:7px 10px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--text);font:inherit;width:240px}
.chctl label{font-size:13px;color:var(--muted)}.chlist{font-size:13px;color:var(--muted);margin-top:6px}
#ch-tip{position:fixed;z-index:50;pointer-events:none;background:var(--card);color:var(--text);border:1px solid var(--line);border-radius:8px;
padding:6px 9px;font-size:12.5px;box-shadow:0 4px 14px rgba(0,0,0,.18);display:none;max-width:260px}
.legend{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:12px;color:var(--muted);margin-top:4px}.legend i{display:inline-block;width:12px;height:3px;margin-right:4px;vertical-align:middle}
"""

REPORT_JS = """
(function(){
  function $(s,r){return (r||document).querySelector(s)}
  function $$(s,r){return Array.prototype.slice.call((r||document).querySelectorAll(s))}
  var tabs=$$('.toptab'), panes=$$('.toppane');
  function showTop(id){tabs.forEach(function(b){b.classList.toggle('on',b.dataset.p===id)});panes.forEach(function(p){p.hidden=(p.id!==id)})}
  tabs.forEach(function(b){b.addEventListener('click',function(){showTop(b.dataset.p)})});
  showTop('p-games');
  var items=$$('.gitem'), secs=$$('.gdetail'), curTab='over', cur=-1;
  function applyTab(s){var ps=$$('.tp',s), has=ps.some(function(p){return p.dataset.t===curTab}); var t=has?curTab:'over';
    $$('.subtabs button',s).forEach(function(b){b.classList.toggle('on',b.dataset.t===t)}); ps.forEach(function(p){p.hidden=(p.dataset.t!==t)})}
  function showGame(i,quiet){if(i<0||i>=items.length)return; cur=i;
    items.forEach(function(b,k){b.classList.toggle('on',k===i)}); secs.forEach(function(s){s.hidden=(s.id!=='g'+i)}); applyTab($('#g'+i));
    if(!quiet&&window.innerWidth<=860){var g=$('.gview'); if(g&&g.scrollIntoView)g.scrollIntoView({block:'start'})}}
  items.forEach(function(b,k){b.addEventListener('click',function(){showGame(k)})});
  secs.forEach(function(s){$$('.subtabs button',s).forEach(function(b){b.addEventListener('click',function(){curTab=b.dataset.t; applyTab(s)})})});
  var first=0; for(var k=0;k<items.length;k++){if(items[k].dataset.played==='0'){first=k;break}} showGame(first,true);
  var q=$('.gsearch'); if(q){q.addEventListener('input',function(){var v=q.value.toLowerCase();
    items.forEach(function(b){b.style.display=(b.dataset.q.indexOf(v)>=0)?'':'none'})})}
  document.addEventListener('keydown',function(ev){if(ev.target&&/input|textarea|select/i.test(ev.target.tagName))return;
    if($('#p-games').hidden)return; if(ev.key==='ArrowDown'||ev.key==='ArrowUp'){ev.preventDefault();
    var step=ev.key==='ArrowDown'?1:-1, j=cur+step; while(j>=0&&j<items.length&&items[j].style.display==='none')j+=step;
    showGame(j); if(items[cur]&&items[cur].scrollIntoView)items[cur].scrollIntoView({block:'nearest'})}});
  $$('.mp-row').forEach(function(tr){tr.addEventListener('click',function(){showTop('p-games'); showGame(parseInt(tr.dataset.g,10))})});
  $$('.secgroup').forEach(function(grp){var ds=$$('details.section',grp).filter(function(d){return d.parentNode===grp}); if(!ds.length)return;
    var nav=document.createElement('div'); nav.className='secnav';
    function pick(n){ds.forEach(function(d,k){d.hidden=(k!==n); d.open=true}); $$('button',nav).forEach(function(b,k){b.classList.toggle('on',k===n)})}
    ds.forEach(function(d,k){var sm=$('summary',d), b=document.createElement('button'); b.type='button'; b.textContent=sm?sm.textContent:'Section '+(k+1);
      b.addEventListener('click',function(){pick(k)}); nav.appendChild(b)}); grp.insertBefore(nav,grp.firstChild); pick(0)});
  $$('table.sortable').forEach(function(tb){var ths=$$('th',tb);
    ths.forEach(function(th,ci){th.addEventListener('click',function(){var rows=$$('tr',tb).slice(1), asc=th.dataset.dir!=='asc';
      ths.forEach(function(x){x.classList.remove('sorted');delete x.dataset.dir}); th.dataset.dir=asc?'asc':'desc'; th.classList.add('sorted');
      rows.sort(function(a,b){var x=a.children[ci], y=b.children[ci]; var xv=x.dataset.v!==undefined?parseFloat(x.dataset.v):x.textContent.toLowerCase();
        var yv=y.dataset.v!==undefined?parseFloat(y.dataset.v):y.textContent.toLowerCase(); if(xv<yv)return asc?-1:1; if(xv>yv)return asc?1:-1; return 0});
      var body=rows[0]&&rows[0].parentNode; rows.forEach(function(r){body.appendChild(r)})})})});
  $$('.tfilter').forEach(function(inp){var tb=document.getElementById(inp.dataset.t); inp.addEventListener('input',function(){var v=inp.value.toLowerCase();
    $$('tr',tb).slice(1).forEach(function(r){r.style.display=r.children[1].textContent.toLowerCase().indexOf(v)>=0?'':'none'})})});
})();
"""



CHART_JS = """
(function(){
  var node=document.getElementById('chart-data'); if(!node)return; var D=JSON.parse(node.textContent); if(!D)return;
  var NS='http://www.w3.org/2000/svg', tip=document.getElementById('ch-tip'), HL=null, ONLY=false;
  var LG={}; try{LG=JSON.parse(document.getElementById('logo-data').textContent)}catch(e){}
  function mark(g,t,x,y,size,cl,tipHtml){var hl=t.id===HL, sz=hl?size*1.6:size, node;
    if(hl)el('circle',{cx:x,cy:y,r:sz/2+4,'class':'hlring'},g);
    if(LG[t.id]){node=el('image',{href:LG[t.id],x:x-sz/2,y:y-sz/2,width:sz,height:sz,'class':'lgi'+cl},g);
      var hit=el('circle',{cx:x,cy:y,r:sz/2,'class':'hit'},g); hover(hit,tipHtml)}
    else{node=el('circle',{cx:x,cy:y,r:hl?7:4,'class':cls(t)},g); hover(node,tipHtml)}
    return node}
  var byName={}; D.teams.forEach(function(t){byName[t.n.toLowerCase()]=t.id});
  function el(t,a,p){var e=document.createElementNS(NS,t); for(var k in a)e.setAttribute(k,a[k]); if(p)p.appendChild(e); return e}
  function txt(p,x,y,s,c,anc){var t=el('text',{x:x,y:y,'class':c||'tick','text-anchor':anc||'start'},p); t.textContent=s; return t}
  function ticks(lo,hi,n){var sp=hi-lo, st=Math.pow(10,Math.floor(Math.log10(sp/n))), e=n/sp*st;
    if(e<=.15)st*=10; else if(e<=.35)st*=5; else if(e<=.75)st*=2; var o=[]; for(var v=Math.ceil(lo/st)*st; v<=hi+1e-9; v+=st)o.push(+v.toFixed(6)); return o}
  function frame(id,W,H,xd,yd,xl,yl,invY,xfmt){var box=document.getElementById(id); box.innerHTML='';
    var s=el('svg',{viewBox:'0 0 '+W+' '+H,'class':'chart'},box), m={l:56,r:16,t:14,b:44};
    var sx=function(v){return m.l+(v-xd[0])/(xd[1]-xd[0])*(W-m.l-m.r)};
    var sy=function(v){var f=(v-yd[0])/(yd[1]-yd[0]); if(invY)f=1-f; return H-m.b-f*(H-m.t-m.b)};
    var xt=ticks(xd[0],xd[1],8); xt.forEach(function(v){el('line',{x1:sx(v),x2:sx(v),y1:m.t,y2:H-m.b,'class':'grid'},s);
      txt(s,sx(v),H-m.b+16,xfmt?xfmt(v):v,'tick','middle')});
    ticks(yd[0],yd[1],7).forEach(function(v){el('line',{x1:m.l,x2:W-m.r,y1:sy(v),y2:sy(v),'class':'grid'},s); txt(s,m.l-6,sy(v)+4,v,'tick','end')});
    txt(s,(m.l+W-m.r)/2,H-6,xl,'axl','middle'); var yl_=txt(s,14,(m.t+H-m.b)/2,yl,'axl','middle');
    yl_.setAttribute('transform','rotate(-90 14 '+((m.t+H-m.b)/2)+')');
    var cp=el('clipPath',{id:id+'-clip'},s); el('rect',{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.t-m.b},cp);
    var g=el('g',{'clip-path':'url(#'+id+'-clip)'},s);
    return {s:s,g:g,sx:sx,sy:sy,m:m,W:W,H:H}}
  function q(arr,f){var a=arr.slice().sort(function(x,y){return x-y}); return a[Math.max(0,Math.min(a.length-1,Math.floor(f*(a.length-1))))]}
  function dom(vals,keep,pad){var all=vals.concat(keep); return [Math.min.apply(null,all)-pad, Math.max.apply(null,all)+pad]}
  function Labeler(F){var boxes=[]; return function(x,y,s,c,force){var w=s.length*6.4+4, h=13, right=x+8+w>F.W-F.m.r;
    var cands=[[y-6],[y+14],[y-18]]; for(var i=0;i<cands.length;i++){var yy=cands[i][0], x0=right?x-8-w:x+7, r=[x0,yy-h+2,x0+w,yy+2];
      if(r[1]<F.m.t||r[3]>F.H-F.m.b)continue; var hit=boxes.some(function(b){return !(r[2]<b[0]||r[0]>b[2]||r[3]<b[1]||r[1]>b[3])});
      if(!hit||(force&&i===cands.length-1)){boxes.push(r); txt(F.g,right?x-8:x+7,yy,s,c,right?'end':'start'); return true}}
    return false}}
  function tipOn(e,h){tip.innerHTML=h; tip.style.display='block'; var x=(e.touches?e.touches[0]:e).clientX, y=(e.touches?e.touches[0]:e).clientY;
    tip.style.left=Math.min(x+14, window.innerWidth-270)+'px'; tip.style.top=(y+14)+'px'}
  function tipOff(){tip.style.display='none'}
  function hover(node,h){node.addEventListener('mousemove',function(e){tipOn(e,h)}); node.addEventListener('mouseleave',tipOff);
    node.addEventListener('click',function(e){tipOn(e,h); e.stopPropagation()})}
  document.addEventListener('click',tipOff);
  function esc(s){return String(s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]})}
  function cls(t){var c='pt'; if(D.trap&&t.in)c+=' in'; if(ONLY&&t.today)c+=' today'; if(t.id===HL)c+=' hl'; else if(ONLY&&!t.today)c+=' dim'; return c}
  function labelSet(key,n){var top=D.teams.slice().sort(function(a,b){return b[key]-a[key]}).slice(0,n).map(function(t){return t.id}); var set={};
    top.forEach(function(i){set[i]=1}); if(HL)set[HL]=1; return set}
  function teamTip(t){return '<b>'+esc(t.n)+'</b> ('+esc(t.rec)+')<br>AdjEM '+(t.em>0?'+':'')+t.em.toFixed(1)+(t.rk?' (#'+t.rk+')':'')+
    '<br>AdjO '+t.o.toFixed(1)+' &middot; AdjD '+t.d.toFixed(1)+'<br>Tempo '+t.t.toFixed(1)+(t.elo?'<br>Elo '+Math.round(t.elo):'')+
    (D.trap?'<br>'+(t.in?'Inside':'Outside')+' the trapezoid':'')+(t.today?'<br><i>playing on this slate</i>':'')}
  function trapezoid(){var T=D.teams.slice().sort(function(a,b){return b.em-a.em}).filter(function(t,i){return i<D.top_n||t.id===HL}), xs=T.map(function(t){return t.t}), ys=T.map(function(t){return t.em});
    var cx=D.champs.map(function(c){return D.t_avg+c.rt}), cy=D.champs.map(function(c){return c.em});
    var hl=D.teams.filter(function(t){return t.id===HL});
    var xd=dom(xs,cx.concat(hl.map(function(t){return t.t})),1), yd=dom(ys,cy.concat(hl.map(function(t){return t.em})),3); yd[1]+=3;
    var F=frame('ch-trap',900,560,xd,yd,'Adjusted tempo (possessions per 40 minutes)','Adjusted efficiency margin (pts / 100 poss)');
    if(D.trap){var tr=D.trap, w=function(y){return tr.w0+tr.k*(y-tr.y0)}, y1=Math.max(tr.y1,yd[1]);
      el('polygon',{points:[[tr.xc-w(tr.y0),tr.y0],[tr.xc+w(tr.y0),tr.y0],[tr.xc+w(y1),y1],[tr.xc-w(y1),y1]].map(function(p){return F.sx(p[0])+','+F.sy(p[1])}).join(' '),'class':'trap'},F.g)}
    el('line',{x1:F.sx(D.t_avg),x2:F.sx(D.t_avg),y1:F.m.t,y2:F.H-F.m.b,'class':'zero'},F.g); el('line',{x1:F.m.l,x2:F.W-F.m.r,y1:F.sy(0),y2:F.sy(0),'class':'zero'},F.g);
    var lab=labelSet('em',10);
    T.slice().sort(function(a,b){return (a.id===HL)-(b.id===HL)||a.today-b.today}).forEach(function(t){
      mark(F.g,t,F.sx(t.t),F.sy(t.em),30,(D.trap&&!t.in?' out':'')+(ONLY&&!t.today&&t.id!==HL?' dim':''),teamTip(t))});
    var L=Labeler(F); T.filter(function(t){return t.id===HL}).forEach(function(t){L(F.sx(t.t)+18,F.sy(t.em)-8,t.n,'lab hl',true)});
    if(!Object.keys(LG).length)T.filter(function(t){return lab[t.id]&&t.id!==HL}).sort(function(a,b){return b.em-a.em}).forEach(function(t){L(F.sx(t.t),F.sy(t.em),t.n,'lab')});
    D.champs.forEach(function(c){var x=F.sx(D.t_avg+c.rt), y=F.sy(c.em);
      var st=el('path',{d:'M'+x+','+(y-7)+' L'+(x+2)+','+(y-2)+' L'+(x+7)+','+(y-2)+' L'+(x+3)+','+(y+1)+' L'+(x+5)+','+(y+6)+' L'+x+','+(y+3)+' L'+(x-5)+','+(y+6)+' L'+(x-3)+','+(y+1)+' L'+(x-7)+','+(y-2)+' L'+(x-2)+','+(y-2)+' Z','class':'champ'},F.g);
      hover(st,'<b>'+(c.s-1)+'-'+String(c.s).slice(2)+' champion: '+esc(c.n)+'</b><br>AdjEM '+(c.em>0?'+':'')+c.em.toFixed(1)+' going into the title game<br>Tempo '+(c.rt>=0?'+':'')+c.rt.toFixed(1)+' vs that season\\'s average (shown at the same gap from this season\\'s)');
      L(x,y+9,"'"+String(c.s).slice(2)+' '+c.n,'clab')});
    var ins=D.teams.filter(function(t){return t.in}).sort(function(a,b){return b.em-a.em});
    document.getElementById('ch-trap-list').innerHTML=D.trap?('<b>'+ins.length+' teams inside the trapezoid:</b> '+ins.map(function(t){return esc(t.n)}).join(', ')):'Not enough past champions in the data to draw the trapezoid.'}
  function landscape(){var T=D.teams.slice().sort(function(a,b){return b.em-a.em}).filter(function(t,i){return i<D.top_n||t.id===HL}), xs=T.map(function(t){return t.o}), ys=T.map(function(t){return t.d});
    var hl=D.teams.filter(function(t){return t.id===HL});
    var xd=dom(xs,hl.map(function(t){return t.o}),2), yd=dom(ys,hl.map(function(t){return t.d}),2);
    var F=frame('ch-land',900,560,xd,yd,'Adjusted offense (pts scored / 100 poss, right = better)','Adjusted defense (pts allowed / 100, up = better)',true);
    var mo=T.reduce(function(a,t){return a+t.o},0)/T.length, md=T.reduce(function(a,t){return a+t.d},0)/T.length;
    el('line',{x1:F.sx(mo),x2:F.sx(mo),y1:F.m.t,y2:F.H-F.m.b,'class':'zero'},F.g); el('line',{x1:F.m.l,x2:F.W-F.m.r,y1:F.sy(md),y2:F.sy(md),'class':'zero'},F.g);
    [-20,-10,0,10,20,30].forEach(function(c){var x0=xd[0], x1=xd[1]; el('line',{x1:F.sx(x0),y1:F.sy(x0-c),x2:F.sx(x1),y2:F.sy(x1-c),'class':'tier'},F.g);
      var xl=yd[0]+c+1.5; if(xl>x0+2&&xl<x1-2)txt(F.g,F.sx(xl),F.sy(yd[0])+14,'AdjEM '+(c>0?'+':'')+c,'tierl','end')});
    var lab=labelSet('em',12);
    T.slice().sort(function(a,b){return (a.id===HL)-(b.id===HL)||a.today-b.today}).forEach(function(t){
      mark(F.g,t,F.sx(t.o),F.sy(t.d),28,(ONLY&&!t.today&&t.id!==HL?' dim':''),teamTip(t))});
    var L=Labeler(F); T.filter(function(t){return t.id===HL}).forEach(function(t){L(F.sx(t.o)+16,F.sy(t.d)-6,t.n,'lab hl',true)});
    if(!Object.keys(LG).length)T.filter(function(t){return lab[t.id]&&t.id!==HL}).sort(function(a,b){return b.em-a.em}).forEach(function(t){L(F.sx(t.o),F.sy(t.d),t.n,'lab')})}
  var PAL=['#c2410c','#2563eb','#16a34a','#9333ea','#db2777','#0891b2','#ca8a04','#4b5563','#dc2626','#65a30d'];
  function eloRace(){var ids=D.teams.filter(function(t){return t.elo&&D.elo[t.id]&&D.elo[t.id].length>1}).sort(function(a,b){return b.elo-a.elo}).slice(0,10).map(function(t){return t.id});
    if(HL&&ids.indexOf(HL)<0&&D.elo[HL])ids.push(HL); if(!ids.length){document.getElementById('ch-elo').innerHTML='<p class="note">No games yet.</p>';return}
    var pts=[]; ids.forEach(function(i){D.elo[i].forEach(function(p){pts.push([Date.parse(p[0]),p[1]])})});
    var xd=[Math.min.apply(null,pts.map(function(p){return p[0]})), Math.max.apply(null,pts.map(function(p){return p[0]}))+86400000];
    var yv=pts.map(function(p){return p[1]}), yd=[Math.min.apply(null,yv)-20, Math.max.apply(null,yv)+20];
    var F=frame('ch-elo',900,460,xd,yd,'Date','Elo rating',false,function(v){var d=new Date(v); return (d.getUTCMonth()+1)+'/'+d.getUTCDate()});
    var leg='', name={}; D.teams.forEach(function(t){name[t.id]=t.n});
    ids.forEach(function(i,k){var col=i===HL?'var(--accent)':PAL[k%PAL.length], ser=D.elo[i];
      var d=ser.map(function(p,j){return (j?'L':'M')+F.sx(Date.parse(p[0]))+','+F.sy(p[1])}).join(' ');
      var pth=el('path',{d:d,'class':'eline'+(i===HL?' hl':''),stroke:col},F.g); var last=ser[ser.length-1];
      hover(pth,'<b>'+esc(name[i])+'</b><br>Elo now '+Math.round(last[1])+'<br>Season start '+Math.round(ser[0][1]));
      leg+='<span><i style="background:'+col+'"></i>'+(LG[i]?'<img src="'+LG[i]+'" style="width:16px;height:16px;vertical-align:middle;margin-right:3px">':'')+esc(name[i])+' '+Math.round(last[1])+'</span>';
      if(LG[i])el('image',{href:LG[i],x:F.sx(Date.parse(last[0]))-11,y:F.sy(last[1])-11,width:22,height:22},F.g)});
    document.getElementById('ch-elo-leg').innerHTML=leg}
  function luck(){var L=D.luck.slice(); if(ONLY)L=L.filter(function(t){return t.today});
    L.sort(function(a,b){return b.v-a.v}); var pick=L.length>24?L.slice(0,12).concat(L.slice(-12)):L; if(HL&&!pick.some(function(t){return t.id===HL})){
      var h=L.filter(function(t){return t.id===HL})[0]; if(h){pick.push(h); pick.sort(function(a,b){return b.v-a.v})}}
    var box=document.getElementById('ch-luck'); box.innerHTML=''; if(!pick.length){box.innerHTML='<p class="note">No games yet.</p>';return}
    var W=900, rowH=20, H=pick.length*rowH+40, m={l:150,r:60,t:10}; var s=el('svg',{viewBox:'0 0 '+W+' '+H,'class':'chart'},box);
    var mx=Math.max.apply(null,pick.map(function(t){return Math.abs(t.v)}))||1, cx=m.l+(W-m.l-m.r)/2, half=(W-m.l-m.r)/2-46;
    el('line',{x1:cx,x2:cx,y1:m.t,y2:H-24,'class':'zero'},s);
    pick.forEach(function(t,i){var y=m.t+i*rowH, w=Math.abs(t.v)/mx*half;
      var b=el('rect',{x:t.v>=0?cx:cx-w,y:y+3,width:Math.max(w,1),height:rowH-6,rx:3,'class':'bar '+(t.v>=0?'pos':'neg')+(t.id===HL?' hl':'')},s);
      hover(b,'<b>'+esc(t.n)+'</b><br>'+t.w+' wins vs '+t.x.toFixed(1)+' expected in '+t.g+' D1 games<br>'+(t.v>=0?'+':'')+t.v.toFixed(1)+' wins of luck');
      txt(s,m.l-(LG[t.id]?28:8),y+rowH/2+4,t.n,'lab'+(t.id===HL?' hl':''),'end');
      if(LG[t.id])el('image',{href:LG[t.id],x:m.l-23,y:y+1,width:rowH-2,height:rowH-2},s); txt(s,t.v>=0?cx+w+5:cx-w-5,y+rowH/2+4,(t.v>=0?'+':'')+t.v.toFixed(1),'tick',t.v>=0?'start':'end')});
    txt(s,cx,H-6,'Wins minus the wins the model expected before each game','axl','middle')}
  function all(){trapezoid(); landscape(); eloRace(); luck()}
  var inp=document.getElementById('ch-find'); inp.addEventListener('change',function(){var v=inp.value.trim().toLowerCase(); HL=byName[v]||null; all()});
  inp.addEventListener('input',function(){if(!inp.value){HL=null; all()}});
  document.getElementById('ch-only').addEventListener('change',function(e){ONLY=e.target.checked; all()});
  var drawn=false; document.querySelectorAll('.toptab').forEach(function(b){b.addEventListener('click',function(){if(b.dataset.p==='p-charts'&&!drawn){drawn=true; all()}})});
  if(!document.getElementById('p-charts').hidden){drawn=true; all()}
})();
"""


def small_logo_url(url):
    m = re.match(r"https?://a\.espncdn\.com(/i/teamlogos/.+)", url or "")
    return f"https://a.espncdn.com/combiner/i?img={m.group(1)}&h={LOGO_PX}&w={LOGO_PX}" if m else url


def download_data_uri(url, timeout=6):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data, ctype = resp.read(), resp.headers.get_content_type()
        if data and ctype.startswith("image/"):
            return f"data:{ctype};base64," + base64.b64encode(data).decode()
    except Exception:
        pass
    return None


def _cached_logo(t, url):
    """Data URI for one team logo, from the disk cache (cbb_data_cache/logos) or downloaded once and saved there."""
    folder = CACHE_DIR / "logos"
    f = folder / f"{t}_{LOGO_PX}.png"
    if f.exists() and f.stat().st_size > 100:
        return "data:image/png;base64," + base64.b64encode(f.read_bytes()).decode()
    if OFFLINE:
        return None
    d = download_data_uri(small_logo_url(url))
    if d and d.startswith("data:image/png;base64,"):
        try:
            folder.mkdir(parents=True, exist_ok=True)
            f.write_bytes(base64.b64decode(d.split(",", 1)[1]))
        except Exception:
            pass
    return d


def load_logos(team_ids, directory):
    """team_id -> image source (embedded data, or a link when offline / LOGO_MODE = "link"). Logos are saved in
    cbb_data_cache/logos, so each one is downloaded only once."""
    if LOGO_MODE == "none":
        return {}
    urls = {t: directory.get(t, {}).get("logo") for t in team_ids}
    urls = {t: u for t, u in urls.items() if u}
    if LOGO_MODE != "embed":
        return {t: small_logo_url(u) for t, u in urls.items()}
    from concurrent.futures import ThreadPoolExecutor
    items = list(urls.items())
    with ThreadPoolExecutor(max_workers=16) as ex:
        got = list(ex.map(lambda kv: _cached_logo(kv[0], kv[1]), items))
    out = {t: (d or small_logo_url(u)) for (t, u), d in zip(items, got)}
    n = sum(1 for v in out.values() if v.startswith("data:"))
    print(f"  Logos: {n} of {len(out)} embedded" + ("" if n == len(out) else " (the rest link online)"))
    return out


def write_html_report(pred, report, st, names, logos, tuning, charts=None):
    e = html.escape
    n_teams = len(st) if st is not None else 360
    logos = logos or {}
    logo_json = json.dumps({str(t): u for t, u in logos.items()}).replace("</", "<\\/")

    def tm(team, big=False, short=True):
        t_ = e(names(team))
        if team in logos:
            return f'<span class="tl"><span class="logo{" lg" if big else ""} logo-{team}"></span>{t_}</span>'
        return t_

    def fmt(v, f):
        return "-" if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else f.format(v)

    def better(a, h, higher=True):
        if a is None or h is None or pd.isna(a) or pd.isna(h) or abs(a - h) < 1e-9:
            return None
        return "a" if (a > h) == higher else "h"

    def cmp_row(label, a_html, h_html, win=None):
        ca = "" if win is None else ("good" if win == "a" else "bad")
        ch = "" if win is None else ("good" if win == "h" else "bad")
        return f'<tr><td class="v {ca}">{a_html}</td><td class="lbl">{e(label)}</td><td class="v {ch}">{h_html}</td></tr>'

    def rank_badge(rk):
        if rk is None or pd.isna(rk):
            return ""
        cls = "good" if rk <= n_teams / 3 else ("bad" if rk > 2 * n_teams / 3 else "mid")
        return f'<span class="rk {cls}">#{int(rk)}</span>'

    def S(team, col):
        if st is None or team not in st.index or col not in st.columns:
            return np.nan
        return st.at[team, col]

    def head(r):
        return (f'<tr><th class="tm">{tm(r["away_id"])}</th><th></th><th class="tm">{tm(r["home_id"])}</th></tr>')

    def ctx(grp):
        return "" if ACTIVE_FEATURE_GROUPS is None or grp in ACTIVE_FEATURE_GROUPS else " (context, not in the model)"

    def model_table(r):
        spec = [("Elo rating", "elo_pre", True, "{:.0f}"),
                ("Adjusted offense (pts / 100 poss)", "adjo", True, "{:.1f}"),
                ("Adjusted defense (pts allowed / 100)", "adjd", False, "{:.1f}"),
                ("Adjusted tempo (poss / 40 min)", "adjt", None, "{:.1f}"),
                ("Offense rating, Four Factors (0-100)", "off_unit", True, "{:.0f}"),
                ("Defense rating, Four Factors (0-100)", "def_unit", True, "{:.0f}"),
                ("Recent form vs expectation (per 100)" + ctx("Recent form"), "form", True, "{:+.1f}"),
                ("Days of rest" + ctx("Rest"), "rest", True, "{:.0f}"),
                ("Missing players' value (Game Score / game)" + ctx("Injuries"), "inj", False, "{:.1f}"),
                ("Games played this season", "gp", None, "{:.0f}")]
        rows = []
        for lbl, k, hi, f in spec:
            av, hv = r.get("a_" + k), r.get("h_" + k)
            sc = 100 if k in ("off_unit", "def_unit") else 1
            av = av * sc if av is not None and not pd.isna(av) else av
            hv = hv * sc if hv is not None and not pd.isna(hv) else hv
            rows.append(cmp_row(lbl, fmt(av, f), fmt(hv, f), None if hi is None else better(av, hv, hi)))
        for k in RANK_STAT_WEIGHTS:
            for side, idx in (("off", 0), ("def", 1)):
                lab, f, scl = RANK_STAT_META[k][idx], RANK_STAT_META[k][2], RANK_STAT_META[k][3]
                av, hv = r.get(f"a_ur_{k}_{side}_pre"), r.get(f"h_ur_{k}_{side}_pre")
                hi = RANK_DIRECTION[k][0 if side == "off" else 1]
                rows.append(cmp_row(lab + (" (adj.)" if OPP_ADJUST else ""), fmt(av * scl if av == av else av, f),
                                    fmt(hv * scl if hv == hv else hv, f), better(av, hv, hi)))
        return f'<table class="cmp">{head(r)}{"".join(rows)}</table>'

    def records_table(r):
        a, h = r["away_id"], r["home_id"]
        if st is None or a not in st.index or h not in st.index:
            return '<p class="note">Season stats are not available yet (non-D1 team or no games played).</p>'
        rows = [cmp_row("Record", e(S(a, "record")), e(S(h, "record"))),
                cmp_row("Home / Road / Neutral", e(f'{S(a, "home_record")} / {S(a, "away_record")} / {S(a, "neutral_record")}'),
                        e(f'{S(h, "home_record")} / {S(h, "away_record")} / {S(h, "neutral_record")}')),
                cmp_row("Conference record", e(S(a, "conf_record")), e(S(h, "conf_record")))]
        for label, col, f in (("Adjusted efficiency margin", "adj_em", "{:+.1f}"), ("Elo rating", "elo", "{:.0f}"),
                              ("Point differential / game", "pd_pg", "{:+.1f}")):
            ra, rh = S(a, col + "_rk"), S(h, col + "_rk")
            rows.append(cmp_row(label, fmt(S(a, col), f) + rank_badge(ra), fmt(S(h, col), f) + rank_badge(rh), better(ra, rh, False)))
        rows.append(cmp_row("Schedule strength (opponents' avg margin)", fmt(S(a, "sos"), "{:+.1f}") + rank_badge(S(a, "sos_rk")),
                            fmt(S(h, "sos"), "{:+.1f}") + rank_badge(S(h, "sos_rk"))))

        def l5(lst):
            if not isinstance(lst, list) or not lst:
                return '<span class="note">no games yet</span>'
            res = ["W" if g_["pf"] > g_["pa"] else "L" for g_ in lst]
            hd = f'{res.count("W")}&ndash;{res.count("L")} &middot; {"".join(res)}'
            body = "".join(
                f'<div><b class="{"gw" if x == "W" else "gl"}">{x}</b> {g_["pf"]}&ndash;{g_["pa"]} '
                f'{"vs" if g_["neutral"] else ("vs" if g_["home"] else "@")} {e(names(g_["opp"]))}'
                f'{rank_badge(S(g_["opp"], "adj_em_rk"))} <span class="note">{g_["day"]:%b %d}</span></div>'
                for g_, x in zip(lst, res))
            return f'<details class="l5"><summary>{hd}</summary>{body}</details>'
        rows.append(cmp_row("Last 5 games (tap to open)", l5(S(a, "last5")), l5(S(h, "last5"))))
        return f'<table class="cmp">{head(r)}{"".join(rows)}</table>'

    MATCHUP_PAIRS = [("Adjusted efficiency ★", "adj_o", "adj_d", "{:.1f}"),
                     ("Four Factors rating ★", "off_unit", "def_unit", "{:.2f}"),
                     ("Points / 100 poss (raw)", "off_eff", "def_eff", "{:.1f}"),
                     ("Effective FG% ★", "off_efg", "def_efg", "{:.1%}"),
                     ("Turnover rate ★", "off_tov", "def_tov", "{:.1%}"),
                     ("Offensive rebound % ★", "off_orb", "def_orb", "{:.1%}"),
                     ("Free-throw rate ★", "off_ftr", "def_ftr", "{:.1%}"),
                     ("2-point FG% ★", "off_two", "def_two", "{:.1%}"),
                     ("3-point FG% ★", "off_three", "def_three", "{:.1%}"),
                     ("3-point attempt rate", "off_3par", "def_3par", "{:.1%}"),
                     ("Steals ★", "off_stl", "def_stl", "{:.1%}"),
                     ("Blocks ★", "off_blk", "def_blk", "{:.1%}")]

    def matchup_panel(off_t, def_t, side_off, side_def):
        if st is None or off_t not in st.index or def_t not in st.index:
            return '<p class="note">Season stats are not available yet.</p>'
        rows, oa, da = [], 0, 0
        for label, oc, dc, f in MATCHUP_PAIRS:
            ro, rd = S(off_t, oc + "_rk"), S(def_t, dc + "_rk")
            win = better(ro, rd, higher=False) if "attempt" not in label else None
            oa += win == "a"
            da += win == "h"
            rows.append(cmp_row(label, fmt(S(off_t, oc), f) + rank_badge(ro), fmt(S(def_t, dc), f) + rank_badge(rd), win))
        verdict = (f"Even matchup ({oa}-{da} by D1 rank)" if oa == da else
                   (f"Edge: {e(names(off_t))} offense, {oa} of {oa + da} categories by D1 rank" if oa > da
                    else f"Edge: {e(names(def_t))} defense, {da} of {oa + da} categories by D1 rank"))
        return (f'<table class="cmp"><tr><th class="tm">{tm(off_t)} offense<br><span class="rec">{side_off}</span></th><th></th>'
                f'<th class="tm">{tm(def_t)} defense<br><span class="rec">{side_def}</span></th></tr>{"".join(rows)}</table>'
                f'<p class="note">{verdict}. Left is the offense, right the defense it faces; ranks are among D1 teams (#1 best). '
                f'★ = stats behind the model\'s ratings. Season-to-date numbers, not adjusted for opponents (the model\'s own '
                f'numbers on the Team stats tab are).</p>')

    def loss_txt(x):
        if x["loss"] < 0:
            return f"+{-x['loss']:.1f} back"
        return "-" if x["loss"] == 0 else f"{x['loss']:.1f}"

    def injury_panel(team, gid):
        info = INJ_NOW.get((int(gid), int(team)))
        if not info or not info["rows"]:
            return f'<div class="panel"><h4>{tm(team)}</h4><p class="note">No player data yet for this team this season.</p></div>'
        body = "".join(
            f'<tr><td><b>{e(x["name"])}</b></td><td class="num">{x["mpg"]:.1f}</td><td class="num">{x["val"]:.1f}</td>'
            f'<td><span class="{"pr full" if x["status"] == "Available" or x["loss"] < 0 else "st out"}">{e(x["status"])}</span></td>'
            f'<td class="num imp {"back" if x["loss"] < 0 else "high" if x["loss"] >= 2 else ("medium" if x["loss"] >= 0.8 else ("low" if x["loss"] > 0 else "none"))}">'
            f'{loss_txt(x)}</td></tr>' for x in info["rows"])
        return (f'<div class="panel"><h4>{tm(team)} rotation</h4><div class="scroll"><table><tr><th>Player</th><th class="num">Min / game</th>'
                f'<th class="num">Value / 40</th><th>Status</th><th class="num">Impact</th></tr>{body}</table></div>'
                f'<p class="note">Bench (backups) value: {info["bench"]:.1f} per 40 minutes. Total missing: <b>{info["loss"]:.1f}</b> '
                'Game Score points per game.</p></div>')

    def inj_count(r):
        n = sum(1 for t in (r["away_id"], r["home_id"]) for x in INJ_NOW.get((int(r["game_id"]), int(t)), {}).get("rows", [])
                if x["status"] != "Available" and x["loss"] >= 0)
        return f' <span class="cnt">{n}</span>' if n else ""

    def why_html(r):
        w = r.get("why")
        if not w:
            return ""
        li = lambda xs: "".join(f"<li>{e(t)}</li>" for t in xs)
        hist = f'<p class="whyhist">{e(w["hist"])}</p>' if w["hist"] else ""
        chips = "".join(f'<span class="shift {"up" if v_ >= 0.5 else ("dn" if v_ <= -0.5 else "")}">{e(n_)} {v_:+.1f}</span>'
                        for n_, v_ in w.get("shifts", []))
        if chips:
            hist += (f'<p class="whyhist" style="margin-bottom:4px">How far each factor moves the pick, in points of win probability '
                     f'versus an even matchup (+ helps {e(names(w["pick"]))}):</p><div class="shifts">{chips}</div>')
        return (f'<div class="whybox"><div class="panel"><h4>Why the model picks {e(names(w["pick"]))}</h4>'
                f'<p class="whyconf"><b>{e(w["word"])}</b> &middot; {w["pp"] * 100:.0f}% chance</p><ul>{li(w["reasons"])}</ul>{hist}</div>'
                f'<div class="panel"><h4>Concerns about this pick</h4><ul>{li(w["concerns"])}</ul>'
                f'<p class="whyhist">The model uses: {e(", ".join(g_.lower() for g_ in (ACTIVE_FEATURE_GROUPS or FEATURE_GROUPS)))}. '
                'Injuries come from who missed each team\'s last game, plus any you add by hand; a new injury or suspension '
                'is not known until a game is missed.</p></div></div>')

    def vegas_box(r):
        sp = r.get("v_spread")
        if sp is None or pd.isna(sp):
            return '<div class="vegas"><span class="k">Vegas</span><span class="sub">no line found</span></div>'
        ml = lambda v: "" if v is None or pd.isna(v) else f" &middot; ML {int(v):+d}"
        num = lambda x: f"{x:+g}" if x != 0 else "PK"
        ph = r["winner"] == r["home_id"]
        a_row = f'<span class="vr{"" if ph else " pk"}">{e(names(r["away_id"]))} {num(sp)}{ml(r.get("v_ml_away"))}</span>'
        h_row = f'<span class="vr{" pk" if ph else ""}">{e(names(r["home_id"]))} {num(-sp)}{ml(r.get("v_ml_home"))}</span>'
        tot = r.get("v_total")
        sub = f'<span class="sub">Total {tot:g}</span>' if tot is not None and not pd.isna(tot) else ""
        fav = None if sp == 0 else (r["home_id"] if sp > 0 else r["away_id"])
        ag = ""
        if fav is not None:
            agree = fav == r["winner"]
            ag = f'<span class="ag {"yes" if agree else "no"}">{"Model agrees on the winner" if agree else "Model picks the underdog"}</span>'
        ps = round(r["pred_margin"] * 2) / 2
        mod = "PK" if ps == 0 else f'{e(names(r["home_id"] if ps > 0 else r["away_id"]))} {-abs(ps):g}'
        mm = f'<span class="sub">Model spread: <b>{mod}</b></span>'
        flag = ""
        vp_ = r.get("v_prob")
        if vp_ is not None and not pd.isna(vp_) and abs(r["p_home"] - vp_) * 100 >= MARKET_FLAG_PTS:
            liked = r["home_id"] if r["p_home"] > vp_ else r["away_id"]
            flag = f'<span class="flag">Model vs Vegas: {e(names(liked))} +{abs(r["p_home"] - vp_) * 100:.0f} pts</span>'
        ats = ""
        if r.get("completed"):
            marg = int(r["home_score"]) - int(r["away_score"])
            ats = ('<span class="sub">Spread: push</span>' if marg == sp else
                   f'<span class="sub">Spread: {e(names(r["home_id"] if marg > sp else r["away_id"]))} covered</span>')
        return f'<div class="vegas"><span class="k">Vegas</span>{a_row}{h_row}{sub}{mm}{ag}{flag}{ats}</div>'

    rows_ = pred["rows"]
    latest_html = ""
    rec0 = report.get("record")
    if rec0 and rec0.get("rows"):
        prev = [x for x in rec0["rows"] if x["day"] < pred["days"][0]]
        if prev:
            ld = max(x["day"] for x in prev)
            lr = [x for x in prev if x["day"] == ld]
            w_ = sum(x["ok"] for x in lr)
            body = "".join(
                f'<tr><td>{tm(x["away"])} @ {tm(x["home"])}</td><td>{tm(x["pick"])}</td><td class="num">{x["conf"] * 100:.0f}%</td>'
                f'<td class="num">{e(x["score"])}</td><td><b style="color:{"#2e9e5b" if x["ok"] else "#c0392b"}">'
                f'{"Correct" if x["ok"] else "Wrong"}</b></td></tr>' for x in sorted(lr, key=lambda x: -x["conf"]))
            latest_html = (f'<details class="edges"><summary style="cursor:pointer"><b>Latest results, {ld:%a %b %d}: {w_}-{len(lr) - w_} '
                           f'({100 * w_ / len(lr):.0f}%)</b> <span class="note">tap to see every game</span></summary>'
                           '<div class="scroll" style="margin-top:8px"><table><tr><th>Game</th><th>Pick</th><th class="num">Confidence</th>'
                           f'<th class="num">Final (away-home)</th><th>Result</th></tr>{body}</table></div>'
                           '<div>D1-vs-D1 games, picked before tip-off by the model trained on earlier seasons (the same picks as the season '
                           'record).</div></details>')
    has_odds = any(r.get("v_prob") == r.get("v_prob") and r.get("v_prob") is not None for r in rows_)
    # ---- where the model and the market disagree most ----
    edge_rows = []
    for gi, r in enumerate(rows_):
        vp_ = r.get("v_prob")
        if r.get("completed") or vp_ is None or pd.isna(vp_):
            continue
        gap = (r["p_home"] - vp_) * 100
        if abs(gap) >= MARKET_FLAG_PTS:
            who = r["home_id"] if gap > 0 else r["away_id"]
            mp_ = r["p_home"] if gap > 0 else 1 - r["p_home"]
            vm_ = vp_ if gap > 0 else 1 - vp_
            edge_rows.append((abs(gap), f'<div><b>{e(names(who))}</b>: model {mp_ * 100:.0f}% vs market {vm_ * 100:.0f}% '
                                        f'({e(names(r["away_id"]))} at {e(names(r["home_id"]))}, {abs(gap):.0f} pts apart)</div>'))
    edges_html = ""
    if edge_rows:
        edge_rows.sort(key=lambda t: -t[0])
        edges_html = ('<div class="edges"><h3>Where the model differs most from Vegas</h3>' + "".join(h for _, h in edge_rows[:8])
                      + '<div>The market knows things this model does not (injuries, lineup news), so treat these as games to look at, '
                        'not automatic bets.</div></div>')

    # ---- game list + game cards ----
    items, cards = [], []
    for gi, r in enumerate(rows_):
        a, h = r["away_id"], r["home_id"]
        conf = max(r["p_home"], 1 - r["p_home"]) * 100
        tier = "Strong" if conf >= 75 else ("Moderate" if conf >= 62 else "Lean")
        home_wins = r["p_home"] >= 0.5
        at = "vs" if r["neutral"] else "@"
        site = (f'Neutral site{": " + e(str(r.get("venue_full_name"))) if isinstance(r.get("venue_full_name"), str) else ""}'
                if r["neutral"] else (e(str(r.get("venue_full_name"))) if isinstance(r.get("venue_full_name"), str) else ""))
        note = r.get("notes_headline")
        note = f" &middot; {e(note)}" if isinstance(note, str) and note and note.lower() != "nan" else ""
        tip = f'{r["tip"]:%a %b %d, %I:%M %p}'.replace(" 0", " ")
        rec = (f'<span class="rec">{e(tip)} &middot; {site}{note}</span>'
               f'<span class="rec">{e(names(a))} {e(S(a, "record") if S(a, "record") == S(a, "record") else "")} &middot; '
               f'{e(names(h))} {e(S(h, "record") if S(h, "record") == S(h, "record") else "")}</span>')
        fin = ""
        if r.get("completed"):
            fa, fh = int(r["away_score"]), int(r["home_score"])
            right = home_wins == (fh > fa)
            fin = f'<span class="fin {"ok" if right else "no"}">FINAL {fa}&ndash;{fh} &middot; {"Pick correct" if right else "Pick wrong"}</span>'
            chip = f'<span class="chip {"ok" if right else "no"}">{"&#10003;" if right else "&#10007;"} {fa}&ndash;{fh}</span>'
        elif r.get("v_prob") == r.get("v_prob") and r.get("v_prob") is not None and abs(r["p_home"] - r["v_prob"]) * 100 >= MARKET_FLAG_PTS:
            chip = '<span class="chip flag">&ne; Vegas</span>'
        else:
            chip = f'<span class="chip">{tier}</span>'
        score_html = (f'{e(names(a))} <span class="n{"" if home_wins else " w"}">{r["score_away"]}</span> &ndash; '
                      f'<span class="n{" w" if home_wins else ""}">{r["score_home"]}</span> {e(names(h))}')
        q = f'{names(a)} {names(h)} {directory_name(a)} {directory_name(h)}'.lower()
        tip_s = f'{r["tip"]:%I:%M %p}'.lstrip("0")
        items.append(f'<button type="button" class="gitem" data-g="{gi}" data-played="{1 if r.get("completed") else 0}" data-q="{e(q)}">'
                     f'<span class="gi-teams">{tm(a)}<span class="at">{at}</span>{tm(h)}</span>'
                     f'<span class="gi-pick">{tm(r["winner"])} <b>{conf:.0f}%</b> &middot; {e(tip_s)}</span>{chip}</button>')
        vline = ""
        vp_ = r.get("v_prob")
        if vp_ is not None and not pd.isna(vp_):
            vline = (f' Vegas (moneyline, no-vig): {e(names(h))} {vp_ * 100:.1f}% (model is {abs(r["p_home"] - vp_) * 100:.1f} points '
                     f'{"higher" if r["p_home"] > vp_ else "lower"} on {e(names(h))}).')
        cards.append(f"""
<section class="gdetail" id="g{gi}" hidden>
<div class="ghead">
  <div class="teams">{tm(a, True)}<span class="at">{at}</span>{tm(h, True)}{rec}</div>
  <div class="pick"><span class="k">Pick</span><span class="who">{tm(r["winner"], True)}</span></div>
  <div class="score"><span class="k">Predicted score</span>{score_html}{fin}</div>
  <div class="conf"><span class="k">Confidence</span><span class="pct">{conf:.1f}%</span><span class="tier">{tier}</span>
    <span class="bar"><span class="fill" style="width:{conf:.1f}%"></span></span></div>
  {vegas_box(r) if has_odds else ""}
</div>
<nav class="subtabs"><button type="button" data-t="over">Overview</button><button type="button" data-t="match">Matchups</button><button type="button" data-t="stats">Model inputs</button><button type="button" data-t="inj">Injuries{inj_count(r)}</button></nav>
<div class="tp" data-t="over">
  {why_html(r)}
  <div class="probline">Win probability: {e(names(a))} {100 - r["p_home"] * 100:.1f}% &middot; {e(names(h))} {r["p_home"] * 100:.1f}%.{vline}</div>
  <div class="panel"><h4>Records and overall quality</h4>{records_table(r)}</div>
</div>
<div class="tp" data-t="match" hidden>
  <div class="probline">Green marks the stronger side for each stat and red the weaker (by D1 rank).</div>
  <div class="cols">
    <div class="panel"><h4>{e(names(a))} offense vs {e(names(h))} defense</h4>{matchup_panel(a, h, "Away" if not r["neutral"] else "", "Home" if not r["neutral"] else "")}</div>
    <div class="panel"><h4>{e(names(h))} offense vs {e(names(a))} defense</h4>{matchup_panel(h, a, "Home" if not r["neutral"] else "", "Away" if not r["neutral"] else "")}</div>
  </div>
</div>
<div class="tp" data-t="inj" hidden>
  <div class="probline">Each team's rotation over its last 5 games. A player who missed the team's last game is presumed out (until he plays
  again); add known injuries in MANUAL_INJURIES at the top of the script. Value / 40 = Game Score per 40 minutes (a box-score rating;
  about 10 is average for a rotation player, 20+ is a star). Impact = his value over the team's backups, scaled by his minutes: what the team loses per game.</div>
  <div class="cols">{injury_panel(a, r["game_id"])}{injury_panel(h, r["game_id"])}</div>
</div>
<div class="tp" data-t="stats" hidden>
  <div class="panel"><h4>Model inputs (pre-game)</h4>{model_table(r)}
  <p class="note">Four Factors numbers are {e(UNIT_WINDOW_TEXT)}. Adjusted efficiency = points per 100 possessions after removing
  opponent strength and home court. Form = how far above (+) or below its expected margin the team played over its last {FORM_GAMES} games.</p></div>
</div>
</section>""")

    # ---- model picks summary ----
    mp_rows = []
    for rank_, (gi, r) in enumerate(sorted(enumerate(rows_), key=lambda t: -max(t[1]["p_home"], 1 - t[1]["p_home"])), 1):
        conf = max(r["p_home"], 1 - r["p_home"]) * 100
        vp_ = r.get("v_prob")
        if vp_ is None or pd.isna(vp_):
            veg = '<span class="mp-v">-</span>'
        else:
            mk = vp_ if r["winner"] == r["home_id"] else 1 - vp_
            veg = f'<span class="mp-v{" far" if abs(r["p_home"] - vp_) * 100 >= MARKET_FLAG_PTS else ""}">Vegas {mk * 100:.0f}%</span>'
        res = ""
        if r.get("completed"):
            fa, fh = int(r["away_score"]), int(r["home_score"])
            ok = (r["winner"] == r["home_id"]) == (fh > fa)
            res = f'<span class="mp-res {"ok" if ok else "no"}">{"&#10003;" if ok else "&#10007;"} {fa}&ndash;{fh}</span>'
        mp_rows.append(f'<tr class="mp-row" data-g="{gi}"><td class="num">{rank_}</td><td class="mp-pick">{tm(r["winner"], True)}</td>'
                       f'<td>{tm(r["away_id"])} <span class="at">{"vs" if r["neutral"] else "@"}</span> {tm(r["home_id"])}</td>'
                       f'<td class="mp-conf"><b>{conf:.1f}%</b><span class="bar"><span class="fill" style="width:{conf:.1f}%"></span></span></td>'
                       f'<td>{e(names(r["away_id"]))} {r["score_away"]} &ndash; {r["score_home"]} {e(names(r["home_id"]))}</td>'
                       f'<td>{veg}</td><td>{res}</td></tr>')
    upsets = [(gi, r) for gi, r in enumerate(rows_) if r.get("v_spread") == r.get("v_spread") and r.get("v_spread") is not None
              and r["v_spread"] != 0 and (r["home_id"] if r["v_spread"] > 0 else r["away_id"]) != r["winner"]]
    ups_html = ""
    if upsets:
        ups_html = ('<h3 style="margin-top:20px">Underdogs the model picks to win</h3><div class="scroll"><table class="mp">'
                    + "".join(f'<tr class="mp-row" data-g="{gi}"><td class="mp-pick">{tm(r["winner"], True)}</td>'
                              f'<td>{tm(r["away_id"])} {"vs" if r["neutral"] else "@"} {tm(r["home_id"])}</td>'
                              f'<td class="num">model {max(r["p_home"], 1 - r["p_home"]) * 100:.0f}%</td>'
                              f'<td class="num">Vegas {(r["v_prob"] if r["winner"] == r["home_id"] else 1 - r["v_prob"]) * 100:.0f}%</td></tr>'
                              for gi, r in upsets) + '</table></div>')
    modelpicks_html = ('<div class="toppane" id="p-modelpicks" hidden><p class="note" style="margin-top:0">Every game, most confident pick '
                       'first. Tap a row to open that game.</p><div class="scroll"><table class="mp"><tr><th class="num">#</th><th>Pick</th>'
                       '<th>Game</th><th>Confidence</th><th>Predicted score</th><th>Market</th><th>Result</th></tr>'
                       + "".join(mp_rows) + '</table></div>' + ups_html + '</div>')

    # ---- record & accuracy ----
    rec = report.get("record")
    record_section = ""
    if rec:
        n_, w_ = rec["n"], rec["w"]
        wk_rows = "".join(f'<tr><td>Week of {k}</td><td class="num">{a_}-{b_ - a_}</td><td class="num">{100 * a_ / b_:.0f}%</td></tr>'
                          for k, a_, b_ in rec["weeks"])
        band_rows = "".join(f'<tr><td>{e(b["label"])} confidence</td><td class="num">{b["w"]}-{b["n"] - b["w"]}</td>'
                            f'<td class="num">{100 * b["w"] / b["n"]:.0f}%</td></tr>' for b in rec["bands"])
        game_rows = "".join(
            f'<tr><td>{x["day"]:%b %d}</td><td>{tm(x["away"])} @ {tm(x["home"])}</td><td>{tm(x["pick"])}</td>'
            f'<td class="num">{x["conf"] * 100:.0f}%</td><td class="num">{e(x["score"])}</td>'
            f'<td style="color:{"#2e9e5b" if x["ok"] else "#c0392b"}"><b>{"Correct" if x["ok"] else "Wrong"}</b></td></tr>'
            for x in reversed(rec["rows"]))
        home_txt = (f' Always picking the home team (true home games only) is {rec["home_w"]}-{rec["home_n"] - rec["home_w"]} '
                    f'({100 * rec["home_w"] / rec["home_n"]:.1f}%).' if rec["home_n"] else "")
        record_section = (
            f'<details class="section"><summary>Record so far in {PREDICT_SEASON}: {w_}-{n_ - w_} ({100 * w_ / n_:.1f}%)</summary><div class="inner">'
            f'<p class="note">Every completed {PREDICT_SEASON} D1-vs-D1 game, picked with only pre-game information by a model trained on '
            f'earlier seasons. Log-loss {rec["ll"]:.3f}. Elo alone is {rec["elo_w"]}-{n_ - rec["elo_w"]} ({100 * rec["elo_w"] / n_:.1f}%).{home_txt}</p>'
            '<div class="cols"><div class="panel"><h4>By week</h4><div class="scroll"><table>' + wk_rows + '</table></div></div>'
            '<div class="panel"><h4>By confidence</h4><div class="scroll"><table>' + band_rows + '</table></div></div></div>'
            '<details style="margin-top:12px"><summary>Most recent games</summary><div class="scroll"><table>'
            '<tr><th>Day</th><th>Game</th><th>Pick</th><th class="num">Confidence</th><th class="num">Final (away-home)</th><th>Result</th></tr>'
            + game_rows + '</table></div></details></div></details>')

    perf = '<p class="note">No completed test games available yet.</p>'
    if report.get("acc") is not None:
        stats = [("Accuracy", f'{report["acc"]:.1%}'), ("Log-loss", f'{report["ll"]:.3f}'),
                 ("Elo-only accuracy", f'{report["elo_acc"]:.1%}'), ("Elo-only log-loss", f'{report["elo_ll"]:.3f}'),
                 ("Home team wins (true home games)", f'{report["base_acc"]:.1%}'),
                 ("Margin error (avg miss, pts)", f'{report["mae_margin"]:.1f}'),
                 ("Total points error (avg miss, pts)", f'{report["mae_total"]:.1f}')]
        stats_html = "".join(f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>' for l, v in stats)
        wf = "".join(f'<tr><td>{s_ - 1}-{str(s_)[2:]}</td><td class="num">{n_:,}</td><td class="num">{a_ * 100:.1f}%</td>'
                     f'<td class="num">{l_:.4f}</td><td class="num">{ea_ * 100:.1f}%</td><td class="num">{el_:.4f}</td></tr>'
                     for s_, n_, a_, l_, ea_, el_ in report.get("wf_table", []))
        cal = "".join(f'<tr><td>{lo * 100:.0f}-{min(hi, 1) * 100:.0f}%</td><td class="num">{n_:,}</td><td class="num">{avg_ * 100:.1f}%</td>'
                      f'<td class="num">{acc_ * 100:.1f}%</td></tr>' for lo, hi, n_, acc_, avg_ in report.get("conf_hist", []))
        el = "".join(f'<tr><td>{e(a_)}</td><td class="num">{n_:,}</td><td class="num">{acc_ * 100:.1f}%</td></tr>'
                     for a_, n_, acc_ in report.get("early_late", []))
        perf = (f'<div class="stats">{stats_html}</div><p class="note">Walk-forward test: each season {report["wf_first"]}-{report["wf_last"]} '
                f'was predicted by a model trained only on earlier seasons ({report["n_test"]:,} D1-vs-D1 games). The live model is fit on all '
                f'{report["n_train"]:,} completed games. A coin flip has log-loss 0.693. Guessing the average margin would miss by '
                f'{report["mae_margin_base"]:.1f} pts and the average total by {report["mae_total_base"]:.1f} pts.</p>'
                '<div class="grid" style="margin-top:12px"><div class="panel"><h4>By season</h4><div class="scroll"><table><tr><th>Season</th>'
                '<th class="num">Games</th><th class="num">Model acc</th><th class="num">Model LL</th><th class="num">Elo acc</th>'
                f'<th class="num">Elo LL</th></tr>{wf}</table></div></div>'
                '<div class="panel"><h4>Calibration</h4><div class="scroll"><table><tr><th>Confidence</th><th class="num">Games</th>'
                f'<th class="num">Predicted</th><th class="num">Won</th></tr>{cal}</table></div>'
                '<p class="note">If the model is well calibrated, "Predicted" and "Won" match.</p>'
                f'<div class="scroll"><table><tr><th>Part of the season</th><th class="num">Games</th><th class="num">Accuracy</th></tr>{el}</table></div></div></div>')
    coefs = report["coefs"].reindex(report["coefs"].abs().sort_values(ascending=False).index)
    top = max(coefs.abs().max(), 1e-9)
    coef_rows = "".join(
        f'<tr><td>{e(FEATURE_LABELS.get(k, k))}</td><td class="num">{v:+.3f}</td>'
        f'<td class="num">{"-" if report["feat_mult"].get(k, 1.0) == 1.0 else "x%g" % report["feat_mult"][k]}</td>'
        f'<td><span class="cbar {"p" if v >= 0 else "n"}" style="width:{abs(v) / top * 140:.0f}px"></span></td></tr>'
        for k, v in coefs.items())
    sub = "".join(f'<div style="display:flex;justify-content:space-between;gap:16px;padding:2px 0 2px 12px;font-size:.92em;opacity:.85">'
                  f'<span>{e(n)}</span><span>{v:.1%}</span></div>' for n, v in sorted(report.get("unit_parts", {}).items(), key=lambda kv: -kv[1]))
    share_rows = ""
    for g_, sv in sorted(report["shares"].items(), key=lambda kv: -kv[1]):
        if g_ == "Four Factors units" and sub:
            share_rows += (f'<tr><td colspan="2"><details><summary style="cursor:pointer;display:flex;justify-content:space-between">'
                           f'<span>{e(g_)} <small>(click for stats)</small></span><span>{sv:.1%}</span></summary>{sub}</details></td></tr>')
        else:
            share_rows += f'<tr><td>{e(g_)}</td><td class="num">{sv:.1%}</td></tr>'
    tune_html = "".join(f"<li>{e(t)}</li>" for t in tuning)
    perf_section = (f'<details class="section"><summary>Model performance and weights</summary><div class="inner">{perf}'
                    f'<div class="grid" style="margin-top:14px"><div class="scroll"><table><tr><th>Feature</th><th class="num">Weight</th>'
                    f'<th class="num">Multiplier</th><th></th></tr>{coef_rows}</table></div>'
                    f'<div class="scroll"><table><tr><th>Factor</th><th class="num">Share of weight</th></tr>{share_rows}</table></div></div>'
                    '<p class="note">Weights are standardized logistic-regression coefficients (no intercept): positive favors the home team. '
                    'The away-offense-vs-home-defense matchup is expected to be negative.</p>'
                    f'<h4 style="margin-top:14px">Settings chosen by the tuning tests</h4><ul class="note">{tune_html}</ul></div></details>')

    ab_section = ""
    ab = report.get("ablation")
    if ab:
        def read_(r, key="alone_vs_none", noise="noise", neg=True):
            v = r[key] if neg else -r[key]
            return '<span class="good-t">clear signal</span>' if v < -2 * r[noise] else '<span class="meh-t">not clearly better than without it</span>'
        grows = "".join(
            f'<tr><td>{e(r["name"])}</td><td>{"yes" if r["in_model"] else "no"}</td><td class="num">{r["alone_ll"]:.4f}</td>'
            f'<td class="num">{r["alone_acc"] * 100:.1f}%</td><td class="num">{r["alone_vs_none"]:+.4f}</td>'
            f'<td class="num">{r["cost_of_removing"]:+.4f} <small>(±{r["cost_noise"]:.4f})</small></td>'
            f'<td>{read_(r, "cost_of_removing", "cost_noise", neg=False)}</td></tr>'
            for r in sorted(ab["groups"], key=lambda r: r["alone_vs_none"]))
        srows = "".join(
            f'<tr><td>{e(r["name"])}</td><td class="num">{RANK_STAT_WEIGHTS.get(r["key"], 0) / sum(RANK_STAT_WEIGHTS.values()) * 100:.0f}%</td>'
            f'<td class="num">{r["alone_ll"]:.4f}</td><td class="num">{r["alone_acc"] * 100:.1f}%</td>'
            f'<td class="num">{r["alone_vs_none"]:+.4f}</td><td>{read_(r)}</td></tr>'
            for r in sorted(ab["stats"], key=lambda r: r["alone_vs_none"]))
        ab_section = (
            '<details class="section"><summary>How well does each factor predict?</summary><div class="inner">'
            f'<p class="note">{ab["n_test"]:,} completed D1 games from {ab["seasons"]}; each season predicted by a model trained on the other '
            f'seasons. Home court alone scores log-loss {ab["none_ll"]:.4f} ({ab["none_acc"] * 100:.1f}% correct); all active factors together '
            f'{ab["full_ll"]:.4f} ({ab["full_acc"] * 100:.1f}%). Lower log-loss is better.</p>'
            '<h4>Factor groups</h4><div class="scroll"><table><tr><th>Group</th><th>In model</th><th class="num">Alone: log-loss</th>'
            '<th class="num">Alone: accuracy</th><th class="num">Alone vs home court only</th><th class="num">Value in the full model</th><th>Read</th></tr>'
            + grows + '</table></div><p class="note">"Value in the full model": how much worse the full model gets WITHOUT the group (for a group '
            'that is not in the model: how much better it gets by ADDING it). Positive = helps. "Clear signal" = more than twice the noise. '
            'The strong factors overlap a lot (Elo, efficiency and Four Factors all measure team quality), so each one adds less on top of the '
            'others than it does alone.</p>'
            '<h4 style="margin-top:16px">Four Factors stats, each alone</h4><div class="scroll"><table><tr><th>Stat</th><th class="num">Weight in the unit rating</th>'
            '<th class="num">Log-loss</th><th class="num">Accuracy</th><th class="num">vs home court only</th><th>Read</th></tr>'
            + srows + '</table></div><p class="note">Each row is a model using only that stat\'s offense-vs-defense edge for both teams '
            '(plus home court). Learned weights only; your multipliers are ignored here.</p></div></details>')
    cp = report.get("compare")
    if cp and cp.get("rows"):
        crow = "".join(f'<tr><td>{e(r["name"])}</td><td class="num">{r["acc"] * 100:.2f}%</td><td class="num">{r["ll"]:.4f}</td>'
                       f'<td class="num">{r["vs_cur"]:+.4f}</td><td class="num">{r["noise"]:.4f}</td></tr>' for r in cp["rows"])
        ab_section += ('<details class="section"><summary>Is a simpler model just as good?</summary><div class="inner">'
                       f'<p class="note">{cp["n"]:,} games, each season predicted by a model trained only on earlier seasons; every row uses the '
                       'same games. "vs current" is the log-loss difference from the shipped model (negative = better); gaps smaller than about '
                       'twice the noise are not reliable.</p><div class="scroll"><table><tr><th>Set-up</th><th class="num">Accuracy</th>'
                       '<th class="num">Log-loss</th><th class="num">vs current</th><th class="num">Noise</th></tr>' + crow + '</table></div></div></details>')

    # ---- teams & ratings ----
    teams_html = '<p class="note">No completed games yet this season.</p>'
    if st is not None:
        def tcell(v, f, rk_=None, sortv=None):
            sv = sortv if sortv is not None else v
            dv = "" if sv is None or pd.isna(sv) else f' data-v="{float(sv):.6g}"'
            return f'<td class="num"{dv}>{fmt(v, f)}{rank_badge(rk_) if rk_ is not None else ""}</td>'
        order = st.sort_values("adj_em", ascending=False)
        cols = [("AdjEM", "adj_em", "{:+.1f}"), ("AdjO", "adj_o", "{:.1f}"), ("AdjD", "adj_d", "{:.1f}"), ("Tempo", "adj_t", "{:.1f}"),
                ("Elo", "elo", "{:.0f}"), ("Off FF", "off_unit", "{:.2f}"), ("Def FF", "def_unit", "{:.2f}"), ("SOS", "sos", "{:+.1f}"),
                ("Pts/G", "ppg", "{:.1f}"), ("Opp/G", "oppg", "{:.1f}")]
        body = "".join(
            f'<tr><td class="num" data-v="{i}">{i}</td><td><b>{tm(t)}</b></td><td data-v="{int(str(row["record"]).split("-")[0])}">{e(row["record"])}</td>'
            + "".join(tcell(row[c], f, row.get(c + "_rk")) for _, c, f in cols) + "</tr>"
            for i, (t, row) in enumerate(order.iterrows(), 1))
        power = (f'<input class="tfilter" data-t="t-power" placeholder="Find a team..."><div class="scroll"><table class="sortable" id="t-power">'
                 f'<tr><th>#</th><th>Team</th><th>Record</th>{"".join(f"<th class=num>{c[0]}</th>" for c in cols)}</tr>{body}</table></div>'
                 '<p class="note">Sorted by adjusted efficiency margin (AdjEM = AdjO - AdjD, points per 100 possessions vs an average D1 team on '
                 'a neutral floor). Off FF / Def FF = Four Factors unit ratings (0-1, 1 = best). SOS = average AdjEM of opponents faced. '
                 'Click a column to sort.</p>')

        def unit_tbl(side, idx):
            heads = "".join(f'<th class="num">{e(RANK_STAT_META[k][idx])}</th>' for k in RANK_STAT_WEIGHTS)
            sub_ = st.dropna(subset=[f"{side}_unit"]).sort_values(f"{side}_unit", ascending=False)
            body_ = "".join(
                f'<tr><td class="num" data-v="{i}">{i}</td><td><b>{tm(t)}</b></td>{tcell(x[side + "_unit"] * 100, "{:.0f}")}'
                + "".join(tcell(x[f"ur_{k}_{side}_pre"] * RANK_STAT_META[k][3], RANK_STAT_META[k][2]) for k in RANK_STAT_WEIGHTS) + "</tr>"
                for i, (t, x) in enumerate(sub_.iterrows(), 1))
            tid = f"t-{side}"
            return (f'<input class="tfilter" data-t="{tid}" placeholder="Find a team..."><div class="scroll"><table class="sortable" id="{tid}">'
                    f'<tr><th>#</th><th>Team</th><th class="num">Rating</th>{heads}</tr>{body_}</table></div>')
        tw = sum(RANK_STAT_WEIGHTS.values())
        recipe = ", ".join(f"{RANK_STAT_NAMES[k]} {w / tw:.0%}" for k, w in RANK_STAT_WEIGHTS.items() if w > 0)
        unote = (f'<p class="note">Rating (0-100) blends each team\'s D1 percentile on: {e(recipe)}. The numbers use {e(UNIT_WINDOW_TEXT)}, '
                 'as of the games shown. Click a column to sort.</p>')
        teams_html = (f'<details class="section"><summary>Power ratings</summary><div class="inner">{power}</div></details>'
                      f'<details class="section"><summary>Offense rankings (Four Factors)</summary><div class="inner">{unit_tbl("off", 0)}{unote}</div></details>'
                      f'<details class="section"><summary>Defense rankings (Four Factors)</summary><div class="inner">{unit_tbl("def", 1)}{unote}</div></details>')

    charts_html = ('<div class="toppane" id="p-charts" hidden><p class="note" style="margin-top:0">No chart data yet this season.</p></div>')
    if charts:
        opts = "".join(f'<option value="{e(t["n"])}">' for t in sorted(charts["teams"], key=lambda t: t["n"]))
        n_ch = len(charts["champs"])
        s0, s1 = (charts["champs"][0]["s"], charts["champs"][-1]["s"]) if n_ch else (0, 0)
        yrs = f'{s0 - 1}-{str(s0)[2:]} through {s1 - 1}-{str(s1)[2:]}' if n_ch else ""
        charts_html = (
            '<div class="toppane" id="p-charts" hidden>'
            '<div class="chctl"><input type="text" id="ch-find" list="ch-teams" placeholder="Highlight a team...">'
            f'<datalist id="ch-teams">{opts}</datalist>'
            '<label><input type="checkbox" id="ch-only"> Only teams playing on this slate</label>'
            '<span class="note" style="margin:0">Hover or tap a logo for details. In the trapezoid chart, faded logos are outside the zone.</span></div>'
            f'<div class="chbox"><h3>Trapezoid of Excellence</h3><p class="desc">The top {TRAP_TOP_N} teams by adjusted efficiency margin, by pace (adjusted tempo) and quality '
            '(adjusted efficiency margin; a highlighted team is shown even if it ranks lower). Title winners almost always combine an elite margin with a tempo that is not extreme, '
            f'and the allowed tempo range widens the better a team is. The green zone is fit to the {n_ch} national champions in this data '
            f'({yrs}, gold stars, placed at the same distance from the average tempo as in their season), using their ratings going into the '
            'title game. Inside = a championship-type profile; it is a historical pattern, not a guarantee.</p>'
            '<div id="ch-trap"></div><div class="chlist" id="ch-trap-list"></div></div>'
            f'<div class="chbox"><h3>Efficiency landscape</h3><p class="desc">The top {TRAP_TOP_N} teams: offense (right = scores more per possession) against defense '
            '(up = allows fewer). Dotted diagonals are equal adjusted efficiency margins; teams in the top right are good on both ends, which '
            'is the profile that tends to hold up in March.</p><div id="ch-land"></div></div>'
            '<div class="chbox"><h3>Elo race</h3><p class="desc">Elo through this season for the current top 10 (plus a highlighted team). '
            'Steep climbs are teams beating good opponents by a lot.</p><div id="ch-elo"></div><div class="legend" id="ch-elo-leg"></div></div>'
            '<div class="chbox"><h3>Luck: wins above or below expectation</h3><p class="desc">Actual wins minus the wins the model expected '
            'before each game (D1 games only). Teams far to the right have won more close or unlikely games than their play suggests; '
            'luck usually does not last, so their record may flatter them. Most extreme 12 at each end.</p><div id="ch-luck"></div></div>'
            '</div><div id="ch-tip"></div>'
            '<script type="application/json" id="chart-data">' + json.dumps(charts).replace("</", "<\\/") + '</script>')
    stamp = datetime.now().strftime("%b %d, %Y at %I:%M %p")
    test_note = (f'<p class="note"><b>Test mode:</b> pretending today is {e(AS_OF_DATE)}; later results are hidden.</p>' if AS_OF_DATE else "")
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>College Basketball Predictions - {e(pred["label"])}</title>
<style>{REPORT_CSS}</style>
<script type="application/json" id="logo-data">{logo_json}</script>
<script>(function(){{var L=JSON.parse(document.getElementById('logo-data').textContent), c=[];
for(var k in L)c.push('.logo-'+k+'{{background-image:url("'+L[k]+'")}}'); var st=document.createElement('style');
st.textContent=c.join(''); document.head.appendChild(st)}})();</script></head><body><div class="wrap">
<h1>College Basketball Predictions: {e(pred["label"])}</h1>
<p class="sub">Generated {stamp}. {len(rows_)} games. Pick a game on the left (or search for a team), then use the tabs. Confidence is the win
probability of the picked team; the predicted score splits a projected total using that probability. Green marks the stronger side, red the weaker.</p>
{test_note}
<nav class="toptabs"><button type="button" class="toptab" data-p="p-games">Games</button><button type="button" class="toptab" data-p="p-modelpicks">Model picks</button><button type="button" class="toptab" data-p="p-charts">Charts</button><button type="button" class="toptab" data-p="p-model">Record &amp; accuracy</button><button type="button" class="toptab" data-p="p-league">Teams &amp; ratings</button></nav>
<div class="toppane" id="p-games">
{latest_html}
{edges_html}
<div class="app"><div class="gside"><input class="gsearch" placeholder="Search a team..."><div class="glist">{"".join(items)}</div></div><div class="gview">{"".join(cards)}</div></div>
</div>
{modelpicks_html}
{charts_html}
<div class="toppane" id="p-model" hidden><div class="secgroup">{record_section}{perf_section}{ab_section}</div></div>
<div class="toppane" id="p-league" hidden><div class="secgroup">{teams_html}</div></div>
<p class="note">Predictions are statistical estimates for fun, not betting advice. Data: ESPN via the hoopR-mbb-data project.</p>
<script>{REPORT_JS}</script>
<script>{CHART_JS}</script>
</div></body></html>"""
    if os.environ.get("CBB_OUTPUT_PATH"):
        path = Path(os.environ["CBB_OUTPUT_PATH"])
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path = Path(__file__).resolve().parent / REPORT_FILENAME
    path.write_text(page, encoding="utf-8")
    return path


_DIR = {}


def directory_name(t):
    v = _DIR.get(t, {})
    return v.get("name", "")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    global PREDICT_SEASON
    t0 = time.time()
    cur = CURRENT_SEASON or season_of(today_local())
    PREDICT_SEASON = cur
    seasons = list(range(FIRST_SEASON, cur + 1))
    print(f"Loading schedules and box scores {seasons[0]}-{seasons[-1]} (the first run downloads ~85 MB, later runs use the cache) ...")
    sched, box, players = load_season_files(seasons, cur)
    t_ = today_local()
    games = build_games(sched, fetch_live_finals([t_ - pd.Timedelta(days=1), t_]))
    _DIR.update(team_directory(sched, box))
    names = lambda t: _DIR.get(t, {}).get("short", str(t))
    print(f"  {len(games):,} games ({int(games['completed'].sum()):,} completed), {len(box):,} team box scores, "
          f"{len(players):,} player box scores")

    print("Building per-game team stats (possessions, efficiency, Four Factors) ...")
    gs = build_game_stats(box, games)
    tg = build_team_games(games, gs)
    d1set = set(zip(tg.loc[tg["d1"] == 1, "team"], tg.loc[tg["d1"] == 1, "season"]))
    conf_map = team_conferences(tg)
    hp = tg[tg["is_home"] == 1][["game_id", "poss", "tempo"]]
    games = games.merge(hp, on="game_id", how="left")
    games["both_d1"] = [((h, s) in d1set) and ((a, s) in d1set) for h, a, s in zip(games["home_id"], games["away_id"], games["season"])]
    tg = add_rest(tg)

    # phantom rows (one per D1 team of the current season, at the first day being predicted) carry each team's
    # CURRENT ratings for the Teams & ratings tables; they never enter the model
    days = pick_days(games)
    ph_day = days[0] if days else today_local()
    cur_d1 = sorted({t for t, s in d1set if s == cur})
    ph = pd.DataFrame({"game_id": [-(i + 1) for i in range(len(cur_d1))], "season": cur, "day": ph_day,
                       "tip": ph_day - pd.Timedelta(minutes=1), "team": cur_d1, "opp": -1, "is_home": -1, "neutral": 0,
                       "conf_game": 0, "conf_id": np.nan, "pts_for": np.nan, "pts_against": np.nan, "completed": False,
                       "d1": 1, "opp_d1": 0})
    tg = pd.concat([tg, ph], ignore_index=True).sort_values(["team", "tip", "game_id"]).reset_index(drop=True)
    tg["sgi"] = tg.groupby(["team", "season"]).cumcount()
    tg["gp"] = tg.groupby(["team", "season"])["completed"].transform(lambda x: x.astype(int).cumsum().shift(1)).fillna(0)

    tuning = []
    unit_res = choose_unit_config(tg[tg["game_id"] > 0].reset_index(drop=True))
    tuning.append(f"Four Factors numbers: {UNIT_WINDOW_TEXT}" + (f" (best of {len(unit_res)} settings tried)" if unit_res else ""))
    tg = add_unit_inputs(tg)
    tg = add_unit_ranks(tg)

    print("Building Elo and efficiency ratings ...")
    elo_kw, elo_info = choose_elo_params(games, d1set, conf_map)
    elo, elo_ratings = compute_elo(games, d1set, conf_map, **elo_kw)
    tuning.append(f"Elo: K {elo_kw['k']:.1f}, home court {elo_kw['hfa']:.0f} Elo points, margin of victory {'on' if elo_kw['mov'] else 'off'}, "
                  f"{elo_kw['carry']:.0%} of last season kept (rest regressed toward the "
                  f"{'conference' if elo_kw['regress'] == 'conf' else 'D1'} average), early-season K x{elo_kw['early']:g}")
    eff_kw, eff_rms = choose_eff_params(games, d1set)
    eff, eff_now = compute_eff_ratings(games, d1set, **eff_kw)
    tuning.append(f"Efficiency ratings: speed {eff_kw['lr']:g}, {eff_kw['carry']:.0%} carried into the next season, home court "
                  f"{eff_kw['hfa']:g} pts per 100 possessions" + (f" (margin error {eff_rms:.1f} pts)" if eff_rms else ""))
    tg = add_form(tg, eff)
    tg = add_continuity(tg, players)
    tg = add_injuries(tg, players, games, days, _DIR)
    tg["elo_dev"] = [ELO_PRIOR_DEV.get((t_, s_), 0.0) for t_, s_ in zip(tg["team"], tg["season"])]
    if tg["rot_cont"].notna().any():
        cm = tg.loc[tg["d1"] == 1, "rot_cont"]
        print(f"  Roster continuity: on average {cm.mean():.0%} of a team's minutes go to players back from last season "
              f"({tg.loc[(tg['d1'] == 1) & (tg['season'] == cur), 'rot_cont'].mean():.0%} this season)")
    phantom = tg[tg["game_id"] < 0]
    tg = tg[tg["game_id"] > 0]

    print("Building matchup features ...")
    mg = build_matchups(games, tg, elo, eff)
    feats = all_model_features()
    mg[feats + TOTAL_FEATS] = mg[feats + TOTAL_FEATS].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    model, report = train_and_evaluate(mg)
    report["record"] = season_record(report["record_model"], mg)
    print_season_record(report["record"])
    st = build_season_team_stats(tg, games, elo_ratings, eff_now, cur, phantom)
    pred = predict_upcoming(model, mg, report, names, st)
    if pred is None:
        return
    print("Loading team logos ...")
    ids = {t for r in pred["rows"] for t in (r["home_id"], r["away_id"])}
    logos = load_logos(ids | (set(st.index) if st is not None else set()), _DIR)
    try:
        charts = build_chart_data(st, games, eff, elo, mg, report, names, ids)
    except Exception as ex:
        print(f"  ! charts skipped ({ex})")
        charts = None
    path = write_html_report(pred, report, st, names, logos, tuning, charts)
    print(f"\nShareable report written to:\n  {path}   ({time.time() - t0:.0f}s total)")
    print("Send that single .html file to a friend; it opens in any web browser.")
    if OPEN_REPORT_IN_BROWSER:
        webbrowser.open(path.as_uri())


PREDICT_SEASON = CURRENT_SEASON

if __name__ == "__main__":
    main()