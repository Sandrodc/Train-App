# Prediction Model Plan

> How we build a delay and connection prediction model that the whole app can run on.
> Companion to [APP_IDEA.md](APP_IDEA.md).

_Last updated: 4 November 2026_

**Goal:** a model that predicts a *range* of outcomes and is honestly calibrated. When it says 80%, it must be right about 80% of the time. That honesty is the product: the Commute Confidence Score is only worth something if users learn to trust it.

---

## Core principles

- **Predict distributions, not single numbers.** "Arrives 08:14, likely between 08:12 and 08:19" beats "08:14". Connection risk and confidence scores all come from the distribution.
- **Calibration over accuracy.** A slightly less accurate but well-calibrated model beats a sharp, overconfident one. Wrong-and-confident kills trust fastest.
- **Point-in-time correctness.** Every training example only uses what was knowable at the moment of prediction. Leaking future data is the #1 way models look great offline and fail live.
- **Beat a named benchmark.** The bar is NS's own real-time prediction. If we can't beat it at a given horizon, show NS's number and spend effort elsewhere.
- **Win where NS is weakest.** NS predicts well once a train is already late. Our edge is 30–90 minutes *before* departure, when the board still says "on time", and on transfers.
- **Evaluation before models.** Build the scoring harness first so every model is judged the same way.

---

## Step 1 — Define exactly what we predict

| Target | Type | Output | Feeds |
| --- | --- | --- | --- |
| Arrival delay at a station | Regression | Quantiles p10 / p50 / p90 (minutes) | Predicted arrival, delay prediction |
| Departure delay at a station | Regression | Quantiles p10 / p50 / p90 (minutes) | Connection risk |
| Cancellation / partial cancellation | Classification | Probability 0–1 | Confidence score, alternatives |

Every prediction has a **horizon**: how long before scheduled departure we predict. Train and evaluate per horizon bucket:

| Horizon | Knowable inputs | Use |
| --- | --- | --- |
| T−24h → T−2h | History, calendar, weather forecast, planned works | Pre-commute notification |
| T−90 → T−15 min | + live network state (active disruptions, upstream delays) | **Our sweet spot** |
| T−15 min → arrival | + the train's own live delay | Match NS, add transfer advice |

**V1 scope:** NS trains on 5–10 corridors that interviewees actually ride (e.g. Rotterdam–Amsterdam, Utrecht hub routes). Use the whole network for features, predict only on these.

---

## Step 2 — Gather the data

**Big unlock:** we don't need to wait months for history. [Rijden de Treinen](https://www.rijdendetreinen.nl/en/open-data) publishes every Dutch train service since 2019 and every disruption since 2011, free with attribution.

| Source | What it gives | Role | When |
| --- | --- | --- | --- |
| [RdT train archive](https://www.rijdendetreinen.nl/en/open-data/train-archive) | Planned vs actual times, delays, cancellations per stop, 2019–now (monthly CSVs, ~30 MB each) | Training labels + history features | Week 1 |
| [RdT disruptions](https://www.rijdendetreinen.nl/en/open-data/disruptions) | All disruptions since 2011: cause, stations, lines, duration (CC BY 4.0) | Disruption features, propagation research | Week 1 |
| NDOV Loket live feeds (InfoPlus via ZeroMQ) | Live departures, trip details, platform changes, revisions | Live features + our own archive of prediction revisions | Week 2 |
| NS API (Reisinformatie, Virtual Train) | Crowd forecasts, train composition, station facilities | Crowding + boarding position | Later |
| KNMI open data | Hourly observed weather + forecasts | Wind, storms, heat, leaves season | Week 2 |
| Calendar | School/public holidays, events, planned engineering works | Demand + capacity | Week 2 |

The RdT archive does **not** contain two things, so start logging them ourselves now:

1. **How predictions evolved.** The archive stores final outcomes. We need what the board said at T−60, T−30, T−10, to benchmark against NS and train horizon-aware models.
2. **Network state snapshots.** Every 1–2 minutes, the delay of every running train: the raw material for "how late is the network right now".

[GoTrain](https://www.rijdendetreinen.nl/en/open-data) (RdT's open-source NDOV processor, Go, GPL v3) is a strong starting point for the live ingest.

---

## Step 3 — Store it so the past can be replayed

Design storage around one question: *"What did we know at 07:12 on 4 March?"*

- **Raw layer (immutable):** every live message as received, with a `received_at` timestamp. Append only. Parquet, partitioned by day.
- **Clean layer:** one row per *train × station × service day*: planned/actual times, platform, cancellation flag. Built from RdT history + our live archive. Keys: service date, train number, station code.
- **Snapshot layer:** prediction revisions and network state keyed by `as_of`. Every feature query takes an `as_of` and only reads rows with `received_at <= as_of`.
- **Tooling:** start with **DuckDB over Parquet** (one file, fast SQL, great with Python). Move to Postgres + TimescaleDB if needed.

One shared station-code table (UIC codes + NS short codes like `UT`, `ASD`, `RTD`) saves weeks of join pain.

---

## Step 4 — Build the scoreboard before any model

One script that scores any predictor the same way, per horizon and per corridor.

| Metric | Measures | Used for |
| --- | --- | --- |
| MAE of p50 (minutes) | How close the middle guess is | Arrival/departure delay |
| Pinball (quantile) loss | Quality of the whole range | p10/p50/p90 delay |
| Interval coverage | Does p10–p90 contain the truth ~80% of the time? | Calibration of ranges |
| Brier score + reliability curve | Are probabilities honest? | Cancellation, connection made |
| Surprise rate | Share of trips >5 min worse than our p90 | The trust metric users feel |

**Rules:**
- **Split by time, never randomly.** Train 2019–2024, validate early 2025, test on the most recent months. Walk-forward backtests.
- **Flag 2020–2021.** COVID timetables and ridership distort patterns; test with and without.
- **Always report three columns:** naive baseline ("the timetable is right"), NS's official prediction at that horizon, and our model.

---

## Step 5 — Statistical baselines (Phase 1, no ML)

These ship as the first reliability scores *and* become the bar ML must beat.

1. **Empirical delay distribution** per train number × station × weekday, last 90 days → "82% on time" directly.
2. **Hierarchical fallback** for sparse cases: train → line × hour → line → station, blended by sample size.
3. **Recency weighting:** exponential decay (~30-day half-life) so timetable changes and new works show up quickly.
4. **Connection miss rate** per transfer pair: how often did the feeder arrive later than the connecting departure minus walking time?

If a well-tuned baseline is within ~10% of LightGBM later, ship the simple one: explainable and robust.

---

## Step 6 — Features (where the real edge lives)

| Group | Examples | Horizon |
| --- | --- | --- |
| Schedule | Train type (IC, Sprinter, ICD), hour, weekday, stop index, scheduled dwell, timetable buffer | All |
| History | Step 5 stats as features: train p50/p90 last 30/90 days, line + station punctuality | All |
| Calendar | Holidays, school breaks, events, Monday/Friday effect, leaves season (Oct–Nov), DST change | All |
| Weather | Wind + gusts, rain, temperature extremes, snow/ice, storm warnings | All |
| Planned works | Works on route or neighbouring lines, single-track sections, bus replacement | All |
| Network state | Avg delay on same line now, delay of the trainset's *previous trip*, active disruptions within N stations | T−90 and closer |
| Own state | Current delay, delay change last 10 min, number of revisions, NS's predicted delay | T−15 and closer |

**Likely highest-value feature: the delay of the train's previous run.** Trainsets turn around at the terminus, so a late inbound means a late outbound. Linking those trips is fiddly but worth it.

Using NS's own predicted delay as a feature is smart: the model learns when to trust it and when to correct it.

---

## Step 7 — ML models (Phase 2)

LightGBM is the default: fast on tabular data, handles missing values, trains on a laptop.

1. **Delay quantiles:** LightGBM with `objective="quantile"` (alpha 0.1 / 0.5 / 0.9). Try one model per horizon bucket vs. horizon as a feature.
2. **Cancellation:** LightGBM binary classifier. Cancellations are rare, so judge on Brier score and calibration, not accuracy.
3. **Calibrate after training:** conformal prediction (e.g. conformalized quantile regression) on a recent held-out window. This is what makes "80% chance" true.
4. **Handle the long tail separately:** most trains are 0–3 min late; disruptions create 30+ min outliers. Clip or log-transform the target; let cancellation/disruption models cover extremes.
5. **Explain every prediction:** store SHAP top-3 reasons. They become user-facing copy like "Reason: signal disruption near Utrecht".

Track every experiment (MLflow or a simple CSV log). Retrain weekly on a rolling window.

---

## Step 8 — Connection risk and Commute Confidence Score

These are products of the model, not separate models.

**Connection risk.** A connection is made when:

```
feeder arrival + walk time ≤ connecting departure
```

- Sample both predicted delay distributions (Monte Carlo, ~1,000 draws) and count how often this holds.
- **Don't assume independence.** On a bad morning both trains run late together, which often *saves* the connection. Model the joint delay, or at least the historical correlation for that pair.
- **Walk time** per platform pair per station (same-platform cross transfer at Utrecht ≈ 1 min, far platforms ≈ 4–6 min). Hand-build for V1 corridors.
- NS sometimes holds connecting trains: learn hold rates per transfer pair from history.

**Commute Confidence Score (0–100)** = probability the user arrives within X minutes of plan (X = their tolerance, default 5) × 100.
Simple, explainable, and testable: commutes scored 80 should arrive on time about 80% of the time. Avoid a hand-weighted blend of factors; it can't be validated.

---

## Step 9 — Run it live and keep it honest

- **Prediction ledger:** log every prediction (inputs, outputs, model version, `as_of`), then join the actual outcome. Ground truth for live performance and future training.
- **Live dashboard:** coverage, MAE and surprise rate per corridor per day, next to NS's numbers. Alert when coverage drops below ~75%.
- **Graceful degradation:** if the feed drops or the model is unsure, fall back to the Step 5 baseline and say so in the UI.
- **Shadow mode first:** run silently for 2–4 weeks before users see it.
- **Weekly retrain:** promote only if it beats the current model on the latest test week.

---

## Step 10 — Disruption propagation (Phase 3, the moat)

Goal: *"Signal issue near Utrecht. 67% chance your commute is affected in 45 minutes."* Start only once Steps 1–9 are solid.

1. **Network graph:** stations as nodes, track segments as edges, plus trainset rotations linking trips.
2. **Learn spread patterns** from 15 years of RdT disruptions joined with service outcomes: per cause × location, how far and fast delays spread and how long recovery takes.
3. **Start simple:** a disruption *duration* model (by cause) + a lookup of affected lines. That alone powers useful alerts.
4. **Then graph models:** neighbour-aggregated features, or a graph neural network if the simple approach plateaus.

Disruption duration prediction is underrated: "expected to last until ~09:10" is exactly what commuters lack today.

---

## Timeline

Rough pace for a solo side project (~8–10 h/week):

| Weeks | Milestone | Steps |
| --- | --- | --- |
| 1–2 | History downloaded and cleaned, live logger running | 2–3 |
| 3–4 | Scoreboard + baselines; first reliability numbers for our corridors | 4–5 |
| 5–8 | Features + LightGBM; beat baseline at T−60 | 6–7 |
| 9–10 | Connection risk + confidence score, then shadow mode | 8–9 |
| Month 4+ | Disruption duration and propagation | 10 |

## First week

- [x] Download the RdT train archive (Jan 2023 – Oct 2026, monthly) and the disruptions dataset (2011–2025)
- [x] Load into DuckDB; build the station-code table
- [ ] Pick 5–10 corridors from interviewees' commutes _(10 placeholder corridors in `pipeline/corridors.py` until interviews are done)_
- [x] Compute on-time % per train for those corridors (first reliability score)
- [x] Live logger running 24/7 on a Hetzner Arm server since 4 Nov 2026 12:18 (systemd, survives reboots; see `deploy/README.md`)
- [x] Confirm NDOV Loket live-feed terms _(CC0 on the best-effort network, max 1 connection per stream; RdT: CC BY 4.0)_

### Progress log

**4 Nov 2026 — history loaded, first reliability numbers**

- 84.2M stop rows (Jan 2023 – Oct 2026) in `data/clean/stops/` as monthly Parquet, ~845 MB. Built in a few minutes on a laptop.
- First run of `pipeline.reliability` (weekdays, 3 Aug – 30 Oct 2026, domestic trains only), arrival at destination:

  | Corridor | Trips | 0–1 min | 2–5 min | >5 min | Cancelled | p90 delay |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | Den Haag C → Utrecht C | 9,245 | 82% | 10% | 3% | 5% | 2 min |
  | Rotterdam C → Utrecht C | 7,351 | 76% | 11% | 6% | 8% | 3 min |
  | Rotterdam C → Amsterdam C | 4,931 | 75% | 13% | 5% | 7% | 3 min |
  | Utrecht C → Amsterdam C | 7,910 | 69% | 18% | 8% | 5% | 5 min |

  Same departure slot can differ a lot: RTD→ASD 07:11 IC is 81% on time, the 07:41 IC 68%. But see the scoreboard result below: much of that per-slot difference is noise.

**Data notes we learned the hard way**

- **Replacement train numbers.** During disruptions NS cancels e.g. train 2216 and runs it as 302216. Counting per train number doubles the cancellations (first pass showed 9–18%). The reliability script now collapses all variants of one departure slot into one trip; it counts as cancelled only if every variant was. Any future label building must do the same.
- **Delays are clipped at 0.** RdT has no negative delays, so early running is invisible. Fine for "late" targets, but p10 quantiles will sit on 0.
- **Cancellations are heavy-tailed by day.** On RTD→ASD, 9 Sep 2026 alone had 97 cancelled stops. Per-day averages hide this; the confidence score must come from distributions, as planned.
- **Disruptions dataset ends 31 Dec 2025.** No disruption labels for 2026 yet. Either train/validate disruption features on 2023–2025 only, or ask RdT for an up-to-date extract (they offer one to professional users).
- **Stations table is the Sept 2023 snapshot.** 68 of 629 station codes in the services data don't match it: metro stops (Rotterdam Beurs, Capelsebrug), foreign stations (Berlin Hbf, Düsseldorf-Bilk) and stations opened later. None affect the V1 corridors.
- **International trains are mixed in.** The 07:07 Eurostar showed up on RTD→ASD (59% on time). Labels now keep only domestic products (Sprinter, Intercity, Intercity direct, Eurocity Direct) and drop Eurostar/ICE/EuroCity/night trains and replacement buses. This mattered: RTD→ASD went from 68% to 75% on time, and from 12% to 5% more than 5 min late.
- **Bucket definition.** "On time" here means 0–1 min late, matching the 1-minute granularity of the data. The plan's example used "on time / 1–5 / >5"; revisit when we pick user-facing copy.

**4 Nov 2026 — scoreboard and Step 5 baselines**

- `pipeline.labels` builds one shared label table, `corridor_trips` (1.35M trips, 2023–2026). It is the single place where the variant-collapsing and domestic-only rules live.
- `pipeline.scoreboard` is the Step 4 harness. It uses a walk-forward, time-based split: **validation** Nov 2025 – Apr 2026 (for tuning) and **test** May – Oct 2026 (touched once). Every predictor gets a **leakage check**: a few days are re-predicted with history cut off at that day, and the predictions must be identical. Results are appended to `experiments/log.csv`.
- `pipeline.baselines` holds the Step 5 baselines as histogram predictors, with a 3-level hierarchy: departure slot → corridor × hour → corridor, each shrunk toward its parent with `k` pseudo-trips. Optional exponential recency weighting.
- Test-split result, horizon day-ahead (as_of = 00:00 on the service day):

  | Model | Pinball | MAE p50 | p10–p90 coverage | Above p90 | Surprise (>p90+5) | Brier cancel |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | Naive ("timetable is right") | 0.570 | 1.14 | 73% | 27% | 5.9% | 0.063 |
  | Corridor average, 90 days | 0.470 | 1.14 | 91% | 9% | 3.1% | 0.059 |
  | Slot only, 90 days | 0.496 | 1.17 | 89% | 11% | 3.1% | 0.062 |
  | **Hierarchical, k=30, half-life 60 d** (tuned on validation) | **0.466** | 1.14 | 91% | 9% | **2.9%** | 0.059 |

**What the baselines tell us**

1. **Day-ahead, history alone barely beats "this corridor's usual spread".** The tuned hierarchy wins on every split, but only ~1% over the corridor average. Slot-only is *worse* than the corridor average, so ~60 trips per departure slot is too little to separate the 07:11 from the 07:41. **Product implication:** per-train reliability scores must be shrunk toward the corridor (as `hier_tuned` does). Never show raw per-slot percentages.
2. **The median is always 0, so MAE can't move.** The value of a day-ahead model is in the range: pinball drops 18% vs naive, and the surprise rate halves (5.9% → 2.9%). That is the trust metric, and it already works.
3. **Upper tail is calibrated; the interval looks over-wide.** 9% of trips land above p90 (target 10%). Coverage of 91% vs a nominal 80% comes from integer minutes and the pile-up at 0: most trips sit exactly on q10 = q50 = 0. Judge calibration on "above p90", not raw coverage, for this target.
4. **Cancellations are not predictable day-ahead from history.** The Brier score is 0.059 vs 0.063 for "never cancelled", and the reliability curve is flat: predicted 12% → observed 10%, predicted 21% → observed 13%. Cancellations are incident-driven. The obvious missing day-ahead input is **planned engineering works**, which are known days in advance and probably explain a big share. Get them from the NS API, or reconstruct them from the RdT data (whole days with a corridor-wide cancellation pattern).
5. This confirms the plan's bet. **The edge has to come from Step 6 live features** (network state at T−90 min, previous-trip delay) and planned works, not from fancier history models.

**Connection miss rate (Step 5.4)** — `pipeline.transfers`, last 180 days, weekdays, 3 min walk assumed at Utrecht:

| Planned buffer at Utrecht | Den Haag → UT → Amersfoort | Eindhoven → UT → Amersfoort | Rotterdam → UT → Arnhem |
| --- | ---: | ---: | ---: |
| 3–4 min | 17–28% missed | 13% | — |
| 5–7 min | 3–4% | 7% | 4–8% |
| 10+ min | 0–4% | 1–5% | 0–4% |

A planned 3-minute change at Utrecht is missed more than 1 in 4 times; at 6 minutes it is 1 in 40. That is exactly the "Connection risk: Moderate" moment from the product brief, and it comes straight from history. Caveats: the 3 min walk is a placeholder (real Utrecht platform pairs range from 1 to 6 min), and with 1-minute delay granularity, borderline cases are ambiguous.

**4 Nov 2026 — Step 6 features and Step 7 models**

Key unlock: **the RdT archive is enough to rebuild live state for training.** It stores the actual time of every stop, so "what was knowable at 07:12" is every event with actual time ≤ 07:12. We can train and backtest the T−60 / T−15 horizons now. The live logger is still needed for serving, and for NS's own predictions as a benchmark.

- `pipeline.external` fetches KNMI daily weather (gust, wind, rain, min/max temperature at De Bilt, Rotterdam, Schiphol, Eindhoven) and builds a calendar: public holidays, regional school holidays, days since the DST switch, leaves season. KNMI's hourly endpoint returned no data, so V1 is daily. School holidays come from rijksoverheid, which only covers school years from 2024-25, so earlier dates are left *unknown* (NULL) rather than guessed.
- `pipeline.features` builds 23 day-level features and 18 live features per trip, for three horizons: `day` (00:00), `t60`, `t15`:
  - *Day:* schedule, the tuned Step 5 baseline (walk-forward), yesterday on the corridor, a planned-works proxy (today's scheduled trips vs usual), calendar, weather.
  - *Live:* own train's last known delay and how long ago it was seen; delay, >5 min share and cancel share of departures at the origin, the destination and network-wide in the last 30/60 min; earlier trips on the same corridor (arrival delay, cancellations in the last 90 min).
  - **Leakage contract:** an event is known when its *actual* time ≤ as_of. A cancellation counts as known 5 min after its planned time, because the archive doesn't say when NS decided. `--check` recomputes 400 random rows by brute force: all match.
- `pipeline.model`: per horizon, HistGradientBoosting quantile regressors (p10/p50/p90) and a cancellation classifier. LightGBM needs a system OpenMP library that isn't installed; sklearn's HistGB is the same algorithm family. Conformal offsets for p10/p90 are learned on the last 8 weeks before the training cutoff (Step 7.3). A Platt correction for cancellations was tried and turned off: no Brier gain, and it flattened day-ahead scores.
- `persist_last_delay`: a stand-in for NS's propagation, i.e. the history distribution shifted by the train's last known delay.

**Test results** (May – Oct 2026, held out, trained on Apr 2023 – Apr 2026):

| Horizon | Model | Pinball | Above p90 | Surprise (>p90+5) | Brier cancel |
| --- | --- | ---: | ---: | ---: | ---: |
| any | Naive timetable | 0.570 | 27.4% | 5.9% | 0.063 |
| any | Best Step 5 baseline (`hier_tuned`) | 0.466 | 9.1% | 2.9% | 0.059 |
| day-ahead | GBM | 0.462 | 9.0% | 2.9% | 0.060 |
| T−60 | Persistence (NS-like) | 0.524 | 8.6% | 2.8% | 0.059 |
| T−60 | GBM | 0.456 | 8.9% | 2.8% | **0.046** |
| T−15 | Persistence (NS-like) | 0.549 | 7.9% | 2.4% | 0.059 |
| T−15 | GBM | **0.439** | 8.8% | **2.5%** | **0.039** |

Cancellation calibration at T−15 (validation, as shipped): predicted 1% → observed 1%, 7% → 7%, 14% → 14%, 29% → 32%, 65% → 71%. Honest enough to show users.

**What we learned**

1. **Day-ahead, the ML model doesn't beat the baseline** (0.462 vs 0.466, same Brier). With today's day-level inputs, history is all there is. Ship the Step 5 baseline for day-ahead (plan rule: "if a well-tuned baseline is within ~10%, ship the simple one").
2. **Live state is where the edge is, as the plan predicted.** Cancellation error drops 22% at T−60 and 34% at T−15. Delay ranges get 6% tighter at T−15 while staying calibrated (8.8% above p90, target 10%).
3. **Top drivers at T−15** (permutation importance): delay range ← own train's last known delay, then how long since we last saw it, corridor, history. Cancellations ← *share of earlier trips on this corridor cancelled in the last 90 min* (by far), then minutes since the train was last seen. Disruptions persist, and the app can say so.
4. **Top drivers day-ahead:** history, max temperature, departure time, previous day's delays, leaves season. Weather is *observed* weather standing in for the forecast, so its day-ahead value is slightly flattered.
5. **The planned-works proxy doesn't work.** Today's scheduled-trip count vs usual adds ~nothing. Real planned-works data is still the missing day-ahead signal.
6. **Shifting history by the current delay is worse than history alone** on pinball (0.549 vs 0.466 at T−15). Delays partly recover en route, so naive propagation over-predicts. The GBM learns how much of a delay sticks.
7. **Data quirk:** a few services carry a service_date one day off from their stop times (e.g. train 29607 on 22/23 Oct 2024). Always filter on stop times, not service_date, when slicing by time.

**Quality review (4 Nov 2026)**

*Weather barely helps, for structural reasons.* Removing all six weather features changes day-ahead pinball by 0.2% and T−15 by nothing. Day-ahead cancellation Brier moves either way depending on the split. On extreme-weather days (gust ≥ 20 m/s, frost < −3 °C, heat ≥ 30 °C) weather improves the delay range by ~1%. Why:
1. **Too few examples.** Severe storms (gust ≥ 25 m/s) double cancellations (13.5% vs 5.4%), but there were only 15 such days in 4 years, and 1 in the test window.
2. **Daily max is too coarse.** A storm at 18:00 doesn't touch the 07:41. Hourly weather matched to the trip's time is needed.
3. **NS plans around forecast storms.** On code-orange/red days NS publishes a reduced timetable, so trains are *removed*, not cancelled, and the labels never see the damage.
4. **The worst days aren't weather.** The top cancellation days (9 Sep 2026, 6 and 10 Jun 2025: ~100% cancelled, 0% delayed) look like strikes. Infrastructure and rolling-stock faults dominate the rest.
5. **At T−15, live state already contains the weather's effect** (earlier trains late or cancelled), so weather is redundant there.

*Decision quality on the test window* (what a user would feel):

| When | Warning rule | Trips warned | Right when warning | Problems caught |
| --- | --- | ---: | ---: | ---: |
| Day-ahead | cancel risk ≥ 20% | 0.1% | 13% | 0.2% of cancellations |
| T−60 | cancel risk ≥ 40% | 2.1% | 72% | 24% of cancellations |
| T−15 | cancel risk ≥ 40% | 3.9% | 69% | 43% of cancellations |
| T−15 | cancel risk ≥ 20% | 6.8% | 52% | 56% of cancellations |
| T−15 | worst case > 5 min late | 11.9% | 38% | 38% of bad trips (cancelled or > 5 min late) |

Cancellations are clustered: 33% happen on corridor-days where more than 30% of trips were cancelled, and 10% on one network-wide day (9 Sep 2026). The model mostly catches disruptions that are already unfolding; isolated one-off cancellations stay unpredictable.

*Against the benchmark:* we beat the naive timetable and the history baselines, but **we can't yet say whether we beat NS.** NS's own predictions weren't archived. NS will also already show many of the cancellations we "predict" at T−15 ("rijdt niet"), so part of our 43% catch rate may be information the user already has. The live logger is what answers this.

**Product implications**

- Pre-commute notification (day-ahead): use `hier_tuned` reliability and ranges. Don't promise day-ahead cancellation warnings except for planned works.
- From ~60 min before departure the model adds real value: "Earlier trains on your route are being cancelled; 45% chance yours is too" is now a calibrated statement.

**Next**
- [ ] Planned works (real data: NS API disruptions/works endpoint or a scraped archive), still the biggest missing day-ahead signal
- [ ] Live logger (Step 2): needed to *serve* these features, and for NS's own predictions as the true benchmark column
- [ ] T−90 / T−30 horizons and one model with horizon as a feature (Step 7.1 comparison)
- [ ] Previous-trip linkage (trainset turnarounds at the origin terminus): 43% of trips have no own-train signal at T−15 because they start at the origin
- [ ] Archived weather *forecasts* instead of observations for day-ahead
- [ ] Weekly retrain + promotion rule (Step 9); hand-build Utrecht walk times; real corridors from interviews

**4 Nov 2026 — hourly weather: snow, ice, timing, forecasts**

- `pipeline.weather` replaces the daily weather with two point-in-time sources tied to the trip's own stations and hours:
  - **Observed, KNMI hourly** (nearest station to origin and destination): snow hours and ice-formation hours in the last 24 h, minimum temperature in the last 12 h, rain in the last 6 h, max gust in the last 3 h, fog hours. Only hours that ended by `as_of` are used.
  - **Forecast, Open-Meteo "previous runs"** (`*_previous_day1`, the value forecast ~24 h earlier), from 19 Jan 2024: around departure at the origin and arrival at the destination, it gives gust, rain, snowfall, min/max temperature, freezing-rain / snow / thunder weather codes, plus snowfall in the 24 h and frost hours in the 12 h before departure.
  - The old daily weather (same-day observations: a "perfect forecast" that wasn't knowable day-ahead) is kept only for comparison.
- Leakage check extended: observed snow hours and overnight minimum are recomputed by brute force from the raw hourly table, and forecast fields are checked to be empty when their day-1 run would have been issued after `as_of`. 600/600 rows match.
- Forecast skill check (Utrecht, daily): the forecast caught the snow on 5 and 7 Jan 2026 but underestimated 3–4 Jan and missed the −7.7 °C frost on 6 Jan (forecast −1.1 °C). Day-to-day correlation with observations: 0.93 for temperature, 0.87 for gusts.

**Result: still almost no gain**, now with an explanation (GBM without post-calibration, Brier = cancellation error):

| Split, horizon | Subset | No weather | Old daily | New hourly + forecast |
| --- | --- | --- | --- | --- |
| Validation (winter), day-ahead | all trips: Brier | 0.0690 | 0.0695 | **0.0680** |
| Validation, day-ahead | winter-ish trips (14.5% cancelled): pinball / above p90 | 0.566 / 10.5% | 0.562 / 10.3% | 0.565 / **9.9%** |
| Test (summer), day-ahead | windy trips: pinball / above p90 | 0.557 / 11.0% | 0.553 / 9.6% | **0.550 / 9.4%** |
| Test, day-ahead | windy trips: Brier | **0.0828** | 0.0826 | 0.0918 |
| Either split, T−15 | all trips | 0.449 / 0.439 pinball | same | same |

The new weather slightly improves the honesty of the delay range in bad weather (fewer trips above p90) and day-ahead cancellation error overall. It doesn't help single out winter cancellations, and it over-warns on windy summer days.

**Why: the training data contains no winter where snow mattered.**

| Winter | Snow hours | Ice hours | Snowy/icy days | Cancelled on those days | Other days |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022-23 | 223 | 223 | 18 | 4.1% | 5.8% |
| 2023-24 | 206 | 105 | 11 | 7.6% | 5.1% |
| 2024-25 | 197 | 493 | 25 | 5.0% | 5.1% |
| **2025-26** | **587** | 169 | 21 | **18.5%** | 6.0% |

In the three training winters, snowy and icy days were no worse than normal days. The model correctly learned "snow doesn't matter", and was then surprised by January 2026. Even within that winter the snow *amount* doesn't explain it: 2 Jan had 52 snow-hours and 1% cancelled, while 5–7 Jan had similar snow and 63–71% cancelled. The damage came from **several days of snow and frost in a row**: frozen points and overhead wires, NS taking trains out of service. History alone can't teach a model a pattern it has seen once.

**What would work for winter instead**
1. **NS's own winter announcements.** NS publishes an adjusted winter timetable the evening before. On the live feed these show up as cancellations hours ahead, which our T−60/T−15 features already pick up (the corridor cancel share drives cancellation prediction). For day-ahead, find where NS's advance announcements appear on the feed (probably the InfoPlus LAB/STB envelopes on the same endpoint; **to verify**: the logger doesn't subscribe to them yet) and use them as a feature.
2. **Multi-day accumulation features** (snow hours in the last 72 h, consecutive frost days), so the model *can* represent "day 3 of a cold spell" once more winters are in the data.
3. **A product-level rule until then:** on forecast snow or freezing rain, widen the day-ahead range and say why ("Snow forecast: NS often runs fewer trains"), rather than pretending the model knows.

Default model weather is now `hourly`: equal or slightly better than the old daily features, and it no longer uses same-day observations it couldn't have known. Open-Meteo's free API is non-commercial; a launch needs their paid plan or KNMI's own forecast data.

**4 Nov 2026 — live logger**

- `pipeline.live` subscribes to NS InfoPlus on the NDOV best-effort ZeroMQ feed: departures (DVS), arrivals (DAS) and full trips (RIT), about 12 messages/s at midday. It writes every 5 minutes:
  - **raw:** every message with `received_at`, append-only. Stored as decompressed XML in zstd Parquet, ~10× smaller than keeping each message's own gzip: ~0.2 GB/day.
  - **board:** one row per DVS/DAS message: NS's predicted delay, planned/actual platform, train status, change codes, cause. This is the "what did NS show at 07:12?" table (Step 3 snapshot layer).
- Change codes confirmed from the feed's own passenger texts: 10 delay, 20 platform change ("gewijzigd vertrekspoor"), 32 "Rijdt niet", 34 "Rijdt niet verder dan … door een defecte trein".
- Reconnects after 3 min of silence, flushes on SIGTERM, writes files atomically. A parse error never drops the raw message.
- `pipeline.ns_benchmark` replays the board at T−60 / T−15 for each corridor trip. It measures (a) NS's predicted arrival delay vs outcome, and (b) **which cancellations NS had already announced by then**: the question that decides whether our cancellation model adds anything. Until RdT publishes November, outcomes come from the feed's last word per stop.
- Deployment: on a Hetzner cost-optimized Arm server (2 vCPU, 4 GB, 38 GB disk) since 4 Nov 2026 12:18. Set up with `deploy/setup_server.sh <ip>` (firewall, automatic security updates, systemd service). Verified: data written every 5 min, comes back by itself after a reboot. Check with `deploy/server_status.sh`, copy data with `deploy/pull_data.sh`. Mac logging covered 11:14–12:18 that day.

**To finish the head-to-head (model vs NS on the same trips):** after ~2–4 weeks of logging and RdT's November file, rebuild labels/features for those dates and add NS as a predictor column in the scoreboard. Later, the same board data becomes live *features* (NS's prediction as an input, as Step 6 suggests).

---

## Sources

- [Rijden de Treinen — open data overview](https://www.rijdendetreinen.nl/en/open-data)
- [Rijden de Treinen — train archive](https://www.rijdendetreinen.nl/en/open-data/train-archive)
- [Rijden de Treinen — disruptions dataset](https://www.rijdendetreinen.nl/en/open-data/disruptions)
- [NS API portal](https://apiportal.ns.nl/)
