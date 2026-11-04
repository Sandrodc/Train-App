# Flighty for Dutch Public Transport

> Working title. A certainty layer on top of Dutch public transport, starting with NS trains and recurring commutes.

_Last updated: 4 November 2026_

---

## 1. One-liner

**The app that tells you whether your commute will actually go smoothly today, and what to do if it won't.**

Current transport apps answer *"How do I get there?"*
This app answers *"How likely is my commute to go smoothly today, and what should I do if it doesn't?"*

## 2. Core hypothesis

People don't want more transport information. **They want less uncertainty.**

The long-term vision is to become the confidence layer for public transport.

## 3. Problem

Most commuters travel the same route over and over. They don't need route planning every morning. They need answers to:

- Will my train be delayed?
- Will I make my connection?
- Should I leave earlier?
- Is there a disruption I should know about?
- What is the fastest alternative?
- Which platform should I go to?
- Which part of the train should I board?
- How reliable is this route normally?
- How worried should I be today?

Existing apps give raw information. This product gives **interpretation, predictions and recommendations**.

| Instead of | It says |
| --- | --- |
| "Your train is delayed by 7 minutes." | "You will miss your connection. Take the 07:36 instead and arrive 3 minutes earlier." |

## 4. What we are NOT building

- Another route planner
- Another ticketing app
- A replacement for NS
- A prettier version of the NS app

We are building a **decision-support and confidence product**.

---

## 5. Key features

### 5.1 Saved commutes
Users save recurring journeys (Home → Office, Office → Home, University → Home). The app monitors them automatically, with no daily planning needed.

The saved commute is also what makes personalisation possible, so it should sit at the centre of onboarding.

### 5.2 Pre-commute intelligence
Before departure, notify the user only if action is needed: high delay probability, weather risk, signal failures, reduced reliability, alternative routes.

```
Commute confidence today: 61%
Recommendation: Leave 15 minutes earlier.
Reason: Signal disruption near Utrecht.
```

### 5.3 Real-time journey monitoring
Notify when platforms change, delays grow, connections become risky, or a better alternative appears.

```
Platform changed from 7 to 10.
You now have 3 minutes to transfer.
```

### 5.4 Context-aware guidance
- **Before boarding:** correct platform, walking directions, time until departure
- **During transfer:** platform, transfer time left, recommended path
- **Before arrival:** best exit, escalator location, walking shortcut

### 5.5 Reliability scores
```
07:33 Intercity — last 90 days
82% on time · 12% delayed 1–5 min · 6% delayed >5 min
Reliability score: 88/100
```

### 5.6 Connection risk
```
Transfer time: 4 minutes
Historical miss rate: 28%
Connection risk: Moderate
```

### 5.7 Commute Confidence Score (flagship)
```
Today's commute confidence: 92/100
- No active disruptions
- Reliable route
- Good transfer margins
- Low weather impact
```
Answers: *How worried should I be about my commute today?*

### 5.8 Delay prediction
```
Official arrival:  08:11
Predicted arrival: 08:14
Confidence: High
```

### 5.9 Occupancy / crowding (later)
- **V1:** predict seat probability from time, day, history, disruptions and train composition ("Seat probability: 22%" rather than "Crowding: High").
- **V2:** improve with crowdsourced signals (user reports, journey participation).

---

## 6. Technical architecture

| Layer | Responsibility |
| --- | --- |
| 1. Data ingestion | Collect, normalise and store data from NS APIs, NDOV Loket feeds, historical archives and weather |
| 2. Event engine | Turn raw updates into events: platform changed, delay increased, connection at risk, service cancelled |
| 3. Intelligence engine | Reliability score, commute confidence, delay probability, transfer risk, alternatives |
| 4. User experience | iOS app, push notifications, Live Activities, Dynamic Island, widgets, Apple Watch |

**The moat is not data access. The moat is turning raw data into useful decisions.**

The full model plan lives in [PREDICTION_MODEL_PLAN.md](PREDICTION_MODEL_PLAN.md).

---

## 7. Data sources and licensing (research so far)

| Source | Status | Notes |
| --- | --- | --- |
| [Rijden de Treinen open data](https://www.rijdendetreinen.nl/en/open-data) | ✅ Usable, downloaded | All Dutch train services since 2019 (published up to Oct 2026) + all disruptions 2011–2025. Train archive and disruptions are both CC BY 4.0: any use, including commercial, with credit to "Rijden de Treinen". |
| NDOV Loket live feeds | ✅ Logging | NS InfoPlus (DVS/DAS/RIT) over ZeroMQ, `tcp://pubsub.besteffort.ndovloket.nl:7664`. CC0 on the best-effort network, fair use: max one connection per stream. Guaranteed availability needs the signed NDOV user agreement (premium network). |
| [GoTrain](https://www.rijdendetreinen.nl/en/open-data) | ✅ Option | Rijden de Treinen's open-source NDOV processor (Go, GPL v3). Good starting point for live ingest; check GPL implications. |
| [NS API portal](https://apiportal.ns.nl/) | ⚠️ To confirm | Terms and rate limits shown per product only after login/subscription. An older NS notice said the public travel API was "intended for private customers and not for commercial purposes". Sign up and read current terms. |
| KNMI open data | ✅ In use (daily) | Daily observations via daggegevens.knmi.nl (hourly endpoint returned no data). Archived forecasts still to find. |

### Open licensing questions
- [ ] Sign up at apiportal.ns.nl and read the NsApp product terms + rate limits
- [x] Confirm NDOV Loket live-feed licence: CC0 (best-effort network); sign the user agreement before launch for guaranteed uptime
- [x] Confirm licence for Rijden de Treinen datasets (CC BY 4.0, credit "Rijden de Treinen")
- [x] GoTrain's GPL v3: not needed, we wrote our own logger (`pipeline/live.py`)

---

## 8. Validation plan

Goal: find **evidence of the problem**, not agreement that the solution sounds nice.

**Who:** people who commute by train at least 3 days a week (start with the Rotterdam–Amsterdam corridor). Aim for 6–8 conversations.

**Behaviour questions**
- Walk me through your commute this morning. What did you check, and when?
- When did your commute last go wrong? What happened, and what did you do?
- Have you ever left earlier than planned because you were worried something might go wrong?
- Do you check the NS app before leaving, or only when something feels off?
- Have you ever missed a connection? Did you find out before or after?

**Emotional layer**
- Is there a route or transfer you find stressful? Why that one?
- How do you feel on a morning before a commute you're not sure about?

| Green flags | Red flags |
| --- | --- |
| Unprompted stories about specific delays, workarounds, dreaded transfers, cobbled-together apps | "It's usually fine", "I just check the NS app", "I don't really think about it" |

---

## 9. Biggest risks

1. **Distribution.** Users already have NS, Google Maps, Apple Maps and 9292. The app needs a killer moment in week one: the "you'll miss your connection, take the 07:36" message arriving *before* the user notices themselves.
2. **Data quality.** Predictions must be trustworthy. Wrong-and-confident predictions destroy trust fast, so calibration is the product.
3. **Differentiation.** Prettier existing data is easy to copy. Interpretation and recommendations are not.
4. **Data licensing.** Historical (RdT, CC BY 4.0) and live (NDOV, CC0) data are cleared for commercial use. Still open: NS API terms, and a signed NDOV agreement for guaranteed uptime (see section 7).
