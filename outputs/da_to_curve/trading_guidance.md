# Prompt Curve Translation: DA-to-Curve View

**Generated:** 2026-02-12 21:14
**Model:** Lightgbm
**Primary product:** Base
**Reference curve:** 7-day trailing average of realised DA blocks
**Signal threshold:** +/-5.0 EUR/MWh

---

## 1. Methodology

### From hourly forecasts to a tradable curve view

1. **Hourly DA forecasts** from the Lightgbm model are aggregated into
   daily block prices — Baseload (hours 1-24), Peakload (hours 9-20, EPEX
   convention), and Offpeak (complement).

2. A **reference curve** is constructed as the 7-day
   trailing average of recently realised DA block prices. This captures the
   recent price level and is used to measure whether the model forecast
   deviates from the prevailing trend. In production, the reference would be
   replaced by the actual EEX Phelix DE prompt-month forward price (German
   Power Baseload/Peakload Calendar Month Futures, CME/EEX).

3. The **spread** (forecast block minus reference curve) measures the
   model's directional view: positive means the model expects delivery
   prices above recent levels; negative means below.

4. When |spread| exceeds 5.0 EUR/MWh, the view is
   classified as **BUY** or **SELL**. Below the threshold it is **HOLD**
   (the model sees no meaningful deviation from recent trend).

### Block definitions (CET/CEST delivery day)

| Block    | Delivery hours      | Description                    |
|----------|--------------------|---------------------------------|
| Baseload | 1-24 (00:00-23:00) | Flat average, all hours          |
| Peakload | 9-20 (08:00-19:00) | Business hours, higher demand    |
| Offpeak  | 1-8, 21-24          | Night + evening, lower demand    |

### Tradable product mapping

The block-level fair value maps directly to exchange-traded products:

| Fair-Value Block | Tradable Product (CME/EEX)                     |
|------------------|-------------------------------------------------|
| Baseload Month   | German Power Baseload Calendar Month Future      |
| Peakload Month   | German Power Peakload Calendar Month Future      |
| Baseload Week    | German Power Baseload Calendar Week Future       |

A baseload month futures contract settles to the arithmetic average of
hourly DA auction prices over the delivery month (sized MWh/h x 24 x days,
adjusted for DST). Our monthly fair value is therefore directly comparable
to the contract's settlement value.

---

## 2. Monthly Fair-Value View

This table shows the model's monthly average forecast vs realised DA
settlement. The **spread** (forecast minus actual) measures forecast bias at
the monthly level — the same granularity as a prompt-month futures contract.

| Month | Actual Base | Fcst Base | Spread | Actual Peak | Fcst Peak |
|-------|------------|----------|--------|------------|----------|
| 2024-01 | 76.6 | 76.6 | +0.0 | 86.2 | 87.2 |
| 2024-02 | 61.3 | 59.3 | -2.1 | 67.5 | 63.8 |
| 2024-03 | 64.7 | 62.9 | -1.8 | 63.7 | 59.6 |
| 2024-04 | 62.4 | 57.8 | -4.5 | 52.8 | 47.4 |
| 2024-05 | 67.2 | 61.6 | -5.6 | 49.3 | 43.6 |
| 2024-06 | 72.9 | 67.9 | -5.0 | 52.5 | 48.5 |
| 2024-07 | 67.7 | 65.5 | -2.2 | 46.7 | 46.1 |
| 2024-08 | 82.0 | 76.7 | -5.3 | 59.7 | 57.2 |
| 2024-09 | 78.3 | 75.7 | -2.6 | 70.9 | 67.6 |
| 2024-10 | 86.1 | 81.3 | -4.8 | 93.4 | 87.7 |
| 2024-11 | 113.9 | 107.7 | -6.2 | 131.7 | 123.3 |
| 2024-12 | 108.3 | 102.0 | -6.4 | 134.2 | 123.1 |
| 2025-01 | 114.1 | 109.3 | -4.9 | 131.0 | 124.4 |
| 2025-02 | 128.5 | 126.4 | -2.1 | 137.0 | 133.8 |
| 2025-03 | 94.6 | 92.7 | -1.9 | 85.4 | 85.4 |
| 2025-04 | 77.9 | 77.5 | -0.5 | 55.2 | 55.9 |
| 2025-05 | 67.3 | 65.1 | -2.3 | 35.4 | 34.9 |
| 2025-06 | 64.0 | 57.3 | -6.7 | 29.8 | 21.8 |
| 2025-07 | 87.8 | 84.0 | -3.8 | 69.1 | 67.0 |
| 2025-08 | 77.0 | 76.7 | -0.3 | 54.7 | 52.6 |
| 2025-09 | 83.5 | 80.4 | -3.1 | 75.8 | 71.4 |
| 2025-10 | 84.5 | 82.5 | -2.0 | 92.1 | 87.2 |
| 2025-11 | 101.9 | 100.1 | -1.8 | 116.5 | 113.4 |
| 2025-12 | 93.5 | 92.6 | -0.8 | 105.8 | 104.8 |
| 2026-01 | 110.1 | 115.3 | +5.2 | 124.4 | 129.7 |
| 2026-02 | 111.2 | 116.3 | +5.0 | 120.9 | 126.8 |

The average absolute spread (baseload) is a measure of how close the model's
monthly view is to realised settlement — this is the residual error a prompt
trader would face when using this forecast as a fair-value anchor.

---

## 3. Signal Quality

The forecast's directional accuracy is assessed by checking whether the
model correctly predicts the sign of (actual block − reference curve).
This is a **directional** measure, not a P&L claim.

| Metric                     | Value                        |
|----------------------------|------------------------------|
| Out-of-sample days         | 774   |
| Days with active signal    | 657 (84.9%) |
| Directional accuracy       | 97%                    |
| Max consecutive mispredictions | 2 days |

**Note:** This accuracy is measured against a 7-day
trailing average, which is a slow-moving, backward-looking reference — not a
real forward price. Against a live EEX forward (which already embeds market
expectations, supply/demand outlook, and risk premia), directional accuracy
would be lower and edge smaller. The metric demonstrates the model carries
signal relative to a naive baseline, not that these hit rates are achievable
in live trading.

---

## 4. How to Use This View

### DA-to-Prompt Positioning

Each morning before the DA auction closes:

1. Run the forecast model for the next 24 hours.
2. Aggregate into baseload / peakload blocks → this is your **fair value (FV)**.
3. Obtain the current prompt-month forward price from EEX/EPEX (**MKT**).
4. Compute **edge = FV − MKT**.
5. **If edge > +5.0 EUR/MWh** → the model sees the
   month settling higher than the market is pricing → consider **buying**
   the prompt-month forward.
6. **If edge < −5.0 EUR/MWh** → the model sees the
   month settling lower → consider **selling** the prompt-month forward.
7. The position converges as DA prices materialise during the delivery
   month. Settlement = average hourly DA over the month.

### Calendar Spread (Base vs Peak)

- Compare the forecast *peak premium* (peak − base) against its historical
  distribution.
- If the premium widens beyond the 75th percentile → **sell peak / buy base**
  (mean-reversion).
- If it narrows below the 25th percentile → **buy peak / sell base**.

### Week-Ahead Rolling View

- Average the next 7 days of forecast blocks to build a "week-ahead FV."
- Compare against the week-ahead EEX forward.
- Useful for sizing longer-duration positions with lower transaction costs.

### Practical considerations

- **Transaction costs:** Typical bid-ask spreads on prompt-month DE baseload
  are ~0.10–0.50 EUR/MWh. Only trade when |edge| comfortably exceeds this.
- **Position frequency:** One position per delivery month is more realistic
  than daily re-entry. Use the latest signal before delivery starts.
- **Conviction scaling:** Scale position size by |edge| / threshold —
  larger deviations warrant larger positions.

---

## 5. When to Invalidate (Exit / Distrust the View)

### Block Mae
- **Status:** OK
- **Metric:** Base block MAE (last 14d)
- **Threshold:** > 30.0 EUR/MWh
- **Current:** 6.04 EUR/MWh

### Error Trend
- **Status:** OK
- **Metric:** Error trajectory (2nd half vs 1st half of lookback)
- **Threshold:** Increasing > 20%
- **Current:** Stable

### Consecutive Losses
- **Status:** OK
- **Metric:** Consecutive losing days
- **Threshold:** > 3 days
- **Current:** 1 days

### Extreme Price Event
- **Status:** OK
- **Metric:** Actual / Proxy ratio
- **Threshold:** > 2.0x or < 0.3x (supply shock / demand collapse)
- **Current:** Normal


**Overall recommendation:** MAINTAIN positions

### General Invalidation Rules

1. **Model degradation:** If the rolling 14-day block MAE exceeds
   30.0 EUR/MWh, the forecast is no longer reliable — stop
   using it for positioning and retrain.

2. **Regime change:** If actual DA prices diverge >2x or <0.3x from the
   reference curve (e.g., supply shock, interconnector outage), the model's
   training regime no longer reflects reality. Exit and wait for a new
   stable regime.

3. **Persistent directional failure:** If the forecast mispredicts direction
   on >3 consecutive days, reduce conviction and
   reassess whether a structural shift has occurred.

4. **Fundamental shift:** If renewable penetration changes by >30% week-
   over-week (e.g., extended Dunkelflaute or record solar), the supply stack
   has shifted and the model may need recalibration.

---

## 6. Limitations & Production Notes

- **Reference curve ≠ tradable forward.** The 7-day
  trailing average used here is backward-looking and not itself tradable. In
  production, the benchmark must be an actual EEX/EPEX forward price. The
  signal quality metrics reported above would differ (likely lower accuracy,
  smaller edge) against a real forward.
- **No transaction costs modelled.** Real bid-ask spreads on prompt-month DE
  base are ~0.10–0.50 EUR/MWh.
- **No volume modelling.** We assume flat 1 MW positions; real trading
  requires volume-weighted sizing based on portfolio exposure and market
  liquidity.
- **Model retraining.** The Lightgbm model was trained on data
  through Dec 2023. Periodic retraining (e.g., monthly walk-forward) is
  essential for production use.
- **Single market.** This view covers DE/LU only. Cross-border flows and
  neighbouring zone prices may affect prompt positioning in practice.
