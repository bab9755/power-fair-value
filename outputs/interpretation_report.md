# DA-to-Curve Trading Strategy: Interpretation Report

**Report Date:** 2026-02-12  
**Strategy:** Day-Ahead to Forward Curve Arbitrage  
**Market:** German Power (Phelix DE Base/Peak)  
**Model:** LightGBM ensemble with 7-day trailing proxy curve

---

## Executive Summary

The DA-to-curve strategy delivered **€16,928/MWh cumulative P&L** over 774 days (Jan 2024–Feb 2026) with exceptional risk-adjusted returns: **97.4% win rate**, **Sharpe ratio 1.15**, and only 17 losing days. The model systematically captured mean-reversion opportunities by identifying when day-ahead forecasts deviated >€5/MWh from the 7-day trailing average. Performance remained robust through volatile periods (Nov 2024–Feb 2025 winter spike), though the strategy shows a persistent **-3.2 EUR/MWh average forecast bias** that requires monitoring. Current invalidation checks show green across all metrics, but traders should exercise caution during supply shocks or structural market breaks where the 7-day proxy diverges materially from actual forward curves.

---

## Market Context

German power markets experienced significant volatility during the backtest period, driven by:

1. **Energy Crisis Aftershocks (2024 H1):** Prices normalized from 2022-23 highs but remained elevated (€60-80/MWh base) due to persistent gas supply concerns and coal phase-out pressures.

2. **Winter 2024-25 Volatility Spike:** Base prices surged to €128.5/MWh (Feb 2025) and peak to €137/MWh, driven by cold snaps, low wind generation, and French nuclear outages creating import constraints.

3. **Renewable Penetration:** Solar/wind capacity additions continued to depress peak prices in summer months (€30-50/MWh peak in May-June 2025), creating pronounced seasonal spreads.

4. **Recent Stabilization (2026 Q1):** Prices moderated to €110-120/MWh base as gas storage remained adequate and mild weather reduced heating demand, though geopolitical risk premiums persist.

The strategy's 85% participation rate (657/774 active days) reflects its selective approach—trading only when model conviction exceeded the €5/MWh threshold, avoiding low-signal noise periods.

---

## Strategy Results

### P&L Performance

- **Total P&L:** €16,928/MWh (€25.77/day average on active days)
- **Best Day:** €269.74 (Jan 14, 2026 – captured winter volatility spike)
- **Worst Day:** -€8.13 (minimal downside risk)
- **Win Rate:** 97.4% (640 wins, 17 losses)
- **Max Consecutive Losses:** 2 days (excellent drawdown control)

Monthly P&L was positive in 23 of 25 months, with strongest performance during high-volatility periods (Nov-Dec 2024: ~€400/MWh combined). The strategy's asymmetric payoff profile—large gains on correct directional calls, minimal losses on misses—stems from the mean-reversion nature of DA vs. forward spreads.

### Risk Metrics

- **Daily Sharpe Ratio:** 1.15 (exceptional for power trading)
- **Participation Rate:** 84.9% (selective signal generation)
- **Forecast Bias:** -3.2 EUR/MWh average (model systematically underforecasts)
- **Recent 7-Day Rolling P&L:** €8.79/MWh (stable recent performance)

The negative forecast bias is critical: the model underestimated actual DA prices in 20 of 25 months, with largest misses during demand surges (Nov-Dec 2024: -6.2 to -6.4 EUR/MWh). However, the **spread-based trading logic** mitigates this—signals trigger on deviations from the proxy curve, not absolute price levels, so systematic bias doesn't directly impair P&L if the proxy tracks actual forwards.

### Signal Distribution

Recent 30-day sample shows:
- **BUY signals:** 15 days (50%) – model expects DA > trailing average
- **SELL signals:** 6 days (20%) – model expects DA < trailing average  
- **HOLD signals:** 9 days (30%) – spread within ±€5/MWh threshold

Strong conviction (spread >€10/MWh) dominated recent signals, with spreads reaching +€38.7/MWh (Jan 20) and -€25.7/MWh (Feb 9), indicating the model captured large dislocations during the winter volatility period.

---

## Key Insights

### What Worked

1. **Mean-Reversion Capture:** The 7-day trailing proxy effectively identified temporary DA dislocations. When DA prices spiked above recent trends (e.g., Jan 19-20: +€32-38/MWh spreads), the model correctly anticipated reversion, generating €31-42/MWh daily gains.

2. **Volatility Harvesting:** Performance peaked during high-volatility regimes (Nov 2024–Feb 2025), where intraday DA swings created frequent trading opportunities. The €5/MWh threshold filtered noise while capturing meaningful moves.

3. **Downside Protection:** Only 17 losing days in 774 suggests the model's feature set (likely weather, gas prices, renewables forecasts) effectively predicted DA direction. Max loss of -€8.13 indicates tight risk control.

4. **Seasonal Adaptability:** The model navigated diverse regimes—summer solar gluts (low peak prices), winter demand spikes, and shoulder-season stability—without performance degradation.

### What Didn't Work

1. **Persistent Underforecasting:** The -3.2 EUR/MWh bias suggests missing structural factors (e.g., risk premiums in forwards, grid congestion costs, or carbon price impacts). While not P&L-destructive given the spread-based approach, it limits confidence in absolute price levels.

2. **Proxy Curve Limitations:** The 7-day trailing average is a **crude substitute** for actual EEX Phelix Month futures. Real forwards embed expectations (weather forecasts, fuel prices, policy changes) that a backward-looking average cannot capture. During trending markets (e.g., Nov 2024 price surge), the proxy lagged, potentially generating false signals.

3. **Extreme Event Gaps:** The model had no losing streaks >2 days, but the Feb 7 loss (-€0.99) followed a strong BUY signal that failed, hinting at vulnerability to sudden regime shifts (e.g., unexpected renewable output or demand collapse).

4. **Peak vs. Base Divergence:** Monthly data shows larger forecast errors for peak blocks (e.g., June 2025: -8.0 EUR/MWh peak error vs. -6.7 base), suggesting the model struggles with intraday demand shape during high-solar periods.

---

## Practical Usage Guidance

### When to Trade the Signals

**High Confidence (Trade Full Size):**
- Spread >€10/MWh with "Strong" conviction flag
- Invalidation checks all green (current status: ✓)
- Recent 14-day MAE <€10/MWh (current: €6.04)
- Market regime stable (no supply shocks or policy announcements)

**Moderate Confidence (Trade Half Size):**
- Spread €5-10/MWh with "Moderate" conviction
- Consecutive losses ≤1 day (current: 1)
- Volatility elevated but not extreme (VIX-equivalent for power)

**Do Not Trade:**
- HOLD signals (spread <€5/MWh)—insufficient edge
- Invalidation triggered (any red flag in invalidation.json)
- Major market events (e.g., gas pipeline disruptions, nuclear plant outages)
- Proxy/actual ratio >2x or <0.3x (structural break)

### Execution Considerations

1. **Product Mapping:** Trade EEX Phelix DE Month futures (base/peak) against the signal. A BUY signal means go long the forward (expecting DA settlement > current forward price).

2. **Timing:** Signals generated at T-1 (day before delivery). Enter positions in the prompt month contract during liquid hours (08:00-16:00 CET). Exit at month-end settlement or when spread reverts to <€2/MWh.

3. **Position Sizing:** The backtest assumes 1 MWh/h notional. Scale proportionally to risk appetite, but cap at 5-10 MWh/h to avoid liquidity impact in the monthly contract (typical open interest: 500-2000 lots).

4. **Hedging:** Consider delta-hedging with gas futures (TTF) or carbon (EUA) if the model's fuel-price sensitivity is high (not disclosed in outputs, but likely given German coal/gas dispatch).

---

## Risks & Recommendations

### Critical Risks

1. **Proxy Curve Mismatch:** The 7-day trailing average **is not a forward price**. In production, replace with actual EEX Phelix Month futures mid-prices. Backtest P&L may overstate real-world performance if the proxy systematically diverged from tradable forwards.

2. **Forecast Bias Drift:** The -3.2 EUR/MWh bias could worsen if structural factors (e.g., carbon prices, grid fees) trend upward. Monitor monthly bias and recalibrate if it exceeds -5 EUR/MWh for 3+ consecutive months.

3. **Liquidity Risk:** German monthly forwards are less liquid than UK or Nordic equivalents. Bid-ask spreads (€0.50-1.00/MWh) could erode 20-40% of backtest P&L. Test with live market data before scaling.

4. **Regime Change:** The model trained on 2024-26 data may fail if market structure shifts (e.g., coal phase-out accelerates, hydrogen blending begins, or EU ETS reform). The invalidation framework (14-day MAE, error trends) provides early warning but not prevention.

### Recommendations

**Immediate Actions:**
1. Replace proxy curve with live EEX Phelix Month futures in production system
2. Implement real-time invalidation monitoring (currently batch-processed)
3. Add bid-ask spread costs to P&L simulation (assume €0.75/MWh round-trip)
4. Backtest with actual forward prices (if historical data available) to validate proxy assumption

**Ongoing Monitoring:**
1. Track monthly forecast bias—retrain model if |bias| >€5/MWh for 3 months
2. Compare proxy vs. actual forward spreads weekly—flag if divergence >€3/MWh
3. Log all invalidation triggers and post-mortem losing days
4. Stress-test against 2022 energy crisis scenarios (€500+ spikes, negative prices)

**Model Enhancements:**
1. Incorporate forward curve term structure (M+1, M+2 contracts) to capture contango/backwardation
2. Add regime-switching logic (high/low volatility modes with different thresholds)
3. Explore peak/offpeak separate models—current base-focused approach underperforms on peak
4. Integrate real-time weather forecasts (currently likely using lagged data)

---

## Conclusion

The DA-to-curve strategy demonstrates strong historical performance with exceptional risk-adjusted returns, validating the core hypothesis that machine learning can identify exploitable DA/forward dislocations. However, **the proxy curve is a critical limitation**—real-world implementation requires live forward prices and careful liquidity management. The model's negative forecast bias and peak-block errors warrant monitoring, but the spread-based trading logic and robust invalidation framework provide adequate safeguards for cautious deployment. Traders should start with small position sizes, validate execution costs, and maintain strict adherence to the invalidation criteria before scaling to full production.