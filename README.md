## Benchmark-Neutral Allocations with Factor Tilts

### Objective
To develop an investment strategy that tracks a core benchmark (such as a market-capitalization-weighted portfolio) while incorporating targeted factor or sector tilts. 
This approach allows portfolios to meet specific investor preferences, such as maximizing exposure to short-term indicators or specific asset-pricing factors, without significantly deviating from the overall performance of the benchmark.

---

### Methodology & Framework

* **Core Optimization Problem:** The strategy minimizes tracking error relative to the benchmark while controlling exposure to targeted risk factors via a factor tilt penalty:
    $$\min_{w} \quad \text{Tracking Error} + \text{Factor Tilt Penalty}$$
    * **Tracking Error:** Quantifies active risk and deviations from the benchmark: $TE = (w - w_b)^T\Sigma(w - w_b)$
    * **Tilt Penalty:** Penalizes deviations from target factor exposures: $\text{Penalty} = \lambda_{\text{tilt}}\sum(B^T w - \text{tilt target})^2$
* **Operational Constraints:**
    * Fully invested portfolio: $\sum w_i = 1$
    * Long-short portfolio flexibility: $-1 \le w_i \le 1$

### Data Collection & Processing
* **Data Sourcing:** Stock price and market capitalization data extracted dynamically via a custom `market_data` module using Yahoo Finance.
* **Scope:** Focuses on the top 10 largest companies across five major sectors: Technology, Healthcare, Consumer, Finance, and Energy, covering a historical timeline from January 2020 to February 2025.
* **Multicollinerity & Dimensionality Reduction:**
    * Traditional asset-pricing models often face collinearity challenges (e.g., infinite VIF values between traditional Quality and Volatility metrics).
    * This pipeline implements a **Tucker Factor Model** utilizing Singular Value Decomposition (SVD) to extract independent principal components, successfully preserving 86.52% of the dataset's variance while ensuring regression model stability.
    * A Variance Inflation Factor (VIF) filter is applied; any factor with a VIF exceeding 10 is automatically removed to eliminate redundant data (e.g., handling sector anomalies like *Tucker_Factor_1* in Tech or *ind_mom* in the Consumer sector).

### Factor Signals & Performance Drivers
The pipeline evaluates cross-sectional asset returns using two distinct categories of signals:

1.  **Tucker Factors:** Macro-level dimensional drivers mapping structural stock behavior.
2.  **Short-Term Trading Indicators:** Independent signals capturing alpha from market inefficiencies:
   * *Industry Relative Reversal (STR<sub>i</sub>):* Measures 21-day cumulative returns relative to industry averages.
   * *Industry Momentum (IND_MOM<sub>i</sub>):* Tracks trailing trends among industry peers.
   * *Seasonality (SEA_SAME<sub>i</sub>):* Identifies recurring directional calendar-month patterns over a 10-year rolling window.

---

### Sector Insights & Hyperparameter Controls

Risk budgets and active factor allocations are governed by tuning the tilt intensity parameter ($\lambda$). Over-aggressive tuning can spike baseline portfolio volatility, making precise sector selection critical:

| Sector Focus | Core Drivers & Characteristics | Optimal Control Range ($\lambda$) |
| :--- | :--- | :--- |
| **Technology** | Dominated by momentum trends; highly sensitive to single-stock anomalies (e.g., NVDA, AVGO). Generates the highest factor returns. | $\lambda < 3$ |
| **Healthcare** | Heavily influenced by underlying risk and quality measures (e.g., ABBV, GILD). | $2 < \lambda < 3$ |
| **Consumer** | Align closely with structural size and value components. | $2 < \lambda < 6$ |
| **Finance** | Highly stable sector driven by seasonality effects; factor exposures rarely exceed $\pm1$. | $4 < \lambda < 8$ |
| **Energy** | Strongly exhibits mean-reversion characteristics. | $2.5 < \lambda < 7.5$ |
