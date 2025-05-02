import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.market_data import MarketData
from src.iis_indicators import IndicatorsIIS
from src.tucker_factor import TuckerFactorCalculator
from sklearn.preprocessing import StandardScaler


def calculate_vif(df):
    # Rimuovi colonne con varianza zero
    df = df.loc[:, df.var() != 0]

    vif_data = []
    for i, col in enumerate(df.columns):
        try:
            vif_val = variance_inflation_factor(df.values, i)
            # Se il VIF è infinito o troppo grande
            if np.isinf(vif_val):
                vif_val = np.nan  # oppure np.inf, a seconda di come vuoi gestirlo
        except:
            vif_val = np.nan  # nel caso di errori di calcolo

        vif_data.append((col, vif_val))

    return pd.DataFrame(vif_data, columns=['Feature', 'VIF'])


def combined_vif_analysis(sector=None):
    """
    Analisi combinata del VIF per gli indicatori IIS e i fattori Tucker.
    """
    print("Combined VIF Analysis")
    print("-" * 50)

    # Inizializza le classi necessarie
    indicators = IndicatorsIIS()
    market_data = MarketData()
    tucker_calculator = TuckerFactorCalculator(n_factors=2)

    try:
        # Ottieni i dati di mercato
        prices = market_data.download_prices(sector)
        returns = prices.pct_change().dropna()

        # Prepara i dati per gli indicatori
        if sector:
            industry_data = {ticker: sector for ticker in market_data.get_tickers(sector)}
        else:
            industry_data = {}
            for sector_name, sector_tickers in market_data.sectors.items():
                for ticker in sector_tickers:
                    industry_data[ticker] = sector_name

        # Calcola gli indicatori
        str_signal = indicators.calculate_industry_relative_reversal(returns, industry_data)
        ind_mom = indicators.calculate_industry_momentum(returns, industry_data)
        seasonality = indicators.calculate_seasonality(returns)

        indicators_df = pd.DataFrame({
            'str_signal': str_signal,
            'ind_mom': ind_mom,
            'seasonality': seasonality
        }).dropna()

        # Calcola i fattori Tucker
        original_exposures = tucker_calculator.calculate_factor_exposures(sector)
        tucker_factors = tucker_calculator.fit_tucker(StandardScaler().fit_transform(original_exposures))
        tucker_factors_df = pd.DataFrame(
            tucker_factors,
            columns=[f'Tucker_Factor_{i + 1}' for i in range(tucker_calculator.n_factors)],
            index=original_exposures.index
        )

        # Allineamento degli indici
        combined_data = pd.concat([indicators_df, tucker_factors_df], axis=1, join='inner').dropna()

        # Calcolo del VIF
        vif_results = calculate_vif(combined_data)
        print(vif_results)

        return vif_results

    except Exception as e:
        print(f"Errore durante l'analisi VIF combinata: {str(e)}")
        raise
