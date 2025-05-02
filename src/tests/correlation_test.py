import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.market_data import MarketData
from src.iis_indicators import IndicatorsIIS
from src.tucker_factor import TuckerFactorCalculator
from sklearn.preprocessing import StandardScaler
from src.correlation_calc import combined_vif_analysis


if __name__ == "__main__":
    print("Analisi VIF per l'intero mercato")
    print("=" * 50)
    combined_vif_analysis()

    sectors = ['Technology', 'Healthcare', 'Consumer', 'Finance', 'Energy']
    for sector in sectors:
        print(f"\nAnalisi VIF per il settore {sector}")
        print("=" * 50)
        combined_vif_analysis(sector)
