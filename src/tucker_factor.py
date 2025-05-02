import numpy as np
import pandas as pd
from scipy import linalg
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from src.market_data import MarketData


class TuckerFactorCalculator:
    def __init__(self, n_factors=2):
        self.n_factors = n_factors
        self.market_data = MarketData()

    def fit_tucker(self, X):
        """Implementa il Tucker form model"""
        # Reshape X se necessario per garantire dimensioni corrette
        if isinstance(X, pd.DataFrame):
            X = X.values

        # SVD sulla matrice di input
        U, S, Vt = linalg.svd(X, full_matrices=False)

        # Seleziona i primi n_factors componenti
        self.U = U[:, :self.n_factors]  # Loading matrix per le righe
        self.S = np.diag(S[:self.n_factors])  # Valori singolari
        self.V = Vt[:self.n_factors, :].T  # Loading matrix per le colonne

        # Calcola i fattori preservando le dimensioni originali
        self.factors = np.dot(U[:, :self.n_factors], np.diag(S[:self.n_factors]))

        return self.factors

    def check_multicollinearity(self, exposures):
        """Calcola VIF per verificare la multicollinearità"""
        vif_data = pd.DataFrame()
        vif_data["Factor"] = exposures.columns
        vif_data["VIF"] = [variance_inflation_factor(exposures.values, i)
                           for i in range(exposures.shape[1])]
        return vif_data

    def calculate_factor_exposures(self, sector=None):
        """Calcola le factor exposures usando i dati di MarketData"""
        # Ottieni i prezzi usando MarketData
        prices = self.market_data.download_prices(sector)
        market_caps = self.market_data.download_market_caps(sector)

        # Calcola i rendimenti
        returns = prices.pct_change().dropna()

        # Calcola le metriche base per i fattori
        exposures = pd.DataFrame(index=prices.columns)

        # Size factor (usando market cap)
        exposures['Size'] = pd.Series(market_caps)

        # Value factor (P/B ratio proxy usando market cap / price)
        exposures['Value'] = exposures['Size'] / prices.iloc[-1]

        # Momentum factor (12-month returns excluding most recent month)
        momentum_window = 252  # circa 1 anno di trading days
        exposures['Momentum'] = (prices.iloc[-22] / prices.iloc[-momentum_window - 22] - 1)  # esclude ultimo mese

        # Quality factor (volatilità come proxy inverso della qualità)
        exposures['Quality'] = -returns.std()  # negativo così valori più alti = maggiore qualità

        # Volatility factor
        exposures['Volatility'] = returns.std()

        # Standardizza tutti i fattori
        for col in exposures.columns:
            exposures[col] = (exposures[col] - exposures[col].mean()) / exposures[col].std()

        return exposures.dropna()

    def analyze_factors(self, sector=None):
        """Analizza i fattori per un settore specifico o l'intero mercato"""
        original_exposures = self.calculate_factor_exposures(sector)
        original_vif = self.check_multicollinearity(original_exposures)

        scaler = StandardScaler()
        exposures_std = scaler.fit_transform(original_exposures)
        tucker_factors = self.fit_tucker(exposures_std)

        tucker_factors_df = pd.DataFrame(
            tucker_factors,
            columns=[f'Tucker_Factor_{i + 1}' for i in range(self.n_factors)],
            index=original_exposures.index
        )

        tucker_vif = self.check_multicollinearity(tucker_factors_df)

        # Calcola varianza spiegata
        singvals = np.linalg.svd(exposures_std, compute_uv=False)
        total_var = np.sum(singvals ** 2)
        var_explained = np.sum(singvals[:self.n_factors] ** 2) / total_var

        # Calcola matrice di covarianza
        tucker_cov = np.dot(tucker_factors, tucker_factors.T)
        tucker_cov = pd.DataFrame(
            tucker_cov,
            index=original_exposures.index,
            columns=original_exposures.index
        )

        # Interpreta i fattori
        interpretation = interpret_tucker_factors(self, original_exposures, tucker_factors_df)

        # Analisi valori estremi per settore specifico
        extreme_values = None
        if sector:
            extreme_values = {}
            for col in original_exposures.columns:
                threshold = original_exposures[col].std() * 2
                extreme_stocks = original_exposures[abs(original_exposures[col]) > threshold]
                if not extreme_stocks.empty:
                    extreme_values[col] = extreme_stocks[col].sort_values(ascending=False)

        return {
            'original_exposures': original_exposures,
            'tucker_factors': tucker_factors_df,
            'original_vif': original_vif,
            'tucker_vif': tucker_vif,
            'tucker_cov': tucker_cov,
            'variance_explained': var_explained,
            'interpretation': interpretation,
            'extreme_values': extreme_values
        }

def interpret_tucker_factors(matrix_model, original_exposures, tucker_factors_df):
    """Analizza la relazione tra fattori Tucker e originali"""
    # Calcola correlazioni tra fattori Tucker e originali
    correlations = pd.DataFrame(
        np.corrcoef(tucker_factors_df.T, original_exposures.T)[:2, 2:],
        index=tucker_factors_df.columns,
        columns=original_exposures.columns
    )

    # Calcola i loadings (coefficienti standardizzati)
    loadings = pd.DataFrame(
        index=tucker_factors_df.columns,
        columns=original_exposures.columns
    )

    for tucker_factor in tucker_factors_df.columns:
        for orig_factor in original_exposures.columns:
            reg = LinearRegression()
            X = tucker_factors_df[tucker_factor].values.reshape(-1, 1)
            y = original_exposures[orig_factor].values
            reg.fit(X, y)
            loadings.loc[tucker_factor, orig_factor] = reg.coef_[0]

    return {
        'correlations': correlations.round(3),
        'loadings': loadings.round(3)
    }

