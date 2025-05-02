import numpy as np
import pandas as pd
import statsmodels.api as sm
import pandas_datareader.data as web
from typing import Dict, Tuple


class FamaFrenchAnalysis:
    def __init__(self):
        """
        Inizializza l'analizzatore per il modello Fama-French
        """
        # Mapping dei nomi delle colonne dal dataset di Kenneth French
        self.factor_mapping = {
            'Mkt-RF': 'mkt-rf',  # Market excess return
            'SMB': 'smb',  # Small Minus Big
            'HML': 'hml',  # High Minus Low
            'RMW': 'rmw',  # Robust Minus Weak
            'CMA': 'cma'  # Conservative Minus Aggressive
        }

        self.ff_factors = list(self.factor_mapping.values())

    def get_ff_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Scarica i fattori Fama-French da Kenneth French's data library
        """
        print("\nDownloading Fama-French factors...")
        # Converti le date in datetime
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Aggiungi un po' di margine alle date per assicurarti di avere abbastanza dati
        start_buffer = start - pd.DateOffset(months=1)
        end_buffer = end + pd.DateOffset(months=1)

        try:
            ff_data = web.DataReader(
                'F-F_Research_Data_5_Factors_2x3_daily',
                'famafrench',
                start=start_buffer,
                end=end_buffer
            )[0]

            # Rinomina le colonne secondo il nostro mapping
            ff_data = ff_data.rename(columns=self.factor_mapping)

            # Converti l'indice in datetime per evitare il FutureWarning
            ff_data.index = pd.to_datetime(ff_data.index)

            # Converti i rendimenti in decimali
            ff_data = ff_data / 100.0

            print(f"Downloaded FF data from {ff_data.index.min()} to {ff_data.index.max()}")
            print("\nColumns in FF data:", ff_data.columns.tolist())

            return ff_data

        except Exception as e:
            print(f"Error downloading FF data: {str(e)}")
            return pd.DataFrame()

    def _prepare_data(self, stock_returns: pd.DataFrame, ff_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepara e allinea i dati per l'analisi
        """
        # Debug: mostra i range di date
        print(f"\nStock returns date range: {stock_returns.index.min()} to {stock_returns.index.max()}")
        print(f"FF data date range: {ff_data.index.min()} to {ff_data.index.max()}")

        # Assicurati che gli indici siano datetime
        stock_returns.index = pd.to_datetime(stock_returns.index)
        ff_data.index = pd.to_datetime(ff_data.index)

        # Rimuovi timezone se presente
        if stock_returns.index.tz is not None:
            stock_returns.index = stock_returns.index.tz_localize(None)
        if ff_data.index.tz is not None:
            ff_data.index = ff_data.index.tz_localize(None)

        # Debug: mostra alcuni esempi di date
        print("\nFirst few stock return dates:")
        print(stock_returns.index[:5])
        print("\nFirst few FF dates:")
        print(ff_data.index[:5])

        # Trova le date comuni
        common_dates = stock_returns.index.intersection(ff_data.index)
        print(f"\nFound {len(common_dates)} common trading days")

        if len(common_dates) < 60:  # Almeno 3 mesi di dati
            raise ValueError(f"Insufficient common dates: {len(common_dates)} days")

        # Allinea i dati
        aligned_returns = stock_returns.loc[common_dates]
        aligned_ff = ff_data.loc[common_dates]

        return aligned_returns, aligned_ff

    def estimate_betas(self, stock_returns: pd.DataFrame,
                       ff_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
        """
        Stima i beta per ogni titolo usando OLS
        """
        print("\nEstimating Fama-French betas...")

        # Verifica che tutte le colonne necessarie siano presenti
        missing_factors = [f for f in self.ff_factors if f not in ff_data.columns]
        if missing_factors:
            raise ValueError(f"Missing Fama-French factors: {missing_factors}")

        # Prepara i dati
        returns, factors = self._prepare_data(stock_returns, ff_data)

        betas = pd.DataFrame(index=returns.columns, columns=self.ff_factors)
        regression_stats = {}

        # Aggiungi la costante ai fattori
        X = sm.add_constant(factors[self.ff_factors])

        for ticker in returns.columns:
            try:
                # Rimuovi i NaN per questo ticker
                y = returns[ticker].dropna()
                X_clean = X.loc[y.index]

                if len(y) < 60:  # Richiedi almeno 60 osservazioni
                    print(f"Insufficient data for {ticker}: only {len(y)} observations")
                    continue

                # Fit della regressione OLS
                model = sm.OLS(y, X_clean).fit()

                # Salva i beta
                betas.loc[ticker, self.ff_factors] = model.params[1:]

                # Calcola statistiche
                regression_stats[ticker] = {
                    'r_squared': model.rsquared,
                    'p_values': model.pvalues,
                    'std_errors': model.bse,
                    'nobs': int(model.nobs) if isinstance(model.nobs, (int, float)) else 0
                }

                print(f"Estimated betas for {ticker}: R² = {model.rsquared:.3f}")

            except Exception as e:
                print(f"Error estimating betas for {ticker}: {str(e)}")
                continue

        return betas, regression_stats

    def calculate_ff_covariance(self, stock_returns: pd.DataFrame,
                                start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Calcola la matrice di covarianza basata sul modello Fama-French
        """
        try:
            # 1. Ottieni i fattori FF
            ff_data = self.get_ff_data(start_date, end_date)
            if ff_data.empty:
                raise ValueError("No Fama-French data available")

            # 2. Stima i beta
            betas, stats = self.estimate_betas(stock_returns, ff_data)

            # Se non abbiamo beta validi, usa la matrice storica
            if betas.isnull().all().all():
                print("Warning: No valid betas estimated. Using historical covariance.")
                historical_cov = stock_returns.cov() * 252
                return historical_cov, {
                    'betas': betas,
                    'factor_cov': None,
                    'regression_stats': stats
                }

            # 3. Calcola la matrice di covarianza dei fattori
            factor_cov = ff_data[self.ff_factors].cov() * 252  # Annualizzata

            # 4. Rimuovi eventuali NaN dai beta
            valid_stocks = betas.dropna(how='all').index
            if len(valid_stocks) == 0:
                raise ValueError("No valid betas after removing NaN")

            betas_clean = betas.loc[valid_stocks]

            # 5. Calcola la matrice di covarianza dei titoli
            B = betas_clean.values  # matrice dei beta
            F = factor_cov.values  # matrice di covarianza dei fattori
            C = np.dot(np.dot(B, F), B.T)  # matrice di covarianza dei titoli

            ff_cov = pd.DataFrame(
                C,
                index=valid_stocks,
                columns=valid_stocks
            )

            # 6. Verifica che la matrice sia ben formata
            if ff_cov.isnull().any().any():
                print("Warning: NaN values in FF covariance matrix. Using historical covariance.")
                return stock_returns.cov() * 252, {
                    'betas': betas,
                    'factor_cov': factor_cov,
                    'regression_stats': stats
                }

            return ff_cov, {
                'betas': betas,
                'factor_cov': factor_cov,
                'regression_stats': stats
            }

        except Exception as e:
            print(f"Error calculating FF covariance: {str(e)}")
            print("Falling back to historical covariance")
            return stock_returns.cov() * 252, {
                'betas': pd.DataFrame(),
                'factor_cov': pd.DataFrame(),
                'regression_stats': {}
            }