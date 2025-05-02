import pandas as pd
from typing import Dict
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from src.market_data import MarketData


class IndicatorsIIS:
    def __init__(self):
        """
        Inizializza la classe per il calcolo dei 3 segnali short-term
        """
        self.signals = [
            'str_signal',  # Industry relative reversal
            'ind_mom',  # Industry momentum
            'seasonality',  # Same month seasonality
        ]
        self.market_data = MarketData()

    def normalize_indicators(self, indicators: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Normalizza gli indicatori utilizzando diversi metodi

        Parameters:
        -----------
        indicators : pd.DataFrame
            DataFrame contenente gli indicatori da normalizzare
        method : str
            Metodo di normalizzazione ('standard', 'robust', 'minmax')

        Returns:
        --------
        pd.DataFrame
            DataFrame con gli indicatori normalizzati
        """
        if method not in ['standard', 'robust', 'minmax']:
            raise ValueError("Metodo di normalizzazione non valido. Usa 'standard', 'robust' o 'minmax'")

        normalized = pd.DataFrame(index=indicators.index, columns=indicators.columns)

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:  # minmax
            scaler = MinMaxScaler()

        for col in indicators.columns:
            values = indicators[col].values.reshape(-1, 1)
            normalized[col] = scaler.fit_transform(values).flatten()

        return normalized

    def calculate_industry_relative_reversal(self, returns: pd.DataFrame,
                                             industry_data: Dict[str, str]) -> pd.Series:
        """
        Calcola l'industry relative reversal come:
        rendimento dello stock - rendimento della sua industria nell'ultimo mese
        """
        # Prendi i rendimenti dell'ultimo mese
        last_month_returns = returns.iloc[-21:]  # ~21 giorni di trading

        # Calcola il rendimento mensile composto per ogni stock
        stock_monthly_returns = (1 + last_month_returns).prod() - 1

        # Calcola il rendimento mensile per ogni industria
        industry_returns = {}
        for industry in set(industry_data.values()):
            industry_stocks = [s for s, ind in industry_data.items() if ind == industry]
            industry_returns[industry] = stock_monthly_returns[industry_stocks].mean()

        # Calcola l'industry relative reversal
        str_signal = pd.Series(index=returns.columns)
        for stock in returns.columns:
            stock_industry = industry_data[stock]
            str_signal[stock] = stock_monthly_returns[stock] - industry_returns[stock_industry]

        return str_signal

    def calculate_industry_momentum(self, returns: pd.DataFrame,
                                    industry_data: Dict[str, str]) -> pd.Series:
        """
        Calcola l'industry momentum come media dei rendimenti dei peer
        nell'ultimo mese (escluso lo stock stesso)
        """
        # Prendi i rendimenti dell'ultimo mese
        last_month_returns = returns.iloc[-21:]

        # Calcola il rendimento mensile composto per ogni stock
        stock_monthly_returns = (1 + last_month_returns).prod() - 1

        # Calcola momentum per ogni stock
        momentum = pd.Series(index=returns.columns)

        for stock in returns.columns:
            # Trova i peer nella stessa industria (escluso lo stock stesso)
            industry = industry_data[stock]
            peers = [s for s, ind in industry_data.items()
                     if ind == industry and s != stock]

            # Momentum è la media dei rendimenti dei peer
            if peers:
                momentum[stock] = stock_monthly_returns[peers].mean()
            else:
                momentum[stock] = 0.0

        return momentum

    def calculate_seasonality(self, returns: pd.DataFrame,
                              current_month: int = None) -> pd.Series:
        """
        Calcola l'effetto di stagionalità basato sui rendimenti
        dello stesso mese nei 10 anni precedenti
        """
        if current_month is None:
            current_month = returns.index[-1].month

        next_month = current_month % 12 + 1

        # Filtra solo i rendimenti del mese target negli ultimi 10 anni
        mask = (returns.index.month == next_month)
        target_month_returns = returns[mask]

        # Calcola la media per ogni stock
        # Prendiamo gli ultimi 10 anni (escludendo il più recente se presente)
        years = target_month_returns.groupby(target_month_returns.index.year).mean()
        if len(years) > 1:  # escludiamo l'anno corrente se presente
            years = years.iloc[:-1]
        if len(years) > 10:  # prendiamo solo gli ultimi 10 anni
            years = years.iloc[-10:]

        return years.mean()

    def analyze_indicators(self, sector: str = None):
        """
        Analizza e calcola tutti gli indicatori per un settore specifico o l'intero mercato

        Parameters:
        -----------
        sector : str, optional
            Il settore da analizzare. Se None, analizza l'intero mercato.

        Returns:
        --------
        dict
            Dizionario contenente tutti i risultati dell'analisi
        """
        try:
            # Ottieni i dati necessari
            prices = self.market_data.download_prices(sector)
            returns = prices.pct_change().dropna()

            # Crea dizionario industry_data
            if sector:
                industry_data = {ticker: sector for ticker in self.market_data.get_tickers(sector)}
            else:
                industry_data = {}
                for sector_name, sector_tickers in self.market_data.sectors.items():
                    for ticker in sector_tickers:
                        industry_data[ticker] = sector_name

            # Calcola gli indicatori base
            str_signal = self.calculate_industry_relative_reversal(returns, industry_data)
            ind_mom = self.calculate_industry_momentum(returns, industry_data)
            seasonality = self.calculate_seasonality(returns)

            # Combina gli indicatori
            all_indicators = pd.DataFrame({
                'str_signal': str_signal,
                'ind_mom': ind_mom,
                'seasonality': seasonality
            })

            # Calcola le normalizzazioni
            methods = ['standard', 'robust', 'minmax']
            normalized_results = {}
            for method in methods:
                normalized = self.normalize_indicators(all_indicators, method)
                normalized_results[method] = normalized

            # Analisi dei valori estremi
            extreme_values = {}
            for col in all_indicators.columns:
                threshold = all_indicators[col].std() * 2
                extremes = all_indicators[abs(all_indicators[col]) > threshold]
                if not extremes.empty:
                    extreme_values[col] = extremes[col].sort_values(ascending=False)

            return {
                'raw_indicators': all_indicators,
                'normalized': normalized_results,
                'str_signal': str_signal,
                'ind_mom': ind_mom,
                'seasonality': seasonality,
                'extreme_values': extreme_values
            }

        except Exception as e:
            raise Exception(f"Errore durante l'analisi degli indicatori: {str(e)}")
