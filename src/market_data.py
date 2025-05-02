import pandas as pd
import yfinance as yf
from typing import List, Dict
import os
import json
from datetime import datetime


class MarketData:
    def __init__(self):
        self.start_date = '2020-01-01'
        self.end_date = '2025-02-01'

        # Cache directories
        self.cache_dir = 'data_cache'
        self.prices_cache_file = os.path.join(self.cache_dir, 'prices_cache.csv')
        self.market_caps_cache_file = os.path.join(self.cache_dir, 'market_caps.json')

        # Ensure cache directory exists
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.sectors = {
            'Technology': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'AMD', 'ADBE', 'CRM', 'CSCO', 'INTC', 'ORCL'],
            'Healthcare': ['JNJ', 'LLY', 'ABBV', 'MRK', 'BMY', 'PFE', 'TMO', 'AMGN', 'DHR', 'GILD'],
            'Consumer': ['AMZN', 'WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'TGT', 'SBUX'],
            'Finance': ['JPM', 'BAC', 'WFC', 'MS', 'GS', 'BLK', 'C', 'V', 'MA', 'AXP'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'SHEL', 'OXY', 'MPC', 'PSX', 'VLO']
        }

        # Load cached data on initialization
        self._prices_data = None
        self._market_caps_data = None
        self.load_cached_data()

    def get_tickers(self, sector: str = None) -> List[str]:
        if sector:
            return self.sectors.get(sector, [])
        return [ticker for tickers in self.sectors.values() for ticker in tickers]

    def load_cached_data(self):
        """Load cached data if available"""
        try:
            if os.path.exists(self.prices_cache_file):
                self._prices_data = pd.read_csv(self.prices_cache_file, index_col=0, parse_dates=True)
                print("Loaded prices from cache")

            if os.path.exists(self.market_caps_cache_file):
                with open(self.market_caps_cache_file, 'r') as f:
                    cache_data = json.load(f)
                    if datetime.now().strftime('%Y-%m-%d') == cache_data['date']:
                        self._market_caps_data = cache_data['market_caps']
                        print("Loaded market caps from cache")
        except Exception as e:
            print(f"Error loading cached data: {str(e)}")
            self._prices_data = None
            self._market_caps_data = None

    def save_prices_cache(self, data: pd.DataFrame):
        """Save prices data to cache"""
        try:
            data.to_csv(self.prices_cache_file)
            print("Saved prices to cache")
        except Exception as e:
            print(f"Error saving prices cache: {str(e)}")

    def save_market_caps_cache(self, data: Dict[str, float]):
        """Save market caps data to cache"""
        try:
            cache_data = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'market_caps': data
            }
            with open(self.market_caps_cache_file, 'w') as f:
                json.dump(cache_data, f)
            print("Saved market caps to cache")
        except Exception as e:
            print(f"Error saving market caps cache: {str(e)}")

    def download_prices(self, sector: str = None) -> pd.DataFrame:
        """Get prices data, using cache if available"""
        try:
            # If we have cached data and sector is specified, filter it
            if self._prices_data is not None:
                if sector:
                    tickers = self.get_tickers(sector)
                    return self._prices_data[tickers]
                return self._prices_data

            # If no cached data, download it
            tickers = self.get_tickers(sector)
            data = yf.download(tickers, start=self.start_date, end=self.end_date, progress=False)
            if data.empty:
                raise Exception("No data downloaded")

            prices = data['Close']

            # Cache the full dataset if we downloaded all sectors
            if sector is None:
                self._prices_data = prices
                self.save_prices_cache(prices)

            return prices

        except Exception as e:
            print(f"Error downloading prices: {str(e)}")
            return pd.DataFrame()

    def download_market_caps(self, sector: str = None) -> Dict[str, float]:
        """Get market caps data, using cache if available"""
        try:
            # If we have cached data from today and sector is specified, filter it
            if self._market_caps_data is not None:
                if sector:
                    return {ticker: cap for ticker, cap in self._market_caps_data.items()
                            if ticker in self.get_tickers(sector)}
                return self._market_caps_data

            # If no cached data, download it
            tickers = self.get_tickers(sector)
            market_caps = {}

            for ticker in tickers:
                try:
                    stock = yf.Ticker(ticker)
                    market_cap = stock.info.get('marketCap')
                    if market_cap:
                        market_caps[ticker] = market_cap
                except Exception as e:
                    print(f"Error for {ticker}: {str(e)}")
                    continue

            # Cache the full dataset if we downloaded all sectors
            if sector is None:
                self._market_caps_data = market_caps
                self.save_market_caps_cache(market_caps)

            return market_caps

        except Exception as e:
            print(f"Error downloading market caps: {str(e)}")
            return {}