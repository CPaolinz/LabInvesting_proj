import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.market_data import MarketData
from src.iis_indicators import IndicatorsIIS
from src.tucker_factor import TuckerFactorCalculator
from src.tests.freturns_sector_test import FactorReturnsSector
from src.tests.matrix_calc import MatrixCalc
import signal


class BenchmarkNeutralOptimizer:
    class TimeoutException(Exception):
        pass

    def __init__(self, factor_returns, asset_cov_matrix):
        self.R_f = factor_returns
        self.SIGMA = asset_cov_matrix

        # Regularize the covariance matrix
        epsilon = 1e-5
        self.SIGMA += np.eye(self.SIGMA.shape[0]) * epsilon

        # Cache for computed values
        self._cache = {}

    def _timeout_handler(self, signum, frame):
        raise self.TimeoutException("Optimization timed out!")

    def _get_cached_value(self, key, sector):
        """Helper method to manage cached values"""
        cache_key = f"{key}_{sector}"
        return self._cache.get(cache_key)

    def _set_cached_value(self, key, sector, value):
        """Helper method to set cached values"""
        cache_key = f"{key}_{sector}"
        self._cache[cache_key] = value
        return value

    def indicators(self, sector):
        cached = self._get_cached_value('indicators', sector)
        if cached is not None:
            return cached

        indicators = IndicatorsIIS()
        indicators_f = indicators.analyze_indicators(sector)
        combined_indicators = pd.concat([
            indicators_f['str_signal'],
            indicators_f['ind_mom'],
            indicators_f['seasonality']
        ], axis=1, keys=['str_signal', 'ind_mom', 'seasonality'])

        return self._set_cached_value('indicators', sector, combined_indicators)

    def tucker_factors(self, sector):
        cached = self._get_cached_value('tucker', sector)
        if cached is not None:
            return cached

        tucker_calculator = TuckerFactorCalculator(n_factors=2)
        tucker_f = tucker_calculator.analyze_factors(sector)
        return self._set_cached_value('tucker', sector, tucker_f['tucker_factors'])

    def combine_betas(self, sector):
        cached = self._get_cached_value('betas', sector)
        if cached is not None:
            return cached

        indicators_df = self.indicators(sector)
        tucker_factors_df = self.tucker_factors(sector)
        X_F = pd.concat([indicators_df, tucker_factors_df], axis=1).dropna()

        return self._set_cached_value('betas', sector, X_F)

    def benchmark_w(self, sector):
        cached = self._get_cached_value('benchmark_weights', sector)
        if cached is not None:
            return cached

        market_data = MarketData()
        caps = market_data.download_market_caps(sector)
        caps = {ticker: cap for ticker, cap in caps.items() if isinstance(cap, (int, float))}

        total_cap = sum(caps.values())
        if total_cap == 0:
            raise ValueError("Total market capitalization is zero")

        w_b = pd.Series({ticker: cap / total_cap for ticker, cap in caps.items()})
        w_b.index.name = 'Ticker'

        return self._set_cached_value('benchmark_weights', sector, w_b)

    def tracking_error(self, w, sector):
        w_b = self.benchmark_w(sector)

        if not isinstance(w, pd.Series):
            w = pd.Series(w, index=w_b.index)

        common_tickers = w_b.index.intersection(self.SIGMA.index)
        w = w.loc[common_tickers]
        w_b = w_b.loc[common_tickers]
        aligned_SIGMA = self.SIGMA.loc[common_tickers, common_tickers]

        assert w.shape[0] == aligned_SIGMA.shape[0], "Dimension mismatch"
        return (w - w_b).T @ aligned_SIGMA @ (w - w_b)

    def factor_tilt_penalty(self, w, tilt_target, lambda_tilt, sector):
        beta = self.combine_betas(sector)
        result = beta.T @ w
        tilt_target = pd.to_numeric(tilt_target, errors='coerce') if tilt_target is not None else 0
        return lambda_tilt * np.sum((result - tilt_target) ** 2)

    def portfolio_volatility(self, w):
        assert w.shape[0] == self.SIGMA.shape[0], "Dimension mismatch"
        return np.sqrt(w.T @ self.SIGMA @ w)

    def factor_based_volatility(self, w, sector):
        beta = self.combine_betas(sector)
        SIGMA_f = beta.cov()
        return np.sqrt(w.T @ beta @ SIGMA_f @ beta.T @ w)

    def objective_function(self, w, tilt_target, lambda_tilt, sector):
        """
        Calculate the objective function value combining tracking error and factor tilt penalty
        """
        te = self.tracking_error(w, sector)
        penalty = self.factor_tilt_penalty(w, tilt_target, lambda_tilt, sector)
        return te + penalty

    def calculate_tilt(self, sector=None):
        """Calculates the tilt target using cached values where possible"""
        cached = self._get_cached_value('tilt_target', sector)
        if cached is not None:
            return cached

        w_b = self.benchmark_w(sector)
        tucker_factors = self.tucker_factors(sector)
        str_signal = tucker_factors['Tucker_Factor_2']

        common_tickers = w_b.index.intersection(str_signal.index)
        if len(common_tickers) == 0:
            raise ValueError("No common tickers between benchmark weights and signal")

        w_b_aligned = w_b.loc[common_tickers]
        str_signal_aligned = str_signal.loc[common_tickers]
        tilt = str_signal_aligned.T @ w_b_aligned

        return self._set_cached_value('tilt_target', sector, tilt)

    def optimize_portfolio(self, sector, lambda_tilt=0.2, long_only=True):
        w_b = self.benchmark_w(sector)
        n_assets = len(w_b)
        initial_weights = w_b.copy()
        tilt_target = self.calculate_tilt(sector)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(n_assets)] if long_only else [(-1, 1) for _ in range(n_assets)]

        signal.signal(signal.SIGALRM, self._timeout_handler)
        signal.alarm(120)

        try:
            result = minimize(
                self.objective_function,
                initial_weights,
                args=(tilt_target, lambda_tilt, sector),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 50, 'disp': True, 'ftol': 1e-3}
            )
            signal.alarm(0)

            if result.success:
                optimized_weights = pd.Series(result.x, index=w_b.index, name='Optimized Weights')
                port_volatility = self.portfolio_volatility(optimized_weights.values)
                factor_volatility = self.factor_based_volatility(optimized_weights.values, sector)
                return optimized_weights, port_volatility, factor_volatility
            else:
                print(f"⚠️ Optimization failed: {result.message}")
                return None

        except self.TimeoutException:
            print("⚠️ Optimization timed out!")
            return None
        except Exception as e:
            print(f"⚠️ Optimization error: {e}")
            return None