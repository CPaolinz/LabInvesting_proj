import pandas as pd
import numpy as np
from src.market_data import MarketData
from src.iis_indicators import IndicatorsIIS
from src.tucker_factor import TuckerFactorCalculator
from src.correlation_calc import combined_vif_analysis
import traceback


class FactorReturnsSector:
    def __init__(self):
        self.market_data = MarketData()

    def hist_returns(self, s):
        prices = self.market_data.download_prices(s)
        returns = prices.pct_change().dropna()
        returns = returns.iloc[-21:]
        return returns

    def indicators(self, s):
        indicators = IndicatorsIIS()
        output = indicators.analyze_indicators(s)
        combined_indicators = pd.concat([
            output['str_signal'],
            output['ind_mom'],
            output['seasonality']
        ], axis=1, keys=['str_signal', 'ind_mom', 'seasonality'])
        return combined_indicators

    def tucker_factors(self, s):
        tucker_calculator = TuckerFactorCalculator()
        output = tucker_calculator.analyze_factors(s)
        return output['tucker_factors']

    def vif(self, sector):
        return combined_vif_analysis(sector)

    def calculate_factor_returns(self, s):
        try:
            indicators_df = self.indicators(s)
            print("\nIndicators")
            print(indicators_df.head())

            tucker_factors_df = self.tucker_factors(s)
            print("\nTucker Factors")
            print(tucker_factors_df.head())

            print("\nFactors returns Calculating")
            print("=" * 50)

            X_F = pd.concat([indicators_df, tucker_factors_df], axis=1).dropna()

            vif_s = self.vif(s)

            # Standardize column names to lowercase
            vif_s.columns = [col.lower() for col in vif_s.columns]

            if 'vif' in vif_s.columns and 'feature' in vif_s.columns:
                high_vif_features = vif_s[vif_s['vif'] > 10]['feature'].tolist()
                X_F = X_F.drop(columns=high_vif_features, errors='ignore')
            else:
                print("Warning: VIF output missing expected columns.")

            R = self.hist_returns(s).mean()

            ones_vector = np.ones_like(R)
            Y = R - ones_vector

            factor_returns_per_company = pd.DataFrame(index=X_F.index, columns=X_F.columns)

            for j in X_F.index:
                x = X_F.loc[j].values.reshape(1, -1)
                y = Y.get(j, np.nan)

                if pd.notna(y) and x.size > 0:
                    try:
                        factor_return = np.linalg.lstsq(x, np.array([y]), rcond=None)[0]
                        factor_returns_per_company.loc[j] = factor_return.flatten()
                    except Exception as inner_e:
                        print(f"Error in regression for {j}: {str(inner_e)}")

            print(f"\nFactors returns Output for {s}")
            print("=" * 50)
            print(factor_returns_per_company.head())

            return factor_returns_per_company

        except Exception as e:
            print(f"Error during factor returns calculation for sector {s}: {str(e)}")
            traceback.print_exc()
            return None


if __name__ == "__main__":
    sectors = ['Technology', 'Healthcare', 'Consumer', 'Finance', 'Energy']
    for sector in sectors:
        print(f"\nFactor returns for: {sector}")
        print("=" * 50)
        factor_returns_calculator = FactorReturnsSector()
        results = factor_returns_calculator.calculate_factor_returns(sector)

