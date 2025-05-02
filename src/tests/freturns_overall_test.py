import pandas as pd
import numpy as np
from src.market_data import MarketData
from src.iis_indicators import IndicatorsIIS
from src.tucker_factor import TuckerFactorCalculator

class FactorReturnsOverall:
    def __init__(self):
        self.market_data = MarketData()

    def hist_returns(self):
        # Get prices and market caps using MarketData
        prices = self.market_data.download_prices()
        # Calculate returns
        returns = prices.pct_change().dropna()
        returns = returns.iloc[-21:]
        return returns

    def indicators(self):
        indicators_iis = IndicatorsIIS()
        indicators_f = indicators_iis.analyze_indicators()
        output = indicators_f

        # Combine all outputs into a single DataFrame with custom column names
        combined_indicators = pd.concat([
            output['str_signal'],
            output['ind_mom'],
            output['seasonality']
        ], axis=1, keys=['str_signal', 'ind_mom', 'seasonality'])  # Adding column names

        # Flatten MultiIndex if needed (in case each DataFrame had its own columns)
        #combined_indicators.columns = combined_indicators.columns.get_level_values(-1)

        return combined_indicators

    def tucker_factors(self, s):
        tucker_calculator = TuckerFactorCalculator()
        output = tucker_calculator.analyze_factors(s)
        return output['tucker_factors']

    def calculate_factor_returns(self):
        try:
            # Extract indicators
            indicators_df = self.indicators()
            print("\nIndicators")
            print(indicators_df.head())

            # Extract Tucker factors
            tucker_factors_df = self.tucker_factors()
            print("\nTucker Factors")
            print(tucker_factors_df.head())

            print("\nFactors returns Calculating")
            print("=" * 50)

            # Combine features
            X_F = pd.concat([indicators_df, tucker_factors_df], axis=1).dropna()

            # Historical returns
            R = self.hist_returns().mean()

            # Target variable: Y = R - 1
            ones_vector = np.ones_like(R)
            Y = R - ones_vector

            # Initialize an empty DataFrame to store factor returns per company
            factor_returns_per_company = pd.DataFrame(
                index=X_F.index,
                columns=X_F.columns
            )

            # Run OLS regression for each company
            for j in factor_returns_per_company.index:
                x = X_F.loc[j].values.reshape(1, -1)
                y = Y.get(j, np.nan)  # Safely get the value

                if pd.notna(y):
                    # Perform least squares regression
                    factor_return = np.linalg.lstsq(x, np.array([y]), rcond=None)[0]

                    # Store the result
                    factor_returns_per_company.loc[j] = factor_return.flatten()

            print("\nFactors returns Output")
            print("=" * 50)
            print(factor_returns_per_company.head())

            return factor_returns_per_company

        except Exception as e:
            print(f"Error during factor returns calculation: {str(e)}")
            raise

if __name__ == "__main__":
    factor_returns_calculator = FactorReturnsOverall()
    factor_returns_calculator.calculate_factor_returns()




