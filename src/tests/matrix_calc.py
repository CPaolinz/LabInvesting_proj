import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.market_data import MarketData
from src.fama_french import FamaFrenchAnalysis
from src.iis_indicators import IndicatorsIIS
from src.tucker_factor import TuckerFactorCalculator
from src.correlation_calc import combined_vif_analysis
from sklearn.covariance import LedoitWolf
import os

class MatrixCalc:
    def __init__(self):
        self.market_data = MarketData()
        self.fama_french = FamaFrenchAnalysis()
        # Create plots directory if it doesn't exist
        self.plots_dir = 'matrix_plots'
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)

    def vif(self, S=None):
        return combined_vif_analysis(S)

    def ff_mat(self, stock_returns: pd.DataFrame, start_date: str, end_date: str):
        ff_cov, _ = self.fama_french.calculate_ff_covariance(stock_returns, start_date, end_date)
        return ff_cov

    def indicators(self, S=None):
        indicators_iis = IndicatorsIIS()
        indicators_f = indicators_iis.analyze_indicators(S)
        combined_indicators = pd.concat([
            indicators_f['str_signal'],
            indicators_f['ind_mom'],
            indicators_f['seasonality']
        ], axis=1, keys=['str_signal', 'ind_mom', 'seasonality'])
        return combined_indicators

    def tucker_factors(self, S=None):
        tucker_calculator = TuckerFactorCalculator()
        tucker_f = tucker_calculator.analyze_factors(S)
        return tucker_f['tucker_factors']

    def combine_betas(self, S = None):
        try:
            indicators_df = self.indicators(S)
            tucker_factors_df = self.tucker_factors(S)

            X_F = pd.concat([indicators_df, tucker_factors_df], axis=1).dropna()

            #Calculate vif
            vif_s = self.vif(S)

            # Standardize column names to lowercase
            vif_s.columns = [col.lower() for col in vif_s.columns]

            if 'vif' in vif_s.columns and 'feature' in vif_s.columns:
                high_vif_features = vif_s[vif_s['vif'] > 10]['feature'].tolist()
                X_F = X_F.drop(columns=high_vif_features, errors='ignore')
            else:
                print("Warning: VIF output missing expected columns.")

            return X_F

        except Exception as e:
            print(f"Error during factor merge: {str(e)}")
            raise

    def exposure_mat(self, S = None):
        try:
            betas = self.combine_betas(S)
            betas = betas.clip(lower=-3, upper=3)

            print(f"\n Our betas: \n {betas}")

            # Limit extreme values
            valid_stocks = betas.index

            lw = LedoitWolf()
            lw.fit(betas)
            factor_cov = lw.covariance_
            #Normalize the covariance matrix
            factor_cov = factor_cov / np.linalg.norm(factor_cov)

            B = betas.values
            F = factor_cov
            C = np.dot(np.dot(B, F), B.T)

            ff_cov = pd.DataFrame(C, index=valid_stocks, columns=valid_stocks)
            return ff_cov
        except Exception as e:
            print(f"Error calculating exposure matrix: {str(e)}")
            raise

    def historic_mat(self,S=None):
        try:
            prices = self.market_data.download_prices(S)
            returns = prices.pct_change().dropna()
            historic = returns.cov()
            return historic
        except Exception as e:
            print(f"Error calculating historical matrix: {str(e)}")
            raise

    def save_heatmap(self, data, title, filename):
        """Save heatmap plot to file"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title(title)
        plt.tight_layout()  # Ensure the plot fits in the figure
        plt.savefig(os.path.join(self.plots_dir, filename))
        plt.close()  # Close the figure to free memory
        print(f"Saved plot: {filename}")


if __name__ == "__main__":
    mat = MatrixCalc()

    try:
        historic = mat.historic_mat()

        prices = mat.market_data.download_prices()
        returns = prices.pct_change().dropna()
        start_date = mat.market_data.start_date
        end_date = mat.market_data.end_date

        ff = mat.ff_mat(returns, start_date, end_date)
        exposure = mat.exposure_mat()

        print(f"\n Matrix from historical data: \n {historic}")
        print(f"\n Matrix from Fama French data: \n {ff}")
        print(f"\n Matrix from our exposure data: \n {exposure}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    print("=" * 50)

    try:
        sectors = ['Technology', 'Healthcare', 'Consumer', 'Finance', 'Energy']
        for sector in sectors:
            print(f"\nMatrices for : {sector}")
            print("-" * 50)

            # Historic
            historic = mat.historic_mat(sector)
            mat.save_heatmap(
                historic,
                f"Historical Matrix Heatmap for: {sector}",
                f"historic_matrix_{sector}.png"
            )
            print(f"\n Matrix from historical data: \n {historic}")

            # Fama French
            prices = mat.market_data.download_prices(sector)
            returns = prices.pct_change().dropna()
            start_date = mat.market_data.start_date
            end_date = mat.market_data.end_date

            ff = mat.ff_mat(returns, start_date, end_date)
            mat.save_heatmap(
                ff,
                f"Fama French Matrix Heatmap for: {sector}",
                f"fama_french_matrix_{sector}.png"
            )
            print(f"\n Matrix from Fama French data: \n {ff}")

            # Exposure
            exposure = mat.exposure_mat(sector)
            mat.save_heatmap(
                exposure,
                f"Exposure Matrix Heatmap for: {sector}",
                f"exposure_matrix_{sector}.png"
            )
            print(f"\n Matrix from our exposure data: \n {exposure}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")