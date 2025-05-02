import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.tests.freturns_sector_test import FactorReturnsSector
from src.tests.matrix_calc import MatrixCalc
from portfolio import BenchmarkNeutralOptimizer


class OptimizedLambdaAnalyzer:
    def __init__(self, sector):
        self.sector = sector
        # Initialize components once
        self.factor_returns_calculator = FactorReturnsSector()
        self.returns = self.factor_returns_calculator.calculate_factor_returns(sector)
        self.cov_mat = MatrixCalc()
        self.asset_cov_matrix = self.cov_mat.historic_mat(sector)
        # Initialize portfolio calculator once
        self.portfolio_calculator = BenchmarkNeutralOptimizer(self.returns, self.asset_cov_matrix)
        # Calculate tilt target once
        self.tilt_target = self.portfolio_calculator.calculate_tilt(sector)
        print(f"Initialized analyzer for {sector} with tilt target: {self.tilt_target}")

    def analyze_lambda_effect(self, lambda_values=None):
        """
        Analyze how lambda affects volatilities with the pre-calculated tilt target.
        """
        if lambda_values is None:
            lambda_values = np.arange(0, 12, 0.1)

        results = {
            'lambda': [],
            'portfolio_volatility': [],
            'factor_volatility': [],
            'tilt_target': []
        }

        total_lambdas = len(lambda_values)
        for idx, lambda_val in enumerate(lambda_values, 1):
            print(f"Processing lambda = {lambda_val:.1f} ({idx}/{total_lambdas})")

            try:
                # Qui usiamo 'sector' invece di 'S'
                result = self.portfolio_calculator.optimize_portfolio(
                    sector=self.sector,  # Cambiato da S a sector
                    lambda_tilt=lambda_val,
                    long_only=False
                )

                if result is not None:
                    weights, port_vol, factor_vol = result
                    results['lambda'].append(lambda_val)
                    results['portfolio_volatility'].append(port_vol)
                    results['factor_volatility'].append(factor_vol)
                    results['tilt_target'].append(self.tilt_target)

            except Exception as e:
                print(f"Error for lambda={lambda_val}: {str(e)}")
                continue

        return pd.DataFrame(results)

    def plot_lambda_effect(self, results_df):
        # Il resto del codice rimane invariato
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.7

        fig, ax1 = plt.subplots()

        # Portfolio Volatility
        color1 = '#1f77b4'
        ax1.set_xlabel('Lambda (λ)', fontsize=12, weight='bold')
        ax1.set_ylabel('Portfolio Volatility', color=color1, fontsize=12, weight='bold')
        line1 = ax1.plot(results_df['lambda'],
                         results_df['portfolio_volatility'],
                         color=color1,
                         marker='o',
                         markersize=4,
                         linestyle='-',
                         linewidth=2,
                         label='Portfolio Volatility')
        ax1.tick_params(axis='y', labelcolor=color1)

        # Factor Volatility
        ax2 = ax1.twinx()
        color2 = '#d62728'
        ax2.set_ylabel('Factor Volatility', color=color2, fontsize=12, weight='bold')
        line2 = ax2.plot(results_df['lambda'],
                         results_df['factor_volatility'],
                         color=color2,
                         marker='s',
                         markersize=4,
                         linestyle='-',
                         linewidth=2,
                         label='Factor Volatility')
        ax2.tick_params(axis='y', labelcolor=color2)

        ax1.set_xlim(0, 12)
        ax1.set_xticks(np.arange(0, 12, 1))
        ax1.set_xticks(np.arange(0, 12, 1), minor=True)

        plt.title(f'Volatility Analysis for {self.sector} Sector\nCalculated Tilt Target = {self.tilt_target:.4f}',
                  fontsize=14,
                  weight='bold',
                  pad=20)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   ncol=2, fontsize=10)

        plt.tight_layout()
        plt.savefig(f'lambda_effect_analysis_{self.sector}.png', bbox_inches='tight', dpi=300)
        plt.close()


if __name__ == "__main__":
    sectors = ['Technology', 'Healthcare', 'Consumer', 'Finance', 'Energy']
    LAMBDA_VALUES = np.arange(0, 12, 0.1)  # Reduced step size for testing

    for SECTOR in sectors:
        print(f"\n{'=' * 50}")
        print(f"Starting analysis for {SECTOR} sector")
        print(f"{'=' * 50}")

        # Initialize analyzer for this sector
        analyzer = OptimizedLambdaAnalyzer(SECTOR)

        # Run analysis
        results_df = analyzer.analyze_lambda_effect(LAMBDA_VALUES)

        # Save results
        results_df.to_csv(f'lambda_effect_results_{SECTOR}.csv', index=False)
        print(f"Saved results to lambda_effect_results_{SECTOR}.csv")

        # Create plot
        analyzer.plot_lambda_effect(results_df)
        print(f"Created plot in lambda_effect_analysis_{SECTOR}.png")