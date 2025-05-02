import pandas as pd
from src.tucker_factor import TuckerFactorCalculator, interpret_tucker_factors

def test_tucker_factors(sector=None):
    """
    Test completo per TuckerFactorCalculator che analizza i fattori
    sia per l'intero mercato che per settori specifici
    """
    print("Test Tucker Factor Analysis")
    print("-" * 50)

    tucker_calculator = TuckerFactorCalculator(n_factors=2)

    try:
        print("\nCalcolo Factor Exposures...")
        results = tucker_calculator.analyze_factors(sector)

        print("\nVIF per i fattori originali:")
        print(results['original_vif'])

        print("\nVIF per i fattori Tucker:")
        print(results['tucker_vif'])

        print("\nMatrice di correlazione tra fattori originali:")
        print(results['original_exposures'].corr().round(3))

        print("\nMatrice di correlazione tra fattori Tucker:")
        print(results['tucker_factors'].corr().round(3))

        print(f"\nVarianza spiegata dai fattori Tucker: {results['variance_explained']:.2%}")

        print("\nMatrice di covarianza basata sui fattori Tucker:")
        print(results['tucker_cov'].round(3))

        print("\nCorrelazioni tra fattori Tucker e originali:")
        print(results['interpretation']['correlations'])

        print("\nLoadings dei fattori Tucker:")
        print(results['interpretation']['loadings'])

        for factor in results['tucker_factors'].columns:
            print(f"\nInterpretazione {factor}:")
            correlations = abs(results['interpretation']['correlations'].loc[factor]).sort_values(ascending=False)
            print(f"Fattori più correlati:")
            for orig_factor, corr in correlations[:2].items():
                print(f"- {orig_factor}: {corr:.3f}")

        if sector:
            print(f"\nAnalisi dettagliata per il settore {sector}:")
            print("\nStatistiche descrittive dei fattori originali:")
            print(results['original_exposures'].describe().round(3))

            if results['extreme_values']:
                print("\nTitoli con valori estremi nei fattori:")
                for factor, values in results['extreme_values'].items():
                    print(f"\nFattore {factor} - Titoli con valori estremi:")
                    print(values)

        return results

    except Exception as e:
        print(f"Errore durante il test: {str(e)}")
        raise


if __name__ == "__main__":
    print("\nAnalisi per l'intero mercato")
    print("=" * 50)
    results_all = test_tucker_factors()

    sectors = ['Technology', 'Healthcare', 'Consumer', 'Finance', 'Energy']
    sector_results = {}

    for sector in sectors:
        print(f"\n\nAnalisi per il settore {sector}")
        print("=" * 50)
        sector_results[sector] = test_tucker_factors(sector)

        if len(sector_results) > 1:
            print(f"\nConfronto varianza spiegata tra settori:")
            var_explained = {s: r['variance_explained'] for s, r in sector_results.items()}
            print(pd.Series(var_explained).round(3))