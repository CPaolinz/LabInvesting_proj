from src.iis_indicators import IndicatorsIIS

def test_indicators(sector: str = None):
    """
    Test completo per gli indicatori IIS
    """
    print("Test Indicators IIS")
    print("-" * 50)

    indicators = IndicatorsIIS()

    try:
        results = indicators.analyze_indicators(sector)

        print("\nCalcolo degli indicatori...")

        print("\nIndustry Relative Reversal - Statistiche:")
        print(results['str_signal'].describe().round(4))

        print("\nIndustry Momentum - Statistiche:")
        print(results['ind_mom'].describe().round(4))

        print("\nSeasonality - Statistiche:")
        print(results['seasonality'].describe().round(4))

        print("\nMatrice di correlazione tra indicatori:")
        print(results['raw_indicators'].corr().round(3))

        print("\nTest di normalizzazione...")
        for method, normalized in results['normalized'].items():
            print(f"\nStatistiche dopo normalizzazione {method}:")
            print(normalized.describe().round(4))

            print(f"\nCorrelazioni dopo normalizzazione {method}:")
            print(normalized.corr().round(3))

        print("\nAnalisi dei valori estremi:")
        for col, extremes in results['extreme_values'].items():
            if not extremes.empty:
                print(f"\nValori estremi per {col}:")
                print(extremes)

        return results

    except Exception as e:
        print(f"Errore durante il test: {str(e)}")
        raise


if __name__ == "__main__":
    print("\nAnalisi per l'intero mercato")
    print("=" * 50)
    results_all = test_indicators()

    for sector in ['Technology', 'Healthcare', 'Consumer', 'Finance', 'Energy']:
        print(f"\n\nAnalisi per il settore {sector}")
        print("=" * 50)
        test_indicators(sector)