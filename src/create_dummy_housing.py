from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 2000  # number of rows


def main() -> None:
    # Feature distributions (simple but semi-realistic)
    lot_area = RNG.normal(7500, 2500, N).clip(1500, 20000)
    bedrooms = RNG.integers(1, 6, N)
    bathrooms = RNG.integers(1, 4, N)
    sqft_living = RNG.normal(1800, 600, N).clip(400, 4500)
    house_age = RNG.integers(0, 120, N)
    distance_to_city = RNG.normal(10, 5, N).clip(0, 50)

    # Target: synthetic price formula + noise
    base_price = 50_000
    price = (
        base_price
        + 20 * lot_area
        + 30_000 * bedrooms
        + 40_000 * bathrooms
        + 120 * sqft_living
        - 500 * house_age
        - 1_000 * distance_to_city
        + RNG.normal(0, 25_000, N)  # noise
    )

    df = pd.DataFrame(
        {
            "lot_area": lot_area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "house_age": house_age,
            "distance_to_city": distance_to_city,
            "target": price,  # <-- important: this matches train.py expectation
        }
    )

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "housing.csv"
    df.to_csv(csv_path, index=False)

    print(f"✅ Wrote synthetic housing dataset to: {csv_path}")
    print(f"Rows: {len(df)}, columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
