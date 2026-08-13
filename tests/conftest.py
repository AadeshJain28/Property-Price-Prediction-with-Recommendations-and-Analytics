from __future__ import annotations

import pandas as pd
import pytest

from property_price.config import Config


@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config.load()


@pytest.fixture(scope="session")
def raw(cfg) -> pd.DataFrame:
    return pd.read_csv(cfg.data_path("raw"))


@pytest.fixture
def toy() -> pd.DataFrame:
    """Six listings, two of which share an identical feature vector.

    Hand-checked: rows 0 and 1 differ only in price, so any correct grouping puts
    them in the same group and a correct duplicate count reports exactly 1.
    """
    return pd.DataFrame(
        {
            "property_type": ["flat"] * 5 + ["house"],
            "availability": ["Ready To Move"] * 6,
            "location": ["whitefield", "whitefield", "hebbal", "hebbal", "jayanagar", "jayanagar"],
            "area_type": ["Super built-up  Area"] * 6,
            "bedroom": [2, 2, 3, 4, 3, 5],
            "bath": [2, 2, 2, 3, 2, 4],
            "balcony": [1, 1, 2, 2, 1, 3],
            "built_up_area": [1000.0, 1000.0, 1400.0, 1800.0, 1300.0, 2600.0],
            "area": [1000.0, 1000.0, 1400.0, 1800.0, 1300.0, 2600.0],
            "price": [0.50, 0.62, 0.90, 1.30, 0.85, 2.40],
            "price_per_sqft": [5000.0, 6200.0, 6428.0, 7222.0, 6538.0, 9230.0],
        }
    )
