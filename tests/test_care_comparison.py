import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.compare_care_models import FAMILIES, fast_alerts, fit, score_values
from scripts.train_care_baseline import episode_starts, persistent_alerts


def test_vectorized_alerts_match_original_with_gaps():
    rng = np.random.default_rng(42)
    steps = rng.choice([600, 1200, 3600], 2000, p=[0.95, 0.04, 0.01])
    times = pd.to_datetime(np.cumsum(steps), unit="s")
    for required in [1, 6, 12]:
        for probability in [0.1, 0.5, 0.9, 1]:
            exceed = rng.random(2000) < probability
            expected = persistent_alerts(times, exceed, required)
            actual, starts = fast_alerts(times, exceed, required)
            np.testing.assert_array_equal(expected, actual)
            np.testing.assert_array_equal(episode_starts(times, expected), starts)


def test_all_families_ignore_excluded_turbine():
    rng = np.random.default_rng(42)
    cases = []
    for asset in [0, 1, 2]:
        for label in ["normal", "anomaly"]:
            cases.append(
                dict(
                    asset=asset,
                    label=label,
                    fit_idx=np.arange(100),
                    x=pd.DataFrame(rng.normal(int(label == "anomaly"), 1, (100, 3))),
                )
            )
    with threadpool_limits(limits=2):
        for family in FAMILIES:
            before, _ = fit(cases, {2}, family)
            for c in cases:
                if c["asset"] == 2:
                    c["x"].iloc[:] += 1e7
            after, _ = fit(cases, {2}, family)
            np.testing.assert_allclose(
                score_values(before, cases[0]["x"], family),
                score_values(after, cases[0]["x"], family),
                atol=1e-10,
            )
