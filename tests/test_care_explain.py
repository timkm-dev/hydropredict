from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from hydropredict.analysis.care_explain import (
    at_least,
    chance_of_hit,
    contribution_shares,
    display_component,
    fault_component,
    sensor_component,
    top_sensors,
)


@pytest.mark.parametrize(
    ("text", "component"),
    [
        ("Transformer failure", "transformer"),
        ("high temperature in transformer cell", "transformer"),
        ("Hydraulic group", "hydraulic"),
        ("Gearbox bearings damaged", "gearbox"),
        ("Generator bearing failure", "generator"),
        ("Turbine is stopped due to a main bearing damage", "rotor bearing"),
        ("Turbine is in standstill since 01.08 due to rotorbearing damage.", "rotor bearing"),
        ("Rotor Bearing 2 - Damage", "rotor bearing"),
        ("Pitch failure - defect fan on pitch motor", None),
    ],
)
def test_fault_components(text, component):
    assert fault_component(text) == component


@pytest.mark.parametrize(
    ("text", "component"),
    [
        ("Temperature in HV transformer phase L1", "transformer"),
        ("Transformator core temperature", "transformer"),
        ("Temperature oil in hydraulic group", "hydraulic"),
        ("Gearbox oil tank temperature", "gearbox"),
        ("Temperature in generator bearing 2 (Drive End)", "generator"),
        ("Active power - generator connected in star", None),
        ("Rotor bearing temperature 1", "rotor bearing"),
        ("Rotor rpm", None),
    ],
)
def test_sensor_components(text, component):
    assert sensor_component(text) == component


def test_shares_are_causal_and_sum_to_one():
    times = pd.date_range("2023-01-01", periods=8, freq="10min")
    errors = np.array([[1, 1], [1, 3], [np.nan, np.nan], [9, 1], [1, 1], [1, 1], [1, 1], [1, 1.0]])
    shares = contribution_shares(times, errors, ["a", "b"], 2)
    np.testing.assert_allclose(shares.iloc[1], [1 / 3, 2 / 3])
    assert shares.iloc[2].isna().all()
    np.testing.assert_allclose(shares.dropna().sum(axis=1), 1)
    for n in range(1, 9):
        pd.testing.assert_frame_equal(
            contribution_shares(times[:n], errors[:n], ["a", "b"], 2), shares.iloc[:n]
        )
    assert top_sensors(shares.iloc[3], 1) == [("a", 0.9)]


def test_chance_of_hit_matches_enumeration():
    sensors, matching = 9, 2
    draws = list(combinations(range(sensors), 3))
    expected = sum(any(i < matching for i in d) for d in draws) / len(draws)
    assert chance_of_hit(sensors, matching) == pytest.approx(expected)
    assert chance_of_hit(10, 0) == 0


def test_poisson_binomial_tail():
    assert at_least([0.5, 0.5], 1) == pytest.approx(0.75)
    assert at_least([0.2, 0.5, 0.9], 0) == pytest.approx(1)
    assert at_least([0.2, 0.5, 0.9], 3) == pytest.approx(0.09)


@pytest.mark.parametrize(
    ("text", "component"),
    [
        ("Hydraulic pressure Aggregate System pressure", "Hydraulic system"),
        ("Temperature in generator bearing 2 (Drive End)", "Generator"),
        ("Active power - generator connected in star", "Electrical"),
        ("Internal consumption voltage L1", "Electrical"),
        ("Axis 2 Voltage battery 50%", "Pitch and hub"),
        ("Wind speed anemometer 2 calculated", "Operating conditions"),
        ("Rotor rpm", "Other"),
    ],
)
def test_display_components(text, component):
    assert display_component(text) == component
