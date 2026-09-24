"""The exit gate compares the winner with the runner-up across choice counts."""

import math

from scripts.select_exit_gate import top_two_gap
from nerqova.early_exit import required_exit_gap


def test_clear_winner_with_fifteen_options_can_exit_below_high_absolute_confidence():
    probabilities = [0.6] + [0.4 / 14] * 14
    assert max(probabilities) < 0.9
    assert top_two_gap(probabilities) > 2.5
    assert math.isclose(top_two_gap([0.5, 0.5]), 0.0)
    assert required_exit_gap(15, 5.0, 2.5) == 2.5
    assert required_exit_gap(3, 5.0, 2.5) == 5.0
