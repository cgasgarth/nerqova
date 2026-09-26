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


def test_transferred_head_keeps_training_origin_and_enforces_serving_pin(tmp_path):
    import pytest
    import torch
    from nerqova.early_head import PairHead
    from nerqova.early_exit import EarlyExitDecisionModel

    path = tmp_path / 'transferred.pt'
    artifact = {'head': PairHead(dimension=16, width=8).state_dict(),
                'head_type': 'pair', 'layer': 16, 'temperature': 1.0,
                'train_manifest': {'checkpoint_revision': 'source-revision'},
                'serving_manifest': {'checkpoint_revision': 'target-revision',
                                     'max_state_tokens': 384}}
    torch.save(artifact, path)
    model = EarlyExitDecisionModel.__new__(EarlyExitDecisionModel)
    model.load_exit(path, 2.0, 'target-revision')
    assert model.max_exit_state_tokens == 384
    assert torch.load(path, weights_only=True)['train_manifest']['checkpoint_revision'] == 'source-revision'
    with pytest.raises(ValueError, match='another checkpoint'):
        model.load_exit(path, 2.0, 'source-revision')
