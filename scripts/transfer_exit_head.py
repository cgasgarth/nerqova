"""Package an unchanged readout for evaluation against a new checkpoint.

Training provenance remains intact. This command does not establish quality;
the new serving target requires calibration, development, and locked checks.
"""

import argparse
from pathlib import Path

import torch
from kev.checkpoint import Checkpoint
from kev.suite import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--max-state-tokens', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.max_state_tokens < 1 or args.out.exists():
        parser.error('context must be positive and output must not exist')
    artifact = torch.load(args.source, map_location='cpu', weights_only=True)
    checkpoint = Checkpoint(args.run)
    training = artifact['train_manifest']
    if artifact['head_type'] != 'pair' or training['base_revision'] != checkpoint.meta.base_revision:
        parser.error('transfer requires the same base revision and a pair readout')
    artifact['serving_manifest'] = {
        'checkpoint_revision': Path(checkpoint.path).name,
        'base_revision': checkpoint.meta.base_revision,
        'max_state_tokens': args.max_state_tokens,
        'method': 'unchanged_readout_transfer',
        'source_sha256': digest(args.source),
        'source_checkpoint_revision': training['checkpoint_revision'],
    }
    for key in ('calibration', 'calibration_manifest'):
        if key in artifact:
            artifact[f'source_{key}'] = artifact.pop(key)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.out)
    print(f'{args.out}: {digest(args.out)}')


if __name__ == '__main__':
    main()
