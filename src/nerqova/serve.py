"""Serve a Kev-format checkpoint through Nerqova's Apple Silicon scorer."""

import argparse
from pathlib import Path

from .checkpoint import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Kev-format checkpoint directory or Hub id[@revision]")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--packed-delta", action="store_true",
                        help="use the experimental packed Metal DeltaNet on Kev-4B")
    parser.add_argument("--exit-head", help="trained same-backbone early-exit pointer head")
    parser.add_argument("--exit-gap", type=float, help="minimum log odds of the top choice over the runner-up")
    parser.add_argument("--exit-gap-wide", type=float,
                        help="minimum log odds for choices with at least 14 options")
    parser.add_argument("--verify-head", help="earlier trained head that must agree with the exit head")
    parser.add_argument("--verify-threshold", type=float,
                        help="minimum verifier confidence margin above uniform chance")
    args = parser.parse_args()

    import uvicorn
    from kev.serve import Server, app
    from kev.suite import digest
    import mlx.core as mx

    # Keep reusable Metal buffers bounded across changing decision shapes.
    mx.set_cache_limit(1024 ** 3)

    checkpoint, tok, model = load_model(
        args.run, packed_delta=args.packed_delta,
        exit_head=args.exit_head, exit_gap=args.exit_gap, exit_gap_wide=args.exit_gap_wide,
        verify_head=args.verify_head, verify_threshold=args.verify_threshold,
    )
    app.state.server = Server(checkpoint, tok, model, "mps")
    app.state.nerqova_info = {
        "engine": "nerqova-early" if args.exit_head else "nerqova-packed" if args.packed_delta else "nerqova",
        "checkpoint_revision": Path(checkpoint.path).name,
        "exit_head_sha256": digest(args.exit_head) if args.exit_head else None,
        "exit_gap": args.exit_gap,
        "exit_gap_wide": args.exit_gap_wide,
        "verify_head_sha256": digest(args.verify_head) if args.verify_head else None,
        "verify_threshold": args.verify_threshold,
    }

    @app.get("/v1/nerqova")
    def runtime_info():
        return app.state.nerqova_info

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
