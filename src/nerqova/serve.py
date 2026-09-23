"""Serve a Kev-format checkpoint through Nerqova's Apple Silicon scorer."""

import argparse

from .checkpoint import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Kev-format checkpoint directory or Hub id[@revision]")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--packed-delta", action="store_true",
                        help="use the experimental packed Metal DeltaNet on Kev-4B")
    args = parser.parse_args()

    import uvicorn
    from kev.serve import Server, app

    checkpoint, tok, model = load_model(args.run, packed_delta=args.packed_delta)
    app.state.server = Server(checkpoint, tok, model, "mps")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
