"""Serve a trained student through Kev's typed System One HTTP API."""

import argparse

from .checkpoint import load_student


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Kev-format checkpoint directory or Hub id[@revision]")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    args = parser.parse_args()

    import uvicorn
    from kev.serve import Server, app

    checkpoint, tok, model = load_student(args.run)
    app.state.server = Server(checkpoint, tok, model, "mps")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
