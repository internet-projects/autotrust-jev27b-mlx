"""python -m jev27b_mlx serve <model> [--port 8000]
   python -m jev27b_mlx decide <model> --kind choice --state "..." --question "..." --options a b c"""
import argparse, json


def main():
    ap = argparse.ArgumentParser(prog="python -m jev27b_mlx")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="HTTP server with POST /v1/decide")
    s.add_argument("model"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000)
    s.add_argument("--memory-limit-gb", type=float, default=None)
    d = sub.add_parser("decide", help="one decision from the command line")
    d.add_argument("model"); d.add_argument("--kind", default="choice", choices=["choice", "noul"])
    d.add_argument("--state", required=True); d.add_argument("--question", required=True); d.add_argument("--options", nargs="*")
    a = ap.parse_args()
    if a.cmd == "serve":
        from .server import serve
        serve(a.model, a.host, a.port, a.memory_limit_gb)
    else:
        from .engine import Jev27
        j = Jev27(a.model)
        print(json.dumps(j.decide(a.state, Jev27.question(a.kind, a.question, a.options)), indent=1))


if __name__ == "__main__":
    main()
