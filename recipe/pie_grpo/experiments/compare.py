"""Compare two experiment metrics.json (e.g. pie vs verl-vllm): table + plot."""
from __future__ import annotations

import argparse
import json


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def summary_row(d: dict) -> dict:
    evals = d.get("eval", [])
    hevals = [e["heval"] for e in evals]
    mbpps = [e["mbpp"] for e in evals if e.get("mbpp") is not None]
    mem = d["memory"]["peak_gpu_gb_per_rank"] or [0.0]
    return {
        "backend": d["run"]["backend"],
        "total_s": round(d["timing"]["total_s"], 1),
        "steps_per_hr": round(d["throughput"]["steps_per_hr"], 1),
        "gen_tokens_per_s": round(d["throughput"]["gen_tokens_per_s"], 1),
        "peak_gb": round(max(mem), 2),
        "final_heval": (hevals[-1] if hevals else None),
        "best_heval": (max(hevals) if hevals else None),
        "final_mbpp": (mbpps[-1] if mbpps else None),
    }


def render_table(rows: list[dict]) -> str:
    cols = ["backend", "total_s", "steps_per_hr", "gen_tokens_per_s", "peak_gb",
            "final_heval", "best_heval", "final_mbpp"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}

    def line(vals):
        return " | ".join(str(v).ljust(widths[c]) for c, v in zip(cols, vals))

    out = [line(cols), "-+-".join("-" * widths[c] for c in cols)]
    out += [line([r[c] for c in cols]) for r in rows]
    return "\n".join(out)


def _plot(runs: list[dict], out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for d in runs:
        ev = d.get("eval", [])
        ax1.plot([e["step"] for e in ev], [e["heval"] for e in ev],
                 marker="o", label=d["run"]["backend"])
    ax1.set_title("held-out HumanEval pass@1"); ax1.set_xlabel("step"); ax1.legend()
    backends = [d["run"]["backend"] for d in runs]
    ax2.bar(backends, [d["throughput"]["steps_per_hr"] for d in runs])
    ax2.set_title("steps / hour")
    fig.tight_layout(); fig.savefig(out_png, dpi=120)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", nargs="+", help="two+ metrics.json paths")
    ap.add_argument("--plot", default=None, help="output PNG (default: alongside first)")
    args = ap.parse_args(argv)
    runs = [load(p) for p in args.metrics]
    print(render_table([summary_row(d) for d in runs]))
    out_png = args.plot or (args.metrics[0].rsplit("/", 1)[0] + "/compare.png")
    try:
        _plot(runs, out_png)
        print(f"\nplot: {out_png}")
    except ImportError:
        print("\n(matplotlib not installed — table only; `pip install matplotlib` for the plot)")


if __name__ == "__main__":
    main()
