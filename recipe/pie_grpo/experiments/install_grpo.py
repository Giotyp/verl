"""Register the vendored grpo inferlet on a running Pie server (as grpo@0.1.0).

Run this AFTER `pie serve` is up and BEFORE the training run — train_grpo's
PieRolloutWorker references the program by name ("grpo@0.1.0"), so it must be
installed first. Uses the prebuilt wasm + Pie.toml manifest vendored under
inferlets/grpo/ (no wasm build toolchain needed on the pod).
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib

HERE = pathlib.Path(__file__).parent
WASM = HERE / "inferlets" / "grpo" / "grpo.wasm"
MANIFEST = HERE / "inferlets" / "grpo" / "Pie.toml"


async def _main(uri: str, username: str):
    from pie_client import PieClient
    async with PieClient(uri) as c:
        await c.authenticate(username)
        await c.install_program(str(WASM), str(MANIFEST), force_overwrite=True)
    print(f"installed grpo inferlet ({WASM.name}) on {uri}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ws://127.0.0.1:8080")
    ap.add_argument("--username", default="rl-trainer")
    a = ap.parse_args()
    asyncio.run(_main(a.uri, a.username))


if __name__ == "__main__":
    main()
