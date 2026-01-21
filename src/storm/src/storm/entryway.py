from __future__ import annotations

import os
import sys
import textwrap

from storm.app import hydra_main

VALID = {"run", "init", "visualize"}
_CMD_ENV = "STORM_CMD"


def _print_top_help() -> None:
    print(
        textwrap.dedent(
            """
            Usage:
              storm <command> [hydra overrides...]

            Commands:
              run         Run pipeline
              init        Initialize project/assets
              visualize   Make plots/figures

            Examples:
              storm run seed=1 data=replogle
              storm run -m seed=1,2,3
              storm visualize checkpoint=path/to.ckpt
              storm run --cfg job
              storm run --help
            """
        ).strip()
    )


def main() -> None:
    argv = sys.argv[1:]

    if not argv or argv[0] in {"-h", "--help"}:
        _print_top_help()
        return

    if argv[0] in VALID:
        cmd = argv[0]
        hydra_args = argv[1:]
    else:
        # Allow `storm seed=1 ...` as shorthand for `storm run seed=1 ...`
        cmd = "run"
        hydra_args = argv

    os.environ[_CMD_ENV] = cmd

    # Forward all remaining args to Hydra unchanged
    sys.argv = [sys.argv[0], *hydra_args]
    hydra_main()


if __name__ == "__main__":
    main()