import argparse
from contextlib import ExitStack
from pathlib import Path
import runpy
import sys


ENTRYPOINTS = {
    "spda": "src.medsam2_spda_npz_2D_3D",
    "eval": "src.medsam2_eval_npz_2D_3D",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run experiments.",
        allow_abbrev=False,
    )
    parser.add_argument("command", choices=ENTRYPOINTS)
    parser.add_argument(
        "arguments", nargs=argparse.REMAINDER, help="Arguments for the selected program"
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent
    source_paths = [str(project_root), str(project_root / "src")]
    sys.path[:] = source_paths + [path for path in sys.path if path not in source_paths]

    original_argv = sys.argv
    with ExitStack() as cleanup:
        cleanup.callback(setattr, sys, "argv", original_argv)
        sys.argv = [str(project_root / "main.py"), *args.arguments]
        runpy.run_module(ENTRYPOINTS[args.command], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
