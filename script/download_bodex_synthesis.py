#!/usr/bin/env python3
"""Download BimanBODex synthesis output folders from the server."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_HOST = "166.111.72.150"
DEFAULT_USER = "ymr"
DEFAULT_PORT = 12742
DEFAULT_ZSTD_LEVEL = 1
DEFAULT_ZSTD_THREADS = 0
DEFAULT_REMOTE_OUTPUT_ROOT = (
    "/home/ymr/mingrui/research/project_any_scale_grasp/"
    "BimanBODex/src/curobo/content/assets/output"
)
DEFAULT_LOCAL_OUTPUT_ROOT = (
    "/home/mingrui/mingrui/research/project_any_scale_grasp/"
    "BimanBODex/src/curobo/content/assets/output"
)

SUITE_DATASETS = {
    "shadow": {
        "single": "sim_shadow",
        "dual": "sim_dual_dummy_arm_shadow",
    },
    "leap": {
        "single": "sim_leap",
        "dual": "sim_dual_dummy_arm_leap",
    },
    "leap_sp": {
        "single": "sim_leap_sp",
        "dual": "sim_dual_dummy_arm_leap_sp",
    },
}

GRASP_TYPE_TARGETS = {
    "right_two": ("single", "tabletop_two"),
    "right_three": ("single", "tabletop_three"),
    "right_full": ("single", "tabletop_full"),
    "both_three": ("dual", "tabletop_three"),
    "both_full": ("dual", "tabletop_full"),
}


def status(message: str) -> None:
    """Print a progress message.

    Args:
        message: Human-readable status text.

    Returns:
        None.
    """

    print(f"[download-bodex] {message}", file=sys.stderr, flush=True)


def command_text(command: list[str]) -> str:
    """Format a shell command for readable logging.

    Args:
        command: Command tokens passed to subprocess.

    Returns:
        Shell-escaped command string.
    """

    return " ".join(shlex.quote(part) for part in command)


def require_tool(name: str) -> None:
    """Require a local executable to be available.

    Args:
        name: Executable name to search in PATH.

    Returns:
        None. Raises SystemExit if the executable cannot be found.
    """

    if shutil.which(name) is None:
        raise SystemExit(f"{name} is required but was not found in PATH.")


def zstd_filter(level: int, threads: int) -> str:
    """Build the zstd filter command used by tar.

    Args:
        level: zstd compression level.
        threads: zstd thread count; 0 lets zstd choose automatically.

    Returns:
        A tar filter command string.
    """

    return f"zstd -{level} -T{threads}"


def ssh_target(args: argparse.Namespace) -> str:
    """Build the SSH target string.

    Args:
        args: Parsed command line arguments.

    Returns:
        SSH target in `user@host` format.
    """

    return f"{args.user}@{args.host}"


def ssh_base(args: argparse.Namespace) -> list[str]:
    """Build the common SSH command prefix.

    Args:
        args: Parsed command line arguments.

    Returns:
        SSH command token list without the remote command.
    """

    return ["ssh", "-p", str(args.port), ssh_target(args)]


def select_targets(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    """Resolve requested suite and grasp types to output folders.

    Args:
        args: Parsed command line arguments.

    Returns:
        List of `(grasp_type, dataset_folder, tabletop_folder)` tuples.
    """

    dataset_map = SUITE_DATASETS[args.suite]
    selected = args.grasp_type or list(GRASP_TYPE_TARGETS)
    targets = []
    for grasp_type in selected:
        dataset_kind, tabletop_folder = GRASP_TYPE_TARGETS[grasp_type]
        targets.append((grasp_type, dataset_map[dataset_kind], tabletop_folder))
    return targets


def run_pipeline(first: list[str], second: list[str], first_label: str, second_label: str) -> int:
    """Run a two-command streaming pipeline.

    Args:
        first: Producer command tokens.
        second: Consumer command tokens.
        first_label: Name used in progress logs for the producer.
        second_label: Name used in progress logs for the consumer.

    Returns:
        Process exit code; zero means both commands succeeded.
    """

    start_time = time.monotonic()
    status(f"Starting {first_label}: {command_text(first)}")
    first_proc = subprocess.Popen(first, stdout=subprocess.PIPE)
    assert first_proc.stdout is not None

    status(f"Starting {second_label}: {command_text(second)}")
    second_proc = subprocess.Popen(second, stdin=first_proc.stdout)
    first_proc.stdout.close()

    second_return = second_proc.wait()
    first_return = first_proc.wait()
    elapsed = time.monotonic() - start_time

    if first_return != 0:
        status(f"{first_label} failed with exit code {first_return} after {elapsed:.1f}s.")
        return first_return
    if second_return != 0:
        status(f"{second_label} failed with exit code {second_return} after {elapsed:.1f}s.")
        return second_return
    status(f"Transfer finished successfully in {elapsed:.1f}s.")
    return 0


def confirm_delete_existing(path: Path) -> bool:
    """Ask whether an existing local download folder should be deleted.

    Args:
        path: Existing local folder that would receive the downloaded data.

    Returns:
        True if the caller should delete the folder and continue; False if the
        caller should keep the folder and skip this download target.
    """

    prompt = f"Local folder already exists: {path}\nDelete it before downloading? [y/N] "
    answer = input(prompt).strip().lower()
    return answer in {"y", "yes"}


def prepare_local_target(local_target: Path, dry_run: bool) -> bool:
    """Prepare the exact local experiment folder before extraction.

    Args:
        local_target: Local experiment folder, e.g. `.../sim_shadow/tabletop_two/exp_name`.
        dry_run: Whether commands are only being printed.

    Returns:
        True if the download should continue; False if it should be skipped.
    """

    if not local_target.exists():
        return True
    if dry_run:
        status(f"Dry run: local folder already exists and would prompt before deletion: {local_target}")
        return True
    if not local_target.is_dir():
        raise SystemExit(f"Local target exists but is not a directory: {local_target}")
    if not confirm_delete_existing(local_target):
        status(f"Keeping existing local folder and skipping download: {local_target}")
        return False

    status(f"Deleting existing local folder: {local_target}")
    shutil.rmtree(local_target)
    return True


def download_one(args: argparse.Namespace, dataset_folder: str, tabletop_folder: str) -> int:
    """Download one BimanBODex synthesis output folder.

    Args:
        args: Parsed command line arguments.
        dataset_folder: BimanBODex output dataset folder, such as `sim_shadow`.
        tabletop_folder: Tabletop split folder, such as `tabletop_two`.

    Returns:
        Process exit code; zero means the folder was downloaded successfully.
    """

    remote_parent = f"{args.remote_root.rstrip('/')}/{dataset_folder}/{tabletop_folder}"
    remote_path = f"{remote_parent}/{args.exp_name}"
    local_parent = Path(args.local_root).expanduser().resolve() / dataset_folder / tabletop_folder
    local_target = local_parent / args.exp_name
    local_parent.mkdir(parents=True, exist_ok=True)

    if not prepare_local_target(local_target, args.dry_run):
        return 0

    status(f"Remote source: {ssh_target(args)}:{remote_path}")
    status(f"Local destination: {local_parent}/")

    remote_cmd = (
        f"test -e {shlex.quote(remote_path)} && "
        f"tar -I {shlex.quote(zstd_filter(args.zstd_level, args.zstd_threads))} "
        f"-cf - -C {shlex.quote(remote_parent)} {shlex.quote(args.exp_name)}"
    )
    ssh_cmd = [*ssh_base(args), remote_cmd]
    tar_cmd = ["tar", "-I", "zstd", "-xf", "-", "-C", str(local_parent)]

    if args.dry_run:
        print(command_text(ssh_cmd))
        print(command_text(tar_cmd))
        return 0

    return run_pipeline(ssh_cmd, tar_cmd, "remote archive/compress", "local extract")


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser.

    Args:
        None.

    Returns:
        Configured ArgumentParser instance.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=sorted(SUITE_DATASETS),
        required=True,
        help="Robot suite to download, e.g. shadow, leap, or leap_sp.",
    )
    parser.add_argument("--exp-name", required=True, help="BimanBODex synthesis run folder name.")
    parser.add_argument(
        "--grasp-type",
        choices=list(GRASP_TYPE_TARGETS),
        action="append",
        help="Limit download to one grasp type. Repeat to select multiple. Default: all five grasp types.",
    )
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_OUTPUT_ROOT)
    parser.add_argument("--local-root", default=DEFAULT_LOCAL_OUTPUT_ROOT)
    parser.add_argument("--host", default=os.environ.get("ASG_SERVER_HOST", DEFAULT_HOST))
    parser.add_argument("--user", default=os.environ.get("ASG_SERVER_USER", DEFAULT_USER))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ASG_SERVER_PORT", DEFAULT_PORT)))
    parser.add_argument("--zstd-level", type=int, default=DEFAULT_ZSTD_LEVEL)
    parser.add_argument("--zstd-threads", type=int, default=DEFAULT_ZSTD_THREADS)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--keep-going", action="store_true", help="Continue with later grasp types after a failure.")
    return parser


def main() -> int:
    """Run the downloader.

    Args:
        None.

    Returns:
        Process exit code.
    """

    parser = build_parser()
    args = parser.parse_args()

    if not args.dry_run:
        require_tool("tar")
        require_tool("zstd")
        require_tool("ssh")

    status(f"Suite: {args.suite}")
    status(f"Experiment: {args.exp_name}")
    failed = []
    targets = select_targets(args)
    for index, (grasp_type, dataset_folder, tabletop_folder) in enumerate(targets, start=1):
        status(f"[{index}/{len(targets)}] Downloading {grasp_type}: {dataset_folder}/{tabletop_folder}/{args.exp_name}")
        return_code = download_one(args, dataset_folder, tabletop_folder)
        if return_code != 0:
            failed.append((grasp_type, return_code))
            if not args.keep_going:
                break

    if failed:
        failure_text = ", ".join(f"{name}={code}" for name, code in failed)
        status(f"Download failed: {failure_text}")
        return failed[0][1]
    status("All requested BimanBODex synthesis folders downloaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
