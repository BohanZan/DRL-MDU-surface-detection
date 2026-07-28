"""
RunManager — unified output directory management.

ALL output paths flow through this module. No script hardcodes paths.

Training (creates new run):
    rm = RunManager("results", tag="4mdu")
    # → results/2026-07-26_HHMMSS_4mdu/

Post-training (finds existing run):
    run_dir = find_run("results", tag="4mdu")           # by tag
    run_dir = find_run("results")                       # latest overall
    ckpt = RunManager.checkpoint_path(run_dir, "mappo_best.pt")
    traj = RunManager.trajectory_path(run_dir, "trajectory_mappo.npz")

Convention:
    results/<run_name>/
    ├── parameters.txt
    ├── checkpoints/
    ├── trajectories/
    ├── animations/
    └── plots/
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any, List


def find_run(output_root: str = "results", tag: Optional[str] = None,
             run_name: Optional[str] = None) -> Optional[str]:
    """Find an existing run directory.

    Priority: run_name > tag > latest.

    Args:
        output_root: top-level results directory
        tag: match directories ending with this tag (e.g. "4mdu")
        run_name: exact directory name

    Returns:
        Absolute path to the run directory, or None if not found.
    """
    if not os.path.isdir(output_root):
        return None

    if run_name:
        path = os.path.join(output_root, run_name)
        return os.path.abspath(path) if os.path.isdir(path) else None

    # Search by tag or latest
    subdirs = sorted(
        [d for d in os.listdir(output_root)
         if os.path.isdir(os.path.join(output_root, d))],
        reverse=True  # newest first
    )

    if tag:
        for d in subdirs:
            if d.endswith("_" + tag) or d == tag:
                return os.path.abspath(os.path.join(output_root, d))
        return None

    # Latest
    if subdirs:
        return os.path.abspath(os.path.join(output_root, subdirs[0]))
    return None


def list_runs(output_root: str = "results") -> List[str]:
    """List all run directories, newest first."""
    if not os.path.isdir(output_root):
        return []
    subdirs = sorted(
        [d for d in os.listdir(output_root)
         if os.path.isdir(os.path.join(output_root, d))],
        reverse=True
    )
    return [os.path.abspath(os.path.join(output_root, d)) for d in subdirs]


class RunManager:
    """Creates and manages a timestamped run directory."""

    def __init__(self, output_root: str, tag: str = ""):
        """
        Args:
            output_root: Top-level output directory (e.g., "results").
            tag: Short label appended to timestamp (e.g., "4mdu", "test").
        """
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.run_name = f"{ts}_{tag}" if tag else ts
        self.run_dir = os.path.join(output_root, self.run_name)

        # Subdirectories
        for sub in ["checkpoints", "trajectories", "animations", "plots"]:
            os.makedirs(os.path.join(self.run_dir, sub), exist_ok=True)

    # -- Static path helpers (work with any run_dir, not just self) -----------

    @staticmethod
    def checkpoint_path(run_dir: str, filename: str) -> str:
        return os.path.join(run_dir, "checkpoints", filename)

    @staticmethod
    def trajectory_path(run_dir: str, filename: str) -> str:
        return os.path.join(run_dir, "trajectories", filename)

    @staticmethod
    def animation_path(run_dir: str, filename: str) -> str:
        return os.path.join(run_dir, "animations", filename)

    @staticmethod
    def plot_path(run_dir: str, filename: str) -> str:
        return os.path.join(run_dir, "plots", filename)

    @staticmethod
    def params_path(run_dir: str) -> str:
        return os.path.join(run_dir, "parameters.txt")

    # -- Instance helpers -----------------------------------------------------

    def cp_path(self, filename: str) -> str:
        return self.checkpoint_path(self.run_dir, filename)

    def traj_path(self, filename: str) -> str:
        return self.trajectory_path(self.run_dir, filename)

    def anim_path(self, filename: str) -> str:
        return self.animation_path(self.run_dir, filename)

    def plt_path(self, filename: str) -> str:
        return self.plot_path(self.run_dir, filename)

    # -- Parameter dump ------------------------------------------------------

    def dump_params(self, config, extra: Optional[Dict[str, Any]] = None) -> str:
        """Write parameters.txt to the run directory."""
        path = self.params_path(self.run_dir)
        with open(path, "w", encoding="utf-8") as f:
            f.write(config.dump())
            f.write("\n")
            f.write(f"\n[Run Info]\n")
            f.write(f"  run_name          = {self.run_name}\n")
            f.write(f"  created_at        = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if extra:
                f.write(f"\n[Extra]\n")
                for k, v in extra.items():
                    f.write(f"  {k:<20s} = {v}\n")
            f.write(f"\n{'=' * 70}\n")
        return path
