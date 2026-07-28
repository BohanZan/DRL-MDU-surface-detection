"""
Single source of truth for ALL project configuration.

Every script imports defaults from here. CLI arguments layer on top via
Config.from_args(). No other file should hardcode a parameter value.

Usage:
    # Default config
    cfg = Config()

    # Resolve paths relative to project root
    cfg = cfg.resolve_paths()

    # Get env kwargs
    env = MDUCoverageEnv(**cfg.env_kwargs())

    # CLI override
    cfg = Config.from_args(args)

    # Dump to parameter file
    print(cfg.dump())
"""

import os
import sys
from dataclasses import dataclass, field, fields, asdict
from typing import Tuple, Optional, Dict, Any, ClassVar


# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

def _get_project_root() -> str:
    """Return absolute path to project root (parent of src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# DataPaths — file locations
# ---------------------------------------------------------------------------

@dataclass
class DataPaths:
    """Paths to input data files, relative to project root.

    Call .resolve() to turn them into absolute paths.
    """
    fns: str = "FNS_square_fold-50m.txt"
    solution: str = "Solution.dat"
    polyhedron: str = "polyhedron_bennu.txt"

    def resolve(self, root: Optional[str] = None) -> "DataPaths":
        """Return a new DataPaths with all paths resolved to absolute."""
        root = root or _get_project_root()
        return DataPaths(
            fns=os.path.join(root, self.fns),
            solution=os.path.join(root, self.solution),
            polyhedron=os.path.join(root, self.polyhedron),
        )


# ---------------------------------------------------------------------------
# EnvConfig — MDUCoverageEnv parameters
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    """Environment parameters — passed to MDUCoverageEnv.__init__()."""
    cone_angle_deg: float = 80.0
    cone_range: float = 300.0
    max_steps: int = 200
    coverage_threshold: float = 0.95
    completion_threshold: float = 0.75
    mdu_start_nodes: Tuple[int, ...] = (288, 296, 360, 368)
    num_mdus: int = 4
    r_newly: float = 20.0
    r_completion: float = 10.0
    r_speed: float = 5.0


# ---------------------------------------------------------------------------
# AgentConfig — MAPPO / PPO hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Agent parameters — maps to MAPPOConfig in src/agents/mappo.py."""
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    ent_coef_start: float = 0.05
    ent_coef_end: float = 0.01
    max_grad_norm: float = 0.5
    num_epochs: int = 4
    hidden_dim: int = 64
    batch_size: int = 256
    bptt_len: int = 32


# ---------------------------------------------------------------------------
# TrainConfig — training loop parameters
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training loop parameters."""
    episodes: int = 3000
    seed: int = 42
    device: str = "auto"
    log_every: int = 50


# ---------------------------------------------------------------------------
# Config — top-level composition
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Master configuration composing all sub-configs."""
    paths: DataPaths = field(default_factory=DataPaths)
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # -- Resolve paths -------------------------------------------------------

    def resolve_paths(self, root: Optional[str] = None) -> "Config":
        """Return a new Config with DataPaths resolved to absolute."""
        return Config(
            paths=self.paths.resolve(root),
            env=self.env,
            agent=self.agent,
            train=self.train,
        )

    # -- Convert to kwargs ---------------------------------------------------

    def env_kwargs(self, root: Optional[str] = None) -> Dict[str, Any]:
        """Return kwargs dict for MDUCoverageEnv.__init__().

        Paths are resolved relative to root; MDU start nodes are sliced to
        num_mdus.
        """
        p = self.paths.resolve(root)
        nodes = list(self.env.mdu_start_nodes[:self.env.num_mdus])
        return dict(
            fns_path=p.fns,
            solution_path=p.solution,
            polyhedron_path=p.polyhedron,
            mdu_start_nodes=nodes,
            cone_angle_deg=self.env.cone_angle_deg,
            cone_range=self.env.cone_range,
            max_steps=self.env.max_steps,
            coverage_threshold=self.env.coverage_threshold,
            completion_threshold=self.env.completion_threshold,
            r_newly=self.env.r_newly,
            r_completion=self.env.r_completion,
            r_speed=self.env.r_speed,
            seed=self.train.seed,
        )

    # -- Convert to MAPPOConfig ----------------------------------------------

    def to_mappo_config(self):
        """Build a MAPPOConfig from AgentConfig fields.

        Imported lazily to avoid circular dependency at module level.
        """
        from src.agents.mappo import MAPPOConfig
        return MAPPOConfig(
            lr_actor=self.agent.lr_actor,
            lr_critic=self.agent.lr_critic,
            gamma=self.agent.gamma,
            gae_lambda=self.agent.gae_lambda,
            clip_epsilon=self.agent.clip_epsilon,
            ent_coef_start=self.agent.ent_coef_start,
            ent_coef_end=self.agent.ent_coef_end,
            max_grad_norm=self.agent.max_grad_norm,
            num_epochs=self.agent.num_epochs,
            hidden_dim=self.agent.hidden_dim,
            batch_size=self.agent.batch_size,
            bptt_len=self.agent.bptt_len,
        )

    # -- CLI override --------------------------------------------------------

    # Mapping from argparse attribute name → (sub_config_name, field_name)
    _ARG_MAP: ClassVar[Dict[str, Tuple[str, str]]] = {
        # EnvConfig
        "cone_angle":       ("env", "cone_angle_deg"),
        "cone_range":       ("env", "cone_range"),
        "max_steps":        ("env", "max_steps"),
        "coverage_threshold":  ("env", "coverage_threshold"),
        "completion_threshold": ("env", "completion_threshold"),
        "mdus":             ("env", "num_mdus"),
        "r_newly":          ("env", "r_newly"),
        "r_completion":     ("env", "r_completion"),
        "r_speed":          ("env", "r_speed"),
        # AgentConfig
        "lr_actor":         ("agent", "lr_actor"),
        "lr_critic":        ("agent", "lr_critic"),
        "gamma":            ("agent", "gamma"),
        "epochs":           ("agent", "num_epochs"),
        "hidden":           ("agent", "hidden_dim"),
        "ent_coef_end":     ("agent", "ent_coef_end"),
        "bptt_len":         ("agent", "bptt_len"),
        # TrainConfig
        "episodes":         ("train", "episodes"),
        "seed":             ("train", "seed"),
        "device":           ("train", "device"),
        "log_every":        ("train", "log_every"),
    }

    @classmethod
    def from_args(cls, args, root: Optional[str] = None) -> "Config":
        """Build a Config from defaults, overridden by CLI args.

        Each argparse attribute that is non-None and exists in _ARG_MAP is
        applied on top of the default Config.
        """
        cfg = cls()

        for arg_name, (sub_name, field_name) in cls._ARG_MAP.items():
            if hasattr(args, arg_name):
                val = getattr(args, arg_name)
                if val is not None:
                    sub_cfg = getattr(cfg, sub_name)
                    setattr(sub_cfg, field_name, val)

        # Resolve paths if root provided
        if root:
            cfg = cfg.resolve_paths(root)
        else:
            cfg = cfg.resolve_paths()

        return cfg

    # -- Dump ----------------------------------------------------------------

    def dump(self) -> str:
        """Return a formatted multi-line string of all config values."""
        lines = []
        sep = "=" * 70
        lines.append(sep)
        lines.append("Run Configuration")
        lines.append(sep)

        # DataPaths
        lines.append("\n[Data Paths]")
        lines.append(f"  fns              = {self.paths.fns}")
        lines.append(f"  solution         = {self.paths.solution}")
        lines.append(f"  polyhedron       = {self.paths.polyhedron}")

        # EnvConfig
        lines.append("\n[Environment]")
        for f in fields(EnvConfig):
            val = getattr(self.env, f.name)
            lines.append(f"  {f.name:<20s} = {val}")

        # AgentConfig
        lines.append("\n[Agent / MAPPO]")
        for f in fields(AgentConfig):
            val = getattr(self.agent, f.name)
            lines.append(f"  {f.name:<20s} = {val}")

        # TrainConfig
        lines.append("\n[Training]")
        for f in fields(TrainConfig):
            val = getattr(self.train, f.name)
            lines.append(f"  {f.name:<20s} = {val}")

        lines.append(f"\n{sep}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.dump()
