"""Registry of the Modal training scripts the terminal can drive.

Each entry maps a backbone to its script, Modal app name, outputs volume and
the parameters its (refactored) local_entrypoint accepts.

Grid/tile defaults are fixed at the published ALS operating point (0.25 m grid,
50 m tile) rather than derived from dataset density - the ALS literature picks
grid by target-class size, not point spacing. Batch is the one value scaled to
hardware, because VRAM is the only thing it actually controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ALS_GRID_M = 0.25
ALS_TILE_M = 50.0
KP_TILE_M = 25.0


@dataclass
class ParamSpec:
    flag: str
    label: str
    kind: str
    default: float
    lo: float
    hi: float
    step: float = 1.0
    decimals: int = 0


@dataclass
class Backbone:
    key: str
    label: str
    script: str
    app_name: str
    rec_gpu: str = "A100"
    params: list = field(default_factory=list)

    @property
    def outputs_volume(self) -> str:
        from . import appstate
        return appstate.modal_outputs_volume() or f"{self.app_name}-outputs"

    @property
    def grid_flag(self) -> str:
        """CLI flag carrying the grid size (--sub-grid for randlanet, else --grid)."""
        return "sub-grid" if any(p.flag == "sub-grid" for p in self.params) else "grid"

    @property
    def has_chunk(self) -> bool:
        """Whether the script accepts --chunk-xy (RandLA samples spheres, so no)."""
        return any(p.flag == "chunk-xy" for p in self.params)

    @property
    def batch_default(self) -> int:
        for p in self.params:
            if p.flag == "batch":
                return int(p.default)
        return 1


def _common(epochs_default: int, batch_default: int, steps_default: int = 500,
            chunk: bool = True, tile_m: float = ALS_TILE_M) -> list:
    specs = [
        ParamSpec("epochs", "Epochs", "int", epochs_default, 1, 1000),
        ParamSpec("batch", "Batch size", "int", batch_default, 1, 32),
        ParamSpec("steps-per-epoch", "Steps / epoch", "int", steps_default, 10, 5000),
    ]
    if chunk:
        specs.append(ParamSpec("chunk-xy", "Tile size (m)", "float", tile_m,
                               10.0, 200.0, step=5.0, decimals=0))
    return specs


# keys must match ParamSpec.flag
PARAM_TIPS = {
    "grid": "Voxel size (m) the cloud is thinned to. Smaller keeps more detail but costs memory.",
    "sub-grid": "Voxel size (m) the cloud is thinned to. Smaller keeps more detail but costs memory.",
    "num-points": "Points fed to the model per sample.",
    "epochs": "Full passes over the training set.",
    "batch": "Samples per training step. Higher needs more GPU memory.",
    "steps-per-epoch": "Batches trained per epoch.",
    "chunk-xy": "Square tile size (m) the cloud is cut into for training.",
    "freeze-encoder": "Train only the classifier head (linear probe). Cheaper, less accurate.",
}

BACKBONES: dict[str, Backbone] = {b.key: b for b in [
    Backbone(
        key="ptv3", label="PTv3", script="scripts/modal/modal_train_ptv3.py",
        app_name="ptv3",
        rec_gpu="A100",
        params=[ParamSpec("grid", "Grid size (m)", "float", ALS_GRID_M, 0.02, 3.0,
                          step=0.05, decimals=2)]
               + _common(250, 4),
    ),
    Backbone(
        key="randlanet", label="RandLA-Net", script="scripts/modal/modal_train_randlanet.py",
        app_name="randlanet-cold",
        rec_gpu="A10G",
        params=[ParamSpec("sub-grid", "Sub-grid size (m)", "float", ALS_GRID_M, 0.02, 2.0,
                          step=0.05, decimals=2),
                ParamSpec("num-points", "Points / sample", "int", 45056, 4096, 131072)]
               + _common(250, 6, chunk=False),
    ),
    Backbone(
        key="kpconvx_cold", label="KPConvX-L", script="scripts/modal/modal_train_kpconvx_cold.py",
        app_name="kpconvx-cold",
        rec_gpu="A100-80GB",
        params=[ParamSpec("grid", "Grid size (m)", "float", ALS_GRID_M, 0.1, 5.0,
                          step=0.05, decimals=2)]
               + _common(150, 4, steps_default=300, tile_m=KP_TILE_M),
    ),
    Backbone(
        key="kpconv", label="KPConv", script="scripts/modal/modal_train_kpconv.py",
        app_name="kpconv",
        rec_gpu="A100-80GB",
        params=[ParamSpec("grid", "Grid size (m)", "float", ALS_GRID_M, 0.1, 5.0,
                          step=0.05, decimals=2)]
               + _common(150, 3, steps_default=300, tile_m=KP_TILE_M),
    ),
    # Concerto/Sonata/Utonia upstream weights are CC-BY-NC
    Backbone(
        key="concerto", label="Concerto", script="scripts/modal/modal_train_concerto.py",
        app_name="concerto",
        rec_gpu="A100",
        params=[ParamSpec("grid", "Grid size (m)", "float", ALS_GRID_M, 0.02, 3.0,
                          step=0.05, decimals=2),
                ParamSpec("freeze-encoder", "Freeze encoder (0/1)", "int", 0, 0, 1)]
               + _common(100, 4),
    ),
    Backbone(
        key="sonata", label="Sonata", script="scripts/modal/modal_train_sonata.py",
        app_name="sonata",
        rec_gpu="A100",
        params=[ParamSpec("grid", "Grid size (m)", "float", ALS_GRID_M, 0.02, 3.0,
                          step=0.05, decimals=2),
                ParamSpec("freeze-encoder", "Freeze encoder (0/1)", "int", 0, 0, 1)]
               + _common(100, 4),
    ),
    Backbone(
        key="utonia", label="Utonia", script="scripts/modal/modal_train_utonia.py",
        app_name="utonia",
        rec_gpu="A100",
        params=[ParamSpec("grid", "Grid size (m)", "float", ALS_GRID_M, 0.02, 3.0,
                          step=0.05, decimals=2),
                ParamSpec("freeze-encoder", "Freeze encoder (0/1)", "int", 0, 0, 1)]
               + _common(100, 4),
    ),
]}

GPU_CHOICES = ["A10G", "L4", "L40S", "A100", "A100-80GB", "H100"]

