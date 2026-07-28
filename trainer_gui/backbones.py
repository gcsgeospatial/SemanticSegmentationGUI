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

# modal ALS practice: 0.25 m is the modal published grid (DALES, FRACTAL,
# Vaihingen, H3D); 50 m the modal patch. Both are starting points, not optima.
ALS_GRID_M = 0.25
ALS_TILE_M = 50.0
# KPConv family only: 100 x grid, i.e. the paper's in_radius = 50 x dl as a side.
# Wider wastes memory past the deepest layer's reach and makes the point cap thin.
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
    min_vram_gb: int = 16          # VRAM the default batch needs; scales batch_for_vram
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


# hover help for the per-model parameter rows, keyed by ParamSpec.flag
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
        # 24 GB floor: fp32 non-flash attention leaves 16 GB no margin
        rec_gpu="A100", min_vram_gb=24,
        params=[ParamSpec("grid", "Grid size (m)", "float", ALS_GRID_M, 0.02, 3.0,
                          step=0.05, decimals=2)]
               + _common(250, 4),
    ),
    Backbone(
        key="randlanet", label="RandLA-Net", script="scripts/modal/modal_train_randlanet.py",
        app_name="randlanet-cold",
        rec_gpu="A10G", min_vram_gb=8,
        params=[ParamSpec("sub-grid", "Sub-grid size (m)", "float", ALS_GRID_M, 0.02, 2.0,
                          step=0.05, decimals=2),
                ParamSpec("num-points", "Points / sample", "int", 45056, 4096, 131072)]
               + _common(250, 6, chunk=False),
    ),
    Backbone(
        key="kpconvx_cold", label="KPConvX-L", script="scripts/modal/modal_train_kpconvx_cold.py",
        app_name="kpconvx-cold",
        rec_gpu="A100-80GB", min_vram_gb=24,
        params=[ParamSpec("grid", "Grid size (m)", "float", ALS_GRID_M, 0.1, 5.0,
                          step=0.05, decimals=2)]
               + _common(150, 4, steps_default=300, tile_m=KP_TILE_M),
    ),
    # original KPConv: deformable blocks are heavier, hence min_vram 40 / batch 3
    Backbone(
        key="kpconv", label="KPConv", script="scripts/modal/modal_train_kpconv.py",
        app_name="kpconv",
        rec_gpu="A100-80GB", min_vram_gb=40,
        params=[ParamSpec("grid", "Grid size (m)", "float", ALS_GRID_M, 0.1, 5.0,
                          step=0.05, decimals=2)]
               + _common(150, 3, steps_default=300, tile_m=KP_TILE_M),
    ),
    # Pointcept-SSL encoders (Concerto/Sonata/Utonia): one shared trainer, CC-BY-NC weights; "freeze encoder" = upstream linear-probe protocol
    Backbone(
        key="concerto", label="Concerto", script="scripts/modal/modal_train_concerto.py",
        app_name="concerto",
        rec_gpu="A100", min_vram_gb=24,
        params=[ParamSpec("grid", "Grid size (m)", "float", ALS_GRID_M, 0.02, 3.0,
                          step=0.05, decimals=2),
                ParamSpec("freeze-encoder", "Freeze encoder (0/1)", "int", 0, 0, 1)]
               + _common(100, 4),
    ),
    Backbone(
        key="sonata", label="Sonata", script="scripts/modal/modal_train_sonata.py",
        app_name="sonata",
        rec_gpu="A100", min_vram_gb=24,
        params=[ParamSpec("grid", "Grid size (m)", "float", ALS_GRID_M, 0.02, 3.0,
                          step=0.05, decimals=2),
                ParamSpec("freeze-encoder", "Freeze encoder (0/1)", "int", 0, 0, 1)]
               + _common(100, 4),
    ),
    Backbone(
        key="utonia", label="Utonia", script="scripts/modal/modal_train_utonia.py",
        app_name="utonia",
        rec_gpu="A100", min_vram_gb=24,
        params=[ParamSpec("grid", "Grid size (m)", "float", ALS_GRID_M, 0.02, 3.0,
                          step=0.05, decimals=2),
                ParamSpec("freeze-encoder", "Freeze encoder (0/1)", "int", 0, 0, 1)]
               + _common(100, 4),
    ),
]}

GPU_CHOICES = ["A10G", "L4", "L40S", "A100", "A100-80GB", "H100"]

GPU_VRAM_GB = {"A10G": 24, "L4": 24, "L40S": 48, "A100": 40,
               "A100-80GB": 80, "H100": 80}

# weights + optimizer + CUDA context; batch-independent, so it comes off the top
VRAM_RESERVE_GB = 2.0


def batch_for_vram(b: Backbone, vram_gb: float) -> int:
    """Batch that fits `vram_gb`, scaled linearly from the backbone's known-good
    (batch_default @ min_vram_gb) pair - VRAM tracks total points, and points
    scale with batch. Capped at 2x default: past that the tuned LR stops holding
    (AdamW wants sqrt-scaling) and batch stops being a free hardware knob."""
    head = max(float(vram_gb) - VRAM_RESERVE_GB, 0.0)
    ref = max(float(b.min_vram_gb) - VRAM_RESERVE_GB, 1.0)
    return max(1, min(int(b.batch_default * head / ref), 2 * b.batch_default))
