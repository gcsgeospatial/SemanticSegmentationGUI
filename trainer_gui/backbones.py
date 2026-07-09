"""Registry of the Modal training scripts the terminal can drive.

Each entry maps a backbone to its script, Modal app name, outputs volume and
the parameters its (refactored) local_entrypoint accepts. `ready=False` rows
appear in the UI but can't be launched until the script gains CLI args.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParamSpec:
    flag: str               # CLI flag name without dashes, e.g. "grid"
    label: str              # form label
    kind: str               # "float" | "int"
    default: float
    lo: float
    hi: float
    step: float = 1.0
    decimals: int = 0
    recommend_key: str = ""  # key in dataset_meta recommendations, "" = use default


@dataclass
class Backbone:
    key: str
    label: str
    script: str             # filename at repo root
    app_name: str
    ready: bool = False                # can train via --dataset + param flags
    folder_infer: bool = False         # supports `--mode infer --infer-input <job>`
    grid_kind: str = "grid"            # "grid" | "octree_depth" — drives recommendation math
    grid_clamp: tuple = (0.05, 2.0)    # clamp band for the recommended grid (m)
    grid_mult: float = 1.25            # recommended grid = grid_mult x mean point
                                       # spacing -> occupancy o ~ grid_mult^2 (>1)
    rec_gpu: str = "A100"              # rough recommended GPU for training (tune to your data)
    min_vram_gb: int = 16             # rough min VRAM (GB) for local training (tune)
    params: list = field(default_factory=list)

    @property
    def outputs_volume(self) -> str:
        return f"{self.app_name}-outputs"

    @property
    def grid_flag(self) -> str:
        """CLI flag carrying the grid/sub-grid size (differs per backbone:
        randlanet uses --sub-grid, the voxel backbones use --grid)."""
        for p in self.params:
            if p.recommend_key == "grid":
                return p.flag
        return "grid"

    @property
    def has_chunk(self) -> bool:
        """Whether the script accepts --chunk-xy (RandLA samples spheres, so no)."""
        return any(p.flag == "chunk-xy" for p in self.params)


def _common(epochs_default: int, batch_default: int, steps_default: int = 500,
            chunk: bool = True, chunk_default: float = 50.0) -> list:
    specs = [
        ParamSpec("epochs", "Epochs", "int", epochs_default, 1, 1000),
        ParamSpec("batch", "Batch size", "int", batch_default, 1, 32, recommend_key="batch"),
        ParamSpec("steps-per-epoch", "Steps / epoch", "int", steps_default, 10, 5000),
    ]
    if chunk:
        specs.append(ParamSpec("chunk-xy", "Tile size (m)", "float", chunk_default, 10.0, 200.0,
                               step=5.0, decimals=0, recommend_key="chunk_xy"))
    return specs


BACKBONES: dict[str, Backbone] = {b.key: b for b in [
    Backbone(
        key="ptv3", label="PTv3", script="scripts/modal/modal_train_ptv3.py",
        app_name="ptv3", ready=True, folder_infer=True,
        # 24 GB floor: the script trains fp32 with standard (non-flash) attention
        # (~0.2 MB/voxel retained -> ~15 GB peak at the recommended ~52-58k voxels
        # per forward), and vertical-heavy (forest) tiles collapse less under the
        # 3D voxel dedup than the mean-scene estimate assumes - 16 GB has no margin.
        rec_gpu="A100", min_vram_gb=24,
        # lo 0.15: the script's fixed 80k/15m train crop caps TRAIN density at
        # ~113 pts/m2, so finer grids train on mostly-empty cells while eval runs
        # full (the DG mismatch); at 0.15 train fill is ~92% vs eval ~100%
        grid_clamp=(0.15, 2.0), grid_mult=1.25,
        params=[ParamSpec("grid", "Grid size (m)", "float", 0.05, 0.02, 3.0,
                          step=0.05, decimals=2, recommend_key="grid")]
               + _common(250, 4),
    ),
    Backbone(
        key="randlanet", label="RandLA-Net", script="scripts/modal/modal_train_randlanet.py",
        app_name="randlanet-cold", ready=True, folder_infer=True,
        rec_gpu="A10G", min_vram_gb=8,
        grid_clamp=(0.06, 2.0), grid_mult=1.2,
        params=[ParamSpec("sub-grid", "Sub-grid size (m)", "float", 0.12, 0.02, 2.0,
                          step=0.05, decimals=2, recommend_key="grid"),
                ParamSpec("num-points", "Points / sample", "int", 45056, 4096, 131072,
                          recommend_key="num_points")]
               + _common(250, 6, chunk=False),
    ),
    Backbone(
        key="kpconvx_cold", label="KPConvX-L", script="scripts/modal/modal_train_kpconvx_cold.py",
        app_name="kpconvx-cold", ready=True, folder_infer=True,
        rec_gpu="A100-80GB", min_vram_gb=24,
        # hi 2.0 reproduces the proven g=2.0/chunk=100 recipe at 0.5 pts/m2;
        # lo 0.4 keeps the 2.5g..40g conv-radius ladder spanning real structures
        grid_clamp=(0.4, 2.0), grid_mult=1.5,
        params=[ParamSpec("grid", "Grid size (m)", "float", 2.0, 0.1, 5.0,
                          step=0.1, decimals=2, recommend_key="grid")]
               + _common(150, 4, steps_default=300, chunk_default=100.0),
    ),
    # --- HAG variants (real PDAL HeightAboveGround as an extra input channel) ---
    Backbone(
        key="ptv3_hag", label="PTv3_hag", script="scripts/modal/modal_train_ptv3_hag.py",
        app_name="ptv3-hag", ready=True, folder_infer=True,
        rec_gpu="A100", min_vram_gb=24,   # same fp32 non-flash budget as ptv3
        grid_clamp=(0.15, 2.0), grid_mult=1.25,
        params=[ParamSpec("grid", "Grid size (m)", "float", 0.05, 0.02, 3.0,
                          step=0.05, decimals=2, recommend_key="grid")]
               + _common(250, 4),
    ),
    Backbone(
        key="randlanet_hag", label="RandLA-Net_hag",
        script="scripts/modal/modal_train_randlanet_hag.py",
        app_name="randlanet-cold-hag", ready=True, folder_infer=True,
        rec_gpu="A10G", min_vram_gb=8,
        grid_clamp=(0.06, 2.0), grid_mult=1.2,
        params=[ParamSpec("sub-grid", "Sub-grid size (m)", "float", 0.12, 0.02, 2.0,
                          step=0.05, decimals=2, recommend_key="grid"),
                ParamSpec("num-points", "Points / sample", "int", 45056, 4096, 131072,
                          recommend_key="num_points")]
               + _common(250, 6, chunk=False),
    ),
    # Both KPConvX scripts now have a --dataset training path (canonical scenes;
    # the HAG variant uses the z-scene-min HAG proxy since canonical data has no
    # HAG laz) plus --mode infer --infer-input for folder inference.
    Backbone(
        key="kpconvx_cold_hag", label="KPConvX-L_hag",
        script="scripts/modal/modal_train_kpconvx_cold_hag.py",
        app_name="kpconvx-cold-hag", ready=True, folder_infer=True,
        rec_gpu="A100-80GB", min_vram_gb=24,
        grid_clamp=(0.4, 2.0), grid_mult=1.5,
        params=[ParamSpec("grid", "Grid size (m)", "float", 2.0, 0.1, 5.0,
                          step=0.1, decimals=2, recommend_key="grid")]
               + _common(150, 4, steps_default=300, chunk_default=100.0),
    ),
]}

GPU_CHOICES = ["A10G", "L4", "L40S", "A100", "A100-80GB", "H100"]


def infer_backbones() -> dict[str, Backbone]:
    """Backbones whose script supports arbitrary-folder inference (--infer-input)."""
    return {k: b for k, b in BACKBONES.items() if b.folder_infer}
