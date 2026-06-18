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
    warm: bool
    ready: bool = False                # can train via --dataset + param flags
    folder_infer: bool = False         # supports `--mode infer --infer-input <job>`
    grid_kind: str = "grid"            # "grid" | "octree_depth" — drives recommendation math
    grid_clamp: tuple = (0.05, 1.0)    # clamp band for the recommended grid (m)
    grid_mult: float = 3.0             # recommended grid = grid_mult x mean point spacing
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
    specs.append(ParamSpec("predict-n", "Scenes to predict after training", "int", 1, 0, 50))
    return specs


BACKBONES: dict[str, Backbone] = {b.key: b for b in [
    Backbone(
        key="ptv3_warm", label="PTv3 (warm — Sonata)", script="modal_train_ptv3_warm.py",
        app_name="ptv3-warm-ieee", warm=True, ready=True,
        grid_clamp=(0.05, 0.6),
        params=[ParamSpec("grid", "Grid size (m)", "float", 0.30, 0.02, 3.0,
                          step=0.05, decimals=2, recommend_key="grid")]
               + _common(100, 4),
    ),
    Backbone(
        key="randlanet_warm", label="RandLA-Net (warm)", script="modal_train_randlanet_warm.py",
        app_name="randlanet-warm-ieee", warm=True, ready=True,
        grid_clamp=(0.06, 0.5),
        params=[ParamSpec("sub-grid", "Sub-grid size (m)", "float", 0.30, 0.02, 2.0,
                          step=0.05, decimals=2, recommend_key="grid"),
                ParamSpec("num-points", "Points / sample", "int", 45056, 4096, 131072)]
               + _common(100, 6, chunk=False),   # RandLA samples spheres, no tiling
    ),
    Backbone(
        key="octformer_warm", label="OctFormer (warm)", script="modal_train_octformer_warm.py",
        app_name="octformer-warm-ieee", warm=True, ready=True,
        grid_kind="octree_depth", grid_clamp=(9, 12),
        params=[ParamSpec("octree-depth", "Octree depth", "int", 11, 8, 13,
                          recommend_key="octree_depth")]
               + _common(100, 4, steps_default=300),
    ),
    Backbone(
        key="kpconvx_warm", label="KPConvX-L (warm)", script="modal_train_kpconvx_warm.py",
        app_name="kpconvx-warm-ieee", warm=True, ready=True,
        # in_sub_size wants ~1.1x the native point spacing (not 3x like voxel
        # backbones) — the conv radius is 2.1 grid units on top of it.
        grid_clamp=(0.05, 1.5), grid_mult=1.1,
        params=[ParamSpec("grid", "Base sub-grid / in_sub_size (m)", "float", 0.8, 0.02, 3.0,
                          step=0.05, decimals=2, recommend_key="grid"),
                ParamSpec("epochs", "Epochs", "int", 100, 1, 1000),
                ParamSpec("batch", "Grad-accum batch", "int", 2, 1, 16, recommend_key="batch"),
                ParamSpec("steps-per-epoch", "Optimizer steps / epoch", "int", 300, 10, 5000),
                ParamSpec("chunk-xy", "Tile size (m)", "float", 30.0, 10.0, 200.0,
                          step=5.0, decimals=0, recommend_key="chunk_xy"),
                ParamSpec("predict-n", "Scenes to predict after training", "int", 1, 0, 50)],
    ),
    Backbone(
        key="ptv3", label="PTv3 (cold)", script="modal_train_ptv3.py",
        app_name="ptv3-ieee", warm=False, ready=True, folder_infer=True,
        grid_clamp=(0.05, 0.6),
        params=[ParamSpec("grid", "Grid size (m)", "float", 0.05, 0.02, 3.0,
                          step=0.05, decimals=2, recommend_key="grid")]
               + _common(250, 4),
    ),
    Backbone(
        key="randlanet", label="RandLA-Net (cold)", script="modal_train_randlanet.py",
        app_name="randlanet-cold-ieee", warm=False, ready=True, folder_infer=True,
        grid_clamp=(0.06, 0.5),
        params=[ParamSpec("sub-grid", "Sub-grid size (m)", "float", 0.12, 0.02, 2.0,
                          step=0.05, decimals=2, recommend_key="grid"),
                ParamSpec("num-points", "Points / sample", "int", 45056, 4096, 131072)]
               + _common(250, 6, chunk=False),
    ),
    Backbone(
        key="octformer", label="OctFormer (cold)", script="modal_train_octformer.py",
        app_name="octformer-stpls3d", warm=False, ready=True,
        grid_kind="octree_depth", grid_clamp=(9, 12),
        params=[ParamSpec("octree-depth", "Octree depth", "int", 11, 8, 13,
                          recommend_key="octree_depth")]
               + _common(250, 4, steps_default=300),
    ),
    Backbone(
        key="kpconvx_cold", label="KPConvX-L (cold)", script="modal_train_kpconvx_cold.py",
        app_name="kpconvx-cold-ieee", warm=False, folder_infer=True,
        grid_clamp=(0.5, 3.0),
        params=[ParamSpec("grid", "Grid size (m)", "float", 2.0, 0.1, 5.0,
                          step=0.1, decimals=2, recommend_key="grid")]
               + _common(150, 4, steps_default=300, chunk_default=100.0),
    ),
    # --- HAG variants (real PDAL HeightAboveGround as an extra input channel) ---
    Backbone(
        key="ptv3_hag", label="PTv3 (cold + HAG)", script="modal_train_ptv3_hag.py",
        app_name="ptv3-ieee-hag", warm=False, ready=True, folder_infer=True,
        grid_clamp=(0.05, 0.6),
        params=[ParamSpec("grid", "Grid size (m)", "float", 0.05, 0.02, 3.0,
                          step=0.05, decimals=2, recommend_key="grid")]
               + _common(250, 4),
    ),
    Backbone(
        key="randlanet_hag", label="RandLA-Net (cold + HAG)",
        script="modal_train_randlanet_hag.py",
        app_name="randlanet-cold-ieee-hag", warm=False, ready=True, folder_infer=True,
        grid_clamp=(0.06, 0.5),
        params=[ParamSpec("sub-grid", "Sub-grid size (m)", "float", 0.12, 0.02, 2.0,
                          step=0.05, decimals=2, recommend_key="grid"),
                ParamSpec("num-points", "Points / sample", "int", 45056, 4096, 131072)]
               + _common(250, 6, chunk=False),
    ),
    # KPConvX (cold) stays ready=False (its train_kpconvx has no --dataset path),
    # but both KPConvX scripts now accept --mode infer --infer-input, so they're
    # folder-inferable. Folder inference uses the z-tile-min HAG proxy.
    Backbone(
        key="kpconvx_cold_hag", label="KPConvX-L (cold + HAG)",
        script="modal_train_kpconvx_cold_hag.py",
        app_name="kpconvx-cold-ieee-hag", warm=False, folder_infer=True,
        grid_clamp=(0.5, 3.0),
        params=[ParamSpec("grid", "Grid size (m)", "float", 2.0, 0.1, 5.0,
                          step=0.1, decimals=2, recommend_key="grid")]
               + _common(150, 4, steps_default=300, chunk_default=100.0),
    ),
]}

GPU_CHOICES = ["A10G", "L4", "L40S", "A100", "A100-80GB", "H100"]


def ready_backbones() -> dict[str, Backbone]:
    return {k: b for k, b in BACKBONES.items() if b.ready}


def infer_backbones() -> dict[str, Backbone]:
    """Backbones whose script supports arbitrary-folder inference (--infer-input)."""
    return {k: b for k, b in BACKBONES.items() if b.folder_infer}
