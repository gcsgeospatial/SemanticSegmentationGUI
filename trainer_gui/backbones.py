"""Registry of the Modal training scripts the terminal can drive.

Each entry maps a backbone to its script, Modal app name, outputs volume and
the parameters its (refactored) local_entrypoint accepts.
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
    grid_kind: str = "grid"            # "grid" | "octree_depth" — drives recommendation math
    grid_clamp: tuple = (0.05, 2.0)    # clamp band for the recommended grid (m)
    grid_mult: float = 1.25            # recommended grid = grid_mult x mean spacing
    rec_gpu: str = "A100"
    min_vram_gb: int = 16
    params: list = field(default_factory=list)

    @property
    def outputs_volume(self) -> str:
        return f"{self.app_name}-outputs"

    @property
    def grid_flag(self) -> str:
        """CLI flag carrying the grid size (--sub-grid for randlanet, else --grid)."""
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
        app_name="ptv3",
        # 24 GB floor: fp32 non-flash attention leaves 16 GB no margin
        rec_gpu="A100", min_vram_gb=24,
        # lo 0.15: finer grids hit the 80k/15m train-crop DG mismatch
        grid_clamp=(0.15, 2.0), grid_mult=1.25,
        params=[ParamSpec("grid", "Grid size (m)", "float", 0.5, 0.02, 3.0,
                          step=0.05, decimals=2, recommend_key="grid")]
               + _common(250, 4),
    ),
    Backbone(
        key="randlanet", label="RandLA-Net", script="scripts/modal/modal_train_randlanet.py",
        app_name="randlanet-cold",
        rec_gpu="A10G", min_vram_gb=8,
        grid_clamp=(0.06, 2.0), grid_mult=1.2,
        params=[ParamSpec("sub-grid", "Sub-grid size (m)", "float", 0.30, 0.02, 2.0,
                          step=0.05, decimals=2, recommend_key="grid"),
                ParamSpec("num-points", "Points / sample", "int", 45056, 4096, 131072,
                          recommend_key="num_points")]
               + _common(250, 6, chunk=False),
    ),
    Backbone(
        key="kpconvx_cold", label="KPConvX-L", script="scripts/modal/modal_train_kpconvx_cold.py",
        app_name="kpconvx-cold",
        rec_gpu="A100-80GB", min_vram_gb=24,
        # lo 0.4 keeps the conv-radius ladder spanning real structures
        grid_clamp=(0.4, 2.0), grid_mult=1.5,
        params=[ParamSpec("grid", "Grid size (m)", "float", 2.0, 0.1, 5.0,
                          step=0.1, decimals=2, recommend_key="grid")]
               + _common(150, 4, steps_default=300, chunk_default=100.0),
    ),
    # original KPConv: deformable blocks are heavier, hence min_vram 40 / batch 3
    Backbone(
        key="kpconv", label="KPConv", script="scripts/modal/modal_train_kpconv.py",
        app_name="kpconv",
        rec_gpu="A100-80GB", min_vram_gb=40,
        grid_clamp=(0.4, 2.0), grid_mult=1.5,
        params=[ParamSpec("grid", "Grid size (m)", "float", 2.0, 0.1, 5.0,
                          step=0.1, decimals=2, recommend_key="grid")]
               + _common(150, 3, steps_default=300, chunk_default=100.0),
    ),
    # Pointcept-SSL encoders (Concerto/Sonata/Utonia): one shared trainer,
    # CC-BY-NC weights; "freeze encoder" = upstream linear-probe protocol
    Backbone(
        key="concerto", label="Concerto", script="scripts/modal/modal_train_concerto.py",
        app_name="concerto",
        rec_gpu="A100", min_vram_gb=24,
        grid_clamp=(0.15, 2.0), grid_mult=1.25,
        params=[ParamSpec("grid", "Grid size (m)", "float", 0.5, 0.02, 3.0,
                          step=0.05, decimals=2, recommend_key="grid"),
                ParamSpec("freeze-encoder", "Freeze encoder (0/1)", "int", 0, 0, 1)]
               + _common(100, 4),
    ),
    Backbone(
        key="sonata", label="Sonata", script="scripts/modal/modal_train_sonata.py",
        app_name="sonata",
        rec_gpu="A100", min_vram_gb=24,
        grid_clamp=(0.15, 2.0), grid_mult=1.25,
        params=[ParamSpec("grid", "Grid size (m)", "float", 0.5, 0.02, 3.0,
                          step=0.05, decimals=2, recommend_key="grid"),
                ParamSpec("freeze-encoder", "Freeze encoder (0/1)", "int", 0, 0, 1)]
               + _common(100, 4),
    ),
    Backbone(
        key="utonia", label="Utonia", script="scripts/modal/modal_train_utonia.py",
        app_name="utonia",
        rec_gpu="A100", min_vram_gb=24,
        grid_clamp=(0.15, 2.0), grid_mult=1.25,
        params=[ParamSpec("grid", "Grid size (m)", "float", 0.5, 0.02, 3.0,
                          step=0.05, decimals=2, recommend_key="grid"),
                ParamSpec("freeze-encoder", "Freeze encoder (0/1)", "int", 0, 0, 1)]
               + _common(100, 4),
    ),
]}

GPU_CHOICES = ["A10G", "L4", "L40S", "A100", "A100-80GB", "H100"]
