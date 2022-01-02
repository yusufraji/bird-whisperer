from dataclasses import dataclass


@dataclass
class Data:
    train: str
    val: str
    test: str


@dataclass
class Paths:
    log: str
    data: Data


@dataclass
class Loader:
    batch_size: int
    shuffle: bool
    num_workers: int
    pin_memory: bool
    drop_last: bool


@dataclass
class Model:
    name: str
    path: str
    output_dim: int


@dataclass
class OptimParams:
    lr: float


@dataclass
class Optimizer:
    name: str
    params: OptimParams


@dataclass
class SchedulerParams:
    T_0: int
    eta_min: float


@dataclass
class Scheduler:
    name: str
    params: SchedulerParams


@dataclass
class Trainer:
    gpus: int
    accumulate_grad_batches: int
    fast_dev_run: bool
    num_sanity_val_steps: int
    resume_from_checkpoint: None


@dataclass
class Params:
    epochs: int
    seed: int
    train_loader: Loader
    val_loader: Loader
    test_loader: Loader
    model: Model
    optimizer: Optimizer
    scheduler: Scheduler
    loss: str


@dataclass
class BirdConfig:
    paths: Paths
    params: Params
