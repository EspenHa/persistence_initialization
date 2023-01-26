from dataclasses import dataclass
from enum import Enum

import hydra


class LayerEnum(Enum):
    REZERO = 1
    PRE_NORM = 2
    POST_NORM = 3


class PositionalEncodingEnum(Enum):
    ROTARY = 1
    SINUSOIDAL = 2


class PersistenceInitEnum(Enum):
    SKIP_AND_GATING = 1
    SKIP = 2
    NONE = 3


class ModelEnum(Enum):
    TRANSFORMER = 1
    INFORMER = 2
    AUTOFORMER = 3


class TrainingSetupEnum(Enum):
    EARLY_STOPPING = 1
    MAX_EPOCH = 2


class FrequencyEnum(Enum):
    YEARLY = "Yearly"
    QUARTERLY = "Quarterly"
    MONTHLY = "Monthly"
    WEEKLY = "Weekly"
    DAILY = "Daily"
    HOURLY = "Hourly"


@dataclass
class Config:
    d_model: int = 32

    seed: int = 0

    model: ModelEnum = ModelEnum.TRANSFORMER
    persistence_init: PersistenceInitEnum = PersistenceInitEnum.SKIP_AND_GATING
    pos_enc: PositionalEncodingEnum = PositionalEncodingEnum.ROTARY
    layer: LayerEnum = LayerEnum.REZERO

    freq: FrequencyEnum = FrequencyEnum.MONTHLY
    training_setup: TrainingSetupEnum = TrainingSetupEnum.EARLY_STOPPING


hydra.core.config_store.ConfigStore.instance().store(name="config", node=Config)
