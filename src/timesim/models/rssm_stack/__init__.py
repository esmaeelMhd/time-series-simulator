"""RSSM-first model namespace.

This package provides a clean, architecture-first import surface:

``timesim.models.rssm_stack``
  - encoders
  - cell
  - decoders
  - distributions
  - world_model

All modules currently re-export existing implementations to keep backward
compatibility while the codebase transitions fully to RSSM-first structure.
"""

from .cell import RSSMCell, RSSMOutput, RSSMState
from .decoders import AuxiliaryDecoder, ObjectiveDecoder
from .distributions import diagonal_independent_normal
from .encoders import (
    ControlEncoder,
    ExogenousEncoder,
    ObservationEncoder,
    UniversalSharedEncoder,
    assert_no_shared_encoder_params,
)
from .world_model import RSSMWorldModel

__all__ = [
    "RSSMCell",
    "RSSMOutput",
    "RSSMState",
    "ControlEncoder",
    "ExogenousEncoder",
    "ObservationEncoder",
    "UniversalSharedEncoder",
    "assert_no_shared_encoder_params",
    "ObjectiveDecoder",
    "AuxiliaryDecoder",
    "diagonal_independent_normal",
    "RSSMWorldModel",
]

