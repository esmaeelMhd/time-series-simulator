"""Primary RSSM world-model entrypoint.

This alias makes the intended primary architecture explicit while preserving
backward compatibility with ``LatentSSMWorldModel`` naming.
"""

from __future__ import annotations

from .latent_ssm import LatentSSMWorldModel


class RSSMWorldModel(LatentSSMWorldModel):
    """Alias class for the primary RSSM world model."""


__all__ = ["RSSMWorldModel"]

