"""Scenario Contract Agent — the user->video-generation half of the pipeline.

Turns a free-form user intention into a *contract-based* generation plan:

- world contract: immutable canvas (entities, materials, layout, style)
- object and scene registries: identity anchors for cross-video consistency
- reference assets and identity checks: visual anchors plus post-generation
  compliance criteria
- variation policy: what is allowed to change and by how much
- scenarios: N controlled variations inside the same canvas
- canvas anchor image + per-scenario start frames (visual anchors for
  image-to-video generation)
- a verifier packet per scenario, consumable by the `verifier` package

No video is generated here. The output is a JSON bundle that a video
model consumes and that the verification gate later checks compliance
against.
"""

__version__ = "0.1.0"
