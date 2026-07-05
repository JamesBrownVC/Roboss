"""Digital-twin motion fitting: dog time series -> Go2 joint trajectory that
minimizes foot-path tracking loss inside the MuJoCo twin, then validated and
exported like any other retarget.

Honest scope: SuperAnimal gives 2D monocular keypoints, so this fits the
SAGITTAL-plane gait pattern (fore-aft + vertical foot excursion, stride
timing, per-leg phase) — not true 3D or hip abduction. Gait *parameters*
transfer robustly; exact joint angles do not. The loss the optimizer minimizes
is the normalized foot-trajectory error between the dog and the simulated Go2.
"""

from .gait import DogGait, extract_gait  # noqa: F401
from .fitter import TwinFitResult, fit_twin  # noqa: F401
