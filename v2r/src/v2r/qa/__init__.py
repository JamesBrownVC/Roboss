"""QA package: cross-checks, yield reporting, license audit (master prompt 6.K, 10).

Pure consumers of the interchange schema: everything here reads episode
workspace artifacts plus stage manifests and writes qa/ artifacts and reports.
No third-party tool is involved, so the qa computations run identically in
'real' and 'synthetic' pipeline modes.
"""

from .crosschecks import run_crosschecks  # noqa: F401
from .feasibility import run_feasibility_judge  # noqa: F401
from .license_audit import generate_license_audit  # noqa: F401
from .yield_report import write_yield_report  # noqa: F401

__all__ = [
    "run_crosschecks",
    "run_feasibility_judge",
    "write_yield_report",
    "generate_license_audit",
]
