"""The report state machine.

    draft --(agent)--> verified | needs_review --(human clinician)--> signed_off | rejected

* Only the agent/system may create drafts and record verification results.
* Only an authenticated *human* with the ``clinician`` role may sign off or
  reject. There is no code path by which an agent or system actor can reach
  ``signed_off``; the check lives here, beneath the API, so it cannot be
  bypassed by a different caller.
* ``signed_off`` and ``rejected`` are terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ReportStatus(str, Enum):
    DRAFT = "draft"
    VERIFIED = "verified"
    NEEDS_REVIEW = "needs_review"
    SIGNED_OFF = "signed_off"
    REJECTED = "rejected"


TERMINAL = {ReportStatus.SIGNED_OFF, ReportStatus.REJECTED}
HUMAN_ONLY = {ReportStatus.SIGNED_OFF, ReportStatus.REJECTED}

ALLOWED: dict[ReportStatus, set[ReportStatus]] = {
    ReportStatus.DRAFT: {ReportStatus.VERIFIED, ReportStatus.NEEDS_REVIEW, ReportStatus.SIGNED_OFF,
                         ReportStatus.REJECTED},
    ReportStatus.VERIFIED: {ReportStatus.SIGNED_OFF, ReportStatus.REJECTED},
    ReportStatus.NEEDS_REVIEW: {ReportStatus.SIGNED_OFF, ReportStatus.REJECTED},
    ReportStatus.SIGNED_OFF: set(),
    ReportStatus.REJECTED: set(),
}


class SignOffError(PermissionError):
    pass


@dataclass(frozen=True)
class Actor:
    id: str
    kind: str  # "human" | "agent" | "system"
    role: str = ""  # clinician | admin | viewer (humans only)

    @property
    def is_clinician(self) -> bool:
        return self.kind == "human" and self.role == "clinician"


def transition(current: ReportStatus | str, target: ReportStatus | str, actor: Actor,
               reason: str | None = None) -> ReportStatus:
    current, target = ReportStatus(current), ReportStatus(target)
    if current in TERMINAL:
        raise SignOffError(f"report is already {current.value}; it cannot change")
    if target not in ALLOWED[current]:
        raise SignOffError(f"illegal transition {current.value} -> {target.value}")
    if target in HUMAN_ONLY:
        if actor.kind != "human":
            raise SignOffError(f"{actor.kind} actor '{actor.id}' may not move a report to {target.value}; "
                               "a credentialed clinician must do this")
        if not actor.is_clinician:
            raise SignOffError(f"user '{actor.id}' (role {actor.role or 'none'}) is not a credentialed clinician")
        if target == ReportStatus.REJECTED and not (reason and reason.strip()):
            raise SignOffError("a reason is required to reject a report")
    elif actor.kind == "human":
        raise SignOffError("verification status is recorded by the agent, not set by users")
    return target
