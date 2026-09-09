"""The production team's shared contract, in one place.

Arya owns the definitions (state, roles, critic); this module is the facade
the Producer, the application and the tests import from, so the shapes both
sides depend on are named here and nowhere else.  Nothing is defined here:
a name that is not re-exported from its owning module is not part of the
contract.

Stage order, and what each stage's artifact reference names:

    brief        the creator's brief and the raaga chosen for it
    tune         a MelodyVersion            ref  "tune:v<version>"
    lyrics       a LyricsVersion            ref  "lyrics:v<version>"
    voice        a studio VocalRender       ref  "voice:<take id>"
    beat         a BeatVersion              ref  "beat:v<version>"
    arrangement  an ArrangementVersion      ref  "arrangement:v<version>"
    mix          a MixVersion               ref  "mix:v<version>"

The Producer owns sequencing, rounds, revision and cancellation; the Critic
reviews every stage before the Producer moves on; a stage the Critic did not
accept can only be passed by an explicit Producer decision, recorded as such.
"""
from __future__ import annotations

from .critic import (CodexCliTransport, CodexCritic, Critic, Verdict,
                     review_stage)
from .roles import InstrumentProfile, Role, RoleAssignment, production_team
from .state import PRODUCTION_STAGES, ProjectState, StageRecord

__all__ = [
    "PRODUCTION_STAGES", "ProjectState", "StageRecord",
    "Role", "InstrumentProfile", "RoleAssignment", "production_team",
    "Critic", "Verdict", "CodexCritic", "CodexCliTransport", "review_stage",
]
