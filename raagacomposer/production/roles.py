"""Profiles describe separate musical responsibilities, not separate model downloads."""
from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    PRODUCER = "producer"
    CRITIC = "critic"
    LYRICS = "lyrics"
    SINGER = "singer"
    MELODY = "melody"
    PERCUSSION = "percussion"
    DRONE = "drone"


@dataclass(frozen=True)
class InstrumentProfile:
    key: str
    instrument: str
    role: Role
    description: str = ""

    def __post_init__(self) -> None:
        if not self.key.strip() or not self.instrument.strip():
            raise ValueError("An instrument profile needs a key and instrument")
        if not isinstance(self.role, Role) or self.role not in (Role.MELODY, Role.PERCUSSION, Role.DRONE):
            raise ValueError("An instrument profile needs an instrumental role")


@dataclass(frozen=True)
class RoleAssignment:
    id: str
    role: Role
    profile: InstrumentProfile | None = None

    def __post_init__(self) -> None:
        if not self.id.strip() or not isinstance(self.role, Role):
            raise ValueError("An assignment needs an identity and role")
        instrumental = self.role in (Role.MELODY, Role.PERCUSSION, Role.DRONE)
        if instrumental != (self.profile is not None):
            raise ValueError("Instrumental assignments need an instrument profile")
        if self.profile is not None and self.profile.role != self.role:
            raise ValueError("Assignment and profile roles must agree")


def production_team(singers: int = 1, include_drone: bool = True) -> list[RoleAssignment]:
    if type(singers) is not int or singers not in (1, 2):
        raise ValueError("The production team has one or two singers")
    if type(include_drone) is not bool:
        raise ValueError("include_drone must be a boolean")
    team = [RoleAssignment(role.value, role) for role in (Role.PRODUCER, Role.CRITIC, Role.LYRICS)]
    team += [RoleAssignment(f"singer-{i + 1}", Role.SINGER) for i in range(singers)]
    for key, instrument, role, description in (
        ("melody-1", "flute", Role.MELODY, "Lead melody and breath-shaped answering phrases"),
        ("melody-2", "violin", Role.MELODY, "Sustained counterline and transitions"),
        ("melody-3", "veena", Role.MELODY, "Plucked phrases and rhythmic melodic detail"),
        ("melody-4", "piano", Role.MELODY, "Sparse supporting note patterns"),
        ("percussion-1", "mridangam", Role.PERCUSSION, "Primary tala cycle and phrase support"),
        ("percussion-2", "kanjira", Role.PERCUSSION, "Secondary percussion, leaving space for the singer"),
    ):
        team.append(RoleAssignment(key, role, InstrumentProfile(key, instrument, role, description)))
    if include_drone:
        profile = InstrumentProfile("drone", "tanpura", Role.DRONE, "Optional tonic support")
        team.append(RoleAssignment("drone", Role.DRONE, profile))
    return team
