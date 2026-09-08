from collections import Counter

import pytest

from raagacomposer.music.instruments import get
from raagacomposer.production.roles import InstrumentProfile, Role, RoleAssignment, production_team


@pytest.mark.parametrize("singers,drone", [(1, True), (2, True), (1, False), (2, False)])
def test_team_has_distinct_responsibilities_and_supported_instruments(singers, drone):
    team = production_team(singers, drone)
    counts = Counter(member.role for member in team)
    assert counts == {Role.PRODUCER: 1, Role.CRITIC: 1, Role.LYRICS: 1,
                      Role.SINGER: singers, Role.MELODY: 4, Role.PERCUSSION: 2,
                      **({Role.DRONE: 1} if drone else {})}
    assert len({member.id for member in team}) == len(team)
    assert [member.role for member in team[:2]] == [Role.PRODUCER, Role.CRITIC]
    for member in team:
        if member.profile:
            assert get(member.profile.instrument) is not None
            assert member.profile.role == member.role


@pytest.mark.parametrize("singers", [0, 3, True, 1.5])
def test_no_silent_extra_or_missing_singers(singers):
    with pytest.raises(ValueError):
        production_team(singers=singers)


def test_profile_cannot_be_assigned_to_the_wrong_musician():
    percussion = InstrumentProfile("mridangam", "mridangam", Role.PERCUSSION)
    with pytest.raises(ValueError):
        RoleAssignment("flute", Role.MELODY, percussion)
    with pytest.raises(ValueError):
        RoleAssignment("critic", Role.CRITIC, percussion)
    with pytest.raises(ValueError):
        RoleAssignment("percussion", Role.PERCUSSION)
