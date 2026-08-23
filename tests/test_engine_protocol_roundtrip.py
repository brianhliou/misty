"""Serialization round-trip pins for the engine protocol mirror.

The TS file (`packages/game/src/engine-protocol.ts`) is the canonical
contract; these tests pin the Python mirror's WIRE SHAPE — exact keys,
string square names, optionality — so drift between the TypedDicts, the
serializers, and the TS types fails a test instead of accumulating
silently (the pre-2026-08-22 MoveJson declared int fields under a key the
serializer never emitted).
"""

from __future__ import annotations

import json

from fow_chess.engine_protocol import (
    EngineObservation,
    EngineObservationAck,
    EngineObservationPush,
    EngineTurnResponse,
    Move,
    move_from_json,
    move_to_json,
    observation_ack_from_json,
    observation_ack_to_json,
    observation_push_from_json,
    observation_push_to_json,
    response_from_json,
    response_to_json,
)


def test_move_wire_shape_uses_from_key_and_square_names():
    j = move_to_json(Move(from_square=12, to_square=28))  # e2 -> e4
    assert j == {"from": "e2", "to": "e4"}
    # promotion key is OMITTED when absent (TS `promotion?`), present when set
    jp = move_to_json(Move(from_square=52, to_square=60, promotion="queen"))
    assert jp == {"from": "e7", "to": "e8", "promotion": "queen"}


def test_move_roundtrip_and_legacy_from_key():
    m = Move(from_square=6, to_square=21, promotion=None)
    assert move_from_json(move_to_json(m)) == m
    # input leniency: the historical "from_" key is still accepted
    assert move_from_json({"from_": "g1", "to": "f3"}) == m


def test_response_omits_empty_diagnostics():
    r = EngineTurnResponse(
        protocol_version="1",
        game_id="g",
        session_id="s",
        move=Move(from_square=12, to_square=28),
    )
    j = response_to_json(r)
    assert "diagnostics" not in j  # TS declares `diagnostics?`
    assert response_from_json(j) == r

    r2 = EngineTurnResponse(
        protocol_version="1",
        game_id="g",
        session_id="s",
        move=Move(from_square=12, to_square=28),
        diagnostics={"iters": 4200},
    )
    j2 = response_to_json(r2)
    assert j2["diagnostics"] == {"iters": 4200}
    assert response_from_json(j2) == r2
    # and the whole payload is plain JSON
    json.dumps(j2)


def test_observation_push_roundtrip():
    push = EngineObservationPush(
        protocol_version="1",
        game_id="g",
        engine_id="python-v2-v1.5",
        session_id="s",
        color="white",
        ply=7,
        observation=EngineObservation(
            ply=7,
            kind="own_move",
            visibility_mask=0xFFFF,
            own_move=Move(from_square=12, to_square=28),
        ),
    )
    j = observation_push_to_json(push)
    # dark-chess default is omitted on the wire (TS `gameSpecId?`)
    assert "gameSpecId" not in j
    assert observation_push_from_json(j) == push
    json.dumps(j)


def test_observation_ack_roundtrip():
    ack = EngineObservationAck(protocol_version="1", game_id="g", session_id="s")
    j = observation_ack_to_json(ack)
    assert j["received"] is True
    assert observation_ack_from_json(j) == ack
