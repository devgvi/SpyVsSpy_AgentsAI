"""FINAL-2 gameplay polish (spec FINAL2_GAMEPLAY_POLISH_ROOMS_AND_TRAPS.md, section 39): room dressing, door traps and
trap variants - playable-runtime rules downstream of the frozen policies. Call-free: no provider, no network."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from spy_vs_ai.runtime import config as rc
from spy_vs_ai.runtime import dressing
from spy_vs_ai.runtime import playback as pb
from spy_vs_ai.runtime import traps as tr
from src import constants
from src.actions import ActionType, GameAction
from workshop.policy_emulator import stage4_level_select as ls

ROOT = Path(__file__).resolve().parents[1]
BLACK, WHITE = constants.BLACK_SPY_ID, constants.WHITE_SPY_ID
DOOR = "door_corridor_main_office"  # level_01: corridor (north) / main office (south), open from the start
NORTH = (720.0, 712.0)               # in the corridor, 48 px above the doorway's threshold line (y 760)
SOUTH = (720.0, 812.0)               # in the main office, below it
FAR = (1150.0, 560.0)                # the reception: out of everyone's way
AWAY = (700.0, 200.0)                # the security room: the other way
DT = 1 / 60
FINAL1_HASHES = {
    "workshop/policies/final-release/freeze_manifest.json": "0eceb995c42bb588cb49f7c8f9d94fc76b276325bdfe98d9e821cb5f0458c229",
    "workshop/policy_data/final_research_registry.json": "6c3d253c581c83c15909f88ed3d9bc584fc9e2a0115c6f2d6e4e270040a72f25",
    "workshop/policy_data/final_incident_registry.json": "d5a6a1afcc19c02f98f992571bfd7616c030dd656f5c6bd699c3970e0d1e59e3",
    "workshop/policies/final-release/claims.json": "5236cda681484b65f991e27db8718818ba3788fbe105cf9f5f56a9c245ff7dc1",
}
# the reproduction record is rewritten by every `final_research_release --reproduce` (timestamps, timings, paths of
# benign lock writes); the release manifest does not hash it, so its scientific content is checked instead
REPRODUCTION_RECORD = "workshop/policies/final-release/final_reproduction.json"
GAME_CODE_FINGERPRINT = "sha256:f1a7529308c66f48"
OBSERVATION_FIELDS_SHA = "866b2bbe57ba0778"  # the frozen Observation dataclass field list (60 fields)


@pytest.fixture(autouse=True)
def _restore_level_selection():
    before = constants.LEVEL_PATH
    yield
    constants.LEVEL_PATH = before


def config(enhanced: bool | None = True, *, level: str = "level_01", mode: str = "evaluation", seed: int = 42,
           black: str = "saboteur", white: str = "cautious") -> rc.GameRuntimeConfig:
    data = {"black": {"policy": "b2_v1", "personality": black}, "white": {"policy": "b2_v1", "personality": white},
            "level": level, "seed": seed, "mode": mode, "view": "play", "time_limit": 300}
    if enhanced is not None:
        data["enhanced_gameplay"] = enhanced
    return rc.from_dict(data)


def make_game(enhanced: bool = True, **kwargs):
    """A runtime game with no agent at all (replay mode): the test drives the engine directly."""
    from spy_vs_ai.runtime.session import RuntimeGame, to_live_config
    cfg = config(enhanced, **kwargs)
    with ls.using(cfg.level):
        game = RuntimeGame(to_live_config(cfg), None, runtime_config=cfg, replay=True, record=False)
    return game


def put(game, actor_id: str, xy) -> None:
    actor = game.state.actors[actor_id]
    actor.x, actor.y = xy
    actor.moving = False
    if game.traps is not None:
        game.traps._prev[actor_id] = tuple(xy)
    game.state.update(0.0)


def frame(game, actions: dict[str, GameAction] | None = None, dt: float = DT) -> None:
    for actor_id in (BLACK, WHITE):
        game.state.apply_action(actor_id, (actions or {}).get(actor_id, GameAction(ActionType.WAIT)), dt)
    game.state.update(dt)


def walk(game, actor_id: str, direction, frames: int) -> None:
    for _ in range(frames):
        frame(game, {actor_id: GameAction(ActionType.MOVE, direction=direction)})


def wait(game, seconds: float) -> None:
    for _ in range(int(round(seconds / DT))):
        frame(game)


def armed_game(variant: str, owner: str = BLACK):
    """`variant` armed on DOOR by `owner`; the other spy waits just north of it, the owner far away."""
    game = make_game()
    victim = WHITE if owner == BLACK else BLACK
    put(game, owner, FAR)  # never both on one spot: that would be an encounter (a fight)
    put(game, victim, NORTH)
    trap = game.traps.instantiate(owner, DOOR, variant, source_trap_type="bomb")
    return game, trap


def kinds(game) -> list[str]:
    return [r["kind"] for r in game.traps.records]


# -- frozen interfaces (spec 1, 6, 37, 38) ---------------------------------------------------------------------------

def test_frozen_policy_action_schema_is_unchanged_and_place_trap_is_its_only_trap_action():
    from workshop.policy_emulator import schema
    assert schema.ACTIONS == ("wait", "move", "interact", "inspect", "place_trap", "disarm", "approach_opponent",
                              "avoid_opponent", "enter_safe_code")
    assert {a.value for a in ActionType} == {"move", "interact", "inspect", "place_trap", "disarm", "wait",
                                             "send_message"}
    banned = ("bomb", "electric", "gas", "snare", "door", "explosive")
    assert not [a for a in schema.ACTIONS if any(b in a for b in banned)]


def test_frozen_observation_schema_is_unchanged():
    from src.observation import Observation
    names = [f.name for f in dataclasses.fields(Observation)]
    assert hashlib.sha256(",".join(names).encode()).hexdigest()[:16] == OBSERVATION_FIELDS_SHA
    assert not [n for n in names if any(k in n for k in ("prop", "door_trap", "enhanced", "variant", "gas"))]


def test_the_frozen_game_code_is_untouched():
    from workshop.scripts.tournament import code_fingerprint
    assert code_fingerprint() == GAME_CODE_FINGERPRINT


def test_final1_scientific_hashes_are_unchanged():
    for rel, digest in FINAL1_HASHES.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel


def test_final1_reproduction_record_still_says_reproduced():
    record = json.loads((ROOT / REPRODUCTION_RECORD).read_text())
    manifest = json.loads((ROOT / "workshop/policies/final-release/freeze_manifest.json").read_text())
    assert record["verdict"] == "REPRODUCED (scientific equality)" and record["ok"] is True
    assert record["manifest_hash"] == manifest["manifest_hash"]
    assert record["provider_calls"] == 0 and record["network_attempts"] == 0 and record["credential_reads"] == 0
    assert record["provider_clients_constructed"] == 0
    assert record["frozen_artifacts"]["rewritten"] == []
    assert record["isolation"]["writes_unexplained"] == []


# -- configuration and labels (spec 2, 35, 36) ---------------------------------------------------------------------

def test_enhanced_gameplay_is_on_in_playable_off_in_evaluation_and_recorded_resolved():
    assert rc.enhanced(config(None, mode="playable")) and not rc.enhanced(config(None, mode="evaluation"))
    assert rc.enhanced(config(True)) and not rc.enhanced(config(False, mode="playable"))
    assert config(None, mode="playable").public()["enhanced_gameplay"] is True
    assert config(None).public()["enhanced_gameplay"] is False
    for word, value in rc.ENHANCED_CHOICES.items():
        assert config(None).__class__ is rc.GameRuntimeConfig
        assert rc.from_dict({**config(None).public(), "enhanced_gameplay": word}).enhanced_gameplay is value
    assert rc.problems(dataclasses.replace(config(None), enhanced_gameplay="yes"))
    assert "enhanced_gameplay: auto" in rc.to_yaml(config(None))


def test_cli_flag_and_labels():
    from spy_vs_ai import play
    args = play.build_parser().parse_args(["--black-policy", "b2_v1", "--white-policy", "b1", "--mode", "evaluation",
                                           "--enhanced-gameplay", "on"])
    cfg = play.config_from_args(args)
    assert cfg.enhanced_gameplay is True and rc.enhanced(cfg)
    ex = rc.execution(cfg)
    assert ex["enhanced_gameplay"] is True and rc.ENHANCED_LABEL in ex["notes"]
    assert rc.execution(config(None))["notes"] == []
    from spy_vs_ai.runtime import policy_registry as pr
    a = cfg.agent("black")
    comp = pr.compatibility(a.policy, provider=None, model=None, identity=a.surrogate_identity, level="level_02",
                            mode="evaluation", time_limit=300.0, enhanced=True)
    assert comp.label == pr.COMPATIBLE and any("enhanced gameplay" in r for r in comp.reasons)
    game = make_game()
    assert game.renderer.enhanced and game.traps is not None and game.state.enhanced_traps is game.traps
    assert make_game(False).traps is None and not make_game(False).renderer.enhanced


# -- PLACE_TRAP resolution (spec 7, 8, 13-16, 24) ------------------------------------------------------------------

def _inspected(game, actor_id: str, xy=(720.0, 870.0), target: str = "desk_01"):
    put(game, actor_id, xy)
    assert game.state.current_interaction_target[actor_id] == target
    frame(game, {actor_id: GameAction(ActionType.INSPECT, target_id=target)})
    while game.state.actors[actor_id].is_busy:
        frame(game)


def test_place_trap_remains_valid_and_resolves_to_a_doorway(monkeypatch):
    monkeypatch.setattr(tr, "DOOR_PREFERENCE", {"saboteur": 1.0})
    game = make_game()
    put(game, WHITE, FAR)
    _inspected(game, BLACK)
    black = game.state.actors[BLACK]
    before = dict(black.trap_inventory)
    frame(game, {BLACK: GameAction(ActionType.PLACE_TRAP, target_id="desk_01", trap_type="bomb")})
    placed = [r for r in game.traps.records if r["kind"] == "placed"]
    assert len(placed) == 1 and placed[0]["target_type"] == "doorway" and placed[0]["target_id"] == DOOR
    assert placed[0]["variant"] in dict(tr.FLAVOR["saboteur"]) and placed[0]["source_trap_type"] == "bomb"
    assert black.trap_inventory["bomb"] == before["bomb"] - 1  # the engine's inventory, one unit
    assert abs(black.action_cooldown - (constants.TRAP_PLACEMENT_COST["bomb"] - DT)) < 1e-6  # the engine's cost
    # never in the engine's own trap state (nor its observation)
    assert game.state.trap_on("desk_01") is None and not game.state.self_placed_traps.get(BLACK)
    assert any(e["type"] == "ENHANCED_TRAP_PLACED" for e in game.state.event_log)
    assert not any(e["type"] == "TRAP_PLACED" for e in game.state.event_log)


def test_object_target_keeps_the_engine_trap(monkeypatch):
    monkeypatch.setattr(tr, "DOOR_PREFERENCE", {"saboteur": 0.0})
    game = make_game()
    put(game, WHITE, FAR)
    _inspected(game, BLACK)
    frame(game, {BLACK: GameAction(ActionType.PLACE_TRAP, target_id="desk_01", trap_type="bomb")})
    trap = game.state.trap_on("desk_01")
    assert trap is not None and trap["type"] == "bomb" and trap["owner_id"] == BLACK
    resolved = [r for r in game.traps.records if r["kind"] == "resolved"]
    assert resolved and resolved[0]["target_type"] == "object" and resolved[0]["variant"] == "explosive"
    assert not game.traps.traps


def test_a_refused_place_trap_stays_refused():
    game = make_game()
    put(game, WHITE, FAR)
    put(game, BLACK, (720.0, 870.0))  # beside the desk, never inspected
    inventory = dict(game.state.actors[BLACK].trap_inventory)
    frame(game, {BLACK: GameAction(ActionType.PLACE_TRAP, target_id="desk_01", trap_type="bomb")})
    assert game.state.actors[BLACK].trap_inventory == inventory and not game.traps.records


def test_resolver_is_deterministic_for_a_fixed_seed_and_state():
    def run():
        game = make_game(black="opportunist")
        put(game, WHITE, FAR)
        _inspected(game, BLACK)
        frame(game, {BLACK: GameAction(ActionType.PLACE_TRAP, target_id="desk_01", trap_type="electric")})
        return game.traps.records, [e for e in game.state.event_log if e["type"].startswith("ENHANCED_")]
    assert run() == run()


def test_doorway_eligibility_and_traffic_ranking():
    game = make_game()
    put(game, WHITE, FAR)
    put(game, BLACK, (720.0, 576.0))  # the corridor's middle
    doors = [d.id for d in game.traps.eligible_doors(BLACK)]
    # the security and archives doors start locked; the exit never qualifies
    assert doors == ["door_corridor_main_office", "door_reception_corridor"]  # traffic 6 before traffic 5
    game.state.door_states["door_security_corridor"] = "open"
    assert "door_security_corridor" in [d.id for d in game.traps.eligible_doors(BLACK)]
    put(game, BLACK, (1300.0, 650.0))  # deep in the reception: its one door is too far
    assert game.traps.eligible_doors(BLACK) == []


def test_no_double_trap_on_one_doorway():
    game = make_game()
    put(game, BLACK, (720.0, 576.0))
    game.traps.instantiate(BLACK, DOOR, "gas", source_trap_type="bomb")
    assert DOOR not in [d.id for d in game.traps.eligible_doors(BLACK)]
    assert DOOR not in [d.id for d in game.traps.eligible_doors(WHITE)]
    with pytest.raises(ValueError):
        game.traps.instantiate(WHITE, DOOR, "snare", source_trap_type="snare")


# -- triggering (spec 9, 10, 31) ------------------------------------------------------------------------------------

def test_opponent_crossing_triggers_and_the_doorway_stays_traversable():
    game, trap = armed_game("electric")
    walk(game, WHITE, (0.0, 1.0), 40)
    assert trap.status == "TRIGGERED" and trap.victim_id == WHITE and not trap.armed
    assert kinds(game).count("triggered") == 1
    wait(game, tr.ELECTRIC_STUN + 0.1)  # stunned on the threshold, then on its way
    walk(game, WHITE, (0.0, 1.0), 20)
    assert game.state.actors[WHITE].y > 800  # it got through: risk, not a wall


def test_standing_near_or_moving_along_the_doorway_does_not_trigger():
    game, trap = armed_game("explosive")
    wait(game, 2.0)
    walk(game, WHITE, (1.0, 0.0), 10)
    walk(game, WHITE, (-1.0, 0.0), 10)
    put(game, WHITE, (700.0, 760.0))  # on the threshold line, inside the doorway
    walk(game, WHITE, (1.0, 0.0), 12)
    assert trap.armed and "triggered" not in kinds(game)


def test_owner_is_immune():
    game, trap = armed_game("explosive", owner=BLACK)
    put(game, WHITE, AWAY)
    put(game, BLACK, NORTH)  # the owner walks through its own doorway
    walk(game, BLACK, (0.0, 1.0), 40)
    assert trap.armed and game.state.actors[BLACK].y > 800 and not game.state.actors[BLACK].is_stunned
    game2, trap2 = armed_game("explosive", owner=WHITE)
    walk(game2, BLACK, (0.0, 1.0), 40)  # ... while the opponent sets it off
    assert trap2.status == "TRIGGERED" and trap2.victim_id == BLACK


# -- detection and disarm (spec 11, 12) ----------------------------------------------------------------------------

def test_inspect_near_a_trapped_doorway_can_detect_it_and_a_detector_steps_over(monkeypatch):
    monkeypatch.setattr(tr, "DETECTION_PROBABILITY", {v: 1.0 for v in tr.VARIANTS})
    game, trap = armed_game("snare")
    _inspected(game, WHITE)
    assert WHITE in trap.detected_by and "detected" in kinds(game)
    assert trap in game.traps.visible_to(WHITE) and trap in game.traps.visible_to(BLACK)
    put(game, WHITE, NORTH)
    walk(game, WHITE, (0.0, 1.0), 40)
    assert trap.armed  # a detected trap does not go off (as a detected object trap on a search)


def test_undetected_door_trap_stays_hidden_and_follows_the_existing_disarm_rule():
    game, trap = armed_game("snare")
    assert trap not in game.traps.visible_to(WHITE)
    put(game, WHITE, (720.0, 800.0))
    assert game.state.current_interaction_target[WHITE] == DOOR
    frame(game, {WHITE: GameAction(ActionType.DISARM)})
    assert trap.armed and not [k for k in kinds(game) if "disarm" in k]  # the engine's no-op: nothing to disarm


@pytest.mark.parametrize("p, outcome", [(1.0, "DISARMED"), (0.0, "TRIGGERED")])
def test_detected_door_trap_can_be_disarmed(monkeypatch, p, outcome):
    monkeypatch.setattr(tr, "DISARM_PROBABILITY", {v: p for v in tr.VARIANTS})
    game, trap = armed_game("electric")
    trap.detected_by[WHITE] = 0.0
    put(game, WHITE, (720.0, 800.0))
    frame(game, {WHITE: GameAction(ActionType.DISARM)})
    assert trap.status == outcome and not trap.armed
    white = game.state.actors[WHITE]
    if outcome == "DISARMED":
        assert white.is_busy and not white.is_stunned and "disarmed" in kinds(game)
    else:
        assert white.is_stunned and game.traps.records[-1]["cause"] == "failed_disarm"


def test_the_owner_removes_its_own_door_trap():
    game, trap = armed_game("gas")
    put(game, BLACK, (720.0, 800.0))
    frame(game, {BLACK: GameAction(ActionType.DISARM)})
    assert trap.status == "REMOVED" and not trap.armed


# -- variants (spec 17-23) ------------------------------------------------------------------------------------------

def _crossed(variant: str):
    game, trap = armed_game(variant)
    lives = game.state.lives.get(WHITE)
    for _ in range(40):
        frame(game, {WHITE: GameAction(ActionType.MOVE, direction=(0.0, 1.0))})
        if not trap.armed:
            break
    assert trap.status == "TRIGGERED"
    return game, trap, lives


def test_explosive_effect():
    game, trap, lives = _crossed("explosive")
    white = game.state.actors[WHITE]
    assert game.state.lives[WHITE] == lives - 1 and white.is_stunned and white.stunned_timer <= tr.EXPLOSIVE_STUN
    assert any(e["type"] == "LIFE_LOST" and e["actor_id"] == WHITE for e in game.state.event_log)
    assert game.state.trap_hits[WHITE] == 1


def test_electric_effect():
    game, trap, lives = _crossed("electric")
    white = game.state.actors[WHITE]
    assert game.state.lives[WHITE] == lives and white.is_stunned
    assert tr.ELECTRIC_STUN - 0.1 <= white.stunned_timer <= tr.ELECTRIC_STUN
    wait(game, tr.ELECTRIC_STUN + 0.1)
    assert not white.is_stunned


def test_gas_effect():
    game, trap, lives = _crossed("gas")
    white, black = game.state.actors[WHITE], game.state.actors[BLACK]
    assert game.state.lives[WHITE] == lives and not white.is_stunned
    assert {e.kind for e in game.traps.effects} == {"gas_cloud", "poison"}
    assert white.speed == pytest.approx(constants.PLAYER_SPEED * tr.GAS_SLOW)
    put(game, BLACK, (680.0, 795.0))  # the owner, inside its own cloud (beyond encounter range of the victim)
    assert black.speed == constants.PLAYER_SPEED and not game.state.fights
    wait(game, tr.GAS_SECONDS + 0.1)
    assert not game.traps.effects and white.speed == constants.PLAYER_SPEED
    assert kinds(game).count("effect_ended") == 2


def test_snare_effect():
    game, trap, lives = _crossed("snare")
    white = game.state.actors[WHITE]
    assert white.speed == 0 and not white.is_stunned and game.state.lives[WHITE] == lives
    y = white.y
    walk(game, WHITE, (0.0, 1.0), 30)
    assert white.y == y  # rooted: no movement (other actions stay possible: it is not stunned)
    wait(game, tr.SNARE_SECONDS)
    assert white.speed == constants.PLAYER_SPEED
    walk(game, WHITE, (0.0, 1.0), 10)
    assert white.y > y


def test_alarm_effect_reveals_the_victim_to_the_owner_through_the_existing_observation():
    from src.observation import build_observation
    game, trap, lives = _crossed("alarm")
    assert game.state.actors[BLACK].alarm_reveal_timer > constants.ALARM_REVEAL_DURATION - 0.1
    assert game.state.lives[WHITE] == lives and not game.state.actors[WHITE].is_stunned
    assert build_observation(game.state, BLACK).revealed_opponent is not None
    assert any(e["type"] == "ALARM_REVEALED" and e["victim_id"] == WHITE for e in game.state.event_log)


def test_variants_are_mechanically_distinct_and_legacy_traps_map_to_explosive():
    effects = {v: tr.EFFECT_LABEL[v] for v in tr.VARIANTS}
    assert len(set(effects.values())) == len(tr.VARIANTS) and set(tr.CUE.values()) == {"BOOM", "ZAP", "GAS", "SNARED", "ALARM"}
    assert tr.variant_of_engine_trap("bomb") == "explosive" and tr.variant_of_engine_trap(None) == "explosive"
    assert max(tr.EXPLOSIVE_STUN, tr.ELECTRIC_STUN, tr.GAS_SECONDS, tr.SNARE_SECONDS) <= 4.0  # no long lockout


# -- room dressing (spec 3-6, 30, 32-34) ---------------------------------------------------------------------------

@pytest.mark.parametrize("level_id", rc.LEVELS)
def test_room_dressing_keeps_paths_doors_exit_and_mission_objects_reachable(level_id):
    from src.level import load_level
    level = load_level(ls.level_path(level_id))
    routes = dressing.route_rects(level)
    props = dressing.plan(level, routes=routes)
    report = dressing.validate(level, props, routes=routes)
    assert report["ok"], report["problems"]
    assert report["exit_reachable"]
    for reach in report["reachability"].values():
        assert reach["lost"] == [] and reach["reachable_with_props"] == reach["reachable"]
    assert len(props) >= len(level.rooms)  # richer rooms
    assert sum(1 for r in level.rooms if any(p.room == r.id for p in props)) >= len(level.rooms) * 2 // 3
    assert all(p.category == dressing.DECORATIVE and (ROOT / p.sprite).is_file() for p in props)
    assert dressing.plan(level, routes=routes) == props  # deterministic


def test_decorative_props_do_not_alter_the_frozen_observation_or_the_world():
    from src.llm_controller import serialize_observation
    from src.observation import build_observation
    plain, dressed = make_game(False), make_game(True)
    before = {a: serialize_observation(build_observation(dressed.state, a), live_state=True, live_workshop=True)
              for a in (BLACK, WHITE)}
    objects = [o.id for o in dressed.state.level.objects]
    blocked = set(dressed.state.nav_grid.blocked)
    from workshop.live.overlay import TelemetryView
    r = dressed.renderer
    r.game, r.view = dressed, TelemetryView(dressed.live_config)
    r.render(dressed.state, 0.0, None, None, None)  # draws the props
    assert dressing.props_for(dressed.state.level)
    for a in (BLACK, WHITE):
        after = serialize_observation(build_observation(dressed.state, a), live_state=True, live_workshop=True)
        other = serialize_observation(build_observation(plain.state, a), live_state=True, live_workshop=True)
        assert after == before[a] == other
    assert [o.id for o in dressed.state.level.objects] == objects and set(dressed.state.nav_grid.blocked) == blocked


def test_markers_show_only_known_door_traps():
    game, trap = armed_game("gas")
    r = game.renderer
    assert [t for t, _ in r.known_door_traps(game.state, BLACK)] == [trap]
    assert r.known_door_traps(game.state, WHITE) == []
    trap.detected_by[WHITE] = 1.0
    assert r.known_door_traps(game.state, WHITE) == [(trap, False)]
    assert r.trap_cue(game.state, {"type": "ENHANCED_TRAP_TRIGGERED", "variant": "gas", "victim": WHITE,
                                    "target_type": "doorway", "target_id": DOOR})[1].startswith("GAS! WHITE")


# -- telemetry and replay (spec 27-29, 40) --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    from tools import gameplay_polish_demo
    out = tmp_path_factory.mktemp("gameplay-polish-demo")
    before = constants.LEVEL_PATH
    try:
        report = gameplay_polish_demo.run(out)
    finally:
        constants.LEVEL_PATH = before  # a playback selects its level process-wide (friction 173)
    return report, out


def test_acceptance_demo_call_free(demo):
    report, out = demo
    assert report["ok"], {k: v for k, v in report["steps"].items() if not v["ok"]}
    assert all(Path(p).is_file() for p in report["screenshots"])


def test_new_replay_reproduces_the_door_traps(demo):
    report, out = demo
    bundle = json.loads((out / "replay_match01.json").read_text())
    assert bundle["config"]["enhanced_gameplay"] is True and bundle["enhanced_gameplay"] is True
    saved = [e for e in bundle["telemetry"] if e.get("event_type") == "enhanced_trap"]
    assert {"placed", "triggered"} <= {e["kind"] for e in saved}
    for e in saved:
        assert e["runtime_rule"].startswith("enhanced gameplay")
    result = pb.Playback(pb.load(out / "replay_match01.json")).run_to_end()
    assert result["verdict"] == pb.REPRODUCED and result["checks"]["enhanced_traps_identical"]
    assert result["provider_calls"] == 0
    # a changed trap record is caught
    bundle["enhanced_traps"]["door_traps"][0]["variant"] = "not-a-variant"
    tampered = out / "tampered.json"
    tampered.write_text(json.dumps(bundle))
    assert pb.Playback(pb.load(tampered)).run_to_end()["verdict"] == pb.DIVERGED


def test_legacy_replay_without_trap_metadata_still_loads_and_reproduces(tmp_path):
    from spy_vs_ai.runtime.session import RuntimeMatchRunner
    cfg = dataclasses.replace(config(False), time_limit=30.0, output_dir=str(tmp_path))
    with ls.using(cfg.level):
        runner = RuntimeMatchRunner(cfg, tmp_path, headless=True, verbose=False)
        runner.run()
    bundle = json.loads(Path(runner.replays[0]).read_text())
    for key in ("enhanced_gameplay", "enhanced_traps"):
        bundle.pop(key)
    bundle["config"].pop("enhanced_gameplay")  # as every replay written before this milestone
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(bundle))
    playback = pb.Playback(pb.load(legacy))
    assert playback.config.enhanced_gameplay is False and playback.game.traps is None
    assert playback.run_to_end()["verdict"] == pb.REPRODUCED
    # a legacy playable-mode recording never turns the new rules on either
    bundle["config"]["mode"] = "playable"
    assert pb.Playback(bundle | {"_path": str(legacy)}).config.enhanced_gameplay is False
