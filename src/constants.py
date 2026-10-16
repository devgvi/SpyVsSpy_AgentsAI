"""Tunable values for Spy vs AI - Milestone 1."""

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS = 60

# M4A.1: the embassy is bigger than one screen, so the camera not only
# scrolls but zooms in - world space (collision/pathfinding/level data)
# stays exactly as authored; only rendering scales up. See
# Renderer._compute_camera / the world_surface it draws onto in render().
CAMERA_ZOOM = 2.0

TILE_SIZE = 32

PLAYER_SPEED = 180.0  # pixels per second
PLAYER_SIZE = (80, 76)  # max visual bounding box; actual sprite fits within this, aspect-preserved
PLAYER_COLLISION_SIZE = (28, 28)  # collision footprint, independent of sprite size - around the feet

INTERACTION_DISTANCE = 70

STATUS_MESSAGE_SECONDS = 2.5

BLACK_SPY_ID = "black_spy"
WHITE_SPY_ID = "white_spy"

# M5F.3A: the `owner_id` an environmental trap's dict carries (spec
# sections 1-4) - distinct from both spy ids above, so trap code that
# already treats `owner_id` as an opaque string (`_disarm`'s is_owner
# check, `_search`'s "not your own trap" check) works unchanged, and
# nothing mistakes an environmental trap for either spy's placement.
ENVIRONMENT_OWNER_ID = "environment"

# --- Traps (Milestone 3B; four distinct trap families as of Milestone 4I) ---

# M4I ("Trap Effects & Gameplay Feedback") replaced the two undifferentiated
# M3B placeholders (mechanical_trap/alarm_trap) with four thematically
# distinct types, each with its own visual effect (see src/effects.py,
# TRAP_EFFECT_FRAMES below) and its own tuning - "initial tuning... not
# permanent rules" per that doc's section 12. Every dict below is keyed the
# same way across the whole trap system (state.py, bot_controller.py,
# renderer.py all iterate/index by trap_type generically - M4F's strategic
# trap economy already required that), so adding these two extra flavors on
# top of the original two needed no logic changes anywhere, only data.
TRAP_TYPES = ("bomb", "electric", "snare", "alarm")
DEFAULT_TRAP_TYPE = "bomb"

TRAP_LABELS = {
    "bomb": "Bomb Trap",
    "electric": "Electric Trap",
    "snare": "Snare Trap",
    "alarm": "Alarm Trap",
}

# Starting loadout per spy - one of each, a deliberately even kit now that
# there are four flavors instead of two. Copied (never shared) into each
# Actor - see Actor.trap_inventory's default_factory.
TRAP_LOADOUT = {"bomb": 1, "electric": 1, "snare": 1, "alarm": 1}

# Chance INSPECT reveals an armed enemy trap on the inspected object -
# rough "how easy to spot" per trap's physical footprint (a snare's thin
# wire loop is the hardest to notice; a bomb's wiring is the easiest).
TRAP_DETECTION_PROBABILITY = {"bomb": 0.90, "electric": 0.85, "snare": 0.70, "alarm": 0.75}

# How long a triggered trap immobilizes ("recovery"/"immobilization"/
# "disruption" - Milestone_4I_TRAP_EFFECTS.md's own per-type wording) the
# actor who set it off. Also doubles as each trap's effect-animation total
# duration (src/effects.py stretches its 8 frames across exactly this many
# seconds), so the visual and the penalty always start and end together -
# the milestone's explicit "the player must understand... when the effect
# ends."
TRAP_STUN_SECONDS = {"bomb": 4.0, "electric": 3.0, "snare": 4.0, "alarm": 1.0}

# M4I: placement now costs a different amount per trap type (replacing
# M4D's flat TIME_COST_PLACE_TRAP) - the pricier, more punishing traps
# (bomb) cost more of the shared mission clock to set than the cheap,
# mild one (alarm). Charged in GameState._place_trap; also the "placement
# cost" term ScriptedBotController._trap_value (M4F) weighs expected
# delay against, so a bomb needs more relevance signal to justify than an
# alarm does.
TRAP_PLACEMENT_COST = {"bomb": 4.0, "electric": 3.0, "snare": 2.5, "alarm": 2.0}

# M4I: alarm's distinguishing mechanic - triggering one doesn't just
# briefly disrupt the victim (TRAP_STUN_SECONDS["alarm"]), it also tells
# the trap's *owner* where the victim currently is, for this many
# seconds - "The reveal duration should be configurable... suggested
# initial value 5.0s." See Actor.alarm_reveal_timer / Observation.revealed_opponent.
ALARM_REVEAL_DURATION = 5.0

# Bugfix live flow V2, Part D: spy encounters (live play only - see
# GameState.encounters_enabled / src/encounters.py). Gameplay geometry, not
# sprites: a fight starts when the two feet points (the centres of the
# 28x28 collision boxes) are within ENCOUNTER_DISTANCE - i.e. the collision
# boxes touch - and both spies are in the same room.
ENCOUNTER_DISTANCE = 28.0
FIGHT_BLACK_WIN_PROBABILITY = 0.5  # neutral; no personality or stat changes it
FIGHT_LOSER_STUN_SECONDS = 3.0  # the existing stun mechanic (like a trap)
FIGHT_COOLDOWN_SECONDS = 6.0  # game seconds with no new fight after one
# Bugfix spawn/combat polish (live only, GameState.fight_ko_respawn): the
# loser is knocked out for this many match seconds, then respawns at a safe
# point. No trap life is lost.
FIGHT_KO_SECONDS = 5.0
ENCOUNTER_REARM_DISTANCE = 96.0  # ...and the spies must have separated this far first

# Live workshop feedback: lives. Each trap that goes off under a spy costs
# one life; with none left the spy is out of the match (it has failed).
# Live play only - see GameState.lives_enabled.
SPY_LIVES = 3

# Bugfix live match flow V3: explicit actor lifecycle (GameState.lifecycle).
# Only ACTIVE / OBJECTIVES_COMPLETE may request model decisions.
LIFECYCLE_ACTIVE = "ACTIVE"
LIFECYCLE_OBJECTIVES_COMPLETE = "OBJECTIVES_COMPLETE"  # every item held, still inside
LIFECYCLE_ESCAPED = "ESCAPED"
LIFECYCLE_TIMED_OUT = "TIMED_OUT"
LIFECYCLE_ELIMINATED = "ELIMINATED"
# live, one copy of each item: the opponent escaped with a required item this
# spy lacks, so its mission can never be completed (GameState._check_mission_impossible).
# Scored exactly like any other failure.
LIFECYCLE_MISSION_IMPOSSIBLE = "MISSION_IMPOSSIBLE"
MISSION_IMPOSSIBLE_REASON = "required_item_left_with_escaped_opponent"
TERMINAL_LIFECYCLES = (LIFECYCLE_ESCAPED, LIFECYCLE_TIMED_OUT, LIFECYCLE_ELIMINATED, LIFECYCLE_MISSION_IMPOSSIBLE)
# own-trap memory status (live payload trap_status for the owner's armed trap)
OWN_TRAP_ARMED = "OWN_TRAP_ARMED"

# M4I section 10: "replace generic long-lived STUNNED text with
# trap-specific short feedback... keep messages short and thematic."
TRAP_TRIGGER_MESSAGES = {
    "bomb": "BOOM! Bomb triggered!",
    "electric": "ZAP! Electric trap!",
    "snare": "SNARE! You're trapped!",
    "alarm": "ALARM! Position revealed!",
}

# M4I section 7: each trap's 8-frame activation effect, stretched across
# TRAP_STUN_SECONDS[trap_type] (see src/effects.py). Bomb explodes in
# place (fixed at the trap's object); electric/snare/alarm are drawn
# following the victim, matching that doc's own examples.
TRAP_EFFECT_FRAMES = {
    "bomb": tuple(f"assets/effects/bomb/explosion_{i:02d}.png" for i in range(1, 9)),
    "electric": tuple(f"assets/effects/electric/electric_{i:02d}.png" for i in range(1, 9)),
    "snare": tuple(f"assets/effects/snare/snare_{i:02d}.png" for i in range(1, 9)),
    "alarm": tuple(f"assets/effects/alarm/alarm_{i:02d}.png" for i in range(1, 9)),
}
TRAP_EFFECT_FOLLOWS_ACTOR = {"bomb": False, "electric": True, "snare": True, "alarm": True}
EFFECT_RENDER_SIZE = 96  # world units - each effect frame's max bounding box, aspect-preserved

# Static "armed" icon per trap type - assets/traps/, shown only for a trap
# this actor legitimately knows about (owns or has detected), never a
# hidden enemy trap - see Renderer._draw_object's lock/trap-icon overlay.
TRAP_ICON_SPRITES = {
    "bomb": "assets/traps/trap_bomb.png",
    "electric": "assets/traps/trap_electric.png",
    "snare": "assets/traps/trap_snare.png",
    "alarm": "assets/traps/trap_alarm.png",
}

# Scripted strategy baseline (Milestone_32.md's M3C / MILESTONE_3.md's M3D):
# an object within this distance of the exit door counts as "near the
# exit" for ScriptedBotController's opportunistic trap-placement heuristic.
BOT_TRAP_NEAR_EXIT_DISTANCE = 320.0

# --- Dependencies, keys & knowledge (Milestone 4B) --------------------------

# What a locked door/object's failure message calls the thing you're
# missing - never names the specific item id (that's the actual placement,
# randomized per seed - see src/dependency_generation.py), only the shape
# of the lock, same as a real door would show you.
LOCK_PHRASES = {"key": "a key", "keycard": "a keycard", "code": "the code"}

# --- Time economy (Milestone 4D) --------------------------------------------

# Initial configurable mission deadline (Milestone_4.md). A single shared
# clock, not per-actor - both spies race the same countdown. See
# GameState.mission_time_remaining / new_game's `mission_time` param.
MISSION_TIME = 180.0

# "Initial cost" table (Milestone_4.md's M4D). Charged onto the acting
# actor's own Actor.action_cooldown (see GameState._charge_time) - never
# subtracted from the shared clock directly. The shared clock only ever
# depletes by real elapsed dt (GameState.update); a busy actor still lets
# that dt tick by while unable to act, which is what actually makes an
# action "cost" time. Reused, not reinvented: this is the exact same
# non-blocking timer pattern Milestone 3B's `stunned_timer` already
# established - no time.sleep(), no blocking the game loop.
TIME_COST_SEARCH = 1.0
TIME_COST_INSPECT = 1.5
# PLACE_TRAP's cost varies per trap type as of Milestone 4I - see
# TRAP_PLACEMENT_COST above; there's no longer one flat number for it.
TIME_COST_DISARM = 2.0
# Live game balance (GameState.live_trap_rules; recorded runs keep the costs
# above). Live run 20261001-110229: inspecting cost 0.5 s more than a blind
# search, the cautious spy inspected all 12 objects it searched, detected
# both traps it reached and disarmed both - no trap ever went off. Inspecting
# is now a real cost, and disarming an enemy trap can fail: on failure the
# trap goes off under the spy who tried (stun + one life). Removing your own
# trap always works (you set it).
LIVE_TIME_COST_INSPECT = 4.0
DISARM_SUCCESS_PROBABILITY = {"bomb": 0.55, "electric": 0.65, "snare": 0.75, "alarm": 0.85}
TIME_COST_UNLOCK = 1.0  # a key/keycard door, locked -> open, in one motion
TIME_COST_ENTER_CODE = 1.5  # a code-locked object (the safe), locked -> open
TIME_COST_PICKUP = 0.3  # added on top of TIME_COST_SEARCH when a search yields an item
TIME_COST_OPEN = 0.3  # a plain (never-locked, or already-unlocked) door/exit
# M5F.2 spec section 4: never free - priced in the same table as every
# other timed action, roughly SEARCH-sized (composing/sending a short
# message is not a trivial no-op, but it's not as costly as a physical
# search either).
TIME_COST_SEND_MESSAGE = 0.5

# --- Communication (Milestone 5F.2) ------------------------------------------

# Spec section 3's recommended length; enforced both at model-output
# validation time (src/llm_controller.py) and defensively again at the
# engine (GameState._do_send_message never trusts the caller).
MESSAGE_MAX_LENGTH = 200
# Spec section 21: "last 3-5 messages" - the bound on how many of this
# actor's own received messages ever appear in its Observation at once.
# Older messages still persist in GameState.messages (engine truth) and in
# AgentMemory's own bounded history; only the model-facing window is capped.
RECENT_MESSAGES_LIMIT = 5

# --- Competitive / Cooperative Victory (Milestone 4E) ------------------------

# Initial configurable scoring (Milestone_4.md). Not zero-sum: both spies
# may score positively in the same episode, and one escaping is worth
# something (SCORE_OTHER_ALSO_ESCAPES) to the other rather than nothing or
# a penalty. Trapping is deliberately absent from this table - the spec is
# explicit that sabotage is never directly rewarded, only indirectly, by
# whatever it does to an actor's own mission/escape/time outcomes above.
# See src/scoring.py for how these combine into one actor's total.
SCORE_COMPLETE_OWN_MISSION = 100.0  # collected every required item, independent of escaping
SCORE_ESCAPE_BEFORE_DEADLINE = 100.0  # physically got out while mission_time_remaining > 0
SCORE_ESCAPE_FIRST = 30.0
SCORE_OTHER_ALSO_ESCAPES = 15.0  # the other actor succeeding has positive, not zero-sum, value
SCORE_REMAINING_TIME_BONUS_PER_SECOND = 0.5  # the "configurable" remaining-time bonus, per second banked
SCORE_FAILURE_TO_ESCAPE = -150.0  # the deadline passed and this actor never got out

# --- Strategic Trap Economy (Milestone 4F) ----------------------------------

# trap value ~= expected opponent delay - placement cost - strategic risk
# (Milestone_4.md's exact framing - see ScriptedBotController._trap_value).
# Every weight below feeds a signal this actor is legitimately allowed to
# know: its own search history, the object's distance to the exit, and the
# opponent's position only while visible or last-seen (never their
# inventory, target, private history, or raw trap/content ground truth).
# Deliberately additive rather than a capped 0-1 probability - two weak
# signals should be able to add up to "worth it" even though either alone
# isn't, without needing real probabilistic modeling (M4F: "keep the
# current trap type first," not "build a belief model").
TRAP_VALUE_ITEM_SIGNAL = 0.9  # this object handed *this actor* an item - the opponent needs their own copy of it too
TRAP_VALUE_NEAR_EXIT_SIGNAL = 0.6  # object sits within BOT_TRAP_NEAR_EXIT_DISTANCE of the exit - a likely late-game path
TRAP_VALUE_OPPONENT_PROXIMITY_SIGNAL = 0.8  # opponent is currently visible or last-seen within range of this object
TRAP_VALUE_OPPONENT_PROXIMITY_RANGE = 400.0  # world units counted as "near" for the signal above

# Strategic risk: reasons this actor's own situation makes spending a trap
# less attractive right now, independent of this particular object.
#
# M4I recalibration: bomb/electric/snare's stun durations (3.0-4.0s) are
# 1.2-2.7x M4F's original mechanical_trap (2.5s), which raises every
# type's expected-delay term proportionally - the M4F-era risk penalties
# (1.5 each) no longer outweighed even the cheapest, strongest new type
# (snare: 4.0s stun for only 2.5s placement cost) under the exact
# two-signal scenario M4F's own acceptance test used. Raised so a decline
# is still demonstrable against *every* trap type, snare included, not
# just recalculated to keep one specific old assertion passing.
TRAP_STRATEGIC_RISK_LOW_TIME_THRESHOLD = 30.0  # seconds remaining below which time pressure adds risk
TRAP_STRATEGIC_RISK_LOW_TIME_PENALTY = 4.0
TRAP_STRATEGIC_RISK_MISSION_READY_PENALTY = 4.0  # already holding every required item - every extra second now costs more
TRAP_STRATEGIC_RISK_LAST_ONE_PENALTY = 0.5  # spending this trap type's last copy has an opportunity cost

NAV_GRID_SIZE = TILE_SIZE  # pathfinding grid cell edge, in pixels
WAYPOINT_ARRIVAL_DISTANCE = 6  # how close (px) to a path waypoint counts as "reached it"
MAX_COLLISION_SUBSTEP = 8.0  # px: longer per-frame steps are split into collision-checked sub-steps (Actor.move)

LEVEL_PATH = "data/level_01.json"
ITEMS_PATH = "data/items.json"

MISSION_TITLE = "MISSION"
MISSION_OBJECTIVE = "Find the documents and escape."

PLAYER_SPRITE_DIR = "assets/characters"
_FACINGS = ("down", "up", "left", "right")

# One pose per action the player can see an actor perform, plus idle/walk.
# File naming (both packs): <folder>/<prefix>_<pose>_<facing>.png - e.g.
# assets/characters/black_spy/black_search_down.png. Sprite keys (as
# produced by src/animation.py's PlayerAnimator) are "<pose>_<facing>",
# e.g. "search_down", so this dict's keys line up with sprite_key()'s
# output with no translation needed in renderer.py.
CHARACTER_POSES = ("idle", "walk", "search", "inspect", "place_trap", "disarm")

# actor_id -> (asset subfolder, filename prefix)
_CHARACTER_PACKS = {
    BLACK_SPY_ID: ("black_spy", "black"),
    WHITE_SPY_ID: ("white_spy", "white"),
}

# Looked up by actor id so rendering code never special-cases which spy it's
# drawing - see Renderer._draw_actor.
ACTOR_SPRITES = {
    actor_id: {
        f"{pose}_{facing}": f"{PLAYER_SPRITE_DIR}/{folder}/{prefix}_{pose}_{facing}.png"
        for pose in CHARACTER_POSES
        for facing in _FACINGS
    }
    for actor_id, (folder, prefix) in _CHARACTER_PACKS.items()
}

FLOOR_TILE_SPRITE = "assets/floors/floor.png"
WALL_TILE_SPRITE = "assets/floors/wall.png"
FLOOR_TILE_TARGET_SIZE = 64  # M4A.1: bigger tiles repeat less often, reading as a surface not a grid

INVENTORY_SLOT_SIZE = 44
INVENTORY_SLOT_GAP = 10
INVENTORY_SLOT_MARGIN = 16

# --- Colors (retro, limited palette) ---------------------------------------

COLOR_FLOOR = (58, 44, 40)
COLOR_WALL = (34, 24, 22)
COLOR_RUG = (110, 24, 24)

COLOR_PLAYER = (20, 20, 20)
COLOR_PLAYER_OUTLINE = (235, 235, 235)
COLOR_WHITE_SPY = (225, 222, 214)
COLOR_WHITE_SPY_OUTLINE = (30, 30, 30)
COLOR_FACING_MARKER = (200, 40, 40)

ACTOR_COLORS = {
    BLACK_SPY_ID: (COLOR_PLAYER, COLOR_PLAYER_OUTLINE),
    WHITE_SPY_ID: (COLOR_WHITE_SPY, COLOR_WHITE_SPY_OUTLINE),
}

COLOR_TEXT = (240, 240, 230)
COLOR_PROMPT_BG = (10, 10, 10)
COLOR_HUD_TITLE = (200, 170, 90)

COLOR_MESSAGE_OK = COLOR_TEXT
COLOR_MESSAGE_FAIL = (214, 130, 60)

COLOR_MISSION_COMPLETE = (220, 190, 90)
COLOR_OVERLAY = (0, 0, 0)
OVERLAY_ALPHA = 170

COLOR_SLOT_EMPTY = (50, 45, 40)
COLOR_SLOT_FILLED = (150, 120, 60)
COLOR_SLOT_BORDER = (200, 190, 160)

RUG_COLOR_BORDER = (70, 14, 14)

DEBUG_COLLISION_COLOR = (255, 210, 0)
DEBUG_RENDER_COLOR = (0, 220, 255)
DEBUG_ANCHOR_COLOR = (255, 0, 140)
DEBUG_RADIUS_COLOR = (255, 255, 255)
DEBUG_TEXT_COLOR = (255, 255, 255)
DEBUG_WALL_COLOR = (255, 130, 0)
DEBUG_ROOM_COLOR = (120, 220, 120)
DEBUG_OPENING_COLOR = (80, 180, 255)  # a door's authoritative opening_rect (Mileston_4A2.md)
DEBUG_CLEARANCE_COLOR = (255, 90, 220)  # a door's doorway clearance zone (collision/navigation)
DEBUG_VISUAL_EXCLUSION_COLOR = (255, 220, 60)  # a door's visual-only exclusion zone (M4A.3, composition not collision)

OBJECT_COLORS = {
    "desk": (92, 60, 38),
    "chair": (120, 82, 50),
    "filing_cabinet": (90, 90, 96),
    "bookshelf": (68, 46, 30),
    "sofa": (84, 34, 40),
    "side_table": (110, 74, 46),
    "plant": (40, 92, 48),
    "globe": (46, 84, 120),
    "painting": (170, 140, 70),
    "door": (50, 34, 22),
    "rug": COLOR_RUG,
    "lamp": (200, 170, 90),
    "armchair": (110, 50, 46),
    "statue_bust": (150, 145, 130),
    "safe": (80, 80, 90),
}

OBJECT_LABEL_COLOR = (230, 220, 200)
