# Playing Spy vs AI

The practical manual: installing, starting a match, choosing who plays each spy, the rules and controls, and saving and
replaying matches. If you have not yet, read [the guide](../README.md) first. It explains the game, the screen and
the seven policies in plain words, and this page builds on it.

Contents: [Install](#install) · [Your first match](#your-first-match) · [The command line](#the-command-line) ·
[Configuration files](#configuration-files) · [Choosing the players](#choosing-the-players) ·
[Modes](#modes-playable-or-evaluation) · [During a match](#during-a-match) · [Enhanced gameplay](#enhanced-gameplay) ·
[Saving and replaying](#saving-and-replaying) · [Labels](#scientific-status-and-compatibility-labels)

## Install

Python 3.12, from the repository root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You need pygame for the game, boto3 only for Amazon Bedrock, and optionally PyYAML (configuration files also load
without it). Every command below runs from the repository root.

## Your first match

```bash
python -m spy_vs_ai.play
```

The setup screen opens (the guide shows it, [section 3](../README.md#3-the-interface)). For a first match that calls
nothing and needs no account:

1. Black: policy **B2-v1** (`b2_v1`, the imitator); White: policy **B2-v2** (`b2_v2`). Leave the other fields as
   they are.
2. Keep the defaults below them: Level 1, seed 42, playable mode, play view.
3. Press START MATCH. Press M during the match to open the research view, O for the overview, Esc to quit.

To see a real model play, set a spy's policy to **REAL LLM** (`real_llm`) or **Hybrid-v1** (`hybrid_v1`) and choose
its provider and model. Check that the provider status at the bottom of the screen shows a credential, and see
[PROVIDERS_AND_MODELS.md](PROVIDERS_AND_MODELS.md) to set one up. Calls to Bedrock, OpenAI and Anthropic may be
billed to your account. Before the match starts, the launcher prints which spy calls which provider and model.

## The command line

Everything on the setup screen has a flag, and a match starts directly when a policy is given:

```bash
# call-free: no provider, no credential, nothing is called
python -m spy_vs_ai.play --black-policy b2_v1 --white-policy b2_v2

# the whole runtime offline: the fake provider stands in for a model (no network, zero tokens)
python -m spy_vs_ai.play --config configs/demo_offline.yaml

# Amazon Bedrock: a Hybrid against a REAL LLM
python -m spy_vs_ai.play \
  --black-policy hybrid_v1 --black-provider bedrock --black-model us.anthropic.claude-sonnet-5-5 \
  --white-policy real_llm  --white-provider bedrock --white-model us.openai.gpt-6.1-sol

# a local Ollama model (a spy left unspecified plays the call-free B2-v1)
python -m spy_vs_ai.play --black-policy real_llm --black-provider ollama --black-model qwen
```

| Flags | Sets |
|---|---|
| `--black-policy`, `--black-provider`, `--black-model`, `--black-personality`, `--black-identity`, `--black-option KEY=VALUE`, `--black-max-provider-calls` | one spy (the same with `--white-*`) |
| `--level level_01\|level_02\|level_03`, `--seed`, `--time-limit` | the embassy and the mission clock (300 s by default) |
| `--mode playable\|evaluation`, `--enhanced-gameplay auto\|on\|off` | the rules ([modes](#modes-playable-or-evaluation), [enhanced gameplay](#enhanced-gameplay)) |
| `--view play\|research`, `--speed 0.5\|1\|2`, `--telemetry` / `--no-telemetry` | how the match starts on screen |
| `--save-replay` / `--no-replay`, `--output-dir` | what is saved, and where (default `runs/`) |
| `--setup` | open the setup screen pre-filled from the other flags or a config |
| `--headless` | play without a window and exit (fixed simulated step in evaluation mode) |

Discovery commands call nothing and read no credential:

- `--list-policies`;
- `--list-providers`;
- `--list-models PROVIDER`: known ids, or the models installed on the local server for Ollama;
- `--provider-status`: which credentials are present, by name only.

Any model id is accepted; the lists are suggestions. `--check` validates and describes a configuration without
starting it, and `--save-config PATH` writes it as YAML.

Invalid combinations are refused before anything starts. For example, a provider given to a call-free policy is
refused, as is a Hybrid without a model, an unknown option, or anything that looks like a credential.

## Configuration files

The setup screen, the flags and a YAML file all build the same validated configuration. Flags override the file.

```yaml
black:
  policy: hybrid_v2
  provider: bedrock
  model: us.anthropic.claude-sonnet-5-5
  personality: cautious
white:
  policy: real_llm
  provider: ollama
  model: qwen
  personality: saboteur
level: level_02
seed: 42
mode: playable
speed: 1.0
show_telemetry: true
```

`configs/` has eight examples:

- `demo.yaml`;
- `demo_offline.yaml`, which uses the fake provider;
- `demo_b2_v1_vs_b2_v2.yaml`, call-free;
- `demo_real_vs_real.yaml`, `demo_hybrid_v1_vs_real.yaml`, `demo_hybrid_v2_vs_real.yaml`;
- `demo_ollama_vs_bedrock.yaml`;
- `demo_evaluated_hybrid_v2_vs_real.yaml`, the research settings.

Not every model is available to every account: change the model ids to ones yours has.

## Choosing the players

**Policy** (who decides; the guide explains each one in [section 2](../README.md#2-who-can-play-a-spy)):

| id | Policy | Calls a provider | Scientific status |
|---|---|---|---|
| `scripted` | Scripted bot | no | engine baseline, not a research policy |
| `b1` | B1 (table-v1), the lookup-table imitator | no | superseded by B2 |
| `b2_v1` | B2-v1 (logistic-v1), the statistical imitator | no | confirmed surrogate component of Hybrid-v1 (not adequate alone in closed loop) |
| `hybrid_v1` | Hybrid-v1 / R2.1: B2-v1 plus the router, escalating hard decisions to the LLM | on escalation | CONFIRMED / GENERALIZED RESEARCH BASELINE |
| `b2_v2` | B2-v2 (candidate C1), B2 retrained on the LLM's hard decisions | no | selected candidate component |
| `hybrid_v2` | Hybrid-v2 / R2.1: B2-v2 plus the same router | on escalation | MORE EFFICIENT — NOT SAFELY PROMOTABLE UNDER CONFIRMATORY RULE |
| `real_llm` | REAL LLM: every decision is a model call | every decision | behavioural reference |

Hybrid-v1 and Hybrid-v2 are different frozen policies, and one is never substituted for the other. A Hybrid calls its
provider only when the router escalates. `max_provider_calls` (`--black-max-provider-calls`) caps that per spy, with
a default of 300.

**Personality:** `cautious`, `opportunist`, `racer` or `saboteur` (described in the guide). It is given to the LLM as
priorities, and it selects which trained imitator files a B1, B2 or Hybrid spy uses.

**Surrogate identity:** each imitator learned from one research model, `sonnet55` (Claude Sonnet 5.5) or `gpt61sol`
(GPT-6.1 Sol). It is derived from a Bedrock research model id, or set with `identity` / `--black-identity`. The
default is `sonnet55`.

**Provider and model:** the model id is passed to the provider unchanged. Examples are `us.anthropic.claude-sonnet-5-5`
or `us.openai.gpt-6.1-sol` on Bedrock, an OpenAI or Anthropic model name for the direct APIs, or an installed Ollama
model (the setup screen lists them). Credentials come only from the environment or the AWS credential chain of the
provider you select. They never go in a config file or on the command line, and a match never reads the credential
of a provider it does not use. Details: [PROVIDERS_AND_MODELS.md](PROVIDERS_AND_MODELS.md).

## Modes: playable or evaluation

- **playable** (the default) is smooth live play. The mission clock runs in real time and REAL spies decide while the
  world keeps moving. The banner reads **PLAYABLE MODE - execution semantics differ from the frozen evaluation
  protocol**. With a Hybrid spy, the match keeps the evaluation decision timing, the only one the Hybrid was validated
  with: the world and the clock pause while a model call is in flight.
- **evaluation** uses the frozen research rules. The world stops while a model thinks, the mission clock counts game
  time, and the order in which decisions take effect is seeded. Only this mode, with the 300 s mission, a level the
  policy was evaluated on, and Bedrock with the research model, can earn the EVALUATED label.

## During a match

The guide's [section 3](../README.md#3-the-interface) describes the screen. The keys:

| Key | Action |
|---|---|
| Space | pause / resume (a model call in flight finishes; its answer waits, and no new decision is asked while paused) |
| + / - | speed 0.5x, 1x, 2x |
| M | play view / research view |
| O | overview of the whole embassy |
| P | split view / single view |
| V, T | show / hide the panels |
| Y | AI telemetry overlay |
| I, H | decision inspector, decision history (research view) |
| Tab, Up / Down, Home / End | pick a spy, move through its decision history |
| Z, S, R, Esc | zoom, screenshot, restart, quit |

**The research view** shows, for the selected decision, only the fields its telemetry carries:

- what the spy saw: its room, inventory, the opponent last seen, time left;
- the legal actions;
- the B2 proposal with its confidence and probabilities, and the B1 proposal with the B1 / B2 distance;
- whether R2.1 escalated, and why;
- the provider and model, the action the model returned, its stated `reason` field, the latency, the tokens and the
  physical provider calls;
- any normalized error, repair or fallback;
- the executed action.

Hidden reasoning is never requested or displayed.

**The telemetry overlay** shows, per spy: provider calls, escalations, input and output tokens, average latency,
fallbacks, provider errors and mission progress. Provider errors are shown in normalized form: Provider
authentication failed, Provider unavailable, Model not found, Request timed out, Provider service error, Invalid
model output, Fallback action used. The details stay in the telemetry file.

**At the end**, the match summary shows the winner and the ranking with its score breakdown. Per spy and in total,
it gives the outcome (escaped, timed out or eliminated), decisions, provider calls, escalations, tokens, fallbacks
and the match duration.

## Enhanced gameplay

The enhanced gameplay adds door traps, trap variants and richer rooms. Turn it on or off with "Enhanced gameplay" on
the setup screen, `--enhanced-gameplay auto|on|off`, or `enhanced_gameplay` in a config. `auto` (the default) turns
it on in playable mode and off in evaluation mode. When it is on, the banner reads **Enhanced gameplay - trap
target/type resolution occurs downstream of the frozen policy**, and every research policy is labelled COMPATIBLE /
NOT EVALUATED. None of this was part of the research.

- **The policies are unchanged.** They still have one trap action, PLACE_TRAP, and DISARM. No action or observation
  field was added, and nothing was retrained.
- **Where a trap goes.** When a spy places a trap the game would accept, the game decides where it goes: on the
  object, as before, or on a doorway of the spy's room within 260 px. The exit and locked doors are never used, a
  doorway holds at most one trap, and the busiest doorway is chosen first. The personality sets how often a doorway is
  chosen and which variant goes there: saboteur explosive or gas, cautious snare or alarm, racer electric, opportunist
  any. Object traps keep their usual effects.
- **Door traps** go off when an opposing spy crosses the doorway's threshold; standing near it or walking along it
  does not. The owner is immune, a spy that detected the trap steps over it, and a trapped doorway never blocks.
- **Variants:**
  - **explosive:** one life and a 2 s stun;
  - **electric:** a 3 s stun;
  - **gas:** a 4 s cloud on the doorway and 4 s of half speed;
  - **snare:** rooted for 2.5 s, other actions still possible;
  - **alarm:** the owner sees the victim for 5 s.

  Each one shows a cue naming the trap, the victim and the effect (BOOM / ZAP / GAS / SNARED / ALARM), with an
  animation.
- **Detection and disarming.** Inspecting within 160 px of a trapped doorway also examines it. A spy sees a door-trap
  marker only for its own traps and the ones it detected; in split view, each pane shows its own spy's knowledge.
  DISARM next to a detected door trap removes it, and a failed disarm sets it off.
- **Room dressing.** Decorative props (plants, lamps, crates, paintings, clocks, radios, books, coat stands, ...) are
  drawn only. They are not level objects, they are not in navigation or in any observation, and no action can target
  them. Doors, routes, spawns, the exit and every object's approach stay clear; `python -m tools.dressing_report`
  checks this.

`python -m tools.gameplay_polish_demo` plays a call-free demonstration match with these rules.

## Saving and replaying

Every finished match writes a folder `runs/<YYYYmmdd-HHMMSS>/`:

- `runtime_manifest.json`: the configuration, the policy and artifact hashes, the provider and model ids (never a
  credential), the game code fingerprint, the outcome and the totals;
- `telemetry_match01.jsonl`: every decision;
- `replay_match01.json`: the replay bundle, scrubbed, and refused if anything credential-like would be written;
- `live_summary.json`.

`--no-replay` skips the bundle. Play never writes into the frozen policy folders (`workshop/policies`,
`workshop/policy_data`), and the runtime refuses an output folder inside them.

```bash
python -m spy_vs_ai.replay runs/<run>/replay_match01.json                    # watch it in a window
python -m spy_vs_ai.replay runs/<run>/replay_match01.json --headless --verify  # check it
python -m spy_vs_ai.replay runs/<run>/replay_match01.json --view research --speed 2
```

A replay makes no provider call. It runs under an offline guard: no network, no provider client or SDK, and
credential-like variables removed. It rebuilds the recorded game with no agent, re-issues the recorded engine inputs,
and checks the engine events, the final spy states and the ranking: **REPLAY REPRODUCED** or **REPLAY DIVERGED**.
Long stretches where the world was frozen while a model thought are shortened on screen. How the bundle works is in
[ARCHITECTURE.md](../project/ARCHITECTURE.md#replay-format).

**Example matches.** [examples/replays/](../../examples/replays/README.md) holds seven replays, one per idea: REAL,
the imitator's loops, a Hybrid escalating on disagreement, the new levels, Hybrid-v2, the trap rules, and two LLM
saboteurs. Five of them are frozen research matches, already re-simulated:

```bash
python -m spy_vs_ai.replay examples/replays/02_b2_loop_pathology.json --view research
```

With the full research archive (`workshop_runs/`, which is not part of the lightweight public repository), any frozen
research match can be re-simulated directly, with `python -m spy_vs_ai.replay --historical-examples` and
`--historical <match_dir>`. Each spy gets back the actions its model returned in that match, and the match is shown
only if every historical engine event and the ranking are reproduced exactly.

## Scientific status and compatibility labels

Each spy carries two labels, shown on the setup screen, by `--check`, in its panel and in the manifest:

- its **scientific status** from the frozen research release. For example, Hybrid-v1 is CONFIRMED / GENERALIZED
  RESEARCH BASELINE, and Hybrid-v2 is MORE EFFICIENT — NOT SAFELY PROMOTABLE UNDER CONFIRMATORY RULE (14.6% fewer
  provider calls in the final trial);
- a **compatibility label** for this match. Running in the runtime is not scientific validation:
  - **EVALUATED:** this exact combination of policy, provider, model, level and execution rules was evaluated;
  - **COMPATIBLE / NOT EVALUATED:** it runs through the same controller contract but was never evaluated (the reasons
    are listed);
  - **EXPERIMENTAL:** it depends on an adapter that never ran against a live provider (Anthropic direct);
  - **FAKE PROVIDER:** no model at all;
  - **ENGINE BASELINE:** the scripted bot.

One match is never a measure of model quality, and playable-mode matches are not comparable to the frozen
experiments.
