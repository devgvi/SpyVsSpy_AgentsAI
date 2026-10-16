# Spy vs AI

**Spy vs AI** is a two-spy embassy game in which each spy is played by the AI you choose: a scripted bot, a large
language model, or a small model trained to imitate one. It is also a research project on one question:

> **How much frontier-LLM reasoning does an autonomous agent actually need?**

The game began as a tribute to the spirit of **Spy vs. Spy on the Commodore 64**: two rival spies racing through rooms,
searching furniture, setting traps and sabotaging each other. Spy vs AI keeps that playful idea, but turns it into a
modern AI-agent environment where different models, policies, runtimes and reasoning strategies can be compared.


<p align="left">
  <img src="docs/media/spy_vs_spy_1984_c64.png"
       alt="Original Commodore 64 version"
       width="50%">
</p>

![Split view of a real match: Black (Claude Sonnet 5.5) on the left, White (Claude Opus 5.5) on the right walking into Black's explosive door trap, with each spy's live panel on the right](docs/media/hero_door_trap.png)


### How the project was built

The project itself was also AI-assisted, with different models used for different roles:

- **GPT-5.6 Sol in ChatGPT** was the main research/design partner for the overall project skeleton: architecture, milestone and experiment design, test specifications, analysis framing, documentation, and the paper structure.
- **Claude Sonnet**, through Claude Code, was the primary coding model used to implement the game, runtime, experiments, tests, and tooling.
- Visual assets were generated with **gpt-image-2.5-sunburst**, then post-processed, normalized, resized, and packaged with **Pillow** and the local `spy_asset_factory` pipeline.
- For the scientific agent experiments themselves, the main provider-backed model families were **Claude Sonnet 5.5** and **GPT-6.1 Sol**, both accessed through **Amazon Bedrock**. Historical visual/demo matches also used **Claude Opus 5.5**.

For the scientific agent experiments themselves, the main provider-backed model families were **Claude Sonnet 5.5**
and **GPT-6.1 Sol**, both accessed through **Amazon Bedrock**. Historical visual/demo matches also used
**Claude Opus 5.5**.

This was not a small "few-match" experiment. Across the research line we ran **well over 1,600 matches**. For the
hard-example distillation phase specifically, B2-v2 was evaluated in **864 call-free closed-loop matches**, then
confirmed in **192 real provider-backed matches**. The distilled surrogate did not learn from the match count itself:
it learned from promoted **hard decisions** collected during earlier LLM escalations, and those later match sets tested
whether the distilled behavior actually held up in closed loop.

![Split view of a real match: Black (Claude Sonnet 5.5) on the left, White (Claude Opus 5.5) on the right walking into Black's explosive door trap, with each spy's live panel on the right](docs/media/hero_door_trap.png)

Two spies break into the same embassy. Each one searches the furniture for the mission items, works out the safe
code, sets traps for the other, and tries to escape first. Above is a real match: Claude Sonnet 5.5 (Black) against
Claude Opus 5.5 (White), both saboteurs. Black has rigged the ambassador's office door, and White walks into it.

![Animated highlights of the same match: White triggers the door trap, Black cracks the safe and escapes first](docs/media/match_highlights.gif)

[Watch the 22-second clip (MP4)](docs/media/match_highlights.mp4). These images are re-rendered offline from the
match's replay, with zero model calls.

**New here? Start with [the guide](docs/README.md).** It explains the game, the screen, the seven kinds of player, and
what the research found, then points you to the right page.

## Install

Python 3.12, from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
python -m spy_vs_ai.play                                            # the setup screen: choose everything, START MATCH
python -m spy_vs_ai.play --black-policy b2_v1 --white-policy b2_v2  # call-free: two imitators, no account needed
python -m spy_vs_ai.play --config configs/demo_offline.yaml         # every policy offline, with a fake model
```

A real LLM against a Hybrid on Amazon Bedrock, billed by your provider:

```bash
python -m spy_vs_ai.play --seed 42 \
  --black-policy real_llm  --black-provider bedrock --black-model us.anthropic.claude-sonnet-5-5 --black-personality saboteur \
  --white-policy hybrid_v1 --white-provider bedrock --white-model us.openai.gpt-6.1-sol --white-personality cautious
```

- **Providers:** `bedrock`, `openai`, `anthropic`, `ollama` (local) and `fake`; any model id is accepted. See
  [docs/play/PROVIDERS_AND_MODELS.md](docs/play/PROVIDERS_AND_MODELS.md).
- **Research mode:** press M during a match to see each decision's inputs, the imitator's proposal, the router's
  reason and the model's answer.
- **Replays:** every match is saved in `runs/` and can be watched again offline with
  `python -m spy_vs_ai.replay runs/<run>/replay_match01.json`. Seven [example replays](examples/replays/README.md),
  one per idea, are included.

Everything else (the command line, configuration files, modes, the trap rules) is in
[docs/play/PLAYING_THE_GAME.md](docs/play/PLAYING_THE_GAME.md).

## What the project learned

- **Local action fidelity is not closed-loop behavioural fidelity.** A cheap imitator (B2) predicted individual LLM
  decisions well offline. But on its own in the game, it drifted into states the LLM never visited and into repetition
  loops.
- **Selective escalation works.** Hybrid-v1 lets B2 decide and escalates a state to the LLM only when router R2.1
  sees a repetition loop or a strong B1/B2 disagreement. It recovered much more REAL-like behaviour with 17.5% of
  REAL's provider calls on Level 1. Under environment shift (Levels 2 and 3, never trained on), it generalized better
  than pure B2, at about 36% of REAL's calls (preregistered: decision A, confirmed generalization). Final status:
  **CONFIRMED / GENERALIZED RESEARCH BASELINE**.
- **Distillation cut provider calls by 14.6%.** Hybrid-v2 retrains B2 on the hard decisions the LLM had to make. It
  used 14.6% fewer real provider calls than Hybrid-v1 (95% CI 6.8%-22.0%, P(fewer calls) = 0.9998).
- **But Hybrid-v2 was not safely promotable.** Three of the four preregistered behavioural non-inferiority checks
  passed. On the fourth (Level-2 state visitation), the estimate was +0.0008, but its 95% upper bound of 0.0182
  exceeds the frozen margin of 0.0168. **Decision B: provider reduction is real but Hybrid-v2 is not safely
  promotable.** Final status: **MORE EFFICIENT BUT NOT SAFELY PROMOTABLE UNDER THE CONFIRMATORY RULE**.
- **The distillation gains were heterogeneous across personalities.** Hybrid-v1 → Hybrid-v2 provider calls in the
  confirmatory trial:

  | personality | Hybrid-v1 | Hybrid-v2 |
  |---|---|---|
  | cautious | 802 | 581 |
  | opportunist | 967 | 913 |
  | racer | 732 | 809 |
  | saboteur | 543 | 296 |

  This pattern is consistent with uneven hard-example coverage, but the experiment did not establish that imbalance
  as the causal mechanism.

**MAIN PROVIDER-BACKED RESEARCH LINE CLOSED.** Hybrid-v1 / R2.1 is the default research policy, and Hybrid-v2 stays
runnable. The research is told in [docs/research/STORY.md](docs/research/STORY.md), with every number and its source
in [docs/research/RESULTS.md](docs/research/RESULTS.md). The full study is
[the paper](paper/SPY_VS_AI_PAPER.md). Follow-up ideas are new research, not unfinished work
([remaining uncertainty](docs/research/RESULTS.md#remaining-uncertainty)).

## Reproducing the research

This repository is lightweight: the game, the runtime, the frozen policy artifacts it plays, curated results and
example replays. **The full paper reproduction artifact is distributed separately** and is not published yet. It
holds the frozen reports, manifests, registries, bootstrap outputs, datasets and reproduction scripts.
[docs/research/RESULTS.md](docs/research/RESULTS.md#reproducibility) explains what you can check here and what needs
that artifact.

```bash
python -m tools.validate_public_repo     # offline: artifact hashes, every policy, replays, links, scans, tests
```

## Repository map

```text
spy_vs_ai/          the playable runtime: play.py, replay.py, runtime/, providers/, ui/
src/                the game engine (frozen)
workshop/           the frozen research code and policy artifacts the runtime uses
data/               level 1 and the item catalogue
assets/             game art (assets/README.md)
configs/            example configurations
examples/replays/   seven example replays
docs/               the guide (docs/README.md) and play/, research/, project/, media/
tools/              validation and export, README media, replay curation, call-free demos
tests/              the test suite (python -m pytest -q tests)
```

## License

Released under the [Apache License 2.0](LICENSE). You may use, copy, modify and redistribute this project, provided you keep the license text and copyright notices.
