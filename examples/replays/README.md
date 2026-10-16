# Example replays

Seven matches, one per idea. Each one replays offline: no agent, no provider call, no credential read. Each was
checked to give REPLAY REPRODUCED, meaning the replayed engine events, final state and ranking are identical to the
recording.

```bash
python -m spy_vs_ai.replay examples/replays/03_hybrid_disagreement_escalation.json             # in a window
python -m spy_vs_ai.replay examples/replays/03_hybrid_disagreement_escalation.json --view research
python -m spy_vs_ai.replay examples/replays/03_hybrid_disagreement_escalation.json --headless --verify
```

In the window: Space pause, + / - speed, M research view (decision inspector and history: the surrogate's proposal,
the router's reason, the model's answer), O overview, P split view, Esc quit.

| File | What it demonstrates |
|---|---|
| `01_real_example.json` | **REAL**, where every decision is a model call: Claude Sonnet 5.5 (Black) vs GPT-6.1 Sol (White), both cautious, on Level 2, seed 9009. Black escapes in 121 s. 108 provider calls. This is the behaviour the surrogates try to reproduce. |
| `02_b2_loop_pathology.json` | **The closed-loop pathology**: the same level, seed and start, with B2-v1 alone (no escalation). Black heads for the same doorway 46 times, and 70 of its 160 decisions repeat one of its previous four (2 of 61 in `01`). Both spies time out at 300 s, with 0 provider calls. Good offline prediction does not mean good closed-loop behaviour. |
| `03_hybrid_disagreement_escalation.json` | **Hybrid-v1 / R2.1**: B2-v1 decides, and router R2.1 escalates hard states to the model. Of its 63 escalations, 44 came from a strong B1/B2 disagreement and 19 from a repetition loop. Level 2, seed 9200, cautious. Black escapes. 63 provider calls for 249 decisions. |
| `04_environment_shift.json` | **Environment shift**: Hybrid-v1 / R2.1 on Level 3, a level the surrogates were never trained on (Stage 4C). White escapes. 59 escalations: 53 for disagreement, 6 for repetition. |
| `05_hybrid_v2_efficiency.json` | **Hybrid-v2 / R2.1** (B2-v2, retrained with promoted hard examples, plus the same R2.1) in the same match setup as `03`. 46 provider calls instead of 63. One match proves nothing; the 14.6% average reduction comes from the Stage 5C confirmatory trial. |
| `06_enhanced_trap_gameplay.json` | **Enhanced gameplay**: door traps and trap variants (playable-runtime rules, downstream of the frozen policies). B2-v1 saboteur vs B2-v2 cautious, call-free, Level 1, seed 3. Door traps are placed and one goes off on White. |
| `07_llm_saboteurs_sonnet_vs_opus.json` | **Two REAL LLM saboteurs** in playable mode: Claude Sonnet 5.5 (Black) vs Claude Opus 5.5 (White) on Bedrock, Level 1, seed 42. Black's explosive door trap catches White and Black escapes first. 65 provider calls when played. The README screenshots and clip are rendered from it. |

Provenance:
- `01` to `05` are frozen research matches (Stages 4C and 5C), re-simulated without a provider from their recorded
  model answers.
- `06` is the call-free gameplay-polish acceptance match.
- `07` is a match played on the runtime.

All seven went through `tools.curate_example_replay`. It turned absolute local paths into project-relative ones and
removed the AWS profile names from the provider options. Frames, engine events and verification data are unchanged,
and each bundle's `curation` record lists what was changed.

Your own matches are saved in `runs/<YYYYmmdd-HHMMSS>/replay_match01.json` and replay the same way. A replay bundle
holds no credential. The format is described in
[docs/project/ARCHITECTURE.md](../../docs/project/ARCHITECTURE.md#replay-format).
