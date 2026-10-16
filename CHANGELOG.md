# Changelog

Release notes for the public repository. The intellectual history of the project, from the first prototype to the
playable runtime, is told in [docs/research/STORY.md](docs/research/STORY.md); the documentation starts at
[the guide](docs/README.md).

## Unreleased - planned v1.0.0 (first public release)

- **The game and its runtime:** `python -m spy_vs_ai.play` opens a visual setup screen. Each spy gets its own policy,
  provider, model and personality, set there, on the command line or in a YAML file. There are play and research
  views and a playable or evaluation mode, and the enhanced gameplay adds door traps, trap variants and room dressing.
- **Policies:** scripted, B1, B2-v1, Hybrid-v1 / R2.1, B2-v2, Hybrid-v2 / R2.1 and REAL LLM. The frozen surrogate
  and router artifacts are included byte for byte and hash-checked against the frozen release.
- **Providers:** Amazon Bedrock, OpenAI direct, Anthropic direct (experimental: never run against the live API),
  Ollama, and a fake provider for offline use.
- **Traces and replays:** every match saves a manifest, decision telemetry and a replay bundle under `runs/`.
  `python -m spy_vs_ai.replay` replays a match offline and verifies it reproduces. Seven example replays are in
  `examples/replays/`.
- **Research:** the research story, results and conclusions, the 33 frozen claims, an experiment index and the
  paper (`paper/SPY_VS_AI_PAPER.md`). The final statuses are unchanged: Hybrid-v1 / R2.1 is CONFIRMED / GENERALIZED
  RESEARCH BASELINE, and Hybrid-v2 / R2.1 is MORE EFFICIENT BUT NOT SAFELY PROMOTABLE UNDER THE CONFIRMATORY RULE.
- **Reproducibility:** `python -m tools.validate_public_repo` checks this tree offline. The full paper reproduction
  artifact is distributed separately (not yet published).
- **Not yet decided:** the license. Until one is added, all rights are reserved by the author.
