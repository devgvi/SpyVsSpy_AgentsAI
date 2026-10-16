# Providers and models

A spy has four independent settings: **policy** (how it decides), **provider** (where inference runs), **model** (the
provider's own model id) and **personality**. Only the provider-backed policies use a provider: `hybrid_v1`,
`hybrid_v2` (only when the router escalates) and `real_llm` (every decision). `scripted`, `b1`, `b2_v1` and `b2_v2`
are call-free: they read no credential and open no connection, and the runtime refuses provider settings for them.

```bash
python -m spy_vs_ai.play --list-providers        # credential mechanism and options per provider
python -m spy_vs_ai.play --list-models bedrock   # known ids (any id is accepted)
python -m spy_vs_ai.play --provider-status       # which credential sources are present (names only, never values)
```

Rules that hold for every provider:

- Credentials are read only for the providers a match actually uses, and only by that provider's adapter.
- No credential ever goes in a configuration file or on the command line: the configuration validator refuses
  option names or values that look like secrets.
- Telemetry, manifests and replays are scrubbed with the values of the selected providers' credentials and checked
  for credential-shaped strings before they are written. A key is never printed or stored.
- The model id is passed through unchanged. Lists and the setup screen offer known ids; "other model id..." (setup)
  or `--black-model` / `--white-model` take any id.

## Amazon Bedrock (`bedrock`)

The research provider. Both research models ran here: Claude Sonnet 5.5 and GPT-6.1 Sol.

- **Credentials:** the normal AWS credential chain (shared profile, SSO, environment, instance role) or
  `AWS_BEARER_TOKEN_BEDROCK`. The variables the adapter may read: `AWS_BEARER_TOKEN_BEDROCK`, `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_PROFILE`, `AWS_DEFAULT_PROFILE`, `AWS_REGION`,
  `AWS_DEFAULT_REGION`, `BEDROCK_REGION`, `AWS_CONFIG_FILE`, `AWS_SHARED_CREDENTIALS_FILE`.
- **Options:** `region` (default `us-east-1`), `aws_profile`, `timeout_s` (default 120), `temperature`
  (`0` or `provider-default`; Claude 5.x models on Bedrock reject an explicit temperature), `request_fields`.
- **Models:** any Bedrock model or inference-profile id, for example `us.anthropic.claude-sonnet-5-5`,
  `us.openai.gpt-6.1-sol`, `us.anthropic.claude-opus-5-5`. Which ids your account can use:
  `aws bedrock list-inference-profiles --region us-east-1`.
- **Status:** `--provider-status` reports whether a credential source is present. It never reads a value and never
  calls Bedrock.

```bash
python -m spy_vs_ai.play \
  --black-policy hybrid_v1 --black-provider bedrock --black-model us.anthropic.claude-sonnet-5-5 \
  --black-option aws_profile=my-profile --black-option region=us-east-1 \
  --white-policy real_llm --white-provider bedrock --white-model us.openai.gpt-6.1-sol --white-option aws_profile=my-profile
```

```yaml
black:
  policy: hybrid_v1
  provider: bedrock
  model: us.anthropic.claude-sonnet-5-5
  personality: cautious
  provider_options:
    aws_profile: my-profile
    region: us-east-1
```

**Through Bedrock vs direct.** `provider: bedrock` with `model: us.openai.gpt-6.1-sol` (or an Anthropic id) uses AWS
credentials only. It is a different provider from OpenAI direct or Anthropic direct, with its own credentials,
endpoints and error behaviour; the research results hold for the Bedrock path.

## OpenAI direct (`openai`)

- **Credentials:** `OPENAI_API_KEY`, read only when a spy uses `openai`. No AWS or Anthropic variable is read.
- **Options:** `timeout_s` (default 60), `temperature` (`0` or `provider-default`).
- **Models:** the OpenAI API's own model ids (no built-in list: type the id).
- **Status:** presence of `OPENAI_API_KEY` only.

```bash
python -m spy_vs_ai.play --black-policy real_llm --black-provider openai --black-model <openai-model-id> \
  --white-policy b2_v1
```

```yaml
white:
  policy: real_llm
  provider: openai
  model: <openai-model-id>
  personality: racer
```

## Anthropic direct (`anthropic`)

- **Credentials:** `ANTHROPIC_API_KEY`, read only when a spy uses `anthropic`.
- **Options:** `timeout_s` (default 60), `temperature`.
- **Models:** the Anthropic API's own model ids (type the id).
- **Transport:** the runtime's own Messages API transport (`/v1/messages`). The older engine client posted an
  OpenAI-style body to an endpoint the Anthropic API does not serve.
- **Status:** labelled **EXPERIMENTAL**: this adapter has never run against the live API.

```yaml
black:
  policy: real_llm
  provider: anthropic
  model: <anthropic-model-id>
  personality: opportunist
```

## Ollama (`ollama`)

- **Credentials:** none. A local server, default `http://localhost:11434`.
- **Options:** `base_url`, `timeout_s`.
- **Models:** what the server has installed (`ollama pull <model>`); `--list-models ollama` and the setup screen read
  them from the server's `/api/tags`. Examples: `qwen`, `gemma4`.
- **Limitations:** small local models produce more invalid outputs, so more repairs and fallback actions; long
  matches need enough context length. The Strands runtime's Ollama path refuses models without tool capability
  (`qwen:latest`); the playable runtime uses the native controller.

```bash
ollama serve &   # then: ollama pull qwen
python -m spy_vs_ai.play --config configs/demo_ollama_vs_bedrock.yaml
```

## Fake (`fake`, offline demos and tests)

No network, no credential, no model: a seeded picker of legal actions (`fake-legal`, or `fake-slow` with a delay;
options `latency_s`, `mode`, `fail_every`, `seed`). It exercises the whole runtime (escalations, THINKING, telemetry,
replay) for free and is labelled **FAKE PROVIDER** everywhere: nothing it does says anything about a model.
`configs/demo_offline.yaml` uses it.

## What was evaluated, and what only runs

The runtime labels every spy before the match starts (setup screen, `--check`, the sidebar, the manifest):

| Label | Meaning |
|---|---|
| **EVALUATED** | this policy / provider / model / level / execution combination was part of a frozen research trial |
| **COMPATIBLE / NOT EVALUATED** | it runs through the same controller contract, but was never evaluated (each reason is listed) |
| **EXPERIMENTAL** | it depends on an adapter that has never run against a live provider (Anthropic direct) |
| **FAKE PROVIDER** | no model at all |
| **ENGINE BASELINE** | the scripted bot, never a research policy |

Scientifically evaluated combinations (all others are runtime-compatible but unevaluated):

| Policy | Provider / models | Levels | Evaluated in |
|---|---|---|---|
| `real_llm` | Bedrock: Claude Sonnet 5.5, GPT-6.1 Sol | 1-3 | the REAL arm of Stages 3-5 |
| `hybrid_v1` | Bedrock, the surrogate identity's own model (sonnet55 → Sonnet 5.5, gpt61sol → GPT-6.1 Sol) | 1-3 | Stage 3 M4D, Stage 4C, Stage 5C |
| `hybrid_v2` | Bedrock, the identity's own model | 2-3 | Stage 5C (Amendment 02) |
| `b2_v1` | none | 1-3 | Stage 3 M3 / M4D, Stages 4C and 5B (through Hybrid R0) |
| `b2_v2` | none | 1-3 | Stage 5B (call-free closed loop) |
| `b1` | none | 1 | Stage 3 M3 |

An EVALUATED label also needs the research execution: `--mode evaluation` and the 300 s mission time
(`configs/demo_evaluated_hybrid_v2_vs_real.yaml`). Playable mode, the enhanced gameplay rules, another level, another
time limit, another provider or model all make a spy COMPATIBLE / NOT EVALUATED - and even an EVALUATED runtime match
is one match, not a scientific result.

**Surrogate identity.** The B1 / B2 surrogates were trained per research model and personality
(`sonnet55:<personality>`, `gpt61sol:<personality>`). A surrogate or hybrid spy picks its identity from the Bedrock
model id; for any other model it uses `sonnet55` unless `--black-identity` / `--white-identity` says otherwise. A
hybrid whose identity is not its escalation model's own is labelled accordingly.

**Costs.** Hybrid spies call the provider only on escalation (in Stage 5C about a third to two fifths of REAL's
calls); `max_provider_calls` (config) / `--black-max-provider-calls` caps a hybrid spy's escalations per match.
REAL spies call on every decision. The match summary and `runtime_manifest.json` report calls and tokens per spy.
