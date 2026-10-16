# LinkedIn post — Spy vs AI

I started this project for a slightly nostalgic reason.

I wanted to build a small tribute to **Spy vs. Spy on the Commodore 64** — two rival spies running through rooms, searching furniture, setting traps, sabotaging each other and trying to escape first.

Then I made one change: **what if each spy were controlled by an AI agent?**

That simple idea turned into a much bigger experiment.

The project itself was AI-assisted too, but different models had very different jobs.

**GPT-5.6 Sol in ChatGPT** was my main partner for the overall skeleton of the project: architecture, milestone and experiment design, test specifications, analysis framing, documentation, and eventually the paper structure.

**Claude Sonnet**, through Claude Code, did most of the implementation work: the game, runtime, experiment code, tests, and tooling.

The visual assets were generated with **gpt-image-2.5-sunburst**, then post-processed and normalized with **Pillow** and a local asset-factory pipeline.

Inside the scientific matches themselves, the main provider-backed model families were **Claude Sonnet 5.5** and **GPT-6.1 Sol**, both through **Amazon Bedrock**. Some historical/demo matches also used **Claude Opus 5.5**.


In **Spy vs AI**, each spy can be controlled by a scripted policy, a full LLM, or a smaller model trained to imitate one. You can choose the provider, the model and the personality independently for each spy, then watch the match play out visually while the runtime records the decisions, escalations, latency, tokens, fallbacks and replay trace.

The game itself became the test bed.

A spy has to move through a partially observable embassy, inspect and search objects, remember what it has already seen, find keys and codes, place and disarm traps, react to the other spy, complete the mission and escape.

The research question eventually became:

> **How much frontier-LLM reasoning does an autonomous agent actually need?**

At first, the answer looked simple.

We trained a cheap surrogate policy, B2, to imitate LLM decisions. Offline, it looked good. It predicted individual actions surprisingly well.

But once we let it control the agent by itself, something important happened:

**good local imitation did not mean good agent behaviour.**

The surrogate drifted into states the LLM rarely visited. It got stuck in more repetition loops. Small local differences changed the next room, the next observation, the next available action — and those differences compounded.

That became one of the main lessons of the project:

**model behaviour ≠ agent behaviour ≠ match outcome.**

So instead of asking the LLM to make every decision, we built a Hybrid policy.

The cheap surrogate makes the normal decisions. A router escalates only selected hard states to the frontier model — especially when two cheap policies strongly disagree.

That worked much better.

Under environment shift, the Hybrid stayed substantially closer to the full LLM than the pure surrogate, while using only about **36% of the provider calls** in the confirmatory generalization experiment.

Then we tried one more step.

Every escalation gives us a useful training example: a state the cheap policy found difficult, plus the decision made by the LLM.

So we distilled those hard examples back into a new surrogate, B2-v2.

This took a lot more than a handful of demo games. Across the research line we ran **well over 1,600 matches**. In the distillation phase alone, we used **864 call-free closed-loop matches** to test the B2-v2 candidate, followed by **192 real provider-backed confirmatory matches**. The model was not simply trained on all those matches: it learned from the *hard decisions* collected during earlier LLM escalations, and the later matches were used to test whether that distilled behaviour actually survived in closed loop.

In the final provider-backed confirmatory experiment, the new Hybrid used **14.6% fewer real provider calls** than the previous Hybrid, with a 95% confidence interval of **6.8%–22.0%** and **P(fewer calls) = 0.9998**.

But the result was not uniformly positive.

Three of four behavioural non-inferiority checks passed. One did not clear the frozen confidence bound, even though the point difference was almost zero.

And the efficiency gains varied a lot by personality:

- cautious: **802 → 581** calls
- opportunist: **967 → 913**
- racer: **732 → 809**
- saboteur: **543 → 296**

That was another useful lesson: **hard-example distillation can reduce aggregate cost while still helping some behavioural regimes much more than others.**

So the final conclusion is deliberately nuanced:

**Hybrid-v1 is the confirmed generalized research baseline.  
Hybrid-v2 is more efficient, but not safely promotable under the frozen confirmatory rule.**

The deeper takeaway for me is broader than this game:

> The question may not be “Which frontier model should control the agent?”  
> It may be “Which decisions actually require frontier reasoning — and what is the cheapest mechanism capable of handling the rest without degrading the agent in closed loop?”

There are several directions I could have pushed further:

- add an intermediate small decision model between the surrogate and the frontier LLM;
- learn a value-aware router that escalates based on the expected cost of being wrong;
- balance hard-example collection across personalities instead of optimizing only the pooled average;
- reproduce the same phenomena in a completely different environment, not just another level;
- isolate the **agent harness** itself — memory, observation format, fallback, runtime and execution semantics — to measure how much of what we call “model behaviour” actually comes from the system around the model.

I stopped the main research line there on purpose. The project now has a playable visual game, a reproducible research history, frozen results, replays, and a paper draft.

And yes — you can still just launch two spies, give them different models and personalities, and watch them try to electrocute, gas, trap and outsmart each other.

That part is still fun.

#AI #LLM #AIAgents #GenerativeAI #MachineLearning #AgenticAI #Research #GameAI
