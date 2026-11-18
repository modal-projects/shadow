---
pretty_name: ShadowBench
license: other
license_name: shadowbench
license_link: LICENSE.md
language:
- en
task_categories:
- reinforcement-learning
- image-text-to-text
tags:
- benchmark
- rl-environment
- spatial-reasoning
- embodied-reasoning
- hands
- shadows
- agents
- tool-use
- trajectories
size_categories:
- 10K<n<100K
configs:
- config_name: leaderboard
  default: true
  data_files:
  - split: results
    path: data/leaderboard/leaderboard.parquet
- config_name: episodes
  data_files:
  - split: medium_effort
    path: data/episodes/medium_effort.parquet
  - split: high_effort
    path: data/episodes/high_effort.parquet
  - split: max_effort
    path: data/episodes/max_effort.parquet
- config_name: steps
  data_files:
  - split: medium_effort
    path: data/steps/medium_effort/*.parquet
  - split: high_effort
    path: data/steps/high_effort/*.parquet
  - split: max_effort
    path: data/steps/max_effort/*.parquet
- config_name: transcripts
  data_files:
  - split: medium_effort
    path: data/transcripts/medium_effort/*.parquet
  - split: high_effort
    path: data/transcripts/high_effort/*.parquet
  - split: max_effort
    path: data/transcripts/max_effort/*.parquet
- config_name: tasks
  data_files:
  - split: test
    path: data/tasks/tasks.parquet
---

# ShadowBench

![ShadowBench: a benchmark and RL environment for spatial reasoning about the body](assets/shadowbench.jpg)

ShadowBench is a benchmark and RL environment for spatial reasoning about the body. A model sees a hand-shadow plate
from a nineteenth-century book and poses two articulated 3D hands, standing between a point lamp and a screen, one
tool call at a time, until their shadow matches the animal. The environment refuses any pose a hand could not make,
and the model sees its shadow only when it asks to look.

This repository holds the environment, the 20 certified targets and all 360 episodes from the
[ShadowBench report](https://perpetualresearch.si/shadowbench): every tool call, every image the model received, the pose of both hands after every
call and the hidden score at each step. The [ShadowBench explorer](https://huggingface.co/spaces/perpetual-research/shadowbench-explorer) shows the results and replays any
episode in the browser.

## Task

**Targets.** Twenty plates from three parlour books of hand shadows: fifteen from Henry Bursill's *Hand Shadows to be
Thrown upon the Wall* (1859), four from William Hilliar's *Novel Hand Shadows* (c. 1900) and one from Félicien
Trewey's *The Art of Shadowgraphy* (c. 1900), all in the public domain. In every plate the drawn hands hide part of
their own shadow, so we traced the visible shadow from each scan, completed the hidden parts, and a person corrected
each drawing in a vector editor and certified it. Ten further plates were left out because they repeat a subject or
cannot be completed with confidence.

**Stage.** A point lamp 2.5 m from a screen of 0.8 × 0.6 m, scored at 512 × 384 pixels, and two hands between them,
raised at the start with their palms toward the screen. The hands are the hand-and-forearm model from Ultraleap's
Unity plugin; the same skin is drawn, casts the shadow and is checked for collisions. Each hand has 23 joint
controls (each finger's knuckles and spread, the thumb's two base axes and two hinges, wrist flexion and side bend,
forearm roll) plus a position and an orientation: 58 numbers for the pair.

**Rules.** Every joint has a conservative range, a finger cannot spread as far when it is bent, and the skin may pass
neither through itself nor through the other hand. Motion between poses is checked in small steps, and one call may
change a joint by at most 45°, move a hand by at most 15 cm or turn it by at most 45°. A call that breaks a rule is
refused: it uses a call, returns a short reason and leaves the hands where they were.

An episode allows 100 calls, of which at most 32 may be `look`, and at most 3 book peeks. The model acts only through
these tools:

| Tool | What it does |
| --- | --- |
| `plan` | Say what the silhouette depicts and which parts of the hands will form which features. Required before the first hand movement; can be revised; recorded, never scored. |
| `set_fingers`, `move_hand`, `turn_hand` | Change the pose: set joint angles, translate a hand, rotate a hand about its wrist. |
| `undo` | Reverse the last accepted change. |
| `describe` | Joint angles and their ranges, hand positions and directions, as numbers. |
| `look` | One image of four panels: the target, the shadow, their difference (red missing, blue extra) and the hands, seen from the front, back, side, top or at an angle. |
| `peek_book` | The original book page with the author's drawing of the hands. Each peek multiplies the final score by 0.94. |
| `submit` | End the episode. The environment also submits when the call budget runs out. |

**Score.** The score compares the shadow with the target by area and by outline. Most of it is the overlap divided
by the union; the rest measures whether the edge of the shadow lies where the edge of the target lies, within one,
two and four pixels, which is the part that ears, beaks and legs earn:

$$
\text{score} = 100 \times \left(0.55\,\text{IoU} + 0.45\,F_{\text{outline}}\right) \times \left(0.9 + 0.1\,h\right) \times \left(0.95 + 0.05\,w\right) \times 0.94^{\,p}
$$

where *h* is the share of the target's holes (an eye, say) that the shadow leaves open, *w* is how close the worse
wrist is to neutral by the wrist-position step of the RULA ergonomic assessment, and *p* is the number of peeks. A
plate is solved when the overlap, the outline and the holes each reach 95%. The model never sees its score.

## Results

Six models ran at medium, high and maximum reasoning effort, one episode on every plate, which makes
360 episodes. No model solved a plate. Intervals are 95% bootstrap intervals over plates; costs are
the token cost of an episode in US dollars.

| Model | Medium | High | Max [95% CI] | Cost per episode (medium · high · max) | Time per episode at max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Claude Opus 5.5 | 39.8 | 43.6 | 48.8 [46.3, 51.9] | $0.85 · $1.27 · $7.13 | 54 min |
| Claude Sonnet 5.5 | 17.3 | 40.6 | 48.8 [46.5, 51.1] | $0.030 · $0.41 · $5.23 | 60 min |
| GPT-6.1 Terra | 43.4 | 44.4 | 46.4 [43.7, 49.1] | $0.18 · $0.21 · $0.44 | 22 min |
| GPT-6.1 Sol | 41.4 | 41.9 | 44.7 [41.9, 47.5] | $0.32 · $0.39 · $0.88 | 40 min |
| Claude Haiku 5.5 | 28.5 | 35.6 | 41.3 [37.7, 45.1] | $0.012 · $0.036 · $0.34 | 54 min |
| GPT-6 Luna | 31.6 | 35.0 | 35.1 [31.5, 38.4] | $0.041 · $0.059 · $0.098 | 24 min |

Three references put the scores in context:

| Reference | Score [95% CI] | What it is |
| --- | ---: | --- |
| Featureless blob | 56.9 [53.4, 60.6] | The convex hull of each target: the right area in the right place, with none of the animal's features. |
| Blind search (CMA-ES) | 36.2 [34.5, 37.9] | An evolutionary optimizer over the same 58 pose numbers under the same anatomical rules, told the score of each of about 6,000 poses per plate. |
| Hands at rest | 12.1 [9.1, 15.1] | The starting pose: both hands raised, palms to the screen. |

The best models, Claude Opus 5.5 and Claude Sonnet 5.5 at maximum effort, are tied and average a little under half
of a perfect score, which is less than the featureless blob earns: every shadow covers roughly the right area and
misses the animal's features. The best single episode, Claude Opus 5.5's giraffe at
max effort, scores 68.3.

**Reasoning effort.** More effort helps the Claude models far more than the GPT models, mostly because the Claude
models use more of the episode: at medium effort Claude Sonnet 5.5 submitted without moving a hand in 13
of its 20 episodes. The GPT models use almost the whole budget at every effort.

| Model | Max minus medium [95% CI] | Plates better at max |
| --- | ---: | ---: |
| Claude Sonnet 5.5 | +31.5 [+25.6, +36.7] | 19 of 20 |
| Claude Haiku 5.5 | +12.8 [+9.5, +15.9] | 19 of 20 |
| Claude Opus 5.5 | +9.0 [+5.8, +12.5] | 19 of 20 |
| GPT-6 Luna | +3.5 [−0.4, +7.4] | 14 of 20 |
| GPT-6.1 Sol | +3.3 [+0.1, +6.6] | 14 of 20 |
| GPT-6.1 Terra | +3.0 [+0.5, +5.4] | 16 of 20 |

**What the episodes show.**

- *A collage, not a puppet.* Of the 295 final poses in which both hands cast a shadow, only
  7 hold the hands together, as the books do. Across these poses the hands are a median
  17.8 cm apart, separated mostly along the light (a median 24.7 cm
  in depth), so each hand casts its own part of the animal at its own size.
- *Moves are poorly aimed.* Of the 17,007 hand moves the environment accepted, 41% made the
  shadow better, 32% made it worse and 26% left the score within
  half a point. In 135 of the 360 episodes the submitted shadow scores more than two points below the
  best one the same episode reached.
- *The knowledge is there.* About two thirds of the plans written before any look at the book (169 of
  254) name the right animal, and 318 of the 360 episodes consult
  the book page, almost always before moving a hand.
- *Anatomy pushes back.* The environment refused 786 moves for hands passing through each other,
  687 for a hand passing through itself, 726 for exceeding the per-call limits and
  171 for joint ranges.

The [report](https://perpetualresearch.si/shadowbench) covers these results in detail, with figures and replays of individual episodes.

## What this repository contains

| Config | Splits | Rows | One row is |
| --- | --- | ---: | --- |
| `leaderboard` (default) | `results` | 21 | A model at one effort: mean score with its 95% interval, the components of the score, cost and time per episode, and for maximum effort the gain over medium on the same plates. Three `reference` rows hold the blob, the blind search and the hands at rest. |
| `episodes` | efforts | 360 | One episode: final score and its components, tool use and refusals, plans and the closing message, tokens, cost and time, the hidden score after every call, where the hands ended up, the final pose and its shadow. |
| `steps` | efforts | 30,354 | One call: the tool and its arguments, whether it was accepted, the text and image the model received, the pose of both hands after the call and its hidden score. |
| `transcripts` | efforts | 360 | One conversation: the system and user messages, each model response with its tool calls, token counts and cost, and each tool result linked to its step. |
| `tasks` | `test` | 20 | One plate: the target and its vector drawing, the book page `peek_book` shows, the first observation, the exact prompts and tool schemas, and the reference scores. |

The splits of `episodes`, `steps` and `transcripts` are the three reasoning efforts, `medium_effort`, `high_effort`
and `max_effort`, each with 120 episodes: six models on the 20 plates. `episode_id` (`<model>@<effort>/<task_id>`)
joins the tables and matches the episode ids of the report's player.

- A pose is `{q, positions, quaternions}`: 46 joint angles in radians (23 per hand, right hand first, in the order
  of `shadowbench.native_environment.ALIASES`, relative to the model's relaxed rest pose), the two wrist anchors in
  metres and the two hand orientations as quaternions (w, x, y, z). World axes: x right, y up, z toward the lamp at
  (0, 0, 2.5); the screen is the plane z = 0. `shadowbench.pose.Pose.parse` reads it.
- `steps.hidden_score` is the score of the pose after every call, including the peek penalty so far, and
  `score_before_peeks` is the same without it. The model never sees either. They drive the report's score curves and
  can serve as a dense reward. Step 0 is the start of the episode: the opening message and the first observation.
- `steps.image` is the image the model received from that call: `look` returns the four panels, `peek_book` the book
  page. Images are 1024 × 768 PNGs, exactly as the model saw them.
- In `transcripts`, `step` links each tool result to its row in `steps`, which holds its image. Parallel calls in one
  response can run in a different order from the one in which they were issued; `step` follows the order in which
  they ran. One of the 29,995 tool calls has no step: it came after the episode had ended and was answered "Episode already ended." 13 Claude responses reached the 32,000-token output limit and ended without a call.
  Reasoning text was not recorded.
- Each episode ended with `submit`, except 2 that reached the call budget. `cost_usd` is the token cost of an
  episode in US dollars; `input_tokens` includes cache reads and writes.

```python
from datasets import load_dataset

board = load_dataset("perpetual-research/shadowbench", "leaderboard", split="results").to_pandas()
episodes = load_dataset("perpetual-research/shadowbench", "episodes", split="max_effort").to_pandas()
print(episodes.groupby("model_name")["score"].mean().sort_values(ascending=False))

# every call of Claude Opus 5.5 posing the giraffe, with the image it received and its hidden score
steps = load_dataset("perpetual-research/shadowbench", "steps", split="max_effort", streaming=True)
for s in steps.filter(lambda s: s["episode_id"] == "claude-opus-5.5@max/hilliar_giraffe"):
    print(s["step"], s["tool"], s["accepted"], s["hidden_score"], s["arguments"][:80])
    if s["image"] is not None:
        s["image"].save(f"giraffe_{s['step']:03d}.png")
```

## Use the environment

```bash
hf download perpetual-research/shadowbench --repo-type dataset --local-dir shadowbench
cd shadowbench
uv sync                # add --extra openenv for the OpenEnv server, --extra search for the blind search
```

Each tool call is one step that returns text and sometimes an image, and the score arrives once, at the end of the
episode.

```python
from shadowbench import SYSTEM_PROMPT, TOOL_SPECS, USER_PROMPT, create, tasks

print(tasks())                                  # the 20 certified plates
env = create("bursill_bunny")                   # the certified target the evaluation used
image = env.observe()                           # the first observation: target | shadow | difference | hands
out = env.call("plan", {"subject": "rabbit", "arrangement": "index and middle fingers as the ears"})
out = env.call("set_fingers", {"hand": "right", "joints": {"ring.mcp": 40, "little.mcp": 40}})
out = env.call("look", {"view": "front"})       # out.text, out.image: what the model receives
env.metrics()["score"]                          # the hidden score of the current pose
out = env.call("submit", {})
env.final["score"], env.final["reward"]         # reward = score / 100
```

`env.steps` holds the same per-call log as the `steps` table, and `create(task_id, log_dir)` also writes it to disk
with every image. `SYSTEM_PROMPT`, `USER_PROMPT` and `TOOL_SPECS` are the exact prompts and tool schemas of the
evaluation. Collision checks are CPU-bound; set `OPENBLAS_NUM_THREADS=1` when running several environments at once.

The environment also runs as an [OpenEnv](https://github.com/openenv/OpenEnv) server with an MCP tool interface,
one isolated episode per session and the final reward through its rubric:

```bash
uv sync --extra openenv
uv run shadowbench-openenv --port 8010
```

The published code is the evaluated environment: its fingerprint, `b49f3ca99522…`, hashes the environment's source,
the hand model and the exact versions of NumPy, SciPy, Pillow, OpenCV and python-fcl that `uv sync` installs, and it
equals the `env_fingerprint` recorded in every episode.

## Evaluate a model

`scripts/run_eval.py` runs the benchmark protocol against any OpenAI-compatible endpoint with tool calling and image
input: OpenAI, OpenRouter, Anthropic's OpenAI-compatible endpoint, vLLM, SGLang and others. It runs every plate once,
resumes where it stopped and prints the mean score with a 95% interval that compares directly with the leaderboard.

```bash
OPENAI_API_KEY=... uv run python scripts/run_eval.py --models gpt-6.1-sol --effort medium

OPENROUTER_API_KEY=... uv run python scripts/run_eval.py --base-url https://openrouter.ai/api/v1 \
    --api-key-env OPENROUTER_API_KEY --models claude-opus-5.5=anthropic/claude-opus-5.5 \
    --extra-body '{"reasoning": {"effort": "medium"}}'

uv run python scripts/run_eval.py --base-url http://localhost:8000/v1 --api-key-env NONE --models your-model
```

The protocol is the evaluation's: the same system prompt, opening message and tools; if a model ends its turn
without submitting it is asked to continue, at most three times, and then the pose it reached is scored. Chat
Completions tool messages carry text only, so the images that `look` and `peek_book` return are sent in a user
message right after the tool results. Each episode writes the same per-call log as the `steps` table.

## Reproduce the results

```bash
uv run python scripts/analyze.py       # the tables above and the leaderboard config, from the episode tables
uv run python scripts/replay.py        # re-simulate a recorded episode from the steps table and check every call
uv run python scripts/search_baseline.py --task hilliar_giraffe   # the blind search (uv sync --extra search)
```

`analyze.py` reproduces every published score, interval and paired comparison exactly. `replay.py` replays the
recorded calls of an episode through a fresh environment and checks that each call is accepted or refused as
recorded, returns the same text and leaves the same pose, with the same hidden score and pixel-identical images
(`--episodes` picks episodes, `--all` replays all 360).

## Limits

Each model ran once per plate and effort, so small differences, such as the tie at the top, are not resolved. The
targets are an illustrator's drawings of shadows, so whether real hands, or this hand model, can reproduce each one
exactly is not known, and the best scores are lower bounds on what the hands can do. The hand model has conservative
joint limits, and real hands, with softer skin and more flexible thumbs, can make some shapes it forbids. Whether a
model peeked at the book was its own choice, so comparisons of episodes that peeked with those that did not are
observational. Six models from two laboratories were evaluated, and there is no human baseline yet.

## License

Code: MIT. Episodes, analysis and the certified target drawings: CC BY 4.0. The book plates are in the public
domain. The hand model comes from Ultraleap's Unity plugin under the Apache License 2.0. Details in
[LICENSE.md](LICENSE.md).

## Citation

```bibtex
@misc{perpetual2026shadowbench,
  title  = {ShadowBench: Posing Hands to Cast Shadow Puppets from Nineteenth-Century Books},
  author = {{Perpetual Research}},
  year   = {2026},
  url    = {https://perpetualresearch.si/shadowbench}
}
```

Perpetual is a research lab crafting RL environments for creative intelligence. Contact: founders@vibemotion.ai ·
[@perpetual_si](https://x.com/perpetual_si)
