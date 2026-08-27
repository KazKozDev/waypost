# waypost

A local LLM provider router. One OpenAI-compatible endpoint, behind it —
the free tiers of a dozen providers and a local model as the end of the
cascade.

The goal is not "pick the best model" but **not to hit the quota**. Free
tiers are limited not by money but by RPM/TPM/RPD, so the central object
of the system is accounting for remainders, not a list of models.

```
                 ┌──── CONTROL PLANE (background) ───────────┐
                 │  discovery → probe → health → purge        │
                 │            ↓ write to the registry         │
                 └────────────────┬───────────────────────────┘
                                  │ in-memory snapshot
client → [ingress] → [policy] → [cache L0/L2] → [router] → [executor] → provider
                        │                        │             │
                  PII / injections          filter+scoring   degradation ladder
                  → privacy: strict         + bandit         → local model
```

## Quick start

```bash
pip install -e ".[dev]"
cp .env.example .env          # fill in keys (none is fine too)
python -m waypost.server
```

### Launch by double-click (macOS)

Two `.command` files live in the root — Finder launches them by double
click, opening a normal Terminal window:

| File | What it does |
|---|---|
| `waypost-server.command` | starts the server; if it is already running, says so and does not spawn a second one |
| `waypost-chat.command` | interactive chat; if the server is not responding, tells you to start it |

Both find the project root from their own path, use `.venv/bin/python`
(or the system `python3` if absent) and honor `ROUTER_PORT`. The server
script distinguishes three port situations: its own live server, a
foreign process and a free port — and in each says what to do.

To keep the server handy, drag `waypost-server.command` to the Dock.

### macOS menu-bar app

A native menu-bar app in the Ollama mold: an icon in the top bar, runs
the server from `.venv` as a subprocess, shows status
(🟢 running / 🟠 starting / 🔴 failed) and holds the address
`http://127.0.0.1:8080/v1`.

```bash
./macos/build.sh             # builds build/Waypost.app
open build/Waypost.app        # double-click works too
```

Menu: address (copied), open stats / models, show the log
(`var/server.log`), restart, stop, quit. To autostart — System Settings
→ General → Login Items → `+`.

Any OpenAI client works by changing `base_url`:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")

client.chat.completions.create(
    model="auto",                       # the choice is left to the router
    messages=[{"role": "user", "content": "hi"}],
)
```

Without a single key the server still works: all traffic goes to the
local model. This is not a degradation but a normal mode — convenient to
start from.

## What it can do

| Component | Status |
|---|---|
| OpenAI-compatible facade, streaming | ready |
| Registry from a YAML manifest + auto-discovery | ready |
| Quota accounting (rpm/rpd/tpm) per **key**, survives a restart | ready |
| Capability filter + scoring + bandit | ready |
| Degradation ladder (7 steps), circuit breaker | ready |
| Multiple keys per provider, rotation on 429 | ready |
| TTFT hedging with a quota budget | ready |
| Retry idempotency | ready |
| L0 exact cache · L2 semantic + cross-encoder | ready |
| Prefix discipline, provider-side cache points | ready |
| Cascade verifier (schema, tool-calls, looping, language) | ready |
| PII detector → forces `privacy: strict` | ready |
| Guard: injections in untrusted blocks | ready |
| Local `/v1/embeddings` with microbatching | ready |
| `/v1/rerank` for RAG chunk selection | ready |
| Batch API (`/v1/batches`) with a queue and a window | ready |
| Prometheus metrics (`/metrics`), telemetry in SQLite | ready |
| Control plane: discovery / probe / health / purge | ready |
| Context compression (safe / LLMLingua-2) | present, off by default |
| L1 classifier (embeddings + head) | skeleton, the head trains on your logs |
| Offline: traffic clustering, reward scoring | scripts ready |

## The degradation ladder

The order is strict: each next step costs more than the previous.

1. **Retry on the same key** — only 5xx/timeout, exponential backoff
   with jitter, at most two attempts.
2. **Another key of the same provider** — the model is the same, the
   warmed prefix is not lost. Quota is counted per key, so the step is
   real.
3. **Another provider** — the same model at another host or a neighbor.
4. **Another model of the same tier.**
5. **A local model** — the end of the cascade, always available.
6. **An approximate answer from the semantic cache** with a lowered
   threshold and an `approximate: true` marker. Serving it as fresh is
   not allowed.
7. **A structured error** with diagnostics in the `router` block.

On top: a circuit breaker (closed → open → half-open with cooldown
doubling) and hedging — if the first candidate is silent longer than its
TTFT, a second one starts, the first to answer wins, the loser is
canceled **and returns the reserved quota**. The hedge is bounded to 5%
of requests: it is a second quota spend, not a free speedup.

## API extensions

On top of the OpenAI spec (not forwarded to providers):

| Field | Meaning |
|---|---|
| `privacy: "strict"` | forbid data-training providers → local only |
| `latency_class: "batch"` | speed does not matter, save the quota; hedging is off |
| `session_id` | sticky routing: do not lose the warmed prefix |
| `no_cache` | bypass the cache |
| `idempotency_key` | a retry after a network drop does not charge the quota twice (also via the `Idempotency-Key` header) |

The response contains a `router` block with the decision diagnostics:
provider, model, key index, tier, the fallback path, the attempt count,
cache status, the hedging and escalation facts. Without it, routing
debugging turns into guesswork.

## Endpoints

| Method | Path | Why |
|---|---|---|
| POST | `/v1/chat/completions` | the main one, including SSE streaming |
| POST | `/v1/embeddings` | local embeddings, 25 ms microbatching |
| POST | `/v1/rerank` | RAG chunk selection with a cross-encoder |
| POST | `/v1/batches` | an offline job (inline or JSONL) |
| GET | `/v1/batches[/{id}[/output]]` | status and results |
| POST | `/v1/batches/{id}/cancel` | cancel |
| GET | `/v1/models` | what is available, with the data policy |
| GET | `/v1/stats` | quotas, cache, cascade, hedges, jobs |
| GET | `/v1/discovery` | what auto-discovery found |
| GET | `/v1/pricing` | paid or free, and how it is proven |
| GET/POST | `/v1/jobs`, `/v1/jobs/{name}/run` | control plane |
| GET | `/metrics` | Prometheus |
| GET | `/health` | breakers |

## CLI

```bash
waypost                       # start the server
waypost keys                  # interactively enter all keys
waypost keys list             # which keys are set (masked, with source)
waypost keys keychain --purge # move keys from .env to macOS Keychain
waypost models                # models from the manifest
waypost discover              # auto-discovery of free models
waypost probe                 # probing (TTFT, limits, capability)
waypost stats                 # stats of the running server
waypost jobs [--run probe]    # control-plane background jobs
waypost batch submit job.jsonl
waypost batch status batch_…  # and output / cancel / list
waypost report                # telemetry report (--json/--csv)
waypost train-head --data …   # train the L1 classifier head
```

## Providers and keys

Free tiers only (all without a card, key by email). There are no paid
providers in the manifest — not "disabled", but removed: keeping in the
registry what you are forbidden to use means keeping the risk.

| Provider | Key in `.env` | Where to get | Free limits |
|---|---|---|---|
| OpenRouter | `OPENROUTER_API_KEY` | openrouter.ai/workspaces/default/keys | 20 RPM / 50 RPD |
| Groq | `GROQ_API_KEY` | console.groq.com/keys | 30 RPM, LPU speed |
| Cerebras | `CEREBRAS_API_KEY` | cloud.cerebras.ai | 1M tokens/day |
| SiliconFlow | `SILICONFLOW_API_KEY` | cloud.siliconflow.cn | 30 RPM / 60K TPM |
| Zhipu GLM | `ZHIPU_API_KEY` | open.bigmodel.cn | GLM-4.7-Flash 200K ctx |
| NVIDIA NIM | `NVIDIA_API_KEY` | build.nvidia.com/settings/api-keys | ~40 RPM, no daily cap |
| Mistral | `MISTRAL_API_KEY` | console.mistral.ai/api-keys | ~1B tokens/month |
| Gemini | `GEMINI_API_KEY` | aistudio.google.com/app/apikey | 10 RPM / 250 RPD |

**Multiple keys of one provider.** Add a suffix: `GROQ_API_KEY_2`, `_3` …
This is a separate quota and the second step of the ladder — on a 429 the
router switches to it without changing the model.

**Keychain instead of `.env`.** `waypost keys keychain --purge` moves keys
to the macOS system key store. Lookup order: environment variable →
Keychain (`service="waypost"`, `account=NAME`).

By default **only free models** are used (`ROUTER_FREE_ONLY=true`).

### How paid vs free is decided

A mechanism decides (`waypost/pricing.py`), not human memory, and it
decides **fail-closed: unknown → paid**. Evidence by strength:

| Source | What it is | When it fires |
|---|---|---|
| `billed` | the provider billed: `usage.cost > 0` in the response | on the hot path, in `Executor` |
| `api` | prices from `GET /models` (`pricing.prompt` and the like) | at auto-discovery |
| `id` | a `:free` / `-free` suffix in the identifier | at auto-discovery |
| `manifest` | the `free:` field in `config/providers.yaml` | at registry load |
| `unknown` | declared nowhere | → treated as **paid** |

Merge rules: any "paid" overrides any "free"; on a tie, the stronger
source wins. A bill (`billed`) is irreversible within the process
lifetime.

Practical consequences:

* **`free:` in the manifest is required.** A provider without a
  declaration drops out of selection entirely under
  `ROUTER_FREE_ONLY=true`; the server logs a warning at startup. The
  default used to be "free" — that is exactly how a paid model could end
  up in the plan.
* **There are no paid providers in the manifest at all.** The first line
  of defense is not a filter but the absence of a candidate: OpenAI,
  xAI, DeepSeek and Fireworks are removed from `config/providers.yaml`.
  The lower lines stay in case a free tier stops being free.
* **A provider that starts charging is demoted by itself.** If a
  non-zero price appears in `GET /models`, discovery marks the model paid
  and the router stops selecting it — no manifest edit needed.
* **The last line is the bill in the response.** A non-zero `usage.cost`
  (OpenRouter is asked for it explicitly) marks the model paid at once:
  one paid call is the cost of discovery, the next one bypasses it.
* **An explicit request for a paid model does not pass.** Even if a paid
  offering appears in the registry (e.g. via auto-discovery), naming it
  in `model:` under `free_only` does not bypass the filter, it falls
  back to the auto-plan.

```bash
waypost models          # only what the router is allowed to use
waypost models --all    # paid too, with a reference to the verdict source
curl localhost:8080/v1/pricing   # a full breakdown: who, what, and how proven
```

### Keys in CI

Keys are stored as repository secrets and plugged into the `discover`
job: Settings → Secrets and variables → Actions → New repository secret,
the secret name = the variable name. A provider without a secret is
skipped; an invalid key does not fail the pipeline.

## How decisions are made

**Policy is a filter, not a score.** A score can be outweighed: a
low-quality model wins if the others are busy. A rule cannot be
outweighed — a request with PII physically does not go to the cloud.

**Filter and scoring are separate.** If capability is weighed, a model
without tool calling gets a high speed score and breaks the request.

**Error asymmetry.** Underestimating complexity costs more than
overestimating it: the first costs an escalation and double quota spend.
So low classifier confidence raises the tier rather than lowering it.

**Scoring:**

```
score = w_q · quality(task, model)      # bandit over the manifest
      − w_l · normalize(ttft)
      − w_b · share of the daily quota
      + w_c · affinity to a warmed prefix
      + w_r · measured reliability
```

Weights depend on `latency_class`: interactive cares about TTFT, the
background about saving the quota that interactive will need.

## Guard and privacy

Injections live not in the user message but in **untrusted blocks**:
tool-call results and retrieved RAG chunks. The router is the single
point through which all of this flows, so the detector is here, and it
scans the untrusted zone (`role: tool`, messages named
`context*`/`document*`). The user message is not scanned: "ignore
previous instructions" from the user is just a request.

What is caught: instruction overrides, attempts to extract the system
prompt or keys, coercion to call tools, hidden text (zero-width, HTML
comments, long base64 blocks). Regexes are the base; Llama Prompt Guard
2 plugs in via `model_scorer` and catches rephrasings.

A finding by default does not block the request but **wraps** the block
in an explicit "this is data, not instructions" fence
(`ROUTER_GUARD_ACTION=block` switches the behavior to a refusal).

**PII → `privacy: strict`.** Email, phone, Luhn-valid card, passport,
INN, IP force the local model. `trains_on_data: true` is the default for
all cloud providers: free tiers are paid for with your data.

**The request log is off.** Texts are needed to train the head, cluster
traffic and for offline reward, but these are the most sensitive data
in the system: `ROUTER_ENABLE_PROMPT_LOG=true` is enabled deliberately.

## Caching

- **L0, exact** — SHA-256 of the canonicalized request. Only with
  `temperature = 0`: at a non-zero temperature the user asks for variety,
  and the same answer is not optimization but a bug.
- **L1, provider-side prefix** — needs no code, only discipline.
  `waypost/prefix.py` keeps a stable block order (`system → tools →
  static → history`), places cache points at the end of the static zone
  for providers with an explicit mode (`prompt_cache: explicit`) and
  computes a prefix hash for sticky routing. Switching providers zeroes
  the warmed prefix — so the binding is both by `session_id` and by the
  hash of the static zone.
- **L2, semantic** — a bi-encoder fetches candidates, a cross-encoder
  confirms. Protection against false hits: a namespace by (tier,
  language, prefix hash); numbers and proper nouns are checked **with
  order** ("compare Moscow and Paris" ≠ "compare Paris and Moscow" —
  the sets match, the answers differ); requests with a recency marker
  are not cached; only answers that passed the verifier go into the
  cache. Enabled by `ROUTER_ENABLE_SEMANTIC_CACHE=true`.

A realistic expectation: L0 gives 5-15% hits, L2 — another 10-25% on
repetitive tasks and near zero on creative ones.

## The cascade verifier

A cheap model answers first, the answer is checked, and only on a
failure is the request escalated to a higher tier. Checks in increasing
cost: empty / truncated / looping → JSON and schema → tool-call name and
arguments → the answer language did not match the request language → a
refusal instead of an answer → context groundedness (HHEM, optional) →
instruction following (NLI, optional).

The cascade pays off while the failure share stays below ~30-40%. That
share is **measured** by `/v1/stats → cascade.escalation_rate` (with
`profitable` next to it), not assumed. Above the threshold — the cascade
must be turned off, not tuned.

## Efficiency

**Embedding microbatching.** A 25 ms window glues single calls into one
encoder pass. For chat, batching is a TTFT loss; for embeddings the
opposite: they come in batches, and the latency of one vector matters to
no one. The average batch size is visible in
`/v1/stats → embeddings.avg_batch`.

**Batch API.** An offline job with a window up to 24 h runs in the
background with `latency_class=batch`, bounded parallelism and an item
returned to the queue if it hit a quota. State is in SQLite — a job of
ten thousand requests survives a restart.

**Reranking over compression.** Sending 5 of 20 found chunks is usually
better than compressing all 20: `/v1/rerank`.

**Context compression** (`ROUTER_COMPRESS_MODE`) is off by default on
purpose: it breaks the prefix cache, drops accuracy on code and
structures, and only pays off above a length threshold. The `safe` mode
is a deterministic cleanup of duplicates and whitespace in the dynamic
zone; `llmlingua` is LLMLingua-2. The system prompt and tool
descriptions are never touched.

**The local model parallelism cap** (`concurrency: 1` in the manifest)
is not tuning: mlx-lm holds weights in shared memory, and two heavy
generations at once contend for memory, not speed.

## Observability

```bash
curl localhost:8080/v1/stats     # quotas, cache, cascade, hedges, jobs
curl localhost:8080/metrics      # Prometheus
waypost report --days 7          # telemetry report
```

Metrics: `waypost_requests_total`, `waypost_latency_ms` (a histogram
per provider), `waypost_quota_remaining`, `waypost_breaker_state`,
`waypost_cache_hit_rate`, `waypost_escalations_total`,
`waypost_hedges_total`, `waypost_guard_total`, `waypost_tokens_total`.

The terminal logs say what happened:

```
ROUTE groq/llama-3.3-70b task=code tier=L src=rules attempts=1 path=groq/… → groq 780ms
CACHE exact hit task=chat tier=S
VERIFY fail reason=language_mismatch → escalate
HEDGE openrouter/… silent > 1.5s → in parallel groq/…
GUARD override in untrusted block → fence
attempt groq/…[key0] failed: [Verdict.SWITCH] 429: rate limit
```

## Control plane

Background jobs independent of each other (a failed one does not bother
its neighbors):

| Job | By default | What it does |
|---|---|---|
| `discovery` | every 12 h | looks for new free models at providers |
| `probe` | off | measures TTFT, limits, working capabilities |
| `health` | every 5 min | pulls measured reliability into the registry |
| `purge` | every 6 h | cleans expired idempotency records |

`probe` is off by default because it spends quota. Run it by hand
(`waypost probe --apply`) or enable `ROUTER_ENABLE_PROBE=true`. A
measurement always overrides the manifest: documentation lies more often
than `x-ratelimit-*` headers.

## Training on your own logs

```bash
ROUTER_ENABLE_PROMPT_LOG=true python -m waypost.server   # accumulate traffic
python -m scripts.cluster_logs --clusters 8              # what it consists of
python -m scripts.reward_score --sample 50 --apply       # a dense task×model matrix
python -m scripts.train_head --data var/labels.jsonl     # the L1 head
ROUTER_ENABLE_L1_CLASSIFIER=true python -m waypost.server
```

`cluster_logs` shows the empirical task classes instead of ones invented
in advance, and doubles as a drift detector. A cluster where the rules
give one class but escalations are many is the first candidate for
relabeling.

`reward_score` runs the accumulated requests through several models and
compares the answers with a judge (a local model by default — free and
private), giving the bandit a dense matrix instead of sparse online
signals.

## Manual test: interactive chat

```bash
python -m scripts.chat                 # normal mode
python -m scripts.chat --stream
```

After each answer it shows the router's decision. In-chat: `/model
NAME`, `/privacy strict`, `/temp 0.7`, `/stream on|off`, `/clear`,
`/quit`.

## Tests

```bash
python -m pytest -q          # 165 tests
```

None of them hits the network: upstreams are replaced with
`httpx.MockTransport`. What is checked is behavior, not line coverage —
a 429 switches without a retry, a 5xx retries twice and switches, a 400
fails at once, key exhaustion moves to the second key of the same
provider, `privacy: strict` does not touch the cloud under any scoring,
a canceled hedge returns the quota, the local model does not run two
generations at once.

## Installing extras

```bash
pip install -e ".[ml]"        # real embeddings (model2vec)
pip install -e ".[rerank]"    # the bge-reranker-v2-m3 cross-encoder
pip install -e ".[analysis]"  # HDBSCAN for log clustering
pip install -e ".[compress]"  # LLMLingua-2
```

Everything is optional. Without `[ml]` embeddings are computed by the
lexical fallback — honestly visible in `/v1/stats → embeddings.backend`,
and you cannot tune the semantic-cache threshold against it.

## Further

1. **An L2 classifier** as a runtime teacher for uncertain cases —
   currently an extension point `classify_l2()`.
2. **A native Anthropic adapter** — the only protocol where
   `cache_control` is placed explicitly; currently only supported through
   OpenAI-compatible endpoints (`prompt_cache: explicit`).
3. **HHEM for RAG** — the interface in `Verifier(grounding=…)` is ready,
   the model plugs in with one line; a measurement of the check cost is
   needed.
4. **The provider Batch API discount** — the queue is its own, but jobs
   are still run as ordinary requests, not via provider batch
   endpoints.

The order is exactly this, not by interest: the first three fix what
already works, rather than add new reasons to fail. Every added model is
memory, warmup, updates, and one more point of failure.