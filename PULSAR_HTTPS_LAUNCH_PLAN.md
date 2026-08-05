# Pulsar MCP — HTTPS Arm Launch Plan

**Status:** proposal, not yet approved
**Scope:** PR #37 (`feat/docker-https-cloudrun-mcp`) and the work needed to make Pulsar's MCP server safely reachable over HTTPS
**Basis:** triage of the 16 existing review findings on PR #37 against branch HEAD, plus 83 new findings from a 11-agent review sweep, plus two adversarial passes over the resulting plan
**Author:** synthesized review; every load-bearing claim carries a confidence marker (see [Appendix A](#appendix-a--evidence-confidence))

---

## Part 0 — Exposure check: **RUN, CLEAR**

The question this had to answer: `deploy-cloudrun.yml` fires on `push: main` and `workflow_dispatch` is enabled. The plan assumes nothing is publicly exposed, and the *reason* is an accident — `roles/run.developer` lacks `run.services.setIamPolicy`, so `--allow-unauthenticated` should have failed. That reasoning has a hole: if anyone widened `pulsar-mcp-deployer`'s role to get past a permission error (`roles/editor`, `roles/owner`, `roles/run.admin` — what people actually do), the deploy succeeded and the service is public *right now*, with unsandboxed `ingest_dataset`.

**Checked 2026-08-05. Nothing is deployed and nothing is exposed.** Three independent confirmations:

| Check | Result |
|---|---|
| Artifact Registry images | **0 items, 0 MB.** No image was ever pushed — so no Cloud Run revision can exist. Decisive. |
| `deploy-cloudrun.yml` on the default branch | **Absent** (`gh` returns HTTP 404). The workflow exists only on this feature branch, and pushes there don't match `branches: [main]`. It has never fired. |
| `pulsar-mcp-deployer` project roles | Exactly `artifactregistry.writer`, `iam.serviceAccountUser`, `run.developer` — **never widened**. The escalation branch didn't happen. |

The infrastructure *was* provisioned (`setup_gcloud_wif.sh` ran at `2026-08-04T17:55`): project `pulsar-mcp-prod` is ACTIVE, the AR repo exists, `run.googleapis.com` and `artifactregistry.googleapis.com` are enabled, and all three GitHub secrets are set. So the pipeline is fully wired and one merge to `main` away from firing. **That is the actual state: armed, not fired.**

**Two findings that fall out of this check:**

1. **The WIF script is not dead on arrival after all** — one of the review's blockers claimed the bare `gcloud projects create` yields a parentless, billing-disabled project so `gcloud services enable` hard-fails under `set -e`. Both APIs are enabled and the AR repo was created, so the script ran to completion against a project that had billing. Downgrade that finding from blocker to "requires a billed project; document the prerequisite." *(Whether the project was pre-created or the org supplies a default billing account is unresolved.)*
2. **`sidney@krv.ai` cannot read Cloud Run in `pulsar-mcp-prod`** — `run.services.list`, `.get`, and `.getIamPolicy` are all denied, and the account holds **no direct role binding** on the project. So the person expected to operate this deployment currently cannot inspect it, and could not have run the exposure check above to completion without the Artifact Registry side-channel. Fix the operator's own IAM before Gate 3, or post-deploy verification and incident response both have no eyes.

Re-run before any merge to `main` that carries the deploy workflow:

```bash
P=pulsar-mcp-prod; R=us-central1
gcloud artifacts docker images list "us-central1-docker.pkg.dev/$P/pulsar-mcp" --project="$P"
gcloud run services list --project="$P" --region="$R"
gcloud run services get-iam-policy pulsar-mcp --project="$P" --region="$R" 2>/dev/null | grep allUsers
gcloud run services describe pulsar-mcp --project="$P" --region="$R" \
  --format='value(spec.template.metadata.annotations)' | grep -i invokerIamDisabled
```

The last command matters independently: `--no-invoker-iam-check` makes a service fully public via a *service-spec field* rather than an IAM binding, so `get-iam-policy` shows no `allUsers` and an IAM audit reports the service as private. Grepping the annotation is the only way to see it.

---

## Part 1 — The plain-English version

*For a developer who has not read the 99 findings and wants to know what is actually going on.*

### What was asked for, and what was built

The PR's goal was reasonable: make the Pulsar MCP server reachable over HTTPS so a remote agent can use it. What got built was the infrastructure to do that — a container, a TLS reverse proxy, a Cloud Run deploy pipeline, and IAM provisioning.

The containerization is genuinely good work. The problem is that "make it reachable" was scoped as an infrastructure task, and it is not one.

### The one thing wrong, stated once

**Pulsar's MCP server is a single-user desktop program. The PR deploys it as a multi-user web service. Nothing in between was changed.**

That is the whole finding. Every one of the six blockers is a consequence:

Concretely, the server assumes:

- **One user.** Session state lives in a Python dictionary capped at 3 entries. Cloud Run sends up to 80 concurrent requests to one instance. The 4th user silently deletes the 1st user's loaded dataset and fitted model — mid-workflow, with no error.
- **One process.** `dataset_id` and `run_id` are files on local disk. Cloud Run runs up to 100 instances by default. A handle minted on instance A returns "unknown handle" from instance B, non-deterministically.
- **Disk that persists.** The cache directory points at `/tmp`, which on Cloud Run is a RAM disk that counts against your 2 GiB memory limit and vanishes on every cold start.
- **A trusted caller.** `ingest_dataset("/etc/passwd")` works, by design — under stdio your MCP client already runs as you, so there is no boundary to cross. Once the server answers HTTP, that same code is an arbitrary file read. And `probe_columns` returns the file's contents back to the caller, so it is a complete exfiltration path, not just a disclosure.

None of that is a bug in the code. It is correct code for the deployment it was written for, being put in a deployment it was not.

### Why the PR cannot just be fixed in place

Two reasons, and the second is the one that matters.

**First:** it does not currently run. Every non-stdio transport crashes before binding — `mcp.run(allowed_hosts=[...])` passes an argument FastMCP 3.2.0's HTTP runner does not accept, so it raises `TypeError`. The test suite is green because the one test that covers this replaces `mcp.run` with a mock. So the entire subject of the PR has never executed, in CI or locally.

That is also why five other findings were invisible: with no HTTP path ever running, the session-collapse bug, the eviction DoS, and the path-read surface were all unreachable dead code. Fixing the crash is what makes them live. **That is why the security work is a prerequisite and not an improvement.**

**Second:** the PR is one PR doing six things — a transport change, a cache refactor, a container, a TLS proxy, a CI/CD pipeline, and IAM provisioning. The state and trust model was never *chosen*, because no single PR was ever *about* choosing it. Six reviews got bundled into one, so the architectural decision had nowhere to happen.

### What "production ready" actually requires

Five things, in dependency order. Each is a real gate, not a nice-to-have.

1. **Make the transports work** and stop shipping config that reads like a security control but isn't (`PULSAR_MCP_ALLOWED_HOSTS=*` appears in four places and does nothing).
2. **Confine the filesystem** — one data root, one resolver, for reads and writes both. Right now four separate write paths take a caller-supplied destination with no containment check.
3. **Bound the compute** — the sweep grid is unbounded caller-controlled work, and there is no cap on dataset size either.
4. **Require a caller identity**, and make sessions survive concurrent use.
5. **Then deploy** — private, single-instance, IAM-gated, no public URL.

Steps 1–4 are Python. Step 5 is infrastructure. The PR did step 5 first.

### The decision you actually have to make

**Which MCP client must work at launch?** Everything else follows from it, and it is not a technical question — it's a product one.

- **Claude Code / mcp-remote / Agent SDK** → a static bearer token or `headersHelper` works today, GA. Small change.
- **claude.ai or Claude Desktop connectors** → request-header auth is beta-gated ("contact Anthropic for early access"), so OAuth 2.1 is the only generally-available path. Weeks of work.

Pick wrong and you build the wrong mechanism. Nobody has picked yet.

### The honest shape of the launch

You are not shipping a multi-tenant SaaS. You are shipping **a private, single-instance, single-tenant analysis server that a handful of named operators reach through an authenticated tunnel.** That is a genuinely useful thing and it is achievable soon.

What it is not: durable across restarts, safe for two operators to share without seeing each other's data, or able to scale past one instance. Those are fine limitations — as long as they are written down as commitments rather than discovered later. The failure mode this plan is guarding against is not a missing feature; it is shipping something that *looks* multi-tenant because it has auth, and isn't.

### How long

**30–40 working days** across 11 PRs, of which roughly 6 days are GCP round-trips that cannot be parallelized. The first PR — the one that unblocks the current review — is 1–2 days.

---

## Part 2 — The state and trust model *(the missing spine)*

Both adversarial passes independently flagged the same thing as the single largest gap: four documents assert "single-tenant by construction" and none of them makes the decision. This section makes it. Everything downstream cites it rather than re-asserting it.

It belongs in `docs/source/userGuides/deployment.rst` and in PR 1's description, gated by a test that asserts the section exists and names all four elements.

### 2.1 State locality — what it is today

| Layer | Where it lives | Scope | Survives restart? |
|---|---|---|---|
| Session (`model`, `data`, `clusters`, embeddings) | `_sessions` OrderedDict, `session.py` | Process | No |
| Registry (datasets, runs, cluster assignments, uploads) | JSON files under `PULSAR_MCP_CACHE_DIR` | Instance filesystem | Only if the dir is durable |
| Registry object itself | Module-level singleton, `registry.py:427`, imported by 7 modules | Process | No |
| PCA embedding cache | Session, fingerprinted | Process | No |

**The invariant that matters:** two instances do not share state and do not reconcile. They are **disjoint universes, not eventually consistent.** A handle from one is meaningless to the other. There is no merge, no sync, and no error that says so — just `UNKNOWN_HANDLE`.

### 2.2 Trust boundary, per phase

| Phase | Reachability gate | In-container principal | Distinct principals | Tenancy |
|---|---|---|---|---|
| **0 — launch** | Cloud Run IAM + `gcloud run services proxy` | **None** — the proxy terminates auth | 0 | Single, by construction |
| **1 — next** | IAM + `JWTVerifier` on Google ID tokens | Yes, one shared `sub` | **1** | Single, by construction |
| **2 — later** | OAuth 2.1, per-human sign-in | Yes, per human | N | Genuinely multi-tenant |

### 2.3 The consequence nobody wrote down

**In Phase 0 and Phase 1, the number of distinct principals is 0 and 1 respectively. Therefore ownership enforcement is scaffolding, not a control.**

This is load-bearing and it changes the plan. An `owner` field on four record types, a principal threaded through 12 registry call sites across 7 modules, a salted dataset digest, and a per-principal cache namespace — all of it resolves to a constant while there is one principal. Worse, the tests fabricate two distinct principals, pass, and *certify isolation that does not exist in the deployment being shipped.* That is the exact defect class as the mocked `mcp.run` test: green while pinning a falsehood.

**So:** cut the ownership half of PR 5 and defer it to the OAuth phase, where it becomes meaningful. Keep the session-lifecycle half — that is urgent and real. And add the inverse test: one that *confirms* all operator tokens share one `sub`, so nobody later assumes otherwise.

The single-tenant property must be **documented and asserted**, not implemented and hoped for.

### 2.4 Options considered

| | Approach | Buys | Costs | Verdict |
|---|---|---|---|---|
| **(a)** | Pin single-instance: `--max-instances=1`, small `--concurrency`, in-process state, ceiling documented | Ships now; zero architectural risk; honest | One instance, no durability, no multi-tenancy | **Launch.** |
| **(b)** | Registry behind an interface (`LocalFs` \| GCS \| volume), sessions per-principal | Multi-instance; durable handles | Needs real compare-and-swap, not a filesystem — see below | **12-month target.** |
| **(c)** | Fully stateless, client-carried opaque handles | No server memory; scales freely | Rewrites the workflow contract; every call re-reads the dataset | Rejected — fights the product. |

**Why (b) is not a quick win.** The obvious version — mount a GCS bucket with gcsfuse — is a trap. The registry serializes on `fcntl.flock`, and Cloud Storage FUSE provides no file locking: *"the last write wins and all previous writes are lost."* The mutual exclusion the code assumes would be **silently absent**, turning a loud error into undetectable corruption. It also stages writes in memory, reintroducing the OOM the volume was meant to fix. Real (b) needs Firestore, Cloud SQL, or GCS objects with `ifGenerationMatch` — a separate, larger PR.

### 2.5 The seam to introduce now

So option (b) stays cheap later without building it today, PR 2 introduces exactly two things:

- **`pulsar/mcp/settings.py`** — one frozen dataclass from one validated `load_settings()`, replacing eight scattered `os.environ` reads (two of them at import time). This is what makes configuration injectable instead of ambient.
- **`pulsar/mcp/paths.py`** — one `_resolve_under_root()` used by every path-accepting call site.

That is the whole seam. **Do not** build a second registry backend, an ABC with one implementation, or a per-principal namespace. The module-level `registry` singleton stays (7 import sites); retiring it is part of (b), not a prerequisite for it.

### 2.6 Invariants that keep the model true

Break any of these and the documented model is silently false:

1. `--max-instances=1`. Not tuning — an architectural commitment.
2. Startup **refuses** `FASTMCP_STATELESS_HTTP=true`. Reachable by env var alone, no code change, and it destroys the workflow entirely (§4, DO-NOT list).
3. No gcsfuse-backed cache dir.
4. No `--session-affinity` — it is documented best-effort, broken by high CPU (which a sweep guarantees), and redundant at one instance. Shipping it implies a guarantee that does not exist.
5. `cache_dir` is **instance-local scratch**, never durable storage.

---

## Part 3 — The PR sequence

11 PRs. The current #37 gets rescoped in place as PR 1 rather than closed — it already holds three of the four files PR 1 needs and carries the review history.

| # | PR | Gates public exposure | Depends on | Size |
|---|---|---|---|---|
| 1 | Fix the FastMCP remote-transport boundary *(rescope #37)* | — | — | ~300 ln / 1–2 d |
| 2 | Path sandbox + settings module (read confinement) | ✅ | 1 | ~450 ln / 2–3 d |
| 3 | Write confinement + sweep cost budget + error envelopes | ✅ | 2 | ~450 ln / 2–3 d |
| 4 | Move blocking registry I/O off the event loop | — | — *(parallel)* | ~200 ln / 1 d |
| 4.5 | **Observability**: structured logs, telemetry middleware, `/health` + `/ready`, SIGTERM drain | ✅ | 4, 5 | ~400 ln / 2–3 d |
| 5 | Session lifecycle *(ownership half cut — see §2.3)* | ✅ | 4 | ~300 ln / 2 d |
| 6 | Container image, local-only, built in CI | — | 1, 2, 3 | ~350 ln / 2–3 d |
| 7 | docker-compose + Caddy, loopback default | — | 6 | ~200 ln / 1–2 d |
| 8 | Auth provider (`auth=` kwarg) | ✅ **the gate** | 1, 5 | ~400 ln / 4–6 d |
| 9 | Remote data ingress (upload gate + byte caps) | — | 2, 3 | ~250 ln / 1–2 d |
| 10 | Cloud Run deploy, IAM-only, no public URL | — | 6, 8, 9, 4.5 | ~300 ln / 4–5 d |

PRs 2, 3, 5, 8 must **all** land before any non-loopback binding is documented anywhere. PR 4 is independently valuable and parallelizable.

> **On the reviewer's own proposal** — *"land Docker + compose now, split Cloud Run into its own PR behind auth."* **Endorsed on the split, amended on what "compose now" may mean.** Compose may land as a *local, loopback-bound, single-tenant* container. It may not land as the `https://mcp.yourdomain.com/sse` endpoint the README currently walks the reader through: the Caddyfile has no auth of any kind, compose publishes `80:80`/`443:443`, and that path needs no GCP account — so it is reachable by *more* readers than the Cloud Run path the IAM blocker currently gates. Hence PR 6 depends on 2 and 3, not on 1 alone.

### PR 1 — Fix the FastMCP remote-transport boundary

**Goal:** every `--transport` the CLI advertises actually binds, and nothing in the repo pretends to be a security control.

**Mechanics:** rescope #37 in place. Remove the six deploy files; revert the README deploy section. Say in the PR body that `Dockerfile`/`.dockerignore` return in PR 6, `docker-compose.yml`/`Caddyfile` in PR 7, `deploy-cloudrun.yml`/`setup_gcloud_wif.sh` in PR 10 — otherwise reviewers seeing a diff that deletes `Dockerfile` will assume the work was thrown away.

| File | Change |
|---|---|
| `pulsar/mcp/server.py` | Delete `--allowed-hosts`, its env read, and `allowed_hosts` from `run_kwargs`. Extract `_build_run_kwargs(parsed)` and `_parse_args(argv)` for testability. Default `--host` → `127.0.0.1`. Move the `int()` port parse out of the argparse default into a validated post-parse step. Mark `sse` deprecated in `--help`. Refuse startup if `FASTMCP_STATELESS_HTTP` is truthy *(non-stdio only)*. Log transport / bind / auth-state at startup. |
| `pulsar/mcp/registry.py` | Replace import-time `_CACHE_DIR` + derived globals with a lazy `_cache_dir()`. |
| `pulsar/mcp/session.py` | Wrap `ctx.session_id` in `try/except RuntimeError` → `NO_SESSION_CONTEXT` envelope with `agent_action`. **Only this.** |
| `tests/test_mcp_server_patch.py` | Drop `allowed_hosts=["*"]` from the mocked assertion. Delete the `importlib.reload` (unnecessary once the cache dir is lazy). |
| `tests/test_mcp_transport_args.py` | **New.** Assert `set(_build_run_kwargs(...)) - {"transport"} <= set(inspect.signature(FastMCP.run_http_async).parameters)`, with an explicit guard that no parameter is `VAR_KEYWORD` — a future `**kwargs` sink would make the subset check vacuous. |
| `tests/test_mcp_remote_startup.py` | **New.** Per transport in `{sse, http, streamable-http}`: reserve port 0, `Popen`, poll `create_connection`, assert `proc.poll() is None`, surface child stderr via `pytest.fail`, then `fastmcp.Client` → `ping()` + `list_tools()`. Plus `--path /mcp` handshakes at `/mcp` and 404s at `/sse`; `PULSAR_MCP_PORT=notanint` raises the *specific* message. |
| `README.md`, `CHANGELOG.md` | Revert the deploy section. Restore the persistent-install pointer (one line → `mcp.rst`). Collapse the triple blank line. Real `### Added` / `### Fixed` / `### Changed` / `### Deprecated` entries. |

**Closes:** INLINE 1, 10, 11 *(host half)*, 12, 13, 14, 15 *(blank-line half)* · NIT (b), (e) *(all five sites)* · Review Body 1's test request · `missing-transport-kwarg-contract-test` · `missing-real-bind-startup-test` · `stateless-http-per-request-session-churn` · `changelog-describes-none-of-this-branch`

**Risk if it lands alone.** PR 1 is what makes `--transport streamable-http --host 0.0.0.0` work for the *first* time, and it ships documentation of the new flags. A developer who reads the new `--help` and runs the obvious thing on a shared box gets an unsandboxed arbitrary-read/write server. A startup log line is the right instinct but too thin.

**Therefore:** PR 1 **refuses a non-loopback bind** unless `PULSAR_MCP_I_UNDERSTAND_NO_SANDBOX=1`, until PR 2 lands — then flip to sandbox-by-default and delete the escape hatch. ~6 lines, and it closes the window without gating PR 1 on PR 2.

**Non-goals, stated adversarially:** does not make the server safe to expose — **do not merge a follow-up that documents, publishes, or binds a non-loopback URL until PR 8 lands.** Does not sandbox any path. Does not touch eviction, `PULSAR_MAX_SESSIONS`, or `gc.collect()`. Does not change the default transport away from `stdio`.

### PR 2 — Path sandbox + settings module

**Goal:** one data root, one resolver, one settings object — the trust model becomes chosen rather than an accident of `Path.resolve(strict=True)`.

**New:** `pulsar/mcp/settings.py`, `pulsar/mcp/paths.py`, `tests/test_mcp_path_sandbox.py`, `tests/test_mcp_settings.py`.
**Touched:** `registry.py`, `session.py` (`_read_dataset_file`), `config_tools.py`, `tools/{ingestion,meta,__init__}.py`, `prompts.py`, `docs/source/userGuides/mcp.rst`.

`_resolve_under_root()` returns `mcp_error(error_code="PATH_OUTSIDE_SANDBOX", agent_action=...)`. Enforcement is active only when `transport != "stdio"` (or `PULSAR_MCP_SANDBOX=1`) — **do not "simplify" this by making it unconditional; it breaks the local stdio workflow that is the product's primary use.** `get_runtime_context` stops reporting hardcoded `transport_assumption: "stdio-single-client"`, and gains `pulsar_version` so a remote client can tell which server produced a payload.

**Closes:** BLOCKER 3 *(read half)* · `probe-columns-content-exfiltration` · `ingest-dataset-filesystem-oracle` · `config-yaml-run-data-second-read-vector` · `runtime-context-lies-about-transport` · `env-var-surface-scattered-and-undocumented` · INLINE 10 *(completes)*

**Risk if it lands alone:** a false sense of completion. A read sandbox while `export_dataset_bundle(slug="/tmp/x")` still writes anywhere is not a control. PR 3 must follow before anyone calls the surface sandboxed.

### PR 3 — Write confinement + sweep cost budget + error envelopes

Four write paths get confined to `cache_dir/exports/`: `export_labeled_data(output_path)`, `export_dataset_bundle(output_dir, slug)` — reject `slug` outside `[A-Za-z0-9._-]+`, since an absolute slug currently *replaces* `output_dir` via pathlib semantics — `_auto_save_config` (reject `run.name` with separators), and both `os.getcwd()` fallbacks.

`validate_config_yaml` gains an enforced `dims × seeds × steps` budget → `SWEEP_BUDGET_EXCEEDED`. **Plus a row/byte ceiling at ingest** — the grid budget does not cover input size, and a single 5M-row CSV at `dimensions: [2]`, one seed, 4 steps passes every grid check and OOM-kills a 2 GiB instance in PCA. Noting that gap in a PR body is not closing it.

`AgnosticFastMCP.call_tool` starts reporting `stripped_parameters` and hard-fails on strips of a small privacy/bounding allowlist (`exclude_columns`, `cluster_names`, `output_path`, `output_dir`, `slug`). Today a misspelled `exclude_columns` means PII lands in the exported report with a success response.

**Also here — the 26 bare error handlers.** `grep -rnE 'return mcp_error\("[a-z_]+", str\(e\)\)' pulsar/mcp/tools/` → 26 sites across all 8 tool modules, each emitting a well-formed envelope with `error_code: null` and `agent_action: null`. That is a CLAUDE.md violation on 26 of 30 tools' failure paths, *and* a disclosure surface the sandbox does not close: `reason` is `str(e)`, and Python OS exceptions embed absolute paths (`[Errno 13] Permission denied: '/var/secrets/x.csv'`). Give each a real `error_code` + `agent_action`, and redact `/`-rooted paths from `reason` when a sandbox root is active.

**Closes:** BLOCKER 3 *(write half)* · `export-labeled-data-output-path-unvalidated` · `export-dataset-bundle-slug-traversal` · `auto-save-config-run-name-path-traversal` · `unbounded-sweep-grid-no-cost-cap` · `agnostic-call-tool-silently-strips-restricting-params` · NIT (a) *(`/app` half)*

### PR 4 — Move blocking registry I/O off the event loop

Wrap every synchronous `registry.*` call in `asyncio.to_thread`, highest value first: `save_cluster_assignment` (`clustering.py:365`, which `fsync`s an O(rows) labels JSON under `LOCK_EX`), `save_run` (`sweeping.py:435`), and the `get_dataset` read path reached from six call sites. Switch read-mostly paths to `LOCK_SH`. Drop `gc.collect()` from the eviction path.

**Empirically:** with a worker holding `_locked_registry()` for 2.0 s, an `async def` registry call blocked for 2.006 s and a concurrent heartbeat task got **zero** iterations. The loop was fully frozen, not merely slowed.

**`_locked_registry`'s false re-entrancy must be fixed *in this PR*, not deferred.** It guards with a `threading.RLock` while opening a fresh fd per entry, so the RLock buys nothing at the flock layer and removes the tripwire. Today, accidental nesting inside one coroutine is on one thread and the RLock lets it through, then self-deadlocks on the new fd. **PR 4 makes that non-deterministic** — two calls in one logical operation may land on different `to_thread` pool threads, so the same code deadlocks or doesn't depending on executor scheduling. An unreproducible deadlock is strictly worse than a deterministic one. Downgrade to a plain `threading.Lock` (~2 lines) so nesting fails loudly.

**Correction to carry into the tests:** the registry **does** lock on Windows. `registry.py:364-403` has a full `msvcrt.locking` fallback with a matching `LK_UNLCK` release. An earlier note claiming Windows runs unlocked is false — writing the prescribed `xfail(strict=True)` would fail on xpass. The real nuance, worth one line: `msvcrt.locking(..., LK_LOCK, 1)` retries ~10× over ~10 s then **raises `OSError`**, whereas `fcntl.flock(LOCK_EX)` blocks indefinitely. Windows has an implicit ~10 s lock timeout POSIX does not — which interacts directly with the `LOCK_NB`-plus-bounded-retry change.

**Closes:** `sync-registry-flock-in-async-def` · `gc-collect-inside-event-loop` · `flock-per-fd-rlock-false-reentrancy` · `no-registry-multiprocess-test`

### PR 4.5 — Observability *(new — was entirely unowned)*

The acceptance criteria gate launch on structured logging, a telemetry middleware, `/ready`, and three log-based metrics. The PR split assigned **one** of those five artifacts. `grep -rn "add_middleware\|Middleware" pulsar/` → zero hits; `logger = logging.getLogger(__name__)` at `server.py:15` is the entire observability surface. The primary breakage-detection signal is downstream of a middleware nobody was scheduled to write, so **the whole detection story was unowned.**

**New:** `pulsar/mcp/logging.py` (JSON formatter on `StreamHandler(sys.stdout)`, `severity` + `logging.googleapis.com/trace`), `pulsar/mcp/middleware.py` (`PulsarTelemetryMiddleware`, ~30 lines, `on_call_tool`), `/health` *(moved here from PR 6 so both routes ship together)* and `/ready`, three Cloud Monitoring log-based metrics.

**Plus SIGTERM handling, which does not exist at all.** `grep -rnE "SIGTERM|signal\.|atexit|shutdown|lifespan" pulsar/` → zero hits. Cloud Run sends SIGTERM with a ~10 s grace then SIGKILL — **including on scale-to-zero of an idle instance**, so this is the steady-state path, not a deploy edge case. A sweep 40 s into `await asyncio.to_thread(model.fit, ...)` is killed mid-fit; `save_run` never runs; the `run_id` the client is waiting on never exists. Install a handler that stops admitting calls (`SERVER_SHUTTING_DOWN` + `agent_action`), waits N seconds for in-flight work, and persists completed-but-unsaved runs. Add `stop_grace_period: 30s` to compose.

**Two log-hygiene traps this PR must not walk into.** Existing leaks: `interpreter.py` logs **column names** at WARNING on every statistical failure (8 sites) — schema-identifying in clinical tables; `sweeping.py:82` logs an absolute host path; `session.py:120` logs the session key at INFO. And the plan's *own* telemetry spec mandates `principal` and `session_key` as required fields — which writes an operator identity into every log line. Hash or truncate `session_key`; make `principal` opaque above DEBUG; extend the hygiene test past "no payload bodies" to also assert no column names, no `/`-rooted paths, and no `@`-containing principals at INFO or below.

**Two unverified assumptions to settle here, cheaply, before they gate a launch:**

- **`/ready` is a pull endpoint nobody scrapes.** One instance, CPU-throttled between requests unless `--no-cpu-throttling`, reachable only through a proxy an operator must be running, curl'd once manually at launch. Emit the same fields as a periodic structured log line so they reach Cloud Logging where the metrics already live. Zero new infrastructure.
- **Cloud Run's default startup probe is TCP** (`timeoutSeconds: 240, periodSeconds: 240, failureThreshold: 1`), and a process that binds but serves nothing passes it. The plan's #1 breakage signal needs an `httpGet` probe — but whether httpGet probes work on an IAM-restricted service is **undocumented**, and the failure mode is not an error, it is silent degradation to exactly the default the plan calls inadequate. Verify against a throwaway service; if it doesn't work, promote the log-based metric to primary and say so.

### PR 5 — Session lifecycle *(ownership half cut)*

**Goal:** a session cannot be silently destroyed by another caller.

Add `last_touched` + an idle-TTL reaper. Bound by estimated bytes via the existing `calculate_memory_mb()` rather than a raw count. Add an `in_flight` counter so a session with a running `to_thread` fit is never evicted (refuse the new session with `RESOURCE_BUSY` if nothing is evictable). Delete the dead `or "default"`. Validate `PULSAR_MAX_SESSIONS` in `load_settings()` with a `>= 1` clamp, renamed `PULSAR_MCP_MAX_SESSIONS` with a deprecation fallback. Surface `sessions_active` / `memory_mb` / `evictions` in `get_runtime_context`.

**Cut from this PR per §2.3:** `owner` on four record types, principal-threading through 12 call sites, salted dataset digest, `secrets.token_urlsafe` handle ids. All of it is inert under one principal, and it silently invalidates every persisted handle for **local stdio users** — the product's primary use case — with no `schema_version` field anywhere in `registry.py` and no migration criterion. Take that breaking change when OAuth makes it meaningful.

**Where the eviction DoS actually gets fixed.** The empirically-reproduced attack — three unauthenticated SSE clients sending `mcp-session-id: ATTACK-0/1/2` evict the victim's session, which then returns `data_present: false` with **no error** — is *not* closed by principal-keying the session. Under one shared `sub` the key becomes `sharedSub:<attacker-chosen header>`, and the attacker still fully controls the varying component. What closes it is TTL reaping + byte bounding + in-flight protection, i.e. this PR. Credit it here, or an implementer who lands principal-keying and defers the reaper will believe the DoS is closed.

**The `in_flight` counter must be mutated only on the loop thread.** The natural place to decrement is inside the callable handed to `to_thread` — which violates CLAUDE.md's rule verbatim and races the LRU scan, reintroducing exactly the mid-flight eviction this exists to prevent. Add a test that records the thread ident at each mutation.

**Closes:** BLOCKER 2 · `no-session-reaper-raising-max-sessions-is-unsafe` · `eviction-cannot-free-inflight-fit` · `sse-forged-session-key-eviction-dos` · `session-id-default-fallback-is-dead-code` · `max-sessions-int-env-at-import`

### PR 6 — Container image, local-only, built in CI

`maturin build --release --locked`; `maturin` pinned `>=1.12.6,<2` (the unpinned `pip install maturin` currently bypasses the `maturin>=1.7,<2` bound the project itself declares); runtime Python deps from a hash-pinned `uv export --frozen` instead of resolving ~60 packages from floors at build time while a 1.1 MB `uv.lock` sits unused; `ARG PYTHON_VERSION` so the two `FROM` lines cannot skew; a new `rust-toolchain.toml` pinning the compiler for both image and CI; `mkdir -p` + `chown` for the cache dir **and `/app`** (root-owned today, so any relative-path export fails as uid 1000); `PYTHONUNBUFFERED=1` so startup crashes reach `docker logs`; `.dockerignore` extended, with a comment noting it gates **wheel contents**, not just context size.

Default remote transport in the image becomes `streamable-http` — `sse` is the deprecated MCP transport, and this is the cheapest moment to pick right, before any URL ships.

**CI docker job** on `pull_request` (the image's first build is currently the production deploy step): buildx + `type=gha` cache, `load: true`, `push: false`, then an image-contract check — `--help` exits 0, `import pulsar._pulsar, pulsar.mcp.server` succeeds, `id -u` is 1000, and `smoke_mcp.py --url` handshakes against the running container. `scripts/smoke_mcp.py` gains `--transport`/`--url` modes; it covers only in-process and stdio today.

**Also:** `permissions: contents: read` at the top of `ci.yml` — it is the only workflow with no `permissions:` block, and this PR adds a job to it. `docs.yml` gains `-W --keep-going`; without it a broken cross-reference in the new `deployment.rst` builds green. Supply chain: `.github/workflows` has zero `cargo audit` / `pip-audit` / Trivy / CodeQL and no `dependabot.yml`, and this PR introduces a whole new artifact class (Debian base packages, CPython, OpenSSL, ~60 Python deps, statically-linked Rust crates) — add scanning that fails on HIGH/CRITICAL with a documented allowlist. Licensing: `pyproject.toml` has no `license` field or classifier, and the image redistributes mixed GPL/LGPL Debian packages plus MIT/Apache-2.0 Rust crates whose attribution obligations travel with the static binary — set the field, and generate `THIRD_PARTY_LICENSES.txt` into the runtime stage.

**Risk if it lands alone — the real one.** A container image in the repo is an invitation to `docker run -p 0.0.0.0:8000:8000`. The image cannot prevent that. Every `docker run` snippet binds `127.0.0.1:8000:8000`, and `deployment.rst` opens with a bounded prerequisites block. But **documentation is not the control** — that is why PR 6 depends on PRs 2 and 3, so the artifact itself carries sandbox enforcement whenever `transport != stdio`.

### PR 7 — docker-compose + Caddy, loopback default

`ports: - "127.0.0.1:8000:8000"` — load-bearing, not cosmetic. Named volume at the cache dir so container *recreation* (`up --build`, the documented launch path) stops destroying the registry.

**And the thing everyone missed:** the `pulsar-mcp` service has **no `volumes:` key at all**, so no host file can reach the container. `ingest_dataset("/Users/me/data.csv")` resolves against the container filesystem and returns `FILE_NOT_FOUND`. The documented local recipe **cannot perform the product's primary function.** Add `- ${PULSAR_DATA_DIR:-./data}:/data:ro` and document `ingest_dataset("/data/<file>.csv")`. The only end-to-end proof this deliverable works is: drop a CSV in the host dir, `docker compose up -d`, get a `dataset_id` back.

Caddy moves behind an opt-in profile and, when enabled, carries `basic_auth` (bcrypt from env), `request_body { max_size }`, explicit timeouts, and a `log` block. **Keep `flush_interval -1`** — verified correct for SSE in Caddy 2, do not "clean it up." Pin `caddy:2-alpine`. Document the `localhost` internal-CA reality honestly: Caddy's root lands in the `caddy_data` volume and is never installed on the host, so either ship the `docker compose cp` + per-OS trust-store step or make the local default plain HTTP on loopback and reserve the HTTPS block for a real `DOMAIN`. Add a supported-host-OS table with the bind-mount uid caveat (`useradd -u 1000` + a host mount breaks on native Linux where the user isn't 1000).

### PR 8 — Auth provider

**The enforcement lift is one kwarg.** FastMCP 3.2.0 already wraps the SSE route (`http.py:197`), the SSE message mount (`:210`), and the streamable-HTTP route (`:336`) in `RequireAuthMiddleware` whenever `auth` is set, and 401s already carry a spec-shaped `WWW-Authenticate`. So: `AgnosticFastMCP(..., auth=<provider>)` from `PULSAR_MCP_AUTH=none|jwt|oauth`, with `none` permitted only for stdio.

Provider is `RemoteAuthProvider(token_verifier=JWTVerifier(...), authorization_servers=[...], base_url=...)` — a bare `TokenVerifier` contributes no RFC 9728 metadata routes and is MCP-spec-noncompliant even though Claude Code tolerates it. **Not `StaticTokenVerifier`** — its own docstring says "Never use this in production."

**The decision this PR must state in its description:** static bearer / `headersHelper` is GA in **Claude Code only**. claude.ai and Claude Desktop request-header auth is beta-gated; org-managed connectors attempt OAuth 2.1 + DCR on connect. Name the launch client, or the wrong mechanism gets built.

**Three holes to close, all currently unaddressed:**

- **Revocation.** Google ID tokens are **not revocable**. Removing an operator's `serviceAccountTokenCreator` leaves a ≤1 h tail of full access to every dataset, run, and cluster label. Defensible for an operator-only tool; not defensible that the word "revocation" appears nowhere as a criterion. Document the window; note that a self-issued `JWTVerifier` with per-operator `sub` would give real revocation for the same effort.
- **`headersHelper`'s 10 s timeout.** The prescribed command is a cold `gcloud` invocation (~1.5–3 s startup) + an IAM `generateIdToken` round trip + credential refresh. Not comfortably inside 10 s on a slow link, and the failure mode is Claude Code marking the server as needing auth with no diagnostic. The helper must be a **cached-token wrapper** — write token+expiry to a file, re-mint only within 5 min of expiry. Criterion: <3 s warm, <8 s cold, measured.
- **Client-set overstatement.** Phase 0's supported clients are exactly *Claude Code + mcp-remote + Agent SDK*. claude.ai cannot reach `127.0.0.1` at all. Do not claim universality.

### PR 9 — Remote data ingress

Replace the import-time `_ENABLE_UPLOAD` constant — currently restart-only, invisible in `get_runtime_context`, and it changes which tools appear in `tools/list` — with a runtime settings read. *(Move this to PR 2 alongside the settings module: it is 3 lines and PR 2's honest `get_runtime_context` depends on it.)*

Enforce caps in `append_upload_chunk`: max chunk bytes, max total per `upload_id`, max concurrent staged uploads, plus a TTL reaper — each breach returning `UPLOAD_LIMIT_EXCEEDED` with an `agent_action`. Route writes through `to_thread`. Rewrite `WORKFLOW_PROMPT`'s Cache-Bridge section to branch on transport instead of telling a network client to `cp` a file onto a host it has no shell on. Consider restricting remote uploads to CSV — `_read_dataset_file` dispatches to `pd.read_parquet` on suffix alone, so enabling uploads makes parquet parsing of untrusted bytes reachable.

**Why this is a blocker for usefulness, not just safety:** `PULSAR_MCP_ENABLE_UPLOAD` is set nowhere in the Dockerfile, compose, or the deploy workflow, so the three upload tools are never registered. The only ingress left requires a path already inside the container. **A remote client over HTTPS cannot submit a dataset at all.** The deployed configuration is simultaneously dangerous and useless for its stated purpose.

### PR 10 — Cloud Run deploy, IAM-only, no public URL

**Drop `--allow-unauthenticated`.** That single change **dissolves** INLINE 2 rather than fixing it: with no invoker policy applied at deploy time, the deployer never calls `run.services.setIamPolicy` and the missing permission stops being reachable. Grant `roles/run.invoker` to named principals out-of-band, never in CI.

Add `--service-account=<dedicated minimal runtime SA>`, `--max-instances=1`, small `--concurrency`, `--timeout=3600`, `--no-cpu-throttling`, `PULSAR_MCP_MAX_SESSIONS`, and an **`httpGet` startup probe** (§PR 4.5). **Drop `--session-affinity`** per §2.6.

Gate on CI via `workflow_run` or a folded `needs:`. Add `concurrency: { group: deploy-cloudrun-${{ github.ref }}, cancel-in-progress: true }`. Deploy by `@sha256:` digest, not a mutable tag; push `:latest` only after the deploy verifies. `--no-traffic --tag=sha-…`, probe, then promote. Record the previous revision to `$GITHUB_STEP_SUMMARY` for rollback. Move `id-token: write` to **job** scope. Add an Artifact Registry cleanup policy — every merge pushes an immortal multi-hundred-MB tag today.

**Post-deploy verification must use the auth action's own `token_format`, not `gcloud auth print-identity-token`** — the workflow authenticates via impersonation-based WIF, and `print-identity-token` is documented for user and service-account credentials, not external-account ADC. It would fail on first run, and a verification step that fails on first run gets deleted:

```yaml
- uses: google-github-actions/auth@v2
  id: idtok
  with:
    workload_identity_provider: ${{ secrets.GCP_WORKLOAD_IDENTITY_PROVIDER }}
    service_account: ${{ secrets.GCP_SERVICE_ACCOUNT }}
    token_format: 'id_token'
    id_token_audience: ${{ steps.deploy.outputs.url }}
    id_token_include_email: true
```

Prefer the `--no-traffic --tag` probe route, which avoids granting CI `run.invoker` at all.

**Fix the WIF script — it is dead on arrival today.** The bare `gcloud projects create` produces a parentless, billing-disabled project, and `gcloud services enable run/artifactregistry` then fails `FAILED_PRECONDITION` under `set -e`. Require a pre-existing billed project or link billing explicitly. Scope `roles/iam.serviceAccountUser` to the runtime SA **resource** instead of project-wide (today the deployer can run any container as any SA in the project, including the default Compute SA). Pin the WIF attribute condition to `repository_id` + `refs/heads/main` instead of a squattable repository name that any branch or `workflow_dispatch` satisfies. Drop the `GCP_PROJECT_ID` fallback literal — a missing secret currently retargets the deploy silently.

**Access:** `gcloud run services proxy pulsar-mcp --port=8080` → client points at `http://127.0.0.1:8080/mcp` with no client-side auth config at all.

> ⚠ **Verify this before committing to it.** Google's docs recommend the proxy for testing private services but say **nothing** about streaming, SSE, long-lived connections, or timeouts — the framing is request/response. Pulsar's flagship tool is a minutes-long fit reporting progress over a server→client stream. If the proxy buffers or drops idle streams, Phase 0 ships an access path where `run_topological_sweep` **appears to hang**, with no fallback because Phase 0 has zero client-side auth config. 20-minute test: start the proxy against any private streaming service, `curl -N`, confirm chunks arrive incrementally and survive 60 s idle. If it fails, PR 8 moves from "next" to "this one."

---

## Part 4 — Corrections to the existing review

Nothing on PR #37 is stale, and neither `574b995` nor `4d701e1` fixed anything — `574b995` is a 5-line env read that left import-time resolution, the singleton, and filesystem-locality untouched, and it is what put the cache dir onto Cloud Run tmpfs in the first place. **6 blockers, 4 high, 7 medium, 6 low/nit are live at HEAD.** Only line references drifted (`session.py:112-121` → `111-131` after ruff format).

Eight findings need rewording before someone implements the wrong thing:

| Finding | Correction |
|---|---|
| **BLOCKER 2** *(mechanism)* | Conclusion right, mechanism wrong. `ctx.session_id` **never** returns falsy — it returns a header value, a cached prefix, or a fresh uuid4, or it *raises `RuntimeError`*. So `or "default"` is dead code for HTTP. The real failures: **(a)** an unvalidated client-supplied `mcp-session-id` on SSE — three requests evict every legitimate client's model with no error, empirically reproduced; **(b)** under stateless mode, a *fresh* session per request, so the ingest→sweep→dossier loop cannot complete at all. Keep the severity. |
| **BLOCKER 3** *(both directions)* | Understates: it is arbitrary file **write** too — `export_labeled_data(output_path)`, `export_dataset_bundle(output_dir, slug)`, `_auto_save_config` via `run.name`. Overstates: handle *enumeration* isn't trivial (48 bits of uuid4). The gap is the missing authorization check plus `dataset_id = sha256(path:size:mtime)`, which needs no leak at all. Also missing: `probe_columns` returns file contents verbatim — the exfiltration step, verified against `/etc/hosts`. |
| **INLINE 6** | **Do not apply the suggested COPY reorder.** `pyproject.toml:34` sets `python-packages = ["pulsar"]`, so building before copying `pulsar/` succeeds *with no warning* and yields a wheel with the extension at top-level `_pulsar/`, **no `pulsar` package**, and an entry point still naming `pulsar.mcp.server:main`. `pip install` succeeds; the container dies with `ModuleNotFoundError`. Empirically reproduced. Use a Cargo-only prebuild layer + buildx `type=gha`. |
| **INLINE 2** | Confirmed against the live IAM API (`run.developer` has `getIamPolicy`, not `setIamPolicy`), and `--no-invoker-iam-check` is not an escape hatch — same permission. Narrowest predefined role is `roles/run.editor`, **not** `run.admin`. But **don't grant it** — dropping `--allow-unauthenticated` removes the requirement. There is also a second, independent blocker: domain-restricted sharing (`iam.allowedPolicyMemberDomains`) blocks the `allUsers` binding regardless of role. |
| **INLINE 8** | The "dangling session per probe" does not happen — the `python -c` process exits and closes the socket. The hardcoded-path and Cloud-Run-ignores-`HEALTHCHECK` sub-claims are the load-bearing ones. |
| **INLINE 9** | Trigger is container **recreation**, not restart. `/tmp` is the overlay layer, so a restart *preserves* it; the documented `up --build` recreates. |
| **INLINE 12** | The test does **not** stay green through the fix — `assert_called_with` makes it go RED. The defect is that it is green *today* while pinning a guaranteed crash. |
| **NIT (c)/(e)** | (c) is latent, not breaking — `--port=8000` happens to match. The sharper instance is the healthcheck URL, which hardcodes 8000 while the port is configurable: set `PULSAR_MCP_PORT=9000` and the container is permanently unhealthy, which then blocks Caddy via `depends_on`. (e)'s fourth site is `tests/test_mcp_server_patch.py:71`, not the README — `grep ALLOWED_HOSTS README.md` returns nothing. |

**There is no scope creep.** `gh pr view 37 --json files` returns **exactly 10 files**, all in scope. Local `main` and `origin/main` sit at `8113306` (2026-07-16) while remote `main` is at `7095692` (#34) — the 17 "extra" files in `git diff main...HEAD` belong to already-merged #33/#34. The branch needs no rebase. **Do not split those out, and do not "fix" this by reverting merged work.** Review with `gh pr diff 37`. *(The `conftest.py` → `reference.py` move is verified clean: `pseudo_laplacian_py` was a plain helper, never a fixture; all four import sites updated; 36 tests pass.)*

---

## Part 5 — DO NOT DO

1. **INLINE 6's COPY reorder** — ships a wheel with no `pulsar` package. See Part 4.
2. **Grant the deployer `roles/run.editor`** — technically correct for INLINE 2, wrong move: it hands CI the ability to make any service in the project public. Drop `--allow-unauthenticated` instead.
3. **`--no-invoker-iam-check`** — makes a service public via a service-spec field, so IAM audits report it as private. Enforce `constraints/run.managed.requireInvokerIam` so it can't be set.
4. **`FASTMCP_STATELESS_HTTP=true`** to "fix" the multi-instance handle-404. It is the documented FastMCP answer to the symptom and it destroys the workflow: a fresh transport per request, a new uuid4 per call, and the ingest→sweep→diagnose loop cannot complete. It also accepts an attacker-chosen `mcp-session-id` verbatim and executes a `tools/call` POST with **no initialize handshake** at HTTP 200. PR 1's startup guard prevents this — **do not remove it.**
5. **Raise `PULSAR_MAX_SESSIONS` before PR 5.** Sessions are never removed on disconnect. Setting it to 80 converts silent-eviction corruption into 80 retained DataFrames-plus-models inside 2 GiB — an OOM that drops every co-tenant's work. Reaper + byte budget + in-flight protection first, *then* the number.
6. **Back the registry with gcsfuse.** No file locking; "last write wins and all previous writes are lost." Silently removes the mutual exclusion the code assumes. See §2.4.
7. **Build OAuth 2.1 / DCR / `GoogleProvider` yet.** Weeks of work, unnecessary if Claude Code is the launch client. Decide the client first.
8. **Move session state out of process.** Redis/Firestore keyed by principal is the right answer to genuine multi-instance operation and a large architectural PR. `--max-instances=1` is the honest interim.
9. **Enable `PULSAR_MCP_ENABLE_UPLOAD` on anything reachable before PR 9.** The natural move once someone notices remote clients have no host paths — and without caps it streams unbounded bytes into a RAM disk, under `LOCK_EX`, on the event loop.
10. **Migrate to pyo3 `abi3`.** Would collapse the two-`FROM` coupling, but far larger than the `ARG PYTHON_VERSION` fix that resolves the immediate risk.
11. **Chase these — they are already correct.** Spending review effort here costs credibility: `flush_interval -1` is the right SSE directive; there is **no** unsafe deserialization anywhere (zero `yaml.load`/`pickle`/`eval`/`exec`/`subprocess` in `pulsar/`, 15 `yaml.safe_load` sites, atomic `tmp`+`fsync`+`os.replace` registry writes); `get_topological_skeleton(detail="full")` does **not** bypass `max_nodes`/`max_edges`; `--no-install-recommends` + same-layer apt cleanup are already present; INLINE 14 over-claims (the no-clone rationale survives at `README:27` and `mcp.rst:58`, `pipx` at `mcp.rst:119-126` — one sentence to restore, not a paragraph); the registry **does** lock on Windows.
12. **Fold the `release.yml` `fastmcp>=2.0` duplicates into this chain.** Six sites total; raise the floor once in `pyproject.toml` (PR 6) so the Dockerfile can reference `[mcp]`. The three `release.yml` instances are pre-existing on main and belong in their own follow-up.

---

## Part 6 — Launch gates

Ordered, binary, all green before the first deploy.

**Gate 0 — before any code**
1. ✅ Part 0's exposure check — **done, clear.** Re-run before any merge to `main` carrying the deploy workflow. Also: grant `sidney@krv.ai` read access to Cloud Run in `pulsar-mcp-prod` (currently denied — see Part 0).
2. Ten cheapest GCP verification commands **executed** against a scratch project, output pasted into the PR: `gcloud iam roles describe roles/run.developer`, `gcloud org-policies describe iam.allowedPolicyMemberDomains`, `gcloud projects describe --format='value(parent)'`, `bash scripts/setup_gcloud_wif.sh` on a throwaway. Half a day; converts ~15 inferences into facts and will likely delete two or three findings outright.
3. `gcloud run services proxy` streaming test (§PR 10).
4. `httpGet` startup probe under IAM-only verified (§PR 4.5).
5. **Launch client named** in writing.

**Gate 1 — code correctness**
6. All three transports bind for real in CI and complete an MCP handshake.
7. `_build_run_kwargs` kwargs ⊆ `run_http_async` signature, with the `VAR_KEYWORD` guard.
8. `ingest_dataset` outside the data root → `PATH_OUTSIDE_SANDBOX`; all four write paths confined.
9. Sweep budget and row/byte ceiling enforced, with the estimate in `details`.
10. No `mcp_error(...)` call passes `error_code=None` (AST walk); no `reason` contains a `/`-rooted path.
11. Two sessions isolated; an in-flight session is never evicted; `in_flight` mutated only on the loop thread.
12. Registry calls off the event loop — a concurrent heartbeat task makes progress during a 2 s registry hold.

**Gate 2 — artifact**
13. Image builds on `pull_request`; contract check passes (`--help`, imports, uid 1000, `/health`).
14. Wheel contains the `pulsar` package *(the INLINE 6 trap)*.
15. Image scan clean of HIGH/CRITICAL, or allowlisted with a reason.
16. Compose: CSV in host dir → `ingest_dataset("/data/x.csv")` returns a `dataset_id`. **The only end-to-end proof the local deliverable works.**

**Gate 3 — deploy**
17. 401 without a token; 200 with one; `/.well-known/oauth-protected-resource` → 200.
18. Deploy gated on CI; digest-pinned; concurrency group set; previous revision recorded.
19. `get-iam-policy` shows no `allUsers`; the `invokerIamDisabled` annotation is absent.
20. SIGTERM during an in-flight fit either persists the run or returns `SERVER_SHUTTING_DOWN` — never silent loss.
21. Structured logs reach Cloud Logging; no column names, absolute paths, or raw principals at INFO or below.
22. Billing budget + alert; Artifact Registry cleanup policy.

### Explicit non-guarantees — document these, don't discover them

State does **not** survive cold starts or restarts. The service runs **one instance**; handles do not cross instances. Two operators on one deployment **can see each other's** datasets, runs, and per-row cluster labels. Access revocation is eventually-consistent with a ≤1 h bound and there is **no** immediate-revocation mechanism. A session cannot outlive Cloud Run's request timeout. There is **no** tool-contract stability guarantee across server versions — clients must call `get_runtime_context`. And **the data-classification statement is missing entirely**: `grep -rni "hipaa|phi|baa|de-identif|gdpr|pii"` across the repo returns exactly one hit (`demos.rst:165`, "with synthetic data (no real PHI)") while the repo ships `demos/ehr/`, `demos/heart-attack/`, `demos/healthcare-deserts/`. Datasets, column names, and derived labels persist to `cache_dir` **and to Cloud Logging**; Cloud Run region, Artifact Registry region, and the log bucket region are the residency surface. **If Krv carries BAA or equivalent obligations, this escalates from a documentation gap to a Tier-A blocker** — that determination belongs to the repo owner, not this review, but the absence of the statement is a gap either way.

---

## Part 7 — What can be delegated

Split by whether the work is *verifiable by the agent doing it*.

**Agent-doable — mechanical, test-verifiable (PRs 1–5, 6 partly, 9)**
The `allowed_hosts` removal, lazy `_cache_dir()`, the two broken tests, `--locked`, the `fastmcp` floor, `PYTHONUNBUFFERED`, `.dockerignore`, the healthcheck route, compose volumes and the data bind mount, the `ALLOWED_HOSTS` sweep, the path sandbox and write confinement, the sweep budget, the 26 error envelopes, the `to_thread` offload, the session reaper, docs and changelog. Every one has a test that proves it.

**Human-only — the agent can write it but cannot verify it (PR 10, parts of 6/8)**
Linking a billing account. Creating the runtime SA. Granting `run.invoker` to named principals. Narrowing the WIF trust condition. Every claim in Gate 0. The proxy streaming test and the startup-probe test. An agent will produce confident-looking YAML and confident-looking prose about infrastructure it never touched — and the evidence base has a demonstrated error rate (see Appendix A).

**Decision-only — nobody should delegate these**
Which client must work at launch (gates PR 8's mechanism). Whether single-tenant is acceptable for the intended data. Whether the data-classification statement is a blocker.

---

## Appendix A — Evidence confidence

This review's own evidence base has a **demonstrated error rate**, and the plan should be read accordingly.

One claim tagged `code-read-only` — "Windows registry locking is absent" — was **false**, disproven by reading 20 lines of `registry.py` in this repository. It had already been promoted from a finding to a shipped non-criterion *and* to a test premise before anyone checked. That is one wrong claim about a file the reviewer had open.

Roughly **fifteen further `code-read-only` GCP/IAM claims** gate individual launch-checklist items and **none were executed**: org-policy defaults, `run.developer`'s permission set, DRS enforcement, `gcloud services enable` failure behavior under `set -e`, the default Compute SA's Editor grant (which the evidence itself downgrades to conditional — the automatic grant is off only for orgs created after 2024-05-03, and the script's bare `gcloud projects create` yields a parentless project with no org policy at all), and `--no-invoker-iam-check` semantics.

**The launch checklist's green state may therefore certify facts about a different project than the one being deployed to.** Gate 0 item 2 exists to fix this: half a day of running commands, before any code is written.

Part 0 is the worked example of why that half-day pays. Running four commands disproved the "already public" scenario on three independent grounds, **downgraded one blocker** (the WIF script is not dead on arrival — both APIs are enabled and the AR repo exists, so it ran to completion), and **surfaced a finding nobody predicted** (the intended operator has no read access to Cloud Run in the project). Two of those three outcomes were invisible to code reading.

**Verified empirically** (trust these): the `allowed_hosts` `TypeError`; `run_http_async`'s signature and the absence of a `VAR_KEYWORD` sink; the SSE forged-session-key eviction DoS; per-request session churn under stateless mode; `ctx.session_id` never returning falsy; the INLINE 6 broken-wheel reproduction; the 2.006 s event-loop freeze; `probe_columns` exfiltration against `/etc/hosts`; the export-path traversals; FastMCP 3.2.0's auth inventory and `RequireAuthMiddleware` wiring; custom routes appended outside the `if auth:` branch (so `/health` is genuinely unauthenticated); `roles/run.developer`'s permission set against the live IAM API; `gh pr view 37 --json files` returning 10 files.

**Method:** 11 agents over 3 phases — 1 triage pass over the existing 16 findings, 5 gap hunts, 3 synthesis documents, 2 adversarial passes. 24 triage entries, 83 new findings, ~1.2 M tokens. The state-and-trust-model agent failed mid-stream; §2 is written by hand from its inputs, which is why it is the section most worth a second reader.
