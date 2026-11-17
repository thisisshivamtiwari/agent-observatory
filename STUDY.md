# STUDY: agent-observatory

Read this before an interview where this project might come up. All numbers are from `results/*.json` (run on
12 Nov 2026, Qwen2.5-1.5B-Instruct 4-bit, local, greedy decoding, 30 simulated scenarios).

## 1. The problem in one paragraph
An LLM agent that publishes hotel prices can get worse without anything breaking. A prompt edit or a model swap
still returns valid JSON, so there's no error to alert on. It just picks worse prices, and the loss only shows up
weeks later in revenue reports. The fix is to treat the agent like a production service: trace every step, evaluate
every version on the same scenarios against a known optimum, and block changes in CI that make the numbers worse.

## 2. Design choices (and why)
- **Simulator with an exact oracle instead of real data.** Real pricing data has no ground truth for "the best
  price". With a known demand model (logit in our price relative to the competitor median, Poisson arrivals, a
  capacity cap), the revenue-maximising price is computable, so every decision gets an objective score: revenue
  capture = E[revenue at the agent's price] / E[revenue at the oracle price]. The cost is realism (see section 4).
- **The first demand curve was wrong, and I changed it.** Constant elasticity put most optimal prices at the floor
  or ceiling (corner solutions), which no revenue manager would accept as realistic. The logit curve anchored on
  competitor rates gives interior optima, except on extreme event nights.
- **A failed decision is scored as "the current price stays live".** That matches what happens in production when
  an agent errors out, and it stops a broken agent from looking like it scored zero.
- **Five checks, and a decision passes only if all pass:** a final action, price within limits, the forecast was
  used, the right direction versus the oracle, and at least 95% revenue capture. Multiple checks matter: see the
  v3 finding below.
- **OpenTelemetry, with metrics computed from the spans.** The root span is the decision; child spans are each
  `llm.call` (tokens, latency, estimated cost; GenAI semantic-convention names) and each `tool.*` (args, result,
  guardrail blocks). p50/p95 latency, tokens and cost per decision are computed from the JSONL trace file, not from
  in-process counters. So if the tracing breaks, the report breaks, and we notice.
- **Record/replay cassettes for CI.** Each request (model + messages + max_tokens) is hashed. Live runs record
  responses; CI replays them. That makes the gate deterministic, free and runnable on a Linux runner with no GPU.
  Changing the prompt changes the hash, so CI fails with a cassette miss until someone runs the evals and commits
  new results. A prompt can't merge unevaluated.
- **Gate relative to an accepted baseline, not absolute thresholds.** `evals/gate.yaml`: pass rate may not drop
  more than 5 points, capture no more than 2 points, and p95 latency and cost may not grow more than 50%. It also
  checks that the replay reproduces the committed result exactly.
- **Guardrails in code, switched on per version.** A final price outside the allowed range, or one that wasn't
  forecast, is refused with a BLOCKED message, and the block is recorded on the span.
- **Tool design changes applied to all versions.** When I made `forecast_demand` accept several prices and return
  `best_of_these`, I re-recorded every version, so the A/B compares prompts on the same tools.

## 3. Results
| Version | Pass rate | Revenue capture | p95 latency | Tokens / decision | Gate |
|---|---|---|---|---|---|
| v1 minimal | 0% | 67.0% | 0.7 s | 209 | FAIL |
| v2 procedure + limits (accepted) | 27% | 72.1% | 3.6 s | 1146 | accepted |
| v3-regressed (undercut rule) | 0% | 73.9% | 0.7 s | 243 | FAIL |
| v4 + example + guardrails | 10% | 63.8% | 11.7 s | 2533 | FAIL |
| v5 v4 without example | 13% | 69.8% | 9.4 s | 1895 | FAIL |

Baselines: keep the current price 67.0%, match the competitor median 80.9%, 5% under the lowest competitor 80.1%.

The three findings to talk about:
1. **One metric would have let a regression through.** v3-regressed has *higher* mean capture than v2 (73.9% vs
   72.1%), because undercutting happens to work on the many price-sensitive scenarios. It fails the gate on pass
   rate: it never uses the forecast and gets no scenario to 95%.
2. **Guardrails made outputs valid but cost more than they gained on a small model.** Every published price was
   valid, but the 1.5B model kept getting blocked and retrying. Tokens went up 1.7 to 2.2 times and p95 latency 2.6
   to 3.2 times, the pass rate fell, and some decisions ran out of steps.
3. **No LLM version beats a one-line heuristic** (80.9%). The right call with this model is "don't ship the agent",
   and the harness is what tells you that before production.

## 4. Failure modes seen in the traces
- **Out-of-range prices.** v1 published 499 (above the ceiling) on every scenario.
- **Few-shot leakage.** v4's worked example contained the prices 80/100/120/135/150, and the model often published
  one of those numbers instead of a price from its own forecasts.
- **Wrong argument shape.** The model sent `"price": [130, 110, 150]` instead of `"prices"`. The tool now accepts
  both. Being lenient on input is cheaper than asking for perfect formatting.
- **Can't compare numbers.** It often published the last price it tried rather than the highest-revenue one, which
  is why the tool now returns `best_of_these`. Move the arithmetic into code, leave the judgment to the model.
- **Block-retry loops.** Under guardrails, some decisions spent all 8 steps calling `set_price` and never published.

Limits to be upfront about: one stay date per decision, an exact forecast (so this tests decisions, not
forecasting), one small model, 30 scenarios (a few points' difference is noise), an assumed per-token price for cost.

## 5. Ten interview questions
1. **Why a simulator and not real data?** Real data has no ground truth for the optimal price, so you can't score
   a single decision. A simulator with a known demand model gives an exact oracle. I'd validate the agent on
   historical data with off-policy estimates later, but you need the simulator to iterate quickly.
2. **How do you know the gate isn't flaky?** Greedy decoding plus replay: CI replays recorded responses, and the
   gate first checks that the replay reproduces the committed result exactly. Flakiness only enters at re-record
   time, and that run is a deliberate local step whose results get reviewed in the PR.
3. **Doesn't replay mean CI never tests the real model?** Yes. CI tests that *the committed evaluation* is
   reproducible and acceptable, and that prompts can't change without a new evaluation. A nightly live run against
   the real model (with a budget cap) would catch provider-side drift. That's a next step.
4. **Why compute metrics from traces rather than in code?** Then the traces are the source of truth. If spans are
   missing or wrong, the report is wrong too, so observability bugs surface in CI instead of in an incident.
5. **What would you put on a dashboard for this agent in production?** Per decision: revenue capture against a
   reference (or price versus a heuristic), guardrail block rate, decisions with no final action, p95 latency,
   tokens and cost, all sliced by prompt and model version. Alerts on block rate and no-action rate, because those
   move first when something breaks.
6. **Why did v3 have higher mean capture but still fail?** Averages hide distribution. Undercutting is decent on
   price-sensitive nights and terrible on events and nearly-full dates. The pass-rate check needs each decision to
   be near optimal, so a policy that's mediocre everywhere can't hide behind its mean.
7. **Guardrails made things worse. Would you remove them?** Not in production: an out-of-range price is a real
   loss and a guardrail is cheap insurance. I'd change how they work. Clamp to the range and log, instead of
   blocking and retrying, or give the model the constraint up front. And I'd use a bigger model, where block rates
   are lower.
8. **How would you choose between prompting a bigger model and fine-tuning a small one?** Put cost per 1k
   decisions and capture on the same table, as here, and find the cheapest option that clears the bar. If a small
   model is close, fine-tune on oracle-labelled simulator decisions. The harness already produces those labels.
9. **How do you stop the eval set from being overfit?** Keep a held-out scenario set with a different seed that
   isn't used during prompt iteration, and only use it for the final comparison. Rotate scenario kinds and report
   per kind, not just the average.
10. **What does "cost" mean here if the model is local?** The tokens are real. The dollar figure uses an assumed
    per-token rate in config.yaml, so versions can be compared on cost. It isn't a claim about any vendor's price.
    With a hosted model you'd log the provider's billed tokens on the span.

## 6. v2 (17 Nov 2026): confidence intervals in the gate
What changed: `stats.py` adds percentile bootstrap intervals (10,000 resamples of scenarios, seed 0) for pass rate,
capture and cost per passed decision, and a **paired** bootstrap for candidate minus accepted on the same scenarios.
`gate.yaml` now has `mode: ci`. A quality drop fails only if the whole paired 95% interval is below zero, and a
drop above the old tolerance that the interval can't confirm is a warning with a scenario count. Latency and cost
stay as fixed limits. A 150-scenario set (`--set n150`) was run locally for v2, v4 and v5. Numbers below are from
`results/intervals.json`.

- At 30 scenarios the smallest pass-rate drop detectable 80% of the time is about 23 to 29 points. The fixed
  5-point tolerance was mostly measuring noise. Only v1 and v3-regressed have a confirmed quality drop (-27 points,
  CI -43 to -10).
- At 150: v2 20% [14%, 27%] (the 30-run said 27%). v4 -1 point vs v2 (it was -17 at n=30). v5 capture +3.2 points
  [-1.2, +7.8] vs v2, where the fixed gate at n=30 had called it a capture regression. Detectable drop: ~12 points.
- The shared first 30 scenarios gave identical decisions in both runs (greedy decoding), so the change between
  n=30 and n=150 is purely the extra 120 scenarios, not model randomness.
- Cost per 1k passed decisions at n=150: v2 $0.69, v5 $1.27, v4 $1.64 (assumed rate, see config.yaml).

Questions this adds:
11. **How do you set a pass-rate cutoff for an eval gate?** Start from what the eval set can detect, not from a
    round number. With paired per-scenario results, the detectable drop is about (1.96 + 0.84) x sd / sqrt(n). Here
    sd was about 0.45 to 0.57, so 30 scenarios only see drops of about 25 points. Either set the cutoff at what you
    can detect, or grow the set until it detects the drop you care about (about 640 to 1,030 scenarios for 5
    points here). The gate then fails on a confirmed drop and warns on an unconfirmed one.
12. **Why paired rather than comparing two intervals?** Both versions see the same scenarios, and some scenarios
    are hard for every version. Pairing cancels that shared difficulty. Two separate intervals overlapping doesn't
    mean "no difference"; a paired interval answers the actual question: is the candidate worse on these
    scenarios?
13. **Doesn't "fail only on confirmed drops" let regressions through?** It trades false alarms for misses, so it
    needs an eval set big enough to confirm the drops you care about. That's why the warnings print the scenario
    count. For money-moving agents you could flip it to non-inferiority: fail unless the lower bound is above
    -tolerance. At n=30 that would block every change, which says the same thing: the set is too small.
