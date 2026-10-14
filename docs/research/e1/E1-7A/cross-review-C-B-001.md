# Independent read-only cross-review of C and B

Scope: `counterfactual.py`, `counterfactual_cases.py`,
`counterfactual_audit.py`, their three test modules, and
`carbon_opportunity.py` with its tests. This review did not load a real
workload, environment or Test window and made no changes to those modules.

## Finding requiring correction before the formal run

The original reviewed C audit resource-hours expectation used
`15 * math.ceil(spec.duration / 15)`. A duration
`math.nextafter(15., math.inf)` converts to exactly15 minutes in the original
Pandas Timedelta, even though raw float ceil returns two slots. Thus that audit
could reject a correct original-executor pair. The finding was sent to root;
the owning agent corrected only the new C audit to integer ceil of the original
duration nanoseconds and added a near-boundary test. The source correction was
confirmed on reread at `counterfactual_audit.py` near line202; the owner's test
and campaign gates must supply actual execution evidence. No old risk/history
formula should be altered to address this independent audit discrepancy.

No other blocker was found in the reviewed implementation for the registered
Train cases and current E1 model.

## C invariants confirmed by code

- LocalExecutor still owns arrival, release, ranking, allocation, raw finish,
  physics and integral records. It calls the override only after the original
  current can-fit check. Occurrence and case guards reject absent, repeated or
  capacity-blocked intervention targets.
- Exactly one target is registered per branch; only its Boolean Execute/Defer
  is overridden. All unmatched decisions call the inherited immediate EDF
  decision. Same-step starts remove tasks from pending and add known running
  holds before later candidate risk observations.
- ArrivalHistory observes only snapshot arrivals. Its existing seen-ID guard
  prevents repeated arrival work. The diagnostic computes frozen RiskOnly and
  Task Full signals on the same current state; those signals do not decide the
  intervention or continuation.
- The prefix includes immutable current candidate, actual arrived ledger,
  pending/running ownership, resource capacity/availability, current CI/weather,
  past events/steps, observer history/running/pending state, accumulated energy
  and carbon, and the reachable Python physical-model state. Arrays, RNG state
  and alias relationships are fingerprinted; unsupported opaque objects fail
  closed. Target branch action is deliberately excluded so the two potential
  actions can share one pre-intervention state.
- Both full replays must complete the same finite cohort. Their own Drain
  results remain separate; the slower required endpoint determines common
  accounting, and `extend_idle` continues each branch's persistent power model
  against real frozen samples. Missing Drain coverage raises rather than wraps
  data or leaves unfinished tasks silently excluded.
- Raw finish determines deadline outcome; ceil-grid finish determines resource
  release. Identity uniqueness plus final ledger counts avoid repeated miss
  counting. Candidate/known-other/later-arrival partitions come from the
  observed seen-ID set at intervention, not future outcome labels.

## Known-release screen boundary

`known_release_screen` inspects current running tasks and already-arrived
strict EDF predecessors only. It compares the hypothetical candidate hold NOW
or one grid later at each selected predecessor's latest legal grid start. It
uses actual-known ceil releases and original fsum resources. It excludes
expired or out-of-horizon predecessors and never reads a future arrival or CI.
It is a single-predecessor conflict feature, not an existing reservation,
multi-job feasibility proof, future miss probability or prediction that the
candidate really starts exactly one step later. The output explicitly records
`existing_reservations=0` and `reservation_conflict_assessed=False`.

## B accounting and proxy boundary

Genuine interval accounting recomputes IT/HVAC kWh and each component's kgCO2e
from unchanged measured kW and current CI, checks supplied totals, excludes
transfer energy, preserves zero-load idle power, and reports CI-bin and
energy-weighted CI decompositions. It does not assume HVAC is a fixed base load.

The Train OLS is explicitly descriptive, can flag rank deficiency and negative
dynamic predictions, and provides neither causal per-task power nor HVAC.
Execution-window opportunities use all occupied15-minute slots and future real
CI with explicit `offline_clairvoyant_diagnostic` labels and coverage rejection.
They exclude intercept, HVAC, background and altered subsequent task starts.
Consequently the proxy's sign/size must remain separate from the full paired
LocalExecutor physical result; root's registered real replay is necessary for
any total-cluster carbon claim.

## Nonblocking interpretation and robustness risks

- Prefix equality covers inspected Python state, not arbitrary external side
  effects or global RNGs. The current upstream power path is deterministic with
  frozen reset/input configuration; do not generalize the fingerprint claim to
  unknown external models. Timedelta serialization uses total_seconds rather
  than an exact nanosecond tag, which is harmless for the current15-minute
  physical clock but should be revisited if submicrosecond model state is added.
- The pair audit checks net policy-defer difference equals one and the actual
  target decision; the current wrapper independently guarantees no unmatched
  active defer. An optional stronger artifact guard would assert absolute
  defer counts0/1 to detect hypothetical equally corrupted extra defers in both
  saved branches. This is not a reachable current-policy failure.
- Report negative dynamic IT proxy values and rank warnings if any occur; they
  cannot be presented as physically negative task energy. Reporting selection
  and eligible/coverage denominators prevents a few chosen proxy opportunities
  from being mistaken for general opportunity prevalence.
- The12 predeclared pairs are descriptive interventions under fixed future
  inputs and EDF continuation. Their outcomes do not certify universal causality,
  predictive probabilities, or an online resource-reservation guarantee.

The remaining acceptance gate is the actual registered full Drain/physical
campaign and replay evidence, not this source review alone.
