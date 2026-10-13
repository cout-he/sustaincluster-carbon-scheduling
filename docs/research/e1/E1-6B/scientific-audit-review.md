# Independent scientific audit

`scientific_audit.py` is an offline accounting audit. It requires a completed E1-6B development-Validation campaign and checks 57 method/scenario results plus 12 historical baseline regressions. It runs no simulator, policy, checkpoint inference, training or parameter selection, and has no Test input. Full frozen environmental coverage is used only to check already recorded intervals, never as an online decision input.

The audit derives task features and ownership from the immutable frozen cohort, and derives starts, nominal finishes, deadlines and observed releases independently from each ledger. Nonpreemptive occupancy is `[start, ceil(finish / 15 min))`; SLA uses the nominal finish. It reconstructs all pending/running sets, whole-queue EDF order and stable ties, candidate-by-candidate available resources and capacity-first waiting, and compares those with policy events, policy intervals, the task waiting ledger and interval counts. It verifies no arrival enters Drain and all tasks finish before the common observation end.

The audit recomputes each interval's energy as power in kW times 0.25 h, and carbon as total IT+HVAC kWh times the corresponding frozen CI in g/kWh divided by 1000. It checks the real CI/weather samples, the IT+HVAC sum, zero transmission quantities, units, resource hours, utilization, JCT, waiting, pending mean/max, Drain endpoint and final deadline numerator/denominator. Numeric comparisons use relative tolerance `1e-12` and general absolute tolerance `1e-9`; stricter unit/fraction checks are stated in code.

For actor results it compares the saved real start/defer/capacity sets with the executor records, checks current observed CI and pre-action queue counts, verifies the first miss at `ceil(deadline / 15 min)`, prohibits repeated task penalties, checks the exact final miss set and independently recomputes the registered reward. The reward event belongs to the next boundary after the action interval.

For EDF, EDF+Carbon, RiskOnly and frozen Task Full, full ledger/event/policy-ledger/policy-event bytes must match E1-6A. Physical and policy intervals are compared over their shared endpoint. If both observation ends match, full interval files must match byte-for-byte; if the ends differ, the report explicitly retains both endpoints and totals and only calls the shared prefix a regression. It does not substitute an older-scope carbon total for the new common-window result.

All read input/current-output/historical files have before/after SHA256 guards. Current scientific files are also bound to the already passed campaign's recorded output hashes. The script refuses an existing audit directory; failure retains its evidence rather than replacing results.

After the real Validation campaign passes:

```bash
python research_notes/E1/E1-6B/scientific_audit.py
```

Hand-calculated synthetic checks completed in WSL:

```text
python -m pytest research_notes/E1/E1-6B/test_scientific_audit.py -q \
  -p no:cacheprovider --basetemp research_notes/E1/E1-6B/scientific-audit-tests-tmp-001
...................                                                      [100%]
19 passed in 1.85s
Exit code: 0
```

The synthetic case intentionally uses a tight 0.5 SLA multiplier to exercise a first deadline penalty before grid release; it changes no actual study input. Counterexamples include mutated task duration/deadline, early release, wrong resource occupancy/capacity flags, energy/carbon units, CI, queue/order, repeated or delayed first penalties, reward and Drain boundary.

One historical EDF 1× archive was independently audited read-only to confirm actual schema compatibility. Its 8,708 tasks, 403 misses, 2,156 capacity waits and 223 intervals passed; independent IT/HVAC/total energy were 33,980.60526546528 / 36,221.73815878744 / 70,202.34342425273 kWh, carbon 9,824.889694300633 kg, and maximum resource occupancy discrepancy was exactly zero. `scientific-audit-reference-001/receipt.json` binds unchanged before/after file hashes. No new Validation or simulation was executed for this check.

Passing this audit establishes consistency of stored execution and accounting evidence. It does not prove the IT/HVAC model represents a particular real facility, that a heuristic filter guarantees deadlines, that PPO improves performance, or that a development Validation result generalizes to Test.

## Final measured run

The complete registered Validation matrix was subsequently audited: 57/57 methods passed independent task/resource/EDF/first-miss/IT+HVAC/CI-carbon reconstruction, with 12/12 historical heuristic regressions. All read artifact hashes were identical before and after. The separate 36-method filter diagnostic audit also passed per-candidate frozen budgets, current-only conditions, same-step updates and denominator checks. Exact replay of all 57 scientific results passed. These completed results replace no old provenance and establish accounting consistency, not algorithm superiority or Test generalization.
