"""Shared completion responsibility, separate from task Deadline and V2."""
from dataclasses import dataclass

CLEAR_MINUTE = 3960
HORIZON_MINUTE = 4320
FAILURE = 'EDF_REFERENCE_COMPLETION_GUARD_FAILED'


@dataclass
class CompletionGuard:
    failure_latched: bool = False
    reference_guard_failure_count: int = 0
    first_failure_minute: float | None = None

    @staticmethod
    def accepts(plan):
        return plan.max_release <= CLEAR_MINUTE

    def inspect_reference(self, plan):
        passes = self.accepts(plan)
        if not passes:
            self.reference_guard_failure_count += 1
            if self.first_failure_minute is None:
                self.first_failure_minute = plan.snapshot.minute
            self.failure_latched = True
        return passes
