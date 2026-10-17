"""Reliable auxiliary labels; the teacher never replaces the executing actor."""
import math

import numpy as np


def _get(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def teacher_weight(config, physical_steps=0, total_steps=None):
    options = config.get('teacher', config)
    initial = float(options.get('initial_weight', .1))
    fraction = float(options.get('decay_fraction', .5))
    if not math.isfinite(initial) or initial < 0 or not 0 < fraction <= 1:
        raise ValueError('Invalid frozen teacher decay schedule')
    if total_steps is None:
        total_steps = config.get('budget', {}).get('total_physical_steps', 65536)
    if total_steps <= 0 or physical_steps < 0:
        raise ValueError('Teacher progress requires nonnegative steps and a positive budget')
    return initial * max(0., 1. - physical_steps / (fraction * total_steps))


def teacher_label(mask, risk, carbon, config, *, physical_steps=0, total_steps=None):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (2,) or not mask.any():
        raise ValueError('Task teacher requires a nonempty two-action mask')
    legal = np.flatnonzero(mask)
    threshold = float(config.get('teacher', config).get('threshold_kg', 0.))
    gain = float(_get(carbon, 'robust_carbon_gain_kg', 0.))
    reliable = bool(_get(risk, 'risk_estimation_valid', _get(risk, 'valid', False))
                    and _get(carbon, 'counterfactual_valid', False)
                    and float(_get(risk, 'risk_confidence', _get(risk, 'confidence', 0.))) > 0
                    and float(_get(carbon, 'forecast_confidence', 0.)) > 0
                    and math.isfinite(gain))
    wait = (reliable and gain > threshold
            and float(_get(risk, 'safe_wait_budget_minutes', 0.)) >= 15.
            and float(_get(risk, 'delta_deadline_misses', 1.)) <= 0)
    action = 1 if wait and mask[1] else (0 if mask[0] else int(legal[0]))
    # The optimizer owns the time-decayed auxiliary coefficient. This is only
    # confidence, so the training schedule cannot accidentally be applied twice.
    confidence = min(float(_get(risk, 'risk_confidence', _get(risk, 'confidence', 0.))),
                     float(_get(carbon, 'forecast_confidence', 0.)))
    enabled = config.get('teacher', config).get('enabled', True)
    return dict(action=action, reliable=reliable,
                weight=min(1., confidence) if enabled and reliable and len(legal) > 1 else 0.,
                reason='reliable_positive_robust_gain' if action == 1 and reliable else
                       'forced_legal_action' if len(legal) == 1 else
                       'unreliable_estimate' if not reliable else 'no_supported_carbon_wait')
