"""Train-only Ridge forecasts with embargoed chronological residual calibration.

Index zero is the observed CI at the current decision boundary. Residual samples
are entire horizon vectors, preserving their empirical temporal dependence.
Neither fitting nor calibration opens a non-Train scenario environment.
"""
from bisect import bisect_right
from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CIForecast:
    mean: np.ndarray
    residual_scenarios: np.ndarray
    confidence: float
    valid: bool
    reason: str

    def __post_init__(self):
        mean = np.array(self.mean, dtype=float, copy=True)
        residuals = np.array(self.residual_scenarios, dtype=float, copy=True)
        if (mean.ndim != 1 or not mean.size or residuals.ndim != 2
                or residuals.shape[1] != mean.size or not residuals.shape[0]
                or not np.isfinite(mean).all() or not np.isfinite(residuals).all()
                or (mean < 0).any() or not 0 <= self.confidence <= 1):
            raise ValueError('Invalid CI forecast arrays or confidence')
        mean.flags.writeable = residuals.flags.writeable = False
        object.__setattr__(self, 'mean', mean)
        object.__setattr__(self, 'residual_scenarios', residuals)

    @property
    def scenarios(self):
        """Absolute nonnegative CI trajectories, in g CO2e/kWh."""
        return np.maximum(0., self.mean[None, :] + self.residual_scenarios)


class CIForecaster:
    SCHEMA = 'joint-v3-train-ridge-ci-v1'

    def __init__(self, horizon=16, lags=8, ridge=1., scenarios=8):
        if (isinstance(horizon, bool) or isinstance(lags, bool)
                or int(horizon) != horizon or int(lags) != lags
                or horizon < 2 or lags < 1 or scenarios < 2
                or int(scenarios) != scenarios or not np.isfinite(ridge) or ridge <= 0):
            raise ValueError('CI forecast dimensions and Ridge penalty must be positive')
        self.horizon, self.lags, self.ridge = int(horizon), int(lags), float(ridge)
        self.scenario_count = int(scenarios)
        self._coef = self._x_mean = self._x_scale = None
        self._residuals = np.empty((0, self.horizon))
        self._confidence = 0.
        self.audit = {}

    @staticmethod
    def _time(value):
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError('CI times must be finite timezone-aware timestamps')
        return stamp.tz_convert('UTC')

    def _features(self, values, now):
        hour = now.hour + now.minute / 60
        week = now.dayofweek + hour / 24
        return np.r_[values[-self.lags:][::-1],
                     np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                     np.sin(2*np.pi*week/7), np.cos(2*np.pi*week/7)]

    def fit(self, scenarios):
        scenarios = tuple(scenarios)
        # Validate every role before accessing even the first environment payload.
        if not scenarios or any(getattr(s, 'role', None) != 'train' for s in scenarios):
            raise ValueError('CI fitting and calibration require Train scenarios only')
        observations, environment_fingerprints = {}, set()
        for scenario in scenarios:
            workload = getattr(scenario, 'workload', None)
            arrival_end = self._time(workload.end) if workload is not None else None
            samples = tuple(scenario.environment.samples)
            # Train environments include a long drain tail which can overlap later
            # development dates. Its CI never enters fitting or residual calibration.
            eligible = (s for s in samples if arrival_end is None or self._time(s.time) < arrival_end)
            record = tuple((self._time(s.time), float(s.ci_g_per_kwh)) for s in eligible)
            fingerprint = sha256(repr(record).encode()).hexdigest()
            if fingerprint in environment_fingerprints:
                continue
            environment_fingerprints.add(fingerprint)
            for stamp, value in record:
                if not np.isfinite(value) or value < 0:
                    raise ValueError('Train CI observations must be nonnegative and finite')
                if stamp in observations and observations[stamp] != value:
                    raise ValueError('Train environments disagree on an observed CI timestamp')
                observations[stamp] = value
        ordered = sorted(observations)
        rows = []
        # Build only contiguous 15 minute histories and targets; do not bridge gaps.
        for index in range(self.lags-1, len(ordered)-self.horizon+1):
            left, right = index-self.lags+1, index+self.horizon-1
            if ordered[right]-ordered[left] != pd.Timedelta(minutes=15*(right-left)):
                continue
            lag = np.array([observations[t] for t in ordered[left:index+1]])
            labels = np.array([observations[t] for t in ordered[index:right+1]])
            rows.append((ordered[index], ordered[right], self._features(lag, ordered[index]), labels))
        self._coef = self._x_mean = self._x_scale = None
        self._residuals = np.empty((0, self.horizon))
        self._confidence = 0.
        self.audit = dict(role='train', unique_environments=len(environment_fingerprints),
                          unique_observations=len(observations), fit_rows=0, calibration_vectors=0,
                          input_rule='train_arrival_window_excluding_drain_tail',
                          eligible_ci_sha256=sha256(json.dumps(
                              [(stamp.isoformat(), observations[stamp]) for stamp in ordered],
                              separators=(',', ':')).encode()).hexdigest(),
                          fitting_rule='chronological_80_percent_with_target_embargo')
        if len(rows) < 2*self.horizon + 4:
            self.audit['fallback_reason'] = 'insufficient_train_history'
            return self
        cutoff = rows[int(.8*len(rows))][0]
        fitting = [r for r in rows if r[1] < cutoff]
        calibration = [r for r in rows if r[0] >= cutoff]
        if not fitting or len(calibration) < 2:
            self.audit['fallback_reason'] = 'insufficient_embargoed_calibration'
            return self
        x = np.stack([r[2] for r in fitting])
        y = np.stack([r[3] for r in fitting])
        self._x_mean = x.mean(axis=0)
        self._x_scale = x.std(axis=0)
        self._x_scale[self._x_scale < 1e-10] = 1.
        design = np.column_stack([np.ones(len(x)), (x-self._x_mean)/self._x_scale])
        penalty = np.eye(design.shape[1])*self.ridge
        penalty[0, 0] = 0.
        self._coef = np.linalg.solve(design.T@design+penalty, design.T@y)
        cx = np.stack([r[2] for r in calibration])
        cy = np.stack([r[3] for r in calibration])
        predictions = np.maximum(0., np.column_stack([np.ones(len(cx)),
                                  (cx-self._x_mean)/self._x_scale])@self._coef)
        predictions[:, 0] = cy[:, 0]
        self._residuals = cy-predictions
        # Fixed engineering scale and training labels only; no held-out normalization.
        rms = float(np.sqrt(np.mean(self._residuals[:, 1:]**2)))
        scale = max(float(y.std()), 1.)
        self._confidence = float(1./(1.+rms/scale))
        self.audit.update(fit_rows=len(fitting), calibration_vectors=len(calibration),
                          fit_origin_start=fitting[0][0].isoformat(),
                          fit_label_end=max(r[1] for r in fitting).isoformat(),
                          calibration_origin_start=calibration[0][0].isoformat(),
                          calibration_label_end=calibration[-1][1].isoformat(),
                          calibration_rmse_g_per_kwh=rms)
        return self

    def predict(self, history, now):
        now = self._time(now)
        history = tuple(history)
        # Histories follow the environment's chronological contract. Locate the
        # boundary in logarithmic time so repeated decisions do not rescan a month.
        # Timestamp filtering precedes any future sample's CI access.
        stop = bisect_right(history, now, key=lambda s: self._time(s.time))
        observed = [(self._time(s.time), s) for s in history[max(0, stop-self.lags):stop]]
        current = float(observed[-1][1].ci_g_per_kwh) if observed else 0.
        if not np.isfinite(current) or current < 0:
            current = 0.
        reason = None
        if self._coef is None:
            reason = 'unfitted_train_forecaster'
        elif len(observed) < self.lags:
            reason = 'insufficient_causal_history'
        else:
            stamps = [t for t, _ in observed[-self.lags:]]
            values = np.array([float(s.ci_g_per_kwh) for _, s in observed[-self.lags:]])
            expected = tuple(pd.date_range(now-pd.Timedelta(minutes=15*(self.lags-1)),
                                           now, freq='15min'))
            if tuple(stamps) != expected or not np.isfinite(values).all() or (values < 0).any():
                reason = 'missing_or_invalid_causal_history'
        if reason is not None:
            return CIForecast(np.full(self.horizon, current),
                              np.zeros((self.scenario_count, self.horizon)), 0., False, reason)
        row = np.r_[1., (self._features(values, now)-self._x_mean)/self._x_scale]
        mean = np.maximum(0., row@self._coef)
        mean[0] = current
        # Deterministic whole-vector empirical draws, never independent per-step noise.
        indices = np.linspace(0, len(self._residuals)-1, self.scenario_count).astype(int)
        residuals = self._residuals[indices].copy()
        residuals[:, 0] = 0.
        return CIForecast(mean, residuals, self._confidence, True, 'ok')

    def export_state(self):
        state = dict(schema=self.SCHEMA, horizon=self.horizon, lags=self.lags, ridge=self.ridge,
                     scenarios=self.scenario_count, confidence=self._confidence, audit=self.audit.copy(),
                     coef=None if self._coef is None else self._coef.tolist(),
                     x_mean=None if self._x_mean is None else self._x_mean.tolist(),
                     x_scale=None if self._x_scale is None else self._x_scale.tolist(),
                     residuals=self._residuals.tolist())
        state['identity_sha256'] = sha256(json.dumps(state, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return state

    @property
    def identity_sha256(self):
        return self.export_state()['identity_sha256']

    @classmethod
    def from_state(cls, state):
        raw = dict(state)
        expected = raw.pop('identity_sha256', None)
        actual = sha256(json.dumps(raw, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if expected != actual or raw.get('schema') != cls.SCHEMA or raw['audit'].get('role') != 'train':
            raise ValueError('CI checkpoint identity or Train provenance mismatch')
        model = cls(raw['horizon'], raw['lags'], raw['ridge'], raw['scenarios'])
        model._coef = None if raw['coef'] is None else np.asarray(raw['coef'], dtype=float)
        model._x_mean = None if raw['x_mean'] is None else np.asarray(raw['x_mean'], dtype=float)
        model._x_scale = None if raw['x_scale'] is None else np.asarray(raw['x_scale'], dtype=float)
        model._residuals = np.asarray(raw['residuals'], dtype=float).reshape(-1, model.horizon)
        model._confidence, model.audit = float(raw['confidence']), dict(raw['audit'])
        return model
