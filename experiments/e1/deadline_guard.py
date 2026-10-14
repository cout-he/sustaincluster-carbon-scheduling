"""Frozen RiskOnly buffer; heuristic one-step protection, no SLA guarantee."""
from .cars_state import ArrivalHistory
from .contracts import STEP
from .policies import latest_start

VERSION='e1-7b-deadline-guard-v1'

class DeadlineGuard:
    def __init__(self,start,capacity,enabled=True):
        if not isinstance(enabled,bool):raise ValueError('Guard switch must be Boolean')
        self.enabled=enabled;self.history=ArrivalHistory(start,capacity);self._pressure=None;self._now=None

    def observe(self,snapshot):
        self.history.observe(snapshot.now,snapshot.arrivals)
        self._now=snapshot.now;self._pressure=self.history.pressure(snapshot.now)

    def check(self,spec,now):
        pressure=self._pressure if now==self._now else self.history.pressure(now)
        buffer=pressure.buffer_steps if self.enabled else 0
        solo=now+STEP<=latest_start(spec)
        allowed=solo and now+STEP*(1+buffer)<=latest_start(spec)
        return dict(allowed=allowed,reason='allowed' if allowed else ('no_grid_slack' if not solo else 'deadline_guard'),
            buffer_steps=buffer,solo_slack_minutes=(latest_start(spec)-now).total_seconds()/60,
            rho=pressure.rho,rho_resources=pressure.rho_resources,history_cold=pressure.rho is None)
