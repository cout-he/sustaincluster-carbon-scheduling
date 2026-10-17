"""E0 observations around original SustainCluster; no training or source patches.

Run from any working directory. The runner resolves repository-relative source/data
paths at the repository root, because the original loaders expect that directory.
Observer subclasses call through unchanged and retain data for assertions only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import shlex
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label):
    require(math.isfinite(float(actual)), f"{label}: non-finite {actual}")
    require(math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-6),
            f"{label}: actual={actual}, expected={expected}")


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def task_record(task):
    # The final random integer is identity decoration, not a demand feature.
    return [task.job_name.rsplit("_", 1)[0], str(task.arrival_time),
            float(task.duration), float(task.cores_req), float(task.gpu_req),
            float(task.mem_req), float(task.bandwidth_gb),
            str(task.sla_deadline), int(task.origin_dc_id)]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_classes():
    import numpy as np
    import pandas as pd
    import torch
    import yaml
    from envs.task_scheduling_env import TaskSchedulingEnv
    from rewards.predefined.composite_reward import CompositeReward
    from simulation.cluster_manager import DatacenterClusterManager

    class ObservedCluster(DatacenterClusterManager):
        """Records original calls without regenerating or modifying any tasks."""
        def __init__(self, *args, **kwargs):
            self.cohorts = {}
            self.last_results = None
            super().__init__(*args, **kwargs)

        def get_tasks_for_timestep(self, current_time):
            tasks = super().get_tasks_for_timestep(current_time)
            self.cohorts[str(current_time)] = {
                "count": len(tasks), "sha256": digest([task_record(t) for t in tasks])}
            return tasks

        def step(self, *args, **kwargs):
            results = super().step(*args, **kwargs)
            self.last_results = results
            return results

    return np, pd, torch, yaml, TaskSchedulingEnv, CompositeReward, ObservedCluster


def check_step(cluster, results, reward_fn, reward, info, run_name, step, at):
    """Check bookkeeping and exposed physical units without changing physics."""
    totals = {"site_energy_kwh": 0.0, "site_emissions_kg": 0.0,
              "site_energy_cost_usd": 0.0, "tasks_started": 0,
              "tasks_finished": 0, "sla_met": 0, "sla_violated": 0,
              "running_tasks": 0, "pending_tasks": 0,
              "used_cpu_cores": 0.0, "used_gpus": 0.0, "used_memory_gb": 0.0}
    dc_rows = []
    for name, dc in cluster.datacenters.items():
        details = results["datacenter_infos"][name]
        common, battery = details["__common__"], details["agent_bat"]
        row = {"run": run_name, "step": step, "task_time_utc": str(at), "dc": name}
        for attr, total_attr, task_attr, field in (
            ("available_cores", "total_cores", "cores_req", "used_cpu_cores"),
            ("available_gpus", "total_gpus", "gpu_req", "used_gpus"),
            ("available_mem", "total_mem_GB", "mem_req", "used_memory_gb")):
            available, capacity = float(getattr(dc, attr)), float(getattr(dc, total_attr))
            running_sum = sum(float(getattr(task, task_attr)) for task in dc.running_tasks)
            require(-1e-6 <= available <= capacity + 1e-6, f"{name}/{attr}: out of bounds")
            close(capacity - available, running_sum, f"{name}/{attr}: running resource sum")
            row[field] = capacity - available
            totals[field] += row[field]
        energy = float(battery["bat_total_energy_with_battery_KWh"])
        co2_g = float(battery["bat_CO2_footprint"])
        close(energy, common["energy_consumption_kwh"], f"{name} site energy")
        close(energy, results["energy_usage"][name], f"{name} cluster energy")
        close(co2_g, energy * battery["bat_avg_CI"], f"{name} battery grams CO2")
        close(common["carbon_emissions_kg"], co2_g / 1000.0, f"{name} grams/kg conversion")
        close(common["energy_cost_USD"], energy * common["price_USD_kwh"] / 1000.0,
              f"{name} original $/MWh price conversion")
        close(common["running_tasks"], len(dc.running_tasks), f"{name} running count")
        close(common["pending_tasks"], len(dc.pending_tasks), f"{name} pending count")
        close(common["tasks_assigned"], len(common["routed_tasks_this_step"]), f"{name} starts")
        close(common["finished_tasks_count"], len(common["tasks_finished_this_step_objects"]),
              f"{name} finished count")
        close(common["finished_tasks_count"], sum(common["__sla__"].values()), f"{name} SLA count")
        row.update(site_energy_kwh=energy, site_emissions_kg=co2_g / 1000.0,
                   site_energy_cost_usd=float(common["energy_cost_USD"]),
                   battery_ci_g_per_kwh=float(battery["bat_avg_CI"]),
                   manager_ci_g_per_kwh=float(common["ci"]),
                   price_usd_per_mwh=float(common["price_USD_kwh"]),
                   tasks_started=int(common["tasks_assigned"]),
                   tasks_finished=int(common["finished_tasks_count"]),
                   sla_met=int(common["__sla__"]["met"]),
                   sla_violated=int(common["__sla__"]["violated"]),
                   running_tasks=len(dc.running_tasks), pending_tasks=len(dc.pending_tasks),
                   physical_local_day=int(dc.t_m.day), physical_local_hour=float(dc.t_m.hour),
                   ci_index=int(dc.ci_manager.time_step), weather_index=int(dc.weather_manager.time_step),
                   price_index=int(dc.price_manager.index))
        for field in totals:
            if field not in ("used_cpu_cores", "used_gpus", "used_memory_gb"):
                totals[field] += row[field]
        dc_rows.append(row)
    for raw, field in (("transmission_energy_total_kwh", "transmission_energy_kwh"),
                       ("transmission_emissions_total_kg", "transmission_emissions_kg"),
                       ("transmission_cost_total_usd", "transmission_cost_usd")):
        totals[field] = float(results[raw])
    if info is not None:
        close(info["total_energy_kwh"], totals["transmission_energy_kwh"], "top-level energy is transmission only")
        close(info["total_emissions_kg"], totals["transmission_emissions_kg"], "top-level CO2 is transmission only")
    components = reward_fn.get_last_components()
    require(not reward_fn.use_normalization, "E0 original reward must retain normalize=false")
    close(reward, sum(c["weight"] * components[c["name"]] for c in reward_fn.components),
          "weighted original reward")
    totals["reward"] = float(reward)
    totals.update({"reward_" + key: float(value) for key, value in components.items()})
    return totals, dc_rows


def negative_controls(cluster, results, reward_fn, reward, info, run_name, at):
    """Require meaningful corruptions of copies of real observations to fail."""
    name = next(iter(cluster.datacenters))
    corrupted_results = dict(results)
    corrupted_results["datacenter_infos"] = dict(results["datacenter_infos"])
    corrupted_results["datacenter_infos"][name] = dict(results["datacenter_infos"][name])
    corrupted_results["datacenter_infos"][name]["__common__"] = dict(results["datacenter_infos"][name]["__common__"])
    corrupted_results["datacenter_infos"][name]["__common__"]["carbon_emissions_kg"] += 1.0
    corrupted_cluster = copy.copy(cluster)
    corrupted_cluster.datacenters = dict(cluster.datacenters)
    corrupted_cluster.datacenters[name] = copy.copy(cluster.datacenters[name])
    corrupted_cluster.datacenters[name].available_cores -= 1.0
    cases = [("reject_wrong_carbon_units", cluster, corrupted_results, reward),
             ("reject_resource_bookkeeping_error", corrupted_cluster, results, reward),
             ("reject_wrong_weighted_reward", cluster, results, reward + 1.0)]
    passed = []
    for label, candidate_cluster, candidate_results, candidate_reward in cases:
        try:
            check_step(candidate_cluster, candidate_results, reward_fn, candidate_reward,
                       info, run_name, 0, at)
        except AssertionError:
            passed.append(label)
        else:
            raise AssertionError(f"Negative control was not rejected: {label}")
    return passed


def run_one(run_name, settings, classes, sim, dc_config, reward_config, steps, rows, dc_rows):
    np, pd, torch, _, TaskSchedulingEnv, CompositeReward, ObservedCluster = classes
    began = time.perf_counter()
    seed = int(settings["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    direct = run_name.endswith("_direct")
    strategy = "manual_rl" if run_name == "manual_local_replay" else run_name.removesuffix("_repeat").removesuffix("_direct")
    local_sim = copy.deepcopy(sim)
    local_sim.update({key: settings[key] for key in
                      ("shuffle_datacenters", "single_action_mode", "disable_defer_action")})
    start = pd.Timestamp(year=sim["year"], month=sim["month"], day=sim["init_day"],
                         hour=sim["init_hour"], tz="UTC")
    end = start + pd.Timedelta(minutes=steps * settings["timestep_minutes"])
    logger = logging.getLogger("e0")
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)
    cluster = ObservedCluster(copy.deepcopy(dc_config), simulation_year=sim["year"],
                              init_day=int(sim["month"] * 30.5), init_hour=sim["init_hour"],
                              strategy=strategy, tasks_file_path=sim["workload_path"],
                              shuffle_datacenter_order=False, cloud_provider=sim["cloud_provider"],
                              logger=logger)
    require(all(not dc.use_rl_hvac for dc in cluster.datacenters.values()), "E0 must not load HVAC policies")
    reward_fn = CompositeReward(components=reward_config["components"],
                                normalize=reward_config.get("normalize", False),
                                freeze_stats_after_steps=reward_config.get("freeze_stats_after_steps"))
    env = None
    if direct:
        cluster.reset(seed=seed)
    else:
        env = TaskSchedulingEnv(cluster, start, end, reward_fn, sim_config=local_sim,
                                initial_seed_for_resets=seed)
        obs, reset_info = env.reset(seed=seed)
        require(isinstance(reset_info, dict) and isinstance(obs, list), "reset contract")
        require(all(env.observation_space.contains(o) for o in obs), "reset observation shape/values")
    initial_clocks = {
        name: {"local_day": int(dc.t_m.day), "local_hour": float(dc.t_m.hour),
               "timezone_shift": dc.timezone_shift, "ci_index": int(dc.ci_manager.time_step),
               "weather_index": int(dc.weather_manager.time_step), "price_index": int(dc.price_manager.index)}
        for name, dc in cluster.datacenters.items()}
    first = len(rows)
    negative_checks = []
    for step in range(steps):
        at = start + pd.Timedelta(minutes=step * settings["timestep_minutes"])
        if direct:
            results = cluster.step(at, logger=logger)
            reward = reward_fn(results, [], at)
            info = None
        else:
            close((env.current_time - start).total_seconds(), step * 900, "environment clock")
            actions = []
            if strategy == "manual_rl":
                id_to_action = {dc.dc_id: i for i, dc in enumerate(cluster.datacenters.values())}
                actions = [id_to_action[t.origin_dc_id] for t in env.current_tasks]
            obs, reward, terminated, truncated, info = env.step(actions)
            results = cluster.last_results
            require(bool(terminated) == (step + 1 == steps), "original termination clock")
            require(terminated == truncated, "original done/truncated contract")
            require(all(env.observation_space.contains(o) for o in obs), "step observation shape/values")
        values, details = check_step(cluster, results, reward_fn, reward, info, run_name, step, at)
        if step == 0:
            negative_checks = negative_controls(cluster, results, reward_fn, reward, info, run_name, at)
        cohort = cluster.cohorts[str(at)]
        values.update(run=run_name, step=step, task_time_utc=str(at), incoming_tasks=cohort["count"],
                      cohort_sha256=cohort["sha256"], in_transit_tasks=len(env.in_transit_tasks) if env else 0,
                      deferred_tasks=sum(bool(t.temporarily_deferred) for t in env.current_tasks) if env else 0)
        rows.append(values)
        dc_rows.extend(details)
    run_rows = rows[first:]
    total_incoming = sum(row["incoming_tasks"] for row in run_rows)
    total_started = sum(row["tasks_started"] for row in run_rows)
    total_finished = sum(row["tasks_finished"] for row in run_rows)
    require(total_incoming > 0, "No actual workload tasks in requested interval")
    close(total_incoming, total_started + run_rows[-1]["pending_tasks"] +
          run_rows[-1]["in_transit_tasks"] + run_rows[-1]["deferred_tasks"], "incoming task conservation")
    close(total_started, total_finished + run_rows[-1]["running_tasks"], "started task conservation")
    return {"run": run_name, "status": "passed", "steps": steps,
            "elapsed_seconds": time.perf_counter() - began, "task_start_utc": str(start),
            "task_end_exclusive_utc": str(end), "cluster_random_init_day": cluster.random_init_day,
            "cluster_random_init_hour": cluster.random_init_hour, "initial_physical_clocks": initial_clocks,
            "negative_controls_passed": negative_checks,
            "cohort_sequence_sha256": digest([r["cohort_sha256"] for r in run_rows]),
            **{key: sum(row[key] for row in run_rows) for key in
               ("incoming_tasks", "tasks_started", "tasks_finished", "sla_met", "sla_violated",
                "site_energy_kwh", "site_emissions_kg", "site_energy_cost_usd",
                "transmission_energy_kwh", "transmission_emissions_kg", "transmission_cost_usd", "reward")},
            "final_running_tasks": run_rows[-1]["running_tasks"],
            "final_pending_tasks": run_rows[-1]["pending_tasks"],
            "final_in_transit_tasks": run_rows[-1]["in_transit_tasks"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("e0_config.json"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--strategies", nargs="+")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("artifacts") / "baseline")
    args = parser.parse_args()
    settings = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    steps = args.steps if args.steps is not None else settings["steps"]
    require(steps > 0 and settings["timestep_minutes"] == 15, "positive steps and original 15-minute interval required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    execution = {"started_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
                 "command": shlex.join([sys.executable, *sys.argv]), "python": sys.version,
                 "platform": platform.platform(), "cwd_before": str(Path.cwd()),
                 "repository_root": str(ROOT), "config": settings, "requested_steps": steps}
    write_json(output / "execution.json", execution)
    summaries, rows, dc_rows, checks = [], [], [], []
    os.chdir(ROOT)
    try:
        classes = load_classes()
        yaml = classes[3]
        def read_yaml(path, key):
            return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))[key]
        sim = read_yaml(settings["simulation_config"], "simulation")
        dcs = read_yaml(settings["datacenter_config"], "datacenters")
        reward = read_yaml(settings["reward_config"], "reward")
        execution["input_sha256"] = {path: file_digest(ROOT / path) for path in
                                      (settings["simulation_config"], settings["datacenter_config"],
                                       settings["reward_config"], sim["workload_path"])}
        for strategy in args.strategies or settings["strategies"]:
            started = time.perf_counter()
            try:
                result = run_one(strategy, settings, classes, sim, dcs, reward, steps, rows, dc_rows)
            except Exception as exc:
                result = {"run": strategy, "status": "failed", "elapsed_seconds": time.perf_counter() - started,
                          "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            summaries.append(result)
            print(json.dumps({key: value for key, value in result.items() if key not in
                              ("initial_physical_clocks", "traceback")}), flush=True)
        successful = [item for item in summaries if item["status"] == "passed"]
        if len(successful) > 1:
            checks.append({"check": "identical_incoming_cohorts_across_successful_runs",
                           "passed": len({item["cohort_sequence_sha256"] for item in successful}) == 1,
                           "runs": [item["run"] for item in successful]})
        indexed = {item["run"]: item for item in successful}
        for other in ("local_only_repeat", "local_only_direct"):
            if "local_only" in indexed and other in indexed:
                left = [{k: v for k, v in r.items() if k != "run"} for r in rows if r["run"] == "local_only"]
                right = [{k: v for k, v in r.items() if k != "run"} for r in rows if r["run"] == other]
                checks.append({"check": "local_only_step_exact_equality_with_" + other,
                               "passed": left == right})
        execution["status"] = "passed" if summaries and all(s["status"] == "passed" for s in summaries) and all(c["passed"] for c in checks) else "failed"
    except Exception as exc:
        execution.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        print(execution["traceback"], file=sys.stderr)
    finally:
        execution.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - began)
        write_json(output / "execution.json", execution)
        write_json(output / "summary.json", summaries)
        write_json(output / "checks.json", checks)
        write_csv(output / "summary.csv", [{k: v for k, v in s.items() if k not in ("traceback", "initial_physical_clocks")} for s in summaries])
        write_csv(output / "steps.csv", rows)
        write_csv(output / "datacenters.csv", dc_rows)
    return 0 if execution["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
