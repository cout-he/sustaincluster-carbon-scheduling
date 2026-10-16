"""Development-only model selection and bounded Train-only SLA adaptation."""
import math


def eligible(summary):
    return bool(summary.get('fixed_window_complete') and summary.get('all_arrivals_observed')
        and summary.get('unfinished', 1) == 0 and summary.get('deadline_right_censored', 1) == 0
        and summary.get('temperature_violations', 1) == 0
        and math.isfinite(summary.get('dmr', float('nan'))) and 0 <= summary['dmr'] <= .05)


def aggregate(summaries):
    if not summaries:
        raise ValueError('Selection requires predeclared held-out development scenarios')
    if any(s.get('scenario_role') != 'development' for s in summaries):
        raise ValueError('Repeated model selection is restricted to development, never Train or sealed Validation/Test')
    arrived = sum(s['arrived'] for s in summaries)
    missed = sum(s['deadline_misses'] for s in summaries)
    result = dict(scenario_ids=[s['scenario_id'] for s in summaries], arrived=arrived,
        completed=sum(s['completed'] for s in summaries),
        unfinished=sum(s['unfinished'] for s in summaries),
        censored=sum(s['deadline_right_censored'] for s in summaries),
        temperature_violations=sum(s['temperature_violations'] for s in summaries),
        dmr=missed/arrived if arrived else 0., deadline_misses=missed,
        carbon_kg=math.fsum(s['carbon_kg'] for s in summaries),
        incremental_carbon_kg=math.fsum(s['incremental_carbon_kg'] for s in summaries),
        feasible=all(eligible(s) for s in summaries),
        all_work_accounted=all(s.get('complete_episode_for_constraint', False) for s in summaries))
    result['deadline_key'] = [result['unfinished']+result['censored'],
        int(not result['all_work_accounted']), result['temperature_violations'], result['dmr'], result['carbon_kg']]
    return result


def update_lambda(value, summary, statistics, config):
    maximum, eta, target = config['lambda_max'], config['eta'], config['target_dmr']
    if summary.get('scenario_role') != 'train':
        raise ValueError('Deadline multiplier can only use complete Train episode outcomes')
    reason = None
    if not summary.get('complete_episode_for_constraint'):
        reason = 'incomplete_or_censored'
    elif statistics.get('capacity_pressure', statistics.get('pressure_proxy', 0.)) > config['max_capacity_pressure_for_update']:
        reason = 'high_pressure_proxy_not_a_proof_of_infeasibility'
    elif summary.get('temperature_violations', 0):
        reason = 'thermal_safety_failure'
    updated = value if reason else min(maximum, max(0., value+eta*(summary['dmr']-target)))
    return float(updated), dict(previous=float(value), value=float(updated), target=target,
        observed_dmr=summary['dmr'], applied=reason is None, skipped_reason=reason,
        role='train', interpretation='bounded heuristic; no feasibility guarantee')


def advance_curriculum(state, summary, config):
    result = dict(state)
    successful = eligible(summary)
    result['success_streak'] = result.get('success_streak', 0)+1 if successful else 0
    if (result['success_streak'] >= config['successful_episodes_to_promote']
            and result['stage'] < config['max_stage']):
        result['stage'] += 1
        result['success_streak'] = 0
    return result
