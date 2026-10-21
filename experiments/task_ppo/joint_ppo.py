"""PPO over one ragged autoregressive binary proposal per physical slot.

Conditional inputs are teacher-forced from the recorded original prefix. Joint
likelihoods include every original zero/one, including DDF-rejected proposals.
Rewards, GAE, value samples and clipped ratios are physical-slot quantities.
"""
import math

import torch

from experiments.sca_ppo.ppo import PPOTrainer, compute_gae, _finite_tensor, _boolean_vector


INPUT_DIM=48


def _ragged(task_states, task_actions, input_dim=INPUT_DIM):
    if len(task_states)!=len(task_actions):
        raise ValueError('task_states and task_actions need the same physical-slot count')
    states=[];actions=[]
    for slot_states,slot_actions in zip(task_states,task_actions):
        if len(slot_states)!=len(slot_actions):
            raise ValueError('Each original candidate requires one binary proposal')
        if not len(slot_states):
            states.append(torch.empty((0,input_dim),dtype=torch.float32))
            actions.append(torch.empty(0,dtype=torch.long));continue
        row=_finite_tensor(slot_states,'teacher-forced task states')
        if row.ndim!=2 or row.shape[1]!=input_dim or not (row[:,-1]==1).all():
            raise ValueError('Actor queries require48 features and a final query flag1')
        try:choices=torch.as_tensor(slot_actions,device='cpu').detach().clone()
        except (TypeError,ValueError,RuntimeError) as exc:
            raise ValueError('Original binary proposals must be integer vectors') from exc
        if (choices.dtype not in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8)
                or choices.shape!=(len(row),) or ((choices<0)|(choices>1)).any()):
            raise ValueError('Original proposals must be integer0/1 with the candidate shape')
        states.append(row);actions.append(choices.long())
    return states,actions


def evaluate_joint(model,task_states,task_actions):
    """Return float64 slot log-likelihoods and conditional-entropy sums.

    Empty slots receive arithmetic zero placeholders, never policy samples.
    The entropy sum is a diagnostic for the fixed observed prefixes, not an
    exact autoregressive joint-entropy gradient or a reward/optimizer bonus.
    """
    states,actions=_ragged(task_states,task_actions,model.input_dim)
    slots=len(states)
    logs=torch.zeros(slots,dtype=torch.float64)
    entropies=torch.zeros(slots,dtype=torch.float64)
    selected=[index for index,row in enumerate(states) if len(row)]
    if not selected:return logs,entropies
    flat_states=torch.cat([states[index] for index in selected])
    flat_actions=torch.cat([actions[index] for index in selected])
    owner=torch.cat([torch.full((len(states[index]),),index,dtype=torch.long) for index in selected])
    conditional_logs,conditional_entropy,_=model.evaluate_actions(flat_states,flat_actions)
    logs=logs.index_add(0,owner,conditional_logs.to(torch.float64))
    entropies=entropies.index_add(0,owner,conditional_entropy.to(torch.float64))
    if not torch.isfinite(logs).all() or not torch.isfinite(entropies).all():
        raise ValueError('Joint probabilities and diagnostics must remain finite')
    return logs,entropies


def clipped_joint_objective(new_log_probs,old_log_probs,advantages,clip_range):
    """One ratio and one PPO clip per physical slot, never per candidate."""
    new=torch.as_tensor(new_log_probs,dtype=torch.float64,device='cpu')
    old=torch.as_tensor(old_log_probs,dtype=torch.float64,device='cpu').detach()
    advantage=torch.as_tensor(advantages,dtype=torch.float64,device='cpu').detach()
    if (new.ndim!=1 or not len(new) or old.shape!=new.shape or advantage.shape!=new.shape
            or not math.isfinite(clip_range) or clip_range<0
            or not all(torch.isfinite(x).all() for x in (new,old,advantage))):
        raise ValueError('Joint PPO requires finite equally shaped nonempty slot vectors')
    difference=new-old;ratio=difference.exp()
    if not torch.isfinite(ratio).all():
        raise ValueError('Joint importance ratio is nonfinite; stop, never silently clamp')
    unclipped=ratio*advantage
    clipped=ratio.clamp(1-clip_range,1+clip_range)*advantage
    loss=-torch.minimum(unclipped,clipped).mean()
    kl=((ratio-1)-difference).mean()
    fraction=((ratio-1).abs()>clip_range).to(torch.float64).mean()
    if not all(torch.isfinite(x).all() for x in (loss,kl,fraction)):
        raise ValueError('Joint objective and diagnostics must remain finite')
    return loss,kl,fraction,difference.min(),difference.max()


class JointPPOTrainer(PPOTrainer):
    """Reuse the old shared MLP, GAE and complete checkpoint/RNG lifecycle."""
    checkpoint_schema='task-ppo-v1-joint-pytorch-checkpoint'
    checkpoint_version=1

    def __init__(self,config=None):
        supplied=dict(input_dim=48,action_dim=2,hidden_sizes=[64,64],gamma=1.,
                      lambda_gae=.95,seed=20261021,entropy_coef=0.)
        supplied.update({} if config is None else config)
        if (supplied['input_dim']!=48 or supplied['action_dim']!=2
                or list(supplied['hidden_sizes'])!=[64,64] or supplied['gamma']!=1.
                or supplied['entropy_coef']!=0.):
            raise ValueError('Task-PPO V1 requires48inputs,2actions,64x64,gamma1,entropy_coef0')
        super().__init__(supplied)

    def _prepare_joint(self,batch):
        required={'states','task_states','task_actions','log_probs','values','rewards','terminated'}
        if required-batch.keys():raise ValueError('Missing joint rollout fields: '+str(sorted(required-batch.keys())))
        states=_finite_tensor(batch['states'],'canonical value states')
        if (states.ndim!=2 or not len(states) or states.shape[1]!=48
                or not (states[:,27:]==0).all()):
            raise ValueError('Canonical pre-slot critic queries require27global features and21zeros')
        size=len(states)
        task_states,actions=_ragged(batch['task_states'],batch['task_actions'])
        if len(task_states)!=size:raise ValueError('Task rows cannot change the number of physical slots')
        sampled=torch.tensor([bool(len(row)) for row in task_states],dtype=torch.bool)
        values=_finite_tensor(batch['values'],'values');rewards=_finite_tensor(batch['rewards'],'rewards')
        terminated=_boolean_vector(batch['terminated'],'terminated',size)
        if values.shape!=(size,) or rewards.shape!=(size,):raise ValueError('Rewards and values need one entry per physical slot')
        raw_logs=batch['log_probs']
        if len(raw_logs)!=size:raise ValueError('One joint likelihood entry required per physical slot')
        for index,value in enumerate(raw_logs):
            if sampled[index] and value is None:raise ValueError('Sampled joint proposals require the original log probability')
            if not sampled[index] and value is not None:raise ValueError('Forced/empty slots must not fake a policy log probability')
        old_logs=_finite_tensor([0. if value is None else value for value in raw_logs],
                                'old joint log probabilities',dtype=torch.float64)
        if (old_logs[sampled]>1e-5).any():raise ValueError('Original joint log probabilities cannot be positive')
        if not terminated[-1] and 'last_value' not in batch:raise ValueError('Nonterminal rollout cut requires explicit bootstrap')
        advantages,returns=compute_gae(rewards,values,terminated,batch.get('last_value',0.),
                                      self.config['gamma'],self.config['lambda_gae'])
        policy_advantages=advantages.clone()
        if int(sampled.sum())>1:
            selected=advantages[sampled]
            policy_advantages[sampled]=(selected-selected.mean())/(selected.std(unbiased=False)+1e-8)
        return dict(states=states,task_states=task_states,actions=actions,old_logs=old_logs,
                    sampled=sampled,advantages=policy_advantages,returns=returns)

    def update(self,batch):
        data=self._prepare_joint(batch);states=data['states'];sampled=data['sampled']
        self.model.train()
        before={name:parameter.detach().clone() for name,parameter in self.model.named_parameters()}
        history=[];policy_count=critic_count=conditioned_uses=steps=0
        minima=[];maxima=[]
        for _ in range(self.config['update_epochs']):
            order=torch.randperm(len(states))
            for start in range(0,len(states),self.config['minibatch_size']):
                indices=order[start:start+self.config['minibatch_size']]
                chosen=indices[sampled[indices]]
                _,predicted_values=self.model(states[indices])
                value_loss=.5*(predicted_values-data['returns'][indices]).square().mean()
                zero=torch.zeros((),dtype=torch.float64)
                policy_loss=entropy=kl=clip_fraction=zero
                if len(chosen):
                    positions=chosen.tolist()
                    logs,conditional_entropies=evaluate_joint(self.model,
                        [data['task_states'][i] for i in positions],[data['actions'][i] for i in positions])
                    policy_loss,kl,clip_fraction,low,high=clipped_joint_objective(
                        logs,data['old_logs'][chosen],data['advantages'][chosen],self.config['clip_range'])
                    entropy=conditional_entropies.mean()
                    minima.append(float(low.detach()));maxima.append(float(high.detach()))
                    conditioned_uses+=sum(len(data['task_states'][i]) for i in positions)
                # Entropy is diagnostic only. All optimization terms use slots.
                loss=policy_loss+self.config['value_coef']*value_loss
                if not all(torch.isfinite(item).all() for item in (loss,entropy,kl,clip_fraction)):
                    raise ValueError('Joint PPO loss/diagnostics nonfinite')
                self.optimizer.zero_grad(set_to_none=True);loss.backward()
                gradient=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config['max_grad_norm'],
                                                        error_if_nonfinite=True)
                self.optimizer.step()
                if not all(torch.isfinite(parameter).all() for parameter in self.model.parameters()):
                    raise ValueError('Joint PPO parameters nonfinite after update')
                history.append((len(indices),len(chosen),float(policy_loss.detach()),float(value_loss.detach()),
                    float(entropy.detach()),float(kl.detach()),float(clip_fraction.detach()),float(gradient)))
                policy_count+=len(chosen);critic_count+=len(indices);steps+=1
        self.update_count+=1;self.optimizer_steps+=steps
        deltas={name:float((parameter.detach()-before[name]).square().sum())
                for name,parameter in self.model.named_parameters()}
        def delta(prefix):return math.sqrt(sum(value for name,value in deltas.items() if name.startswith(prefix)))
        def average(column,actor=False):
            denominator=policy_count if actor else critic_count
            return sum(row[column]*row[1 if actor else 0] for row in history)/denominator if denominator else 0.
        metrics=dict(policy_loss=average(2,True),value_loss=average(3),entropy=average(4,True),
            approx_kl=average(5,True),clip_fraction=average(6,True),grad_norm=max(row[7] for row in history),
            optimizer_steps=steps,policy_samples=policy_count,critic_samples=critic_count,
            sampled_slots=int(sampled.sum()),physical_slots=len(states),
            conditioned_decisions=sum(len(row) for row in data['task_states']),conditioned_sample_uses=conditioned_uses,
            actor_parameter_delta_l2=delta('policy_head.'),backbone_parameter_delta_l2=delta('backbone.'),
            value_parameter_delta_l2=delta('value_head.'),parameter_delta_l2=math.sqrt(sum(deltas.values())),
            return_mean=float(data['returns'].mean()),update_count=self.update_count,total_optimizer_steps=self.optimizer_steps,
            joint_logratio_min=min(minima,default=0.),joint_logratio_max=max(maxima,default=0.))
        if not all(math.isfinite(value) for value in metrics.values()):raise ValueError('Joint PPO metrics nonfinite')
        return metrics
