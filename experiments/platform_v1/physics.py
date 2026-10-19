"""Affine IT electricity and a first-law, one-node air-cooled thermal surrogate.

Electric power and heat flow are kW; heat capacity is kJ/K; time is seconds.
All on-site electrical input ultimately becomes heat; no thermal-rating-as-watts.
"""
import math


class PhysicalModel:
    def __init__(self,inventory,cfg):
        self.inv=inventory;self.p=cfg['power'];self.t=cfg['thermal']
        self.c_kj_k=self.t['heat_capacity_kj_per_k_host']*inventory.hosts
        self.ua_kw_k=self.t['ua_w_per_k_host']*inventory.hosts/1000
        self.it_design_kw=self.it_power((inventory.installed_cores,inventory.gpus,inventory.installed_mem_gb))['it_kw']
        self.coil_rating_kw=self.t['coil_sizing_ratio']*self.it_design_kw
        self.air_conductance_kw_k=self.coil_rating_kw/self.t['air_design_delta_k']
        self.airflow_m3_s=self.air_conductance_kw_k/(self.t['air_density_kg_m3']*self.t['air_cp_kj_kg_k'])
        self.supply_fan_kw=self.t['fan_pressure_pa']*self.airflow_m3_s/self.t['fan_efficiency']/1000

    def it_power(self,demand):
        i=self.inv;p=self.p
        limits=(i.installed_cores,i.gpus,i.installed_mem_gb)
        if len(demand)!=3 or any(not math.isfinite(x) or x<0 or x>limit for x,limit in zip(demand,limits)):
            raise ValueError('IT demand outside installed equipment')
        fractions=[x/limit if limit else 0 for x,limit in zip(demand,limits)]
        out={name+'_kw':count*(p[name+'_idle_w']+(p[name+'_reference_full_w']-p[name+'_idle_w'])*u)/1000
             for name,count,u in zip(['cpu','gpu','dimm'],[i.cpu_objects,i.gpus,i.dimm_objects],fractions)}
        out['host_other_kw']=i.hosts*p['host_other_w']/1000
        dc_kw=sum(out.values());out['psu_loss_kw']=dc_kw*(1/p['psu_efficiency']-1)
        out['rack_switch_kw']=i.racks*p['rack_switch_w']/1000
        out['it_kw']=sum(out.values())
        return out

    def transition(self,inlet_c,it_kw,outdoor_c,supply_c,seconds=900):
        t=self.t
        if any(not math.isfinite(x) for x in [inlet_c,it_kw,outdoor_c,supply_c,seconds]) or it_kw<0 or seconds<=0:
            raise ValueError('Nonfinite or negative physical input')
        if not t['outdoor_min_c']<=outdoor_c<=t['outdoor_max_c'] or not t['supply_min_c']<=supply_c<=t['supply_max_c']:
            raise ValueError('Physical input outside declared operating domain')
        evap=supply_c-t['evaporator_approach_k']
        cond=max(outdoor_c+t['condenser_approach_k'],evap+t['minimum_lift_k'])
        cop=max(t['cop_min'],min(t['cop_max'],t['carnot_efficiency']*(evap+273.15)/(cond-evap)))
        temp=inlet_c;peak=inlet_c
        energies={key:0. for key in ['coil_kw','envelope_kw','pump_kw','compressor_kw','condenser_fan_kw','heat_rejection_kw']}
        count=math.ceil(seconds/t['substep_seconds']);dt=seconds/count
        for _ in range(count):
            coil=min(self.coil_rating_kw,max(0.,self.air_conductance_kw_k*(temp-supply_c)))
            envelope=self.ua_kw_k*(outdoor_c-temp)
            flow_m3_s=coil/(t['water_cp_kj_kg_k']*t['water_delta_k']*t['water_density_kg_m3'])
            pump=t['pump_pressure_pa']*flow_m3_s/t['pump_efficiency']/1000
            evaporator=coil+pump  # all pump electricity enters chilled water
            compressor=evaporator/cop
            speed=max(t['condenser_fan_min_speed'],min(1.,evaporator/self.coil_rating_kw)) if evaporator>0 else 0.
            fan=self.coil_rating_kw*t['condenser_fan_electric_fraction_of_coil_rating']*speed**3
            reject=evaporator+compressor+fan
            temp+=(it_kw+self.supply_fan_kw+envelope-coil)*dt/self.c_kj_k
            peak=max(peak,temp)
            for key,value in zip(energies,[coil,envelope,pump,compressor,fan,reject]):energies[key]+=value*dt
        out={key:value/seconds for key,value in energies.items()}
        out.update(supply_fan_kw=self.supply_fan_kw,cop=cop,inlet_end_c=temp,inlet_peak_c=peak,
                   storage_delta_kj=self.c_kj_k*(temp-inlet_c))
        out['hvac_kw']=self.supply_fan_kw+out['pump_kw']+out['compressor_kw']+out['condenser_fan_kw']
        out['balance_residual_kj']=(it_kw+out['hvac_kw']+out['envelope_kw']-out['heat_rejection_kw'])*seconds-out['storage_delta_kj']
        return out

    def safe_transition(self,inlet_c,it_kw,outdoor_c,requested_c):
        t=self.t
        if not math.isfinite(requested_c) or not t['supply_min_c']<=requested_c<=t['supply_max_c']:
            raise ValueError('Requested supply outside action space')
        trial=self.transition(inlet_c,it_kw,outdoor_c,requested_c)
        if trial['inlet_peak_c']<=t['inlet_control_limit_c']+1e-10:return requested_c,trial,False
        lo=t['supply_min_c'];hi=requested_c
        safe=self.transition(inlet_c,it_kw,outdoor_c,lo)
        if safe['inlet_peak_c']>t['inlet_control_limit_c']+1e-10:
            raise ValueError('No safe cooling action at current state/load')
        for _ in range(16):
            mid=(lo+hi)/2;r=self.transition(inlet_c,it_kw,outdoor_c,mid)
            if r['inlet_peak_c']<=t['inlet_control_limit_c']:lo=mid;safe=r
            else:hi=mid
        return lo,safe,True
