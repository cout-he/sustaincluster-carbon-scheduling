"""IT electricity -> heat -> capacity-limited coil/EIR package -> rejection.

All electricity/heat rates kW, heat capacity kJ/K, time seconds. No storage
device: finite room thermal mass is explicitly booked in every transition.
"""
import math
from .chiller import Chiller


class PhysicalModel:
    def __init__(self,inventory,cfg):
        self.inv=inventory;self.cfg=cfg;self.p=cfg['power'];self.t=cfg['thermal'];g=cfg['room']
        self.floor_m2=inventory.racks*g['floor_m2_per_rack']
        self.volume_m3=self.floor_m2*g['height_m']
        air=self.volume_m3*self.t['air_density_kg_m3']*self.t['air_cp_kj_kg_k']
        equipment=(inventory.cpu_only_hosts*g['cpu_host_mass_kg']+inventory.gpu_hosts*g['gpu_host_mass_kg']+inventory.racks*g['rack_mass_kg'])*g['equipment_cp_kj_kg_k']*g['equipment_coupling_fraction']
        slab=self.floor_m2*g['effective_slab_m']*g['concrete_density_kg_m3']*g['concrete_cp_kj_kg_k']
        self.thermal_mass_components=dict(air_kj_k=air,equipment_kj_k=equipment,slab_kj_k=slab)
        self.c_kj_k=air+equipment+slab
        self.envelope_m2=4*math.sqrt(self.floor_m2)*g['height_m']+self.floor_m2
        self.ua_kw_k=self.envelope_m2*g['envelope_u_w_m2_k']/1000
        self.it_design_kw=self.it_power((inventory.installed_cores,inventory.gpus,inventory.installed_mem_gb))['it_kw']
        self.coil_rating_kw=self.t['coil_sizing_ratio']*self.it_design_kw
        self.air_conductance_kw_k=self.coil_rating_kw/self.t['air_design_delta_k']
        self.airflow_m3_s=self.air_conductance_kw_k/(self.t['air_density_kg_m3']*self.t['air_cp_kj_kg_k'])
        self.supply_fan_kw=self.t['fan_pressure_pa']*self.airflow_m3_s/self.t['fan_efficiency']/1000
        self.chiller=Chiller(inventory.chiller_modules,cfg)
        self.water_flow_max_kg_s=inventory.chiller_modules*cfg['chiller']['unit_max_water_kg_s']

    def it_power(self,demand):
        i=self.inv;p=self.p;limits=(i.installed_cores,i.gpus,i.installed_mem_gb)
        if len(demand)!=3 or any(not math.isfinite(x) or not 0<=x<=limit for x,limit in zip(demand,limits)):
            raise ValueError('IT demand outside installed inventory')
        out={name+'_kw':count*(p[name+'_idle_w']+(p[name+'_reference_full_w']-p[name+'_idle_w'])*demand[axis]/limits[axis])/1000
             for axis,name,count in [(0,'cpu',i.cpu_objects),(1,'gpu',i.gpus)]}
        out.update(dimm_kw=i.dimm_objects*p['dimm_base_w']/1000,host_other_kw=i.hosts*p['host_other_w']/1000)
        dc=math.fsum(out.values());out['psu_loss_kw']=dc*(1/p['psu_efficiency']-1)
        out['rack_switch_kw']=i.racks*p['rack_switch_w']/1000;out['it_kw']=math.fsum(out.values())
        return out

    def transition(self,inlet_c,it_kw,outdoor_c,supply_c,seconds=900):
        t=self.t
        if any(not math.isfinite(x) for x in [inlet_c,it_kw,outdoor_c,supply_c,seconds]) or it_kw<0 or seconds<=0:raise ValueError('Invalid physical input')
        if not t['supply_min_c']<=supply_c<=t['supply_max_c'] or not t['outdoor_min_c']<=outdoor_c<=t['outdoor_max_c']:raise ValueError('Outside declared domain')
        water=self.chiller.water_temperature(supply_c);availability=self.chiller.performance(water,outdoor_c,0)
        available=availability['available_capacity_kw'];maxwater=self.water_flow_max_kg_s*t['water_cp_kj_kg_k']*t['water_delta_k']
        alpha=t['pump_pressure_pa']/(t['pump_efficiency']*t['water_density_kg_m3']*t['water_cp_kj_kg_k']*t['water_delta_k']*1000)
        keys=['coil_kw','coil_request_kw','unmet_cooling_kw','envelope_kw','pump_kw','compressor_kw','condenser_fan_kw','heat_rejection_kw','evaporator_kw','package_kw','water_flow_kg_s','actual_supply_c','plr','cycling_fraction','cop']
        integrated={key:0. for key in keys};temp=inlet_c;peak=inlet_c;max_unmet=0.
        n=math.ceil(seconds/t['substep_seconds']);dt=seconds/n
        for _ in range(n):
            demand=min(self.coil_rating_kw,maxwater,max(0,self.air_conductance_kw_k*(temp-supply_c)))
            actual=min(demand,available/(1+alpha));pump=alpha*actual
            ch=self.chiller.performance(water,outdoor_c,actual+pump)
            envelope=self.ua_kw_k*(outdoor_c-temp);unmet=demand-actual
            reject=actual+pump+ch['package_kw']
            flow=actual/(t['water_cp_kj_kg_k']*t['water_delta_k'])
            delivered=temp-actual/self.air_conductance_kw_k
            temp+=(it_kw+self.supply_fan_kw+envelope-actual)*dt/self.c_kj_k;peak=max(peak,temp);max_unmet=max(max_unmet,unmet)
            values=[actual,demand,unmet,envelope,pump,ch['compressor_kw'],ch['condenser_fan_kw'],reject,ch['evaporator_kw'],ch['package_kw'],flow,delivered,ch['plr'],ch['cycling_fraction'],ch['cop']]
            for key,value in zip(keys,values):integrated[key]+=value*dt
        out={key:value/seconds for key,value in integrated.items()}
        out.update(supply_fan_kw=self.supply_fan_kw,available_capacity_kw=available,water_supply_c=water,
                   curve_ambient_c=availability['curve_ambient_c'],low_ambient_boundary=availability['low_ambient_boundary'],
                   curve_domain_valid=availability['curve_domain_valid'],max_unmet_cooling_kw=max_unmet,
                   inlet_end_c=temp,inlet_peak_c=peak,storage_delta_kj=self.c_kj_k*(temp-inlet_c))
        out['hvac_kw']=math.fsum(out[k] for k in ['supply_fan_kw','pump_kw','compressor_kw','condenser_fan_kw'])
        out['balance_residual_kj']=(it_kw+out['hvac_kw']+out['envelope_kw']-out['heat_rejection_kw'])*seconds-out['storage_delta_kj']
        out['chiller_balance_residual_kj']=(out['evaporator_kw']+out['compressor_kw']+out['condenser_fan_kw']-out['heat_rejection_kw'])*seconds
        out['cooling_feasible']=bool(availability['curve_domain_valid'] and peak<=t['inlet_control_limit_c']+1e-9)
        return out

    def safe_transition(self,inlet_c,it_kw,outdoor_c,requested_c):
        t=self.t
        if not math.isfinite(requested_c) or not t['supply_min_c']<=requested_c<=t['supply_max_c']:raise ValueError('Invalid HVAC request')
        count=round((t['supply_max_c']-t['supply_min_c'])/t['safety_grid_c'])
        candidates={requested_c,*[t['supply_min_c']+j*t['safety_grid_c'] for j in range(count+1)]}
        ordered=sorted(candidates,key=lambda x:(abs(x-requested_c),x));trials=[]
        for candidate in ordered:
            result=self.transition(inlet_c,it_kw,outdoor_c,candidate);trials.append((candidate,result))
            if result['cooling_feasible']:return candidate,result,abs(candidate-requested_c)>1e-10
        # Never hide equipment incapacity: propagate actual heat and flag it.
        action,result=min(trials,key=lambda pair:(pair[1]['inlet_peak_c'],pair[1]['inlet_end_c'],abs(pair[0]-requested_c)))
        return action,result,abs(action-requested_c)>1e-10
