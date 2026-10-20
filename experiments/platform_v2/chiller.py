"""Public ElectricEIR air-cooled PACKAGE curves, no out-of-range polynomial use.

Numerical data: LBL Buildings v12.1.0 York YCAL0033EE. Fan ratio is blank in
the reference: retain package COP boundary, split reported components only.
"""
import math


def biquadratic(a,x,y):return a[0]+a[1]*x+a[2]*x*x+a[3]*y+a[4]*y*y+a[5]*x*y


class Chiller:
    def __init__(self,modules,cfg):
        self.modules=modules;self.c=cfg['chiller']
        self.reference_kw=modules*self.c['unit_reference_capacity_kw_th']

    def water_temperature(self,supply_c):
        water=self.c['water_reset_at_supply18_c']+self.c['water_reset_slope']*(supply_c-18)
        if not self.c['water_domain_c'][0]<=water<=self.c['water_domain_c'][1]:raise ValueError('Water reset outside curve domain')
        return water

    def performance(self,water_c,outdoor_c,evaporator_request_kw):
        c=self.c
        if not all(math.isfinite(x) for x in [water_c,outdoor_c,evaporator_request_kw]) or evaporator_request_kw<0:raise ValueError('Invalid chiller input')
        if not c['water_domain_c'][0]<=water_c<=c['water_domain_c'][1]:raise ValueError('Water outside curve domain')
        boundary=outdoor_c<c['ambient_curve_domain_c'][0];valid=outdoor_c<=c['ambient_curve_domain_c'][1]
        curve_ambient=max(outdoor_c,c['ambient_curve_domain_c'][0])
        if valid:
            capacity=self.reference_kw*biquadratic(c['capacity_coefficients'],water_c,curve_ambient)
            eir=biquadratic(c['eir_temperature_coefficients'],water_c,curve_ambient)
            if capacity<=0 or eir<=0:raise ValueError('Nonpositive reference polynomial within domain')
        else:capacity=0.;eir=0.
        actual=min(evaporator_request_kw,capacity*c['plr_max'])
        plr=actual/capacity if capacity else 0.
        operating=max(c['plr_min'],plr) if actual>0 else 0.
        cycling=min(1,plr/c['plr_min']) if actual>0 else 0.
        curve=c['eir_plr_coefficients'];part=curve[0]+curve[1]*operating+curve[2]*operating**2
        package=capacity/c['reference_cop']*eir*part*cycling if actual else 0.
        fan=package*c['package_condenser_fan_share'];compressor=package-fan
        return dict(available_capacity_kw=capacity,evaporator_kw=actual,
                    unmet_evaporator_kw=evaporator_request_kw-actual,package_kw=package,
                    compressor_kw=compressor,condenser_fan_kw=fan,plr=plr,operating_plr=operating,
                    cycling_fraction=cycling,water_supply_c=water_c,curve_ambient_c=curve_ambient,
                    low_ambient_boundary=boundary,curve_domain_valid=valid,
                    cop=actual/package if package else 0.)
