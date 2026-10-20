"""Saved-result arithmetic must reject inconsistent energy and carbon units."""
import unittest
from .analyze import check_row


class SavedArithmeticTests(unittest.TestCase):
    def row(self):
        return dict(status='completed',it_kw_mean='10',supply_fan_kw_mean='1',pump_kw_mean='0.1',
                    compressor_kw_mean='2',condenser_fan_kw_mean='0.2',hvac_kw_mean='3.3',
                    it_kwh='60',hvac_kwh='19.8',total_kwh='79.8',synthetic_carbon_kg='31.92',
                    envelope_kw_mean='0',heat_rejection_kw_mean='12.3',storage_kwh_th='6',
                    synthetic_ci_g_per_kwh='400',mode='unfiltered',inlet_peak_c='28')

    def test_correct_units_and_thermal_storage_close(self):
        a,b=check_row(self.row(),6)
        self.assertLess(a,1e-10);self.assertLess(b,1e-8)

    def test_corrupt_saved_carbon_is_rejected(self):
        row=self.row();row['synthetic_carbon_kg']='31920'
        with self.assertRaises(ValueError):check_row(row,6)

    def test_thermal_storage_is_not_zeroed_to_pass_balance(self):
        row=self.row();row['storage_kwh_th']='0'
        with self.assertRaises(ValueError):check_row(row,6)


if __name__=='__main__':unittest.main()
