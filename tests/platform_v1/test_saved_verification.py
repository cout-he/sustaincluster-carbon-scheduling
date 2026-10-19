"""Postprocessing-only regression; no simulation or data access."""
import unittest
from experiments.platform_v1 import verify_saved


class SavedCarbonTests(unittest.TestCase):
    def test_saved_carbon_is_recomputed_from_power_and_ci(self):
        step={'total_kw':2,'ci_g_per_kwh':200,'carbon_kg':.1}
        self.assertAlmostEqual(verify_saved.independent_carbon([step]),.1)
        with self.assertRaises(ValueError):verify_saved.independent_carbon([dict(step,carbon_kg=999)])
