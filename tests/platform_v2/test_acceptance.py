import tempfile
import unittest
from pathlib import Path
from experiments.platform_v2.acceptance import case_matrix,attach_background,claim_once


class AcceptanceTests(unittest.TestCase):
    def test_exact_matrix_is_36_with_one_predeclared_sla(self):
        cases=case_matrix();self.assertEqual(len(cases),36);self.assertEqual(len(set(cases)),36)
        self.assertEqual({case[3] for case in cases},{'fixed','variable'})

    def test_primary_incremental_carbon_uses_common_fixed_background(self):
        row={'carbon_kg':100.,'total_kwh':200.}
        attach_background(row,{'carbon_kg':60.,'total_kwh':120.},{'carbon_kg':50.,'total_kwh':100.})
        self.assertEqual(row['common_incremental_carbon_kg'],40.)
        self.assertEqual(row['matched_incremental_carbon_kg'],50.)

    def test_once_cannot_be_reclaimed(self):
        root=Path(__file__).resolve().parents[2]/'research_notes/platform-v2/unit-tmp';root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            self.assertTrue(Path(tmp).resolve().is_relative_to(root.resolve()))
            path=Path(tmp)/'marker.json';claim_once(path,{'status':'running'})
            with self.assertRaises(FileExistsError):claim_once(path,{'status':'running'})


if __name__=='__main__':unittest.main()
