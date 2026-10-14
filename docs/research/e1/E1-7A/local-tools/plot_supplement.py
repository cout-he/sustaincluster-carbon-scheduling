"""Two named registered C categories for mechanism illustration, not new cases."""
import json
from pathlib import Path

from experiments.e1.contracts import file_hash
from experiments.e1.plot_diagnostics import _plot_c, verify_campaign

root=Path(__file__).resolve().parent
campaign=root/'C-campaign-001'
receipt=verify_campaign(campaign)
output=root/'figures-supplement-001'
output.mkdir(exist_ok=False)
figures=[_plot_c(campaign,name,output) for name in
         ('load-1.5-gpu_tight','load-1.5-long_slack')]
manifest=dict(split='train',test_access=False,display_only=True,
    selection_sha256=receipt['selection_sha256'],
    explanation='Two already-registered named categories illustrate copy interchange and amplified waiting; no cases reselected or rerun.',
    campaign_receipt_sha256=file_hash(campaign/'receipt.json'),
    plotting_source_sha256=file_hash(root.parents[2]/'experiments/e1/plot_diagnostics.py'),
    helper_sha256=file_hash(Path(__file__)),figures=figures,
    files={p.name:file_hash(p) for p in output.iterdir()})
(output/'figures-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
print(json.dumps(dict(figures=len(figures),case_ids=[f['case_id'] for f in figures])))
