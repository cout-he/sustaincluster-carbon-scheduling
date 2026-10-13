import unittest
from render_corrected_dmr import dmr_limit, build_figure

class FigureBounds(unittest.TestCase):
    def test_all_windows_set_shared_upper_limit(self):
        rows=[dict(split=s, factor=f, policy='edf_backfill', dmr=d)
              for s,f,d in [('train',1.,0.),('train',2.,.07),
                            ('validation',1.,.046),('validation',2.,.393)]]
        figure,axes=build_figure(rows)
        self.assertGreater(axes[0].get_ylim()[1],39.3)
        self.assertEqual(axes[0].get_ylim(),axes[1].get_ylim())
        for ax in axes:
            for line in ax.lines:
                if len(line.get_ydata()):
                    self.assertLessEqual(max(line.get_ydata()),ax.get_ylim()[1])
        import matplotlib.pyplot as plt
        plt.close(figure)
    def test_zero_misses_still_show_five_percent_gate(self):
        self.assertGreater(dmr_limit([dict(dmr=0)]),5)
    def test_invalid_rate_is_rejected(self):
        with self.assertRaises(ValueError):dmr_limit([dict(dmr=1.1)])

if __name__=='__main__':unittest.main()
