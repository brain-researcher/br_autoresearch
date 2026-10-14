"""One focused execution-menu fixture; scientific fitting is mocked."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import continue_exploratory_fmri as driver

class ContinuationTest(unittest.TestCase):
    def test_menu_identity_reuse_and_new_invalid_does_not_block_later_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary); prepared=base/'prepared'; prepared.mkdir()
            prior=base/'prior'; prior.mkdir()
            inventory={'status':'complete','group_mask':'frozen-common-mask.nii.gz',
                       'runs':[{'participant':f'sub-{i:03d}','run':r} for i in range(1,109) for r in range(1,5)]}
            (prepared/'prepared.json').write_text(json.dumps(inventory))
            keys=[{'participant':'sub-001','run':r} for r in (1,2,3)]
            with self.assertRaisesRegex(ValueError,'Duplicate selected'):
                driver.selection_plan(inventory,keys+[keys[0]],prepared,[])
            with self.assertRaisesRegex(ValueError,'outside the prepared panel'):
                driver.selection_plan(inventory,[{'participant':'sub-999','run':1}],prepared,[])
            old=prior/'sub-001_run-1'; old.mkdir(); (old/'effect.npz').write_bytes(b'fixture')
            completion={'status':'complete','prepared':str(prepared.resolve()),'participant':'sub-001',
                        'run':1,'artifacts':['effect.npz']}
            (old/'completion.json').write_text(json.dumps(completion))
            duplicate=base/'duplicate'; (duplicate/'sub-001_run-1').mkdir(parents=True)
            (duplicate/'sub-001_run-1'/'effect.npz').write_bytes(b'fixture')
            (duplicate/'sub-001_run-1'/'completion.json').write_text(json.dumps(completion))
            with self.assertRaisesRegex(ValueError,'Duplicate completed'):
                driver.selection_plan(inventory,keys,prepared,[prior,duplicate])
            bad=prior/'sub-001_run-2.partial'; bad.mkdir()
            with self.assertRaisesRegex(ValueError,'prior partial attempt'):
                driver.selection_plan(inventory,keys,prepared,[prior])
            bad.rmdir()
            selected=base/'selection.json'; selected.write_text(json.dumps(keys))
            calls=[]
            def fit(entry,*args):
                calls.append(entry['run'])
                if entry['run']==2:
                    raise ValueError(driver.COMMON_MASK_FAILURE)
                return {'elapsed_seconds':.01}
            limits=SimpleNamespace(chunk_voxels=4096,max_local_gib=12.)
            with patch.object(driver.runner,'fit_run',side_effect=fit):
                report,status=driver.execute_selection(prepared,selected,[prior],base/'new',lambda:None,limits)
            self.assertEqual(calls,[2,3])
            self.assertEqual(status,2)
            self.assertEqual((report['selected'],report['completed'],report['unavailable'],report['pending']),(3,2,1,0))
            self.assertEqual([r['status'] for r in report['runs']],['reused_complete','unavailable_common_mask','complete'])
            saved=json.loads((base/'new'/'continuation.json').read_text())
            self.assertEqual(saved['unchanged_group_mask'],'frozen-common-mask.nii.gz')
            self.assertEqual(saved['full_panel_required_runs'],432)

if __name__=='__main__':
    driver.runner.require_compute()
    unittest.main()
