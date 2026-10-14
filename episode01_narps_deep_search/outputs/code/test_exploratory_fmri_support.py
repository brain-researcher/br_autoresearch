"""Focused approved-support selection, coefficient reuse and collector fixture."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import nibabel as nib
import exploratory_fmri_runner as runner
import exploratory_fmri_support as support
import summarize_exploratory_fmri as summary


class ApprovedSupportTest(unittest.TestCase):
    def test_input_rule_explicit_override_reuses_saved_coefficients(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); prepared=root/'prepared'; prepared.mkdir()
            original=np.zeros((2,2,2),dtype='uint8'); original.ravel()[:6]=1
            data=np.full((2,2,2,3),10.,dtype=float)
            data.reshape(-1,3)[0]=0.
            data.reshape(-1,3)[1]=[10.,np.nan,10.]
            data.reshape(-1,3)[2]=-1.
            data.reshape(-1,3)[7]=np.nan  # Outside original support never participates.
            excluded,counts=support.invalid_original_indices(data,original==1,chunk_voxels=2)
            self.assertEqual(excluded,[0,1,2])
            self.assertEqual(counts,{'nonfinite_series':1,'finite_nonpositive_mean_series':2,'all_zero_series':1})
            old=prepared/'original.nii.gz'; nib.save(nib.Nifti1Image(original,np.eye(4)),old)
            valid=original.copy(); valid.ravel()[excluded]=0
            new=root/'valid.nii.gz'; nib.save(nib.Nifti1Image(valid,np.eye(4)),new)
            people=[f'sub-{i:03d}' for i in range(1,109)]
            entries=[{'participant':p,'task_version':'EI' if i<54 else 'ER','run':r,
                      'designs':{'unchanged':'design'},'design_diagnostics':'unchanged.json'}
                     for i,p in enumerate(people) for r in range(1,5)]
            inventory={'status':'complete','participants':people,'runs':entries,
                       'group_mask':str(old),'group_mask_voxels':6}
            (prepared/'prepared.json').write_text(json.dumps(inventory))
            manifest={'status':'complete_valid_common_support','rule':support.RULE,
                      'prepared':str(prepared.resolve()),'original_group_mask':str(old.resolve()),
                      'expected_runs':432,'verified_runs':432,'unknown_runs':0,
                      'completed_run_certificates':431,'fully_inspected_failed_runs':1,
                      'mask':str(new.resolve()),'mask_voxels':3,'excluded_voxel_indices':excluded,'excluded_voxels':3}
            approved=root/'support.json'; approved.write_text(json.dumps(manifest))
            self.assertEqual(support.selected_support(prepared,inventory)[:2],(old.resolve(),6))
            selected,n,metadata=support.selected_support(prepared,inventory,approved)
            self.assertEqual((selected,n),(new.resolve(),3))
            self.assertTrue(metadata['amended_after_observed_input_failures'])
            self.assertEqual(inventory['group_mask_voxels'],6)  # Original inventory never changes.
            runs=root/'runs'; final=runs/'sub-001_run-1'; final.mkdir(parents=True)
            effects=np.arange(5*2*8,dtype='float32').reshape(5,2,8)
            artifact=final/'gain_loss_effects.npz'
            np.savez(artifact,effects=effects,voxel_indices=np.arange(8),shape=np.array((2,2,2)),
                     affine=np.eye(4),arms=np.array(runner.ARMS),focal=np.array(runner.FOCAL))
            completion={'status':'complete','prepared':str(prepared.resolve()),'participant':'sub-001',
                        'task_version':'EI','run':1,'group_mask':str(old.resolve()),
                        'designs':entries[0]['designs'],'design_diagnostics':entries[0]['design_diagnostics'],
                        'artifacts':['gain_loss_effects.npz'],
                        'fit_diagnostics':{arm:{'focal_estimable':{'gain':True,'loss':True}} for arm in runner.ARMS}}
            (final/'completion.json').write_text(json.dumps(completion))
            before=artifact.read_bytes()
            with patch.object(runner,'stage_bold',side_effect=AssertionError('Completed run must not read BOLD')):
                result=runner.fit_run(entries[0],prepared,runs,lambda:None,4096,12.,selected,approved_support=approved)
            self.assertEqual(result['status'],'reused_complete')
            self.assertEqual(artifact.read_bytes(),before)
            positions=np.flatnonzero(valid)
            loaded=summary.load_effects(artifact,completion,positions,(2,2,2),np.eye(4))
            np.testing.assert_array_equal(loaded,effects[:,:,positions])
            amended_completion={**completion,'group_mask':str(new.resolve()),'group_mask_voxels':3,
                                'approved_support':str(approved.resolve())}
            with patch.object(summary,'collect_completions',return_value=({('sub-001',1):(artifact,amended_completion)},[])):
                with self.assertRaisesRegex(ValueError,'Only an original-support successful fit'):
                    support.run_statuses(prepared,inventory,[runs],lambda:None)
            # Collector selects the explicit support but still refuses an incomplete full panel.
            result=summary.summarize(prepared,[runs],root/'summary',root/'unused-membership.tsv',lambda:None,approved)
            self.assertEqual(result['status'],'incomplete_cohort')
            self.assertEqual(result['complete_runs'],1)
            self.assertEqual(result['spatial_support']['selection'],'approved_valid_common_support_requested')
            self.assertEqual(result['spatial_support']['status'],'not_applied_incomplete_cohort')
            self.assertEqual(result['spatial_support']['approved_support'],str(approved.resolve()))
            self.assertFalse((root/'summary'/'maps').exists())
            manifest['verified_runs']=431; approved.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                support.selected_support(prepared,inventory,approved)

if __name__=='__main__':
    runner.require_compute()
    unittest.main()
