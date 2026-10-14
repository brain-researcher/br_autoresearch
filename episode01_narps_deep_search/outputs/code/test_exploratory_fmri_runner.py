"""Focused shared-prewhitening and source-I/O checks; compute allocation only."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import run_glm
from nilearn.glm.regression import ARModel
import exploratory_fmri_runner as runner


class SharedARTests(unittest.TestCase):
    def test_invalid_outside_group_retained_as_nan_and_inside_refused(self):
        y=np.array([[0.,10.,np.nan,20.],[0.,12.,15.,22.],[0.,11.,16.,21.]])
        common=np.array([False,True,False,True])
        fitted={arm:(np.full((2,2),i+1.),np.full((2,2),.5)) for i,arm in enumerate(runner.ARMS)}
        with patch.object(runner,'fit_shared_ar_chunk',return_value=(fitted,np.array([.2,.3]))) as fit:
            effects,variances,rho,valid=runner.fit_bold_chunk(y,common,{})
        np.testing.assert_array_equal(valid,[False,True,False,True])
        self.assertEqual(effects.shape,(5,2,4))
        self.assertTrue(np.isnan(effects[:,:,[0,2]]).all())
        self.assertTrue(np.isnan(variances[:,:,[0,2]]).all())
        self.assertTrue(np.isnan(rho[[0,2]]).all())
        for index in range(len(runner.ARMS)):
            np.testing.assert_array_equal(effects[index][:,[1,3]],np.full((2,2),index+1.))
            np.testing.assert_array_equal(variances[index][:,[1,3]],np.full((2,2),.5))
        np.testing.assert_allclose(rho[[1,3]],[.2,.3])
        np.testing.assert_allclose(fit.call_args.args[0],100*(y[:,[1,3]]/y[:,[1,3]].mean(0)-1))
        with patch.object(runner,'fit_shared_ar_chunk') as fit:
            for inside in (0,2):
                invalid_common=common.copy(); invalid_common[inside]=True
                with self.assertRaisesRegex(ValueError,'inside frozen group mask'):
                    runner.fit_bold_chunk(y,invalid_common,{})
            fit.assert_not_called()
            empty=runner.fit_bold_chunk(np.zeros((3,2)),np.zeros(2,dtype=bool),{})
            self.assertTrue(np.isnan(empty[0]).all())
            self.assertTrue(np.isnan(empty[1]).all())
            self.assertTrue(np.isnan(empty[2]).all())
            fit.assert_not_called()

    def test_same_S00_rho_every_arm_and_chunk_invariance(self):
        n=100
        t=np.arange(n,dtype=float)
        base=pd.DataFrame({'constant':np.ones(n),'gain':np.sin(t/7),'loss':np.cos(t/9)})
        extra=np.sin(t/4+.3)
        designs={}
        for i,arm in enumerate(runner.ARMS):
            frame=base.copy()
            if arm!='S00':
                frame['control']=extra*(i+1)
                frame['duplicate_constant']=1.
                frame['absent_response']=0.
            designs[arm]=runner.independent_design(frame)
        rng=np.random.default_rng(417)
        Y=rng.normal(size=(n,7))
        for row in range(1,n):
            Y[row]+=.45*Y[row-1]
        Y+=np.sin(t/7)[:,None]*np.arange(1,8)[None,:]
        output,rho=runner.fit_shared_ar_chunk(Y,designs)
        labels,reference=run_glm(Y,designs['S00'][0],noise_model='ar1',bins=100,n_jobs=1)
        np.testing.assert_allclose(rho,[float(label) for label in labels],rtol=0,atol=3e-8)
        for arm in runner.ARMS:
            for label,source in reference.items():
                selected=labels==label
                fitted=ARModel(designs[arm][0],source.model.rho.copy()).fit(Y[:,selected])
                for i,name in enumerate(runner.FOCAL):
                    contrast=designs[arm][1][name]
                    np.testing.assert_allclose(output[arm][0][i,selected],contrast@fitted.theta,atol=1e-10)
                    np.testing.assert_allclose(output[arm][1][i,selected],(contrast@fitted.cov@contrast)*fitted.dispersion,atol=1e-10)
        first,rho1=runner.fit_shared_ar_chunk(Y[:,:3],designs)
        last,rho2=runner.fit_shared_ar_chunk(Y[:,3:],designs)
        np.testing.assert_array_equal(rho,np.r_[rho1,rho2])
        for arm in runner.ARMS:
            for value in (0,1):
                np.testing.assert_allclose(output[arm][value],np.concatenate([first[arm][value],last[arm][value]],axis=1),atol=1e-10)
        nonestimable=base.assign(gain_copy=base.gain)
        _,contrasts,info=runner.independent_design(nonestimable)
        self.assertIsNone(contrasts['gain'])
        self.assertFalse(info['focal_estimable']['gain'])
        self.assertTrue(info['focal_estimable']['loss'])

    def test_nifti_staging_and_confound_missingness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            data=np.arange(3*4*2*10,dtype='float32').reshape((3,4,2,10))+1
            image=nib.Nifti1Image(data,np.diag([2.,2.,2.,1.]))
            image.header.set_xyzt_units('mm','sec')
            image.header.set_zooms((2.,2.,2.,1.5))
            source=root/'source.nii.gz'
            nib.save(image,source)
            metadata=root/'source.json'
            metadata.write_text(json.dumps({'SliceTimingCorrected':False,'RepetitionTime':1.5}))
            staged=runner.stage_bold(source,root/'local',lambda:None,.01)
            _,header=runner.image_header(staged,metadata)
            self.assertEqual(header['tr_seconds'],1.5)
            np.testing.assert_array_equal(np.asanyarray(nib.load(staged).dataobj),data)
            names=[f'{axis}{suffix}' for axis in ('trans_x','trans_y','trans_z','rot_x','rot_y','rot_z')
                   for suffix in ('','_derivative1','_power2','_derivative1_power2')]
            frame=pd.DataFrame({name:np.zeros(10) for name in names+['white_matter','csf','framewise_displacement']})
            for name in names:
                if '_derivative1' in name:
                    frame.loc[0,name]=np.nan
            frame.loc[0,'framewise_displacement']=np.nan
            frame.loc[3,'framewise_displacement']=.51
            frame['non_steady_state_outlier00']=[1,0,0,0,0,0,0,0,0,0]
            path=root/'confounds.tsv'
            frame.to_csv(path,sep='\t',index=False)
            values,meta=runner.confounds_for_run(path,10)
            self.assertEqual(meta['unique_spike_rows'],[0,3])
            self.assertTrue(np.isfinite(values.to_numpy()).all())
            frame.loc[2,'trans_x_derivative1']=np.nan
            frame.to_csv(path,sep='\t',index=False)
            with self.assertRaisesRegex(ValueError,'Unexpected missing'):
                runner.confounds_for_run(path,10)

if __name__=='__main__':
    runner.require_compute()
    unittest.main()
