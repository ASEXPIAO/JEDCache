"""CPU checks of the shipped experimental controller; no video generation."""
import unittest
import torch
from jedcache import BRANCHES,CachePolicy,ResidualCacheBank,direction_coefficients
class ControllerChecks(unittest.TestCase):
    def setUp(self):
        self.h={b:torch.full((2,3),float(i+1),device='cpu') for i,b in enumerate(BRANCHES)}
        self.bank=ResidualCacheBank();self.bank.begin_task('first')
        self.p=CachePolicy('joint',warmup_steps=2,max_reuse=1)
        # Synthetic state exercises the API; it is not an experimental fit.
        self.p.joint_model.observe([1.,0.,0.,0.,0.,0.,0.,0.],0.5)
    def prime(self):
        for i,b in enumerate(BRANCHES):self.bank.refresh(b,self.h[b],torch.full_like(self.h[b],float(10+i)),1)
    def test_signed_combination(self):
        for target in [10.,7.5]:
            c=direction_coefficients(target,5.)
            actual=sum(c[b]*self.h[b] for b in BRANCHES)
            expected=self.h['target_uncond']+target*(self.h['target_cond']-self.h['target_uncond'])-(self.h['source_uncond']+5*(self.h['source_cond']-self.h['source_uncond']))
            torch.testing.assert_close(actual,expected)
    def test_frozen_shared_action_and_age_guard(self):
        self.prime();a=self.p.joint_model.xtx.clone();z=self.p.joint_model.xty.clone()
        self.p.joint_threshold=1e9
        f=self.bank.feature_values(self.h,direction_coefficients(10,5))
        decision,_=self.p.decide(2,35,self.bank,f)
        self.assertTrue(all(v is False for v in decision.values()))
        for b in BRANCHES:self.bank.reused(b)
        decision,_=self.p.decide(3,35,self.bank,f)
        self.assertTrue(all(decision.values()))
        torch.testing.assert_close(a,self.p.joint_model.xtx);torch.testing.assert_close(z,self.p.joint_model.xty)
    def test_missing_cache_and_warmup(self):
        f=self.bank.feature_values(self.h,direction_coefficients(10,5))
        self.assertTrue(all(self.p.decide(10,35,self.bank,f)[0].values()))
        self.prime();self.p.joint_threshold=1e9
        self.assertTrue(all(self.p.decide(1,35,self.bank,f)[0].values()))
    def test_separate_caches_and_task_reset(self):
        self.prime()
        self.assertEqual(len({self.bank.residual(b).data_ptr() for b in BRANCHES}),4)
        self.bank.begin_task('second')
        self.assertTrue(all(not self.bank.has(b) for b in BRANCHES))
if __name__=='__main__':unittest.main()
