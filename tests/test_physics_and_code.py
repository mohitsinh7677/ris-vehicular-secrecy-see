import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from physical_umi import Config,channel_snapshot,physical_action,secrecy,far_field_distance_m,dist,pl_umi_los,pl_umi_nlos
from run_sweeps import sample_training_config,SPEEDS,DELAYS,P_VALUES,N_VALUES

def test_nominal_3gpp_umi_direct_pathloss():
    c=Config()
    # 3GPP TR38.901 UMi LOS first branch and NLOS under stated nominal distances
    d=100.
    expected_los=32.4+21*np.log10(d)+20*np.log10(28)
    assert abs(pl_umi_los(d,28)-expected_los)<1.e-12
    candidate=22.4+35.3*np.log10(d)+21.3*np.log10(28)
    assert abs(pl_umi_nlos(d,28)-max(expected_los,candidate))<1.e-12

def test_rectangular_ris_far_field_all_sizes():
    for n in N_VALUES:
        c=Config(n=n)
        boundary=far_field_distance_m(c)
        assert boundary<dist(c.ris,c.user)
        assert boundary<dist(c.ris,c.bs)

def test_power_consumption_watts():
    c=Config()
    assert abs(c.pt_w()-10**(-1.5))<1.e-12
    assert abs(c.noise_dbm()-(-87.))<1.e-12
    assert np.isclose(c.total_power_w(True)-c.total_power_w(False),.1+c.n*.001)
    assert c.total_power_w(False) < c.total_power_w(True)

def test_channels_observed_not_eve_and_finite():
    c=Config()
    sample=channel_snapshot(c,np.random.default_rng(771))
    assert sample['Hbr_true'].shape==(128,8)
    assert sample['hru_true'].shape==(128,)
    assert sample['hbu_obs'].shape==(8,)
    assert np.isfinite(sample['Hbr_true']).all()
    w,phi=physical_action(c,np.zeros(2*c.m+c.n))
    assert np.isclose(np.vdot(w,w).real,c.pt_w())
    assert np.allclose(abs(phi),1.)
    val,*_=secrecy(c,sample,w,phi)
    assert np.isfinite(val) and val >=0.

def test_train_excludes_heldout_speeds():
    cfg=Config()
    rng=np.random.default_rng(88)
    seen={sample_training_config(cfg,rng).v_kmh for _ in range(200)}
    assert set(seen) <= {30,45,60,75,90}
    assert 105 not in seen and 120 not in seen

def test_snapshot_reproducibility():
    c=Config(n=32)
    a=channel_snapshot(c,np.random.default_rng(442))
    b=channel_snapshot(c,np.random.default_rng(442))
    assert np.array_equal(a['Hbr_true'],b['Hbr_true'])
    assert np.array_equal(a['hru_obs'],b['hru_obs'])


def test_far_field_256_uses_panel_diagonal():
    # 16 x 16, half-wavelength element-center spacing at 28 GHz.
    c=Config(n=256)
    wavelength=299792458.0/(28.0e9)
    expected=2*(np.hypot(15,15)*wavelength/2)**2/wavelength
    assert np.isclose(far_field_distance_m(c),expected)
    assert 2.3 < far_field_distance_m(c) < 2.5
