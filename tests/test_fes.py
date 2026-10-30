import numpy as np
from numpy.testing import assert_allclose
from bivo4_fes import compute_fes,FES,read_fes
from scipy.special import logsumexp


def data():
    rng=np.random.default_rng(2)
    q=rng.normal(size=(120,2))*.3
    lw=rng.normal(size=len(q))*4
    return q,lw


def test_local_matches_exact_2d():
    q,w=data()
    a=compute_fes(q,w,bounds=[[-1,1],[-1,1]],bins=25,bandwidth=.2,method='local',cutoff=12)
    b=compute_fes(q,w,bounds=[[-1,1],[-1,1]],bins=25,bandwidth=.2,method='exact')
    assert_allclose(a.free_energy,b.free_energy,atol=1e-11)
    assert_allclose(a.local_ess,b.local_ess,atol=1e-9)


def test_local_matches_exact_1d_and_shift():
    q,w=data()
    a=compute_fes(q[:,0],w,bounds=[-1,1],bins=101,bandwidth=.15)
    b=compute_fes(q[:,0],w+10000,bounds=[-1,1],bins=101,bandwidth=.15,method='exact')
    assert_allclose(a.free_energy,b.free_energy,atol=2e-12)


def test_binned_accuracy_and_nonnegative_support():
    q,w=data()
    kw=dict(bounds=[[-1,1],[-1,1]],bins=101,bandwidth=.15)
    a=compute_fes(q,w,method='local',**kw)
    b=compute_fes(q,w,method='binned',**kw)
    populated=a.free_energy<.15
    assert np.max(np.abs(a.free_energy[populated]-b.free_energy[populated]))<.006
    assert np.all(b.local_ess>=0)


def test_conditional_joint_normalization_and_nans():
    q,w=data();q[:10]=np.nan
    a=compute_fes(q,w,bounds=[[-1,1],[-1,1]],bins=21,bandwidth=.2)
    b=compute_fes(q,w,bounds=[[-1,1],[-1,1]],bins=21,bandwidth=.2,log_normalizer=logsumexp(w))
    assert_allclose(a.free_energy,b.free_energy,atol=1e-12)
    assert_allclose(b.log_density-a.log_density,logsumexp(w[10:])-logsumexp(w),atol=1e-12)


def test_rare_weight_not_lost_to_floor():
    q=np.array([[-1.],[1.]])
    a=compute_fes(q,[0,-60],bounds=[-1.2,1.2],bins=241,bandwidth=.03)
    i=np.argmin(abs(a.axes[0]+1));j=np.argmin(abs(a.axes[0]-1))
    assert_allclose(a.free_energy[j]-a.free_energy[i],60*350*8.617333262145e-5,rtol=1e-10)
    assert np.isinf(a.free_energy[120]) # unsupported, not an invented finite floor


def test_save_and_text_units(tmp_path):
    q,w=data();a=compute_fes(q,w,bounds=[[-1,1],[-1,1]],bins=21,bandwidth=.2)
    a.save(tmp_path/'f.npz');b=FES.load(tmp_path/'f.npz')
    assert_allclose(a.free_energy,b.free_energy)
    a.to_plumed(tmp_path/'f.dat',['delta','d'])
    c=read_fes(tmp_path/'f.dat',energy_unit='eV')
    assert_allclose(c.free_energy,a.free_energy,atol=1e-10)


def test_profile_at_finite_node_next_to_missing_cell():
    from bivo4_fes import FES
    z=np.zeros((3,3));z[0,0]=np.inf
    s=FES((np.arange(3.),np.arange(3.)),z,z.copy(),z.copy(),np.zeros((3,3)),{})
    assert_allclose(s.profile([[1.,1.]]),[0.])
    assert np.isnan(s.profile([[.5,.5]])[0])


def test_bad_weights_and_unknown_units_fail(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        compute_fes([0.,1.],[0.,np.nan],bounds=[-1.,2.],bins=31,bandwidth=.2)
    p=tmp_path/'no_units.dat';np.savetxt(p,[[0,0,1],[0,1,2],[1,0,3],[1,1,4]])
    with pytest.raises(ValueError):read_fes(p)
    assert read_fes(p,energy_unit='eV').unit=='eV'


def test_binned_requires_resolved_grid():
    import pytest
    with pytest.raises(ValueError):
        compute_fes([0.,1.],[0.,0.],bounds=[-1.,2.],bins=31,bandwidth=.15,method='binned')
