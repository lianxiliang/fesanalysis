import numpy as np
import pytest
from numpy.testing import assert_allclose
from bivo4_fes import *


def test_shift_invariance_and_extreme_weights():
    lw=np.array([-10000.,-10001.,-10002.])
    assert_allclose(normalized_weights(lw),normalized_weights(lw+10001))
    assert_allclose(sum(populations([1,2,2],lw).values()),1)
    assert 1<weight_diagnostics(lw)['kish_ess']<3


def test_units_and_wall_formula():
    w=opes_logweights([96.4853321233],unit='kJ/mol')
    assert_allclose(w,opes_logweights([1.],unit='eV'),rtol=1e-9)
    assert_allclose(upper_wall([1.,2.,3.],2),[0,0,300])
    assert_allclose(lower_wall([1.,2.,3.],2),[300,0,0])


def test_conditional_population_and_basin():
    p=populations([1,2,1],np.log([1,3,100]),mask=np.array([True,True,False]))
    assert_allclose([p[1],p[2]],[.25,.75])
    d=basin_delta_f([True,False],[False,True],np.log([1,2]))
    assert_allclose(d,-8.617333262145e-5*350*np.log(2))


def test_time_alignment_not_row_alignment():
    c={'time':np.arange(10)*.1}
    assert align_colvar([0,.3,.6,.9],c).tolist()==[0,3,6,9]
    with pytest.raises(ValueError):align_colvar([.05,.15],c)
    with pytest.raises(ValueError):align_colvar([np.nan],c)
    with pytest.raises(ValueError):align_colvar([0,1],{'time':np.array([0,1,1])})


def test_xyz_timestep_stride_and_colvar(tmp_path):
    p=tmp_path/'t.xyz'
    p.write_text('2\nAtoms. Timestep: 100\nO 0 0 0\nH 1 0 0\n2\nAtoms. Timestep: 10100\nO 1 0 0\nH 2 0 0\n')
    f=list(iter_xyz(p,cell=[10,10,10],timestep_ps=.0005))
    assert_allclose([i.time for i in f],[.05,5.05])
    assert xyz_symbols(p).tolist()==['O','H']
    with pytest.raises(ValueError):list(iter_xyz(p))
    c=tmp_path/'COLVAR';c.write_text('#! FIELDS time d1 opes.bias\n0.05 1 -4\n5.05 2 -3\n')
    data=read_colvar(c);assert align_colvar([.05,5.05],data).tolist()==[0,1]


def test_cv_identity_guard():
    from types import SimpleNamespace
    r=SimpleNamespace(d_cv=np.array([2.7,2.8]),donor_cn=np.array([2.,1.]))
    c={'d1':np.array([2.7,4.]),'cn':np.array([2.,1.])}
    with pytest.raises(ValueError):check_cv_alignment(r,c,np.arange(2))


def test_no_forced_site_assignment():
    c=np.array([[.95,.01],[.48,.47],[.02,.01],[.9,.85]])
    assert protonated_site(c,[130,136]).tolist()==[130,-1,-1,-1]


def test_nan_recomputed_cvs_cannot_pass_identity_guard():
    from types import SimpleNamespace
    r=SimpleNamespace(d_cv=np.array([np.nan]),donor_cn=np.array([1.]))
    with pytest.raises(ValueError):
        check_cv_alignment(r,{'d1':np.array([2.7]),'cn':np.array([1.])},np.array([0]))


def test_mdanalysis_lammps_steps_and_id_order(tmp_path):
    import MDAnalysis as mda
    topology=tmp_path/'topology.xyz'
    topology.write_text('3\nreference\nO 0 0 0\nO 2.5 0 0\nH 1 0 0\n')
    text=''
    for step,shift in [(100,0.),(10100,.2)]:
        text+=(f'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n3\n'
               'ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n'
               'ITEM: ATOMS id type x y z\n'
               f'3 2 {1+shift} 0 0\n1 1 {shift} 0 0\n2 1 {2.5+shift} 0 0\n')
    dump=tmp_path/'test.dump';dump.write_text(text)
    u=mda.Universe(str(topology),str(dump),format='LAMMPSDUMP',dt=.0005)
    # Reader buffers are borrowed, so copy positions before advancing.
    records=[(f.time,f.step,f.positions.copy()) for f in iter_mdanalysis(u,timestep_ps=.0005)]
    assert_allclose([r[0] for r in records],[.05,5.05])
    assert [r[1] for r in records]==[100,10100]
    assert_allclose(records[0][2][:,0],[0,2.5,1])
    assert_allclose(records[1][2][:,0],[.2,2.7,1.2],atol=3e-7)


def test_known_biased_gaussian_recovers_target_moment():
    rng=np.random.default_rng(15)
    x=rng.normal(size=100000)
    # Biased ensemble has variance 1; target U/kBT=x^2 has variance 1/2.
    lw=-.5*x*x
    assert abs(weighted_mean(x*x,lw)-.5)<.006
