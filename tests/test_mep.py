import numpy as np
import pytest
from numpy.testing import assert_allclose
from bivo4_fes import String2D,minimax_path


def surface():
    x=np.linspace(-1.4,1.4,141);y=np.linspace(-.5,.5,101)
    X,Y=np.meshgrid(x,y,indexing='ij')
    return x,y,(X*X-1)**2+4*Y*Y


def test_xy_gradient_order_nonsquare_grid():
    x,y,z=surface();a=String2D(x,y,z,indexing='ij');b=String2D(x,y,z.T,indexing='xy')
    pts=np.array([[.3,.1],[-.4,.2]])
    assert_allclose(a.interpolate_gradients_map(pts),b.interpolate_gradients_map(pts))


def test_mep_reverse_and_fixed_ends():
    x,y,z=surface()
    paths=[]
    for a,b in [([-1,0],[1,0]),([1,0],[-1,0])]:
        s=String2D(x,y,z,indexing='ij')
        s.compute_mep(a,b,npts=35,flexible=False,dt=.005,maxsteps=1000,tol=1e-5,progress=False)
        assert s.converged
        assert_allclose(s.mep[[0,-1]],[a,b])
        path,e=s.get_mep_energy_profile();assert abs(e.max()-e[0]-1.)<.01
        paths.append(path)
    assert_allclose(paths[0],paths[1][::-1],atol=1e-12)


def test_parametric_turn_and_zero_gradient():
    x=np.linspace(-2,2,101);y=x.copy();z=np.zeros((101,101))
    s=String2D(x,y,z,indexing='ij')
    s.compute_mep([1,1],[-1,-1],mid=[[1.5,0]],npts=35,flexible=False,progress=False,maxsteps=10)
    assert s.converged
    assert np.max(s.string_traj[0][:,0])>1


def test_no_path_across_missing_cells():
    x,y,z=surface();z[65:75,:]=np.nan
    with pytest.raises(ValueError):minimax_path(x,y,z,[-1,0],[1,0])
    s=String2D(x,y,z,indexing='ij')
    with pytest.raises(ValueError):s.compute_mep([-1,0],[1,0],progress=False)


def test_graph_initializer_supported():
    x,y,z=surface();path=minimax_path(x,y,z,[-1,0],[1,0])
    assert_allclose(path[[0,-1]],[[-1,0],[1,0]],atol=.02)
    assert_allclose(path[:,1],0.,atol=.02)


def test_sparse_string_cannot_skip_a_missing_strip():
    x,y,z=surface();z[70,:]=np.nan
    s=String2D(x,y,z,indexing='ij')
    # All listed images are finite, but their connecting segments cross the hole.
    assert not s._supported([[-1.,0.],[.1,0.],[1.,0.]])
