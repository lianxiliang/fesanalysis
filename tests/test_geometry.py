import numpy as np
import pytest
from numpy.testing import assert_allclose
from bivo4_fes import PTSelection, frame_paths, analyze, Frame, PTResult, as_box
from bivo4_fes.geometry import PATH_COLUMNS, choose_path

BOX=np.array([10.,10.,10.,90.,90.,90.])


def direct_fixture(hx=.98):
    p=np.array([[1.,1.,1.],[3.6,1.,1.],[1.+hx,1.,1.]])
    s=PTSelection.from_symbols(['O','O','H'],1,[2],waters=[],index_base=1)
    return p,s


@pytest.mark.parametrize('hx',[.98,1.30,1.62])
def test_direct_both_sides_and_shared(hx):
    p,s=direct_fixture(hx)
    rows,_=frame_paths(p,BOX,s)
    assert len(rows)==1
    assert_allclose(rows[0,[1,2]],[2*hx-2.6,2.6],atol=1e-10)
    assert rows[0,5]==2


def test_periodic_translation_invariance():
    p,s=direct_fixture()
    rows,_=frame_paths(p,BOX,s)
    p[0]+=[10,-20,30];p[1]+=[-30,20,0];p[2]+=[20,20,-10]
    translated,_=frame_paths(p,BOX,s)
    assert_allclose(rows,translated,atol=1e-12,equal_nan=True)


def test_angle_filter():
    p,s=direct_fixture();p[2]=[1,2,1]
    rows,_=frame_paths(p,BOX,s)
    assert len(rows)==0


def test_indirect_and_reverse():
    # D--H1--W--H2--A: two different H, exactly one water.
    p=np.array([[1.,1,1],[6.2,1,1],[3.6,1,1],[1.98,1,1],[4.58,1,1]])
    s=PTSelection.from_symbols(['O','O','O','H','H'],1,[2],waters=[3],index_base=1)
    for shift in [0.,.64]:
        xyz=p.copy();xyz[3:,0]+=shift
        rows,_=frame_paths(xyz,BOX,s)
        indirect=rows[rows[:,0]==1]
        assert len(indirect)==1
        assert_allclose(indirect[0,[1,2]],[-.64+2*shift,2.6],atol=1e-12)
        assert indirect[0,5]!=indirect[0,6]
        assert indirect[0,4]==2


def test_overlap_candidates_and_selection():
    p=np.array([[1.,1,1],[3.6,1,1],[3.7,1,1],[1.98,1,1]])
    s=PTSelection.from_symbols(['O','O','O','H'],1,[2,3],waters=[],index_base=1)
    rows,_=frame_paths(p,BOX,s)
    assert len(rows)==2
    assert choose_path(rows)[3]==1
    r=analyze([Frame(0,p,BOX)],s,progress=False,keep_paths=True)
    assert r.n_paths[0,0]==2
    assert r.paths.shape==(2,13)


def test_selection_overlap_removed_and_index_base_checked():
    s=PTSelection.from_symbols(['O','O','O','H'],1,[2],waters=[1,2,3])
    assert s.waters.tolist()==[2]
    with pytest.raises(ValueError):
        PTSelection.from_symbols(['O','O','H'],2,[3],waters=[],index_base=1)


def test_reference_backend_matches_orthogonal():
    rng=np.random.default_rng(4)
    for h in [.95,1.26,1.65]:
        p,s=direct_fixture(h);p+=rng.normal(0,.01,p.shape)
        fast,dist=frame_paths(p,BOX,s)
        slow,dist2=frame_paths(p,BOX,s,backend='reference')
        assert_allclose(fast,slow,atol=2e-6,equal_nan=True)
        assert_allclose(dist,dist2,atol=2e-6)


def test_triclinic_periodic_image():
    from MDAnalysis.lib.mdamath import triclinic_vectors
    box=as_box([10,11,12,80,85,65]);p,s=direct_fixture()
    rows,_=frame_paths(p,box,s)
    lattice=triclinic_vectors(box).astype(float)
    p[2]+=lattice[0]+lattice[1]
    shifted,_=frame_paths(p,box,s)
    assert_allclose(rows,shifted,atol=3e-6,equal_nan=True)


def test_roundtrip(tmp_path):
    p,s=direct_fixture();r=analyze([Frame(0,p,BOX,10,.005)],s,progress=False,keep_paths=True)
    r.save(tmp_path/'r.npz');rr=PTResult.load(tmp_path/'r.npz')
    assert_allclose(r.direct,rr.direct,equal_nan=True)
    assert rr.metadata==r.metadata


def test_chunked_geometry_equivalence():
    from bivo4_fes import analyze_chunks
    p,s=direct_fixture()
    frames=[Frame(i,p.copy(),BOX.copy(),i*100,float(i)) for i in range(5)]
    a=analyze(frames,s,progress=False,keep_paths=True)
    chunks=list(analyze_chunks(frames,s,chunk_size=2,progress=False,keep_paths=True))
    b=PTResult.concatenate(chunks)
    assert_allclose(a.paths,b.paths,equal_nan=True)
    assert_allclose(a.direct,b.direct,equal_nan=True)
    assert_allclose(a.frame,b.frame)


def test_reference_selection_removes_endpoint_bridges():
    from bivo4_fes import xyz_symbols
    from pathlib import Path
    path=Path(__file__).parents[1]/'examples/data/sample.xyz.gz'
    if not path.exists():
        pytest.skip('Private example data is not published to git.')
    s=PTSelection.colleague(xyz_symbols(path))
    assert not np.isin(s.waters,np.r_[s.donor,s.acceptors]).any()
    assert s.donor==150
    assert 241 in s.acceptors
