"""Notebook callbacks and cached reselection do not change default behavior."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from bivo4_fes import Frame, PTSelection, analyze, PATH_COLUMNS, PTResult

BOX=np.array([10.,10.,10.,90.,90.,90.])

def fixture():
    xyz=np.array([[1.,1,1],[3.6,1,1],[3.7,1,1],[1.98,1,1]])
    selection=PTSelection.from_symbols(['O','O','O','H'],1,[2,3],waters=[])
    return [Frame(i,xyz.copy(),BOX.copy()) for i in range(3)],selection

def farthest(candidates):
    return candidates[np.argmax(candidates[:, PATH_COLUMNS.index('distance')])]

def test_user_selector_and_original_default():
    frames,s=fixture()
    old=analyze(frames,s,progress=False)
    custom=analyze(frames,s,path_selector=farthest,progress=False)
    assert np.all(old.channel('direct')['acceptor']==1)
    assert np.all(custom.channel('direct')['acceptor']==2)
    assert custom.metadata['selector']=='farthest'

def test_cache_reselection_matches_fresh_callback():
    frames,s=fixture()
    old=analyze(frames,s,keep_paths=True,progress=False)
    new=old.reselect(farthest)
    fresh=analyze(frames,s,path_selector=farthest,progress=False)
    assert_allclose(new.direct,fresh.direct,equal_nan=True)
    assert_allclose(new.indirect,fresh.indirect,equal_nan=True)
    assert np.all(old.channel('direct')['acceptor']==1)
    assert_allclose(old.reselect().direct,old.direct,equal_nan=True)

def test_reselection_survives_cache_roundtrip(tmp_path):
    frames,s=fixture()
    old=analyze(frames,s,keep_paths=True,progress=False)
    old.save(tmp_path/'pt.npz')
    assert_allclose(PTResult.load(tmp_path/'pt.npz').reselect(farthest).direct,
                    old.reselect(farthest).direct,equal_nan=True)

def test_empty_candidates_are_not_given_to_callback():
    frames,s=fixture()
    frames[0].positions[3]=[1.,2.,1.]
    def do_not_call(candidates):
        raise AssertionError('There should be no candidate here.')
    result=analyze(frames[:1],s,keep_paths=True,progress=False,path_selector=do_not_call)
    assert not result.valid('direct').any()
    assert not result.reselect(do_not_call).valid('indirect').any()

def test_missing_cache_error():
    frames,s=fixture()
    r=analyze(frames,s,progress=False)
    with pytest.raises(ValueError,match='keep_paths'):
        r.reselect(farthest)

def test_callback_shape_error():
    frames,s=fixture()
    with pytest.raises(ValueError,match='one row'):
        analyze(frames,s,progress=False,path_selector=lambda rows: rows[:,1])
