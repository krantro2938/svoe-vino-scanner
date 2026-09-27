from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('cv2')
from app.local_features import (
    INDEX_VERSION,
    PARAMETERS,
    LocalFeatures,
    catalog_fingerprint,
    has_side_objects,
    strong_match,
)
from app.models import Wine
from app.config import Settings
from app.retrieval import RecognitionEngine
from app.imaging import DecodedImage
from PIL import Image


def test_strong_gate_rejects_shared_artwork_and_weak_geometry():
    def evidence(inliers, score, second):
        return {'candidates': [dict(slug='a', inliers=inliers, score=score), dict(slug='b', inliers=5, score=second)]}
    assert strong_match(evidence(8, 30, 10)) is None
    assert strong_match(evidence(100, 100, 80)) is None
    assert strong_match(evidence(30, 30, 15))[0] == 'a'


def test_side_object_routing_distinguishes_one_and_three_bottles():
    single = np.full((160, 384, 3), 255, dtype=np.uint8)
    single[:, 160:224] = 0
    shelf = single.copy()
    shelf[:, 20:84] = 0
    shelf[:, 300:364] = 0
    assert not has_side_objects(single)
    assert has_side_objects(shelf)


def write_index(path, catalog, version=INDEX_VERSION):
    rng = np.random.default_rng(42)
    np.savez_compressed(path, descriptors=rng.random((20,128),dtype=np.float32),
                        points=rng.random((20,2),dtype=np.float32),offsets=np.array([0,20]),
                        slugs=np.array(['a']), parameters=json.dumps(PARAMETERS),
                        index_version=version,catalog_sha256=catalog_fingerprint(catalog))
    path.with_suffix('.npz.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest())


def test_index_checks_catalog_version_and_checksum(tmp_path):
    catalog={'a':Wine(slug='a',name='Wine A')}
    path=tmp_path/'index.npz'
    write_index(path,catalog)
    engine=LocalFeatures();engine.load(path,catalog)
    assert engine.slugs==['a']
    with pytest.raises(ValueError,match='catalog mismatch'):
        engine.load(path,{'a':Wine(slug='a',name='Different wine')})
    write_index(path,catalog,version='unknown')
    with pytest.raises(ValueError,match='version'):
        engine.load(path,catalog)
    path.write_bytes(path.read_bytes()+b'corrupted')
    with pytest.raises(ValueError,match='checksum'):
        engine.load(path,catalog)


def test_missing_optional_index_falls_back(tmp_path):
    catalog=tmp_path/'catalog.json'
    catalog.write_text('[{"slug":"a","name":"Wine A"}]')
    engine=RecognitionEngine(Settings(catalog_path=catalog,index_manifest_path=None,
                                      local_feature_index_path=tmp_path/'missing.npz'))
    assert engine._local_features is None
    assert engine.predict(DecodedImage(Image.new('RGB',(64,96)),'PNG',64,96)).wine.slug=='a'


def test_strong_local_match_precedes_ocr_and_is_cached(tmp_path):
    catalog=tmp_path/'catalog.json'
    catalog.write_text('[{"slug":"a","name":"Wine A"},{"slug":"b","name":"Wine B"}]')
    engine=RecognitionEngine(Settings(catalog_path=catalog,index_manifest_path=None))
    class Stub:
        calls=0
        def predict(self,image):
            self.calls+=1
            return {'candidates':[dict(slug='b',inliers=30,score=32),dict(slug='a',inliers=6,score=8)]}
    local=Stub();engine._local_features=local
    image=DecodedImage(Image.new('RGB',(64,96)),'PNG',64,96)
    first=engine.predict(image,'digest');second=engine.predict(image,'digest')
    assert first.wine.slug=='b' and first.method=='local_features'
    assert second is first and local.calls==1
