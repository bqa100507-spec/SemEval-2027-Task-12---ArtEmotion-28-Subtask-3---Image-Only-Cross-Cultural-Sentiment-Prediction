import hashlib
import numpy as np
import pytest
import torch
from PIL import Image
from transformers import CLIPImageProcessor

from st3 import pixel_cache
from st3.pixel_cache import RamPixels
from st3.visual_groups import PixelReader


@pytest.fixture
def readers(tmp_path,monkeypatch):
    monkeypatch.setattr(pixel_cache,'available_ram',lambda:8*1024**3)
    folder=tmp_path/'images';folder.mkdir()
    names=[]
    for i in range(5):
        name=f'{i}.jpg';names.append(name)
        values=np.random.default_rng(i).integers(0,256,(101+i*7,199-i*9,3),dtype=np.uint8)
        Image.fromarray(values).save(folder/name)
    reader=PixelReader(tmp_path,CLIPImageProcessor())
    return reader,RamPixels(reader,names),names


def test_pixels_are_bitwise_identical_for_groupings_and_duplicate_lookups(readers):
    original,cached,names=readers
    for selected in (names,names[::-1],[names[0]],[names[2],names[2],names[0]]):
        assert torch.equal(original(selected,'cpu'),cached(selected,'cpu'))
    assert cached.bytes==5*3*224*224*4
    assert cached._values.dtype==torch.float32 and not cached._values.requires_grad
    cached.cpu(names).zero_()
    assert torch.equal(original.cpu(names),cached.cpu(names))
    with pytest.raises(ValueError,match='absent'):cached.cpu(['missing'])


def test_cache_preserves_rng_and_requires_no_more_image_decodes(readers,monkeypatch):
    original,cached,names=readers
    before=torch.get_rng_state().clone();numpy_before=np.random.get_state()
    again=RamPixels(original,names)
    assert torch.equal(before,torch.get_rng_state())
    assert np.array_equal(numpy_before[1],np.random.get_state()[1])
    monkeypatch.setattr(original,'cpu',lambda n:pytest.fail('Cached epoch must not decode/preprocess again'))
    again.profile=True;again(names,'cpu')
    assert again.stats['decode_seconds']==again.stats['preprocess_seconds']==0


def test_budget_failure_is_explicit_and_never_falls_back(readers,monkeypatch):
    original,_,names=readers
    with pytest.raises(ValueError,match='cache cap'):RamPixels(original,names,max_bytes=1)
    monkeypatch.setattr(pixel_cache,'available_ram',lambda:1)
    with pytest.raises(ValueError,match='insufficient RAM'):RamPixels(original,names)


def test_profile_does_not_change_pixels(readers):
    original,cached,names=readers
    for reader in (original,cached):
        value=reader(names,'cpu');reader.profile=True;reader.reset_stats()
        assert torch.equal(value,reader(names,'cpu'))
        assert reader.stats['transfer_seconds']>=0
    assert original.stats['decode_seconds']>0 and original.stats['preprocess_seconds']>0


def test_loss_gradients_updates_and_dropout_rng_are_exact(readers):
    original,cached,names=readers
    signatures=[]
    for reader in (original,cached):
        torch.manual_seed(2027)
        model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*224*224,8),torch.nn.Dropout(.2))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        for _ in range(3):
            optimizer.zero_grad();output=model(reader(names,'cpu'));output.square().mean().backward();optimizer.step()
        signatures.append(([p.detach().clone() for p in model.parameters()],[p.grad.clone() for p in model.parameters()],torch.get_rng_state()))
    for a,b in zip(signatures[0][:2],signatures[1][:2]):
        assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert torch.equal(signatures[0][2],signatures[1][2])
