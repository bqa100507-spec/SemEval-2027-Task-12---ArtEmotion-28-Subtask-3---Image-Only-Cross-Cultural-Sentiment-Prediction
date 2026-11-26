"""Create an ignored Colab handoff with only the 1,335 allowed training artworks."""

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from st3.baseline import file_sha256
from st3.cli import json_output
from st3.contrast_study import new_output
from st3.data import image_path
from st3.polarity_workflow import read_sealed
from st3.visual_lora_inputs import config022,input_identity


def package(run,data,cache):
    cfg=config022();source=run/'inputs';read_sealed(source,input_identity(cfg))
    rows=pd.read_csv(source/'targets.csv')
    if not rows.role.eq('development').all() or rows.painting.nunique()!=1335:raise ValueError('Locked/incomplete population')
    output=run/'portable_bundle';archive=run/'exp022_colab_inputs.zip'
    if not output.exists():
        with new_output(output) as stage:
            shutil.copytree(source,stage/'inputs')
            for image,digest in rows[['image','image_sha256']].drop_duplicates().itertuples(index=False,name=None):
                old=image_path(data,image)
                if file_sha256(old)!=digest:raise ValueError('Artwork checksum differs')
                new=stage/'data/images'/image;new.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,new)
            relative=Path('huggingface')/('models--'+cfg['clip']['model'].replace('/','--'))/'snapshots'/cfg['clip']['revision']
            for name,digest in cfg['clip']['assets_sha256'].items():
                old=cache/relative/name
                if file_sha256(old)!=digest:raise ValueError('Pinned original CLIP-B asset differs')
                new=stage/'cache'/relative/name;new.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old,new)
            files={p.relative_to(stage).as_posix():file_sha256(p) for p in stage.rglob('*') if p.is_file()}
            json_output({'files_sha256':files,'input_manifest_sha256':file_sha256(source/'run.json'),'paintings':1335,
                'locked_images_included':False,'captions_included':False,'new_pretrained_checkpoints':False},stage/'bundle.json')
    manifest=json.loads((output/'bundle.json').read_text())
    actual={p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file() and p.name!='bundle.json'}
    if actual!=set(manifest['files_sha256']):raise ValueError('Incomplete/extra portable files')
    for name,digest in manifest['files_sha256'].items():
        if file_sha256(output/name)!=digest:raise ValueError('Portable checksum differs: '+name)
    if not archive.exists():
        temporary=archive.with_suffix('.partial')
        try:
            with zipfile.ZipFile(temporary,'w',zipfile.ZIP_STORED) as z:
                for p in sorted(output.rglob('*')):
                    if p.is_file():z.write(p,p.relative_to(output).as_posix())
            temporary.replace(archive)
        finally:
            if temporary.exists():temporary.unlink()
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or set(z.namelist())!=set(manifest['files_sha256'])|{'bundle.json'}:raise ValueError('Invalid handoff ZIP')
        for name,digest in {**manifest['files_sha256'],'bundle.json':file_sha256(output/'bundle.json')}.items():
            with z.open(name) as handle:
                if hashlib.file_digest(handle,'sha256').hexdigest()!=digest:raise ValueError('Handoff ZIP content mismatch: '+name)
    result={'archive':str(archive),'sha256':file_sha256(archive),'bytes':archive.stat().st_size,
        'bundle_manifest_sha256':file_sha256(output/'bundle.json'),'input_manifest_sha256':manifest['input_manifest_sha256']}
    json_output(result,run/'portable_handoff.json');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag,default in (('run-dir','runs/exp022_visual_lora'),('data-dir','data/upstream'),('cache-dir','cache')):
        p.add_argument('--'+flag,type=Path,default=ROOT/default)
    a=p.parse_args();package(a.run_dir.resolve(),a.data_dir.resolve(),a.cache_dir.resolve())
