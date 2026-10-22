"""Colab full-data main experiment. Source A first, exact distances, then B/C/D/E."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import traceback
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from full_analysis import write_json


def split_config(base, split):
    config = {**base,'name':f'colab_full_split_{split}','partition_seed':split,
              'manifest_path':f'data/full/split_{split}/manifest.csv'}
    return config


def expected_count(config):
    return len(config['partition_seeds'])*len(config['sources'])*len(config['depths'])*len(config['seeds'])


def status(out, config, stage, current=None, error=None):
    finished=[]
    for split in config['partition_seeds']:
        for source in config['sources']:
            for seed in config['seeds']:
                for depth in config['depths']:
                    path=out/f'split_{split}/runs/source_{source}/{depth}/seed_{seed}/metrics.json'
                    if path.exists():
                        saved=json.loads(path.read_text())
                        if saved.get('status')=='complete' and saved.get('config')==split_config(config,split):
                            finished.append([split,source,depth,seed])
    payload={'stage':stage,'complete_runs':len(finished),'expected_runs':expected_count(config),
             'current':current,'error':error,'finished':finished,
             'complete':stage=='complete' and len(finished)==expected_count(config)}
    write_json(out/'progress.json',payload)
    print(f"[full status] {stage}: {len(finished)}/{expected_count(config)} complete; current={current}",flush=True)
    return payload


def prepare(root, config, out, image_cache_dir=None):
    from camelyon.data import prepare_full_manifests, prepare_data
    from camelyon.train import build_image_digest_manifest
    reviewed=prepare_full_manifests(root,partition_seeds=tuple(config['partition_seeds']),
                                   revision=config['dataset_revision'],output_dir='data/full')
    if not reviewed['all_ready'] or reviewed['total_images']!=config['expected_images']:
        raise ValueError('Prespecified full-data patient partitions failed validation')
    first=config['partition_seeds'][0]
    manifest=root/f'data/full/split_{first}/manifest.csv'
    if image_cache_dir is not None:
        import pandas as pd
        paths=pd.read_csv(manifest,usecols=['path']).path.astype(str)
        if all((root/relative).is_file() for relative in paths):
            print('[image cache] warm runtime: all PNG files present; retain local images',flush=True)
        else:
            from full_cache import restore_cache
            restored=restore_cache(root,manifest,image_cache_dir,revision=config['dataset_revision'],
                                   expected_images=config['expected_images'])
            print(f'[image cache] complete private cache restored: {restored}',flush=True)
    prepare_data(root,per_slide_class=None,seed=config['sampling_seed'],revision=config['dataset_revision'],
                 partition_seed=first,manifest_path=f'data/full/split_{first}/manifest.csv')
    build_image_digest_manifest(root,manifest,root/config['image_digest_manifest'])
    if image_cache_dir is not None:
        from full_cache import export_cache
        export_cache(root,manifest,image_cache_dir,revision=config['dataset_revision'],
                     expected_images=config['expected_images'])
    for split in config['partition_seeds']:
        folder=out/f'split_{split}'
        folder.mkdir(parents=True,exist_ok=True)
        conf=split_config(config,split)
        path=folder/'config.json'
        if path.exists() and json.loads(path.read_text())!=conf:
            raise ValueError(f'Existing full result config differs: {folder}')
        write_json(path,conf)
        data=root/f'data/full/split_{split}'
        for name in ['manifest.csv','provenance.json','patient_partitions.csv','partition_review.json']:
            if (data/name).exists():
                shutil.copy2(data/name,folder/name)
        import pandas as pd
        frame=pd.read_csv(data/'manifest.csv')
        if len(frame)!=config['expected_images'] or frame.sample_id.duplicated().any():
            raise ValueError('Full image coverage check failed')
        for key in ['patient','slide']:
            if (frame.groupby(['center',key]).partition.nunique()>1).any():
                raise ValueError(f'Leaked {key}')
        counts=frame.groupby(['center','partition','label']).size()
        if len(counts)!=30 or (counts<=0).any():
            raise ValueError('Every center/partition must have both classes')
    status(out,config,'prepared')


def summarize(out, config):
    from full_results import generate
    return generate(out,split_seeds=tuple(config['partition_seeds']),sources=tuple(config['sources']),
                    depths=tuple(config['depths']),model_seeds=tuple(config['seeds']),
                    expected_images=config['expected_images'])


def train_and_analyze(root, config, out):
    from camelyon.train import run_experiment
    from full_analysis import extract_full_reference, compute_full_shift
    features=None
    for source in config['sources']:
        for split in config['partition_seeds']:
            conf=split_config(config,split)
            folder=out/f'split_{split}'
            for seed in config['seeds']:
                for depth in config['depths']:
                    current={'source':source,'partition_seed':split,'model_seed':seed,'depth':depth}
                    status(out,config,'training',current)
                    run_experiment(root,conf,source,depth,seed,folder/f'runs/source_{source}/{depth}/seed_{seed}')
                    status(out,config,'training',current)
            # First complete source-A block precedes distance calculation. Reuse the
            # fixed reference and split-specific exact distances for later sources.
            status(out,config,'distance',{'source':source,'partition_seed':split})
            if features is None:
                features=extract_full_reference(root,conf,out/'reference_cache')
            compute_full_shift(*features,root/conf['manifest_path'],folder/'analysis')
            summarize(out,config)
    return out


def verify(root,config,out):
    from audit_results import audit_results
    for split in config['partition_seeds']:
        conf=split_config(config,split)
        result=audit_results(root/conf['manifest_path'],out/f'split_{split}',conf,main_only=True)
        if not result['complete']:
            raise RuntimeError(f"Split {split} audit failed: {result['failures'][:3]}")
    result=summarize(out,config)
    if not result['complete']:
        raise RuntimeError(f"Full result coverage/stability report incomplete: {result.get('issues')}")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','run','verify','report','all'])
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parent/'configs/full.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--image-cache-dir',type=Path,
                        help='Optional private uncompressed PNG cache; never part of training config/signature')
    args=parser.parse_args()
    root,out=args.root.resolve(),args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    config=json.loads(args.config.read_text())
    plan=out/'full_plan.json'
    if plan.exists() and json.loads(plan.read_text())!=config:
        raise ValueError('Refusing to replace a different full experiment plan')
    write_json(plan,config)
    try:
        if args.stage in ['prepare','all']:prepare(root,config,out,args.image_cache_dir)
        if args.stage in ['run','all']:train_and_analyze(root,config,out)
        if args.stage=='report':summarize(out,config)
        if args.stage in ['verify','all']:
            verify(root,config,out)
            status(out,config,'complete')
    except BaseException as error:
        previous=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else {}
        status(out,config,'interrupted',previous.get('current'),f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':main()
