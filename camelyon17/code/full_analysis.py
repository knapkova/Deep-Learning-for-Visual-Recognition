"""Full-data, bounded-memory fixed ImageNet feature distances; no adaptation."""
from __future__ import annotations
import hashlib
import json
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

REFERENCE = 'torchvision.resnet50.IMAGENET1K_V2.avgpool2048'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)


def extract_full_reference(root, config, cache):
    """Extract each image once. Disk memmap avoids multiple full feature copies."""
    import torch
    from torch.utils.data import DataLoader
    from torchvision.models import resnet50, ResNet50_Weights
    from camelyon.data import PatchDataset
    root, cache = Path(root), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(root/config['manifest_path'], dtype={'sample_id':str})
    frame = frame.sort_values('sample_id', kind='stable').reset_index(drop=True)
    stable = frame[['sample_id','path','center','patient','slide','label']]
    image_records=json.loads((root/config['image_digest_manifest']).read_text())
    # Content identity, not runtime-local mtimes, must survive a Colab reset.
    content_identity=hashlib.sha256(json.dumps(
        {k:v['sha256'] for k,v in image_records['images'].items()},
        sort_keys=True,separators=(',',':')).encode()).hexdigest()
    identity = {'reference': REFERENCE, 'rows':len(frame), 'dimensions':2048,
                'image_size':config['image_size'], 'dataset_revision':config['dataset_revision'],
                'image_content_digest':content_identity,
                'members_sha256':hashlib.sha256(stable.to_csv(index=False).encode()).hexdigest()}
    path, meta, rows_path = cache/'reference_features.npy', cache/'reference_features.metadata.json', cache/'feature_rows.csv'
    if path.exists() and meta.exists() and rows_path.exists():
        saved = json.loads(meta.read_text())
        if saved['identity'] != identity or saved['features_sha256'] != sha256(path) or saved['rows_sha256'] != sha256(rows_path):
            raise ValueError('Full reference cache identity/checksum changed; refusing stale reuse')
        local=root/'data/full/reference_features.npy'
        if not local.exists() or sha256(local)!=saved['features_sha256']:
            shutil.copy2(path,local)
        return local, rows_path
    if not torch.cuda.is_available():
        raise RuntimeError('Full reference extraction requires the authorized Colab GPU')
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.requires_grad_(False).eval().to('cuda')
    local = root/'data/full/reference_features.partial.npy'
    local.parent.mkdir(parents=True, exist_ok=True)
    values = np.lib.format.open_memmap(local, mode='w+', dtype=np.float32, shape=(len(frame),2048))
    loader = DataLoader(PatchDataset(root,frame,image_size=config['image_size'],augment=False),
                        batch_size=config.get('feature_batch_size',128), shuffle=False,
                        num_workers=config.get('num_workers',2),pin_memory=True)
    offset = 0
    with torch.inference_mode():
        for x, _ in loader:
            block = model(x.to('cuda',non_blocking=True)).cpu().numpy()
            if block.shape[1] != 2048 or not np.isfinite(block).all():
                raise ValueError('Invalid fixed reference feature batch')
            values[offset:offset+len(block)] = block
            offset += len(block)
            if offset % (loader.batch_size*100) == 0 or offset == len(frame):
                print(f'[full features] {offset}/{len(frame)}',flush=True)
    values.flush()
    del values, model
    torch.cuda.empty_cache()
    tmp = path.with_suffix('.npy.partial')
    shutil.copy2(local,tmp)
    os.replace(tmp,path)
    stable.to_csv(rows_path,index=False)
    write_json(meta,{'identity':identity,'features_sha256':sha256(path),'rows_sha256':sha256(rows_path),
                     'partition_policy':'Feature extraction is fixed and label-free; each split distance uses train members only.'})
    finished_local=local.with_name('reference_features.npy')
    os.replace(local,finished_local)
    return finished_local,rows_path


def compute_full_shift(feature_path, rows_path, manifest_path, output_dir):
    """Exact marginal W1 over ALL train members, including exact 50/50 class mixture."""
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    identity = {'manifest_sha256':sha256(manifest_path),'rows_sha256':sha256(rows_path),
                'features_sha256':sha256(feature_path),
                'reference':REFERENCE,'metric':'mean_marginal_1d_wasserstein1',
                'population':'all train images; no sample cap'}
    path, meta = output/'representation_shift.csv', output/'representation_shift.metadata.json'
    if path.exists() and meta.exists():
        saved = json.loads(meta.read_text())
        if saved.get('identity') != identity or saved.get('csv_sha256') != sha256(path):
            raise ValueError('Existing distance result does not match full manifest')
        return path
    rows = pd.read_csv(rows_path,dtype={'sample_id':str})
    manifest = pd.read_csv(manifest_path,dtype={'sample_id':str})
    if rows.sample_id.duplicated().any() or manifest.sample_id.duplicated().any() or set(rows.sample_id) != set(manifest.sample_id):
        raise ValueError('Feature rows must match the entire full manifest exactly')
    manifest = manifest.set_index('sample_id').loc[rows.sample_id].reset_index()
    for key in ['center','patient','slide','label']:
        if not np.array_equal(rows[key],manifest[key]):
            raise ValueError(f'Feature/manifest metadata mismatch: {key}')
    x = np.load(feature_path,mmap_mode='r',allow_pickle=False)
    if x.ndim != 2 or len(x) != len(rows):
        raise ValueError('Invalid feature array shape')
    centers = sorted(manifest.center.unique())
    indices,labels,weights = {},{},{}
    for c in centers:
        ix = np.flatnonzero(manifest.center.eq(c)&manifest.partition.eq('train'))
        y = manifest.label.to_numpy()[ix]
        if set(y) != {0,1}:
            raise ValueError(f'Hospital {c}: both train classes required')
        indices[c],labels[c] = ix,y
        weights[c] = np.where(y==0,.5/(y==0).sum(),.5/(y==1).sum())
    pairs = [(a,b) for i,a in enumerate(centers) for b in centers[i+1:]]
    sums = {p:np.zeros(4,dtype=np.float64) for p in pairs}
    for start in range(0,x.shape[1],16):
        end = min(start+16,x.shape[1])
        blocks = {c:np.asarray(x[ix,start:end]) for c,ix in indices.items()}
        if not all(np.isfinite(v).all() for v in blocks.values()):
            raise ValueError('Non-finite features')
        for a,b in pairs:
            for j in range(end-start):
                u,v = blocks[a][:,j],blocks[b][:,j]
                sums[a,b] += [wasserstein_distance(u,v),
                    wasserstein_distance(u,v,u_weights=weights[a],v_weights=weights[b]),
                    wasserstein_distance(u[labels[a]==0],v[labels[b]==0]),
                    wasserstein_distance(u[labels[a]==1],v[labels[b]==1])]
        if end%128 == 0 or end == x.shape[1]:
            print(f'[full W1] {Path(manifest_path).parent.name}: {end}/{x.shape[1]} dimensions',flush=True)
    result=[]
    for (a,b),s in sums.items():
        raw,balanced,c0,c1 = s/x.shape[1]
        for source,target in [(a,b),(b,a)]:
            result.append({'source':source,'target':target,'source_train_n':len(indices[source]),
                'target_train_n':len(indices[target]),'shift_raw':raw,'shift_balanced':balanced,
                'shift_class0':c0,'shift_class1':c1,'shift_classconditional':(c0+c1)/2,
                'samples_per_class':'all','dimensions':x.shape[1],'partition':'train','metric':identity['metric']})
    pd.DataFrame(result).to_csv(path,index=False)
    write_json(meta,{'identity':identity,'csv_sha256':sha256(path),'sample_cap':None,
        'raw':'all train images with their natural class proportions',
        'balanced':'all train images weighted so each class has total weight 0.5 in each hospital',
        'conditional':'each class uses all its train images; target labels only for descriptive analysis',
        'uncertainty':'hospital pairs share patients/domains; no independent-patch significance claim'})
    return path
