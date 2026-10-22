#!/usr/bin/env python3
"""两张单卡worker继续原180主实验；任务锁、原签名/hash校验和单写者最终验收。

不修改正式config或训练器。CUDA_VISIBLE_DEVICES须在导入torch前仅保留一张卡。
软件版本参考JSON使用flat键：python/torch/torchvision/numpy/pandas/scipy/sklearn。
"""
from __future__ import annotations
import argparse
import contextlib
import csv
import datetime
import fcntl
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict,dataclass
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parent


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


class LockBusy(RuntimeError):pass


class FileLock:
    """持有打开的flock直至离开上下文；进程退出由OS释放，不删除锁文件。"""
    def __init__(self,path,*,shared=False,blocking=False):
        self.path=Path(path);self.shared=shared;self.blocking=blocking;self.file=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True);self.file=self.path.open('a+')
        flags=(fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX)|(0 if self.blocking else fcntl.LOCK_NB)
        try:fcntl.flock(self.file.fileno(),flags)
        except BlockingIOError:
            self.file.close();self.file=None;raise LockBusy(f'Already active: {self.path}')
        return self
    def __exit__(self,*args):
        if self.file is not None:fcntl.flock(self.file.fileno(),fcntl.LOCK_UN);self.file.close()


@dataclass(frozen=True)
class Task:
    partition_seed:int
    source:int
    depth:str
    model_seed:int
    @property
    def key(self):return f'split_{self.partition_seed}__source_{self.source}__{self.depth}__seed_{self.model_seed}'
    def folder(self,out):return Path(out)/f'split_{self.partition_seed}'/'runs'/f'source_{self.source}'/self.depth/f'seed_{self.model_seed}'


def ordered_tasks(config):
    sources=list(config['sources']);sources=([0] if 0 in sources else [])+[s for s in sources if s!=0]
    return [Task(split,source,depth,seed) for source in sources for split in config['partition_seeds']
            for seed in config['seeds'] for depth in config['depths']]


def assigned_tasks(config,index,workers=2):
    if workers<1 or not 0<=index<workers:raise ValueError('Invalid worker index/count')
    return [task for ordinal,task in enumerate(ordered_tasks(config)) if ordinal%workers==index]


def validate_plan(config):
    assert set(config['partition_seeds'])=={42,43,44} and set(config['sources'])==set(range(5))
    assert set(config['depths'])=={'head','layer4','layer3_layer4','full'} and set(config['seeds'])=={42,43,44}
    assert len(ordered_tasks(config))==180 and len(set(ordered_tasks(config)))==180,'Plan is not the original 180 tasks'
    assert config['expected_images']==455954 and config['epochs']==20 and config['early_stopping_patience']==5
    assert config.get('per_slide_class') is None and all(config.get(k) is None for k in ['max_train_per_class','max_val_per_class','max_test_per_class'])


def runtime_environment(expected_path=None):
    import torch,torchvision,numpy,pandas,scipy,sklearn
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one visible CUDA GPU required per process')
    versions={'python':platform.python_version(),'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
              'numpy':numpy.__version__,'pandas':pandas.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}
    if expected_path is None:raise ValueError('--expected-environment is required; preserve the original software versions')
    expected=json.loads(Path(expected_path).read_text())
    if not set(versions).issubset(expected):raise ValueError('Expected environment JSON must specify all seven software versions')
    mismatches={key:{'expected':expected[key],'actual':value} for key,value in versions.items() if expected[key]!=value}
    if mismatches:raise RuntimeError(f'Original software environment mismatch: {mismatches}')
    gpu=torch.cuda.get_device_properties(0)
    return {'recorded_at':now(),'hostname':socket.gethostname(),'pid':os.getpid(),'versions':versions,
            'gpu':torch.cuda.get_device_name(0),'gpu_uuid':str(gpu.uuid),'cuda':torch.version.cuda,
            'visible_cuda_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'visible_gpu_count':1,
            'platform':platform.platform(),'core_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in
                     [ROOT/'camelyon/train.py',ROOT/'camelyon/models.py',ROOT/'camelyon/data.py']},
            'heterogeneous_resume_note':'CUDA-to-CUDA resume is supported by the unchanged core signature; different GPUs do not imply bitwise reproducibility'}


class SignatureContext:
    """每split/source只构造一次与训练核心完全相同的输入摘要；不载模型或训练。"""
    def __init__(self,root,config,out=None):self.root=Path(root);self.config=config;self.out=Path(out) if out else None;self.frames={};self.digests={};self.tokens={}
    def signature(self,task):
        import pandas as pd
        import torch
        from camelyon import train
        from full_run import split_config
        conf=split_config(self.config,task.partition_seed)
        manifest=self.root/conf['manifest_path']
        token=(manifest.stat().st_size,manifest.stat().st_mtime_ns)
        if task.partition_seed in self.tokens and self.tokens[task.partition_seed]!=token:raise RuntimeError('Frozen manifest changed during worker execution')
        if task.partition_seed not in self.frames:
            if self.out is not None:
                provenance=json.loads((self.out/f'split_{task.partition_seed}'/'provenance.json').read_text())
                if sha256(manifest)!=provenance['manifest_sha256']:raise ValueError('Runtime manifest differs from persisted original split provenance')
            self.frames[task.partition_seed]=pd.read_csv(manifest,dtype={'sample_id':str});self.tokens[task.partition_seed]=token
        key=(task.partition_seed,task.source)
        if key not in self.digests:
            frame=self.frames[task.partition_seed];sample_seed=conf.get('sampling_seed',42)
            parts=[train._frame_for(frame,'train',task.source,conf.get('max_train_per_class'),sample_seed),
                   train._frame_for(frame,'val',task.source,conf.get('max_val_per_class'),sample_seed),
                   train._frame_for(frame,'test',None,conf.get('max_test_per_class'),sample_seed)]
            self.digests[key]=train._dataset_digest(self.root,manifest,parts,conf.get('image_digest_manifest'))
        spec={'config':conf,'source':str(task.source),'depth':task.depth,'seed':task.model_seed,
              'dataset_digest':self.digests[key],'engine_version':train.ENGINE_VERSION,'torch_version':str(torch.__version__)}
        return hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()


def verified_completed(folder,signature):
    from camelyon import train
    train._check_identity(Path(folder),signature)
    # Verifies ALL original completion artifacts, including best.pt bytes and signature.
    return train._completed_cache(Path(folder),signature)


def validate_inputs(root,conf,expected_images):
    import pandas as pd
    manifest=Path(root)/conf['manifest_path']
    provenance=json.loads((manifest.parent/'provenance.json').read_text())
    if provenance['selected_images']!=expected_images or provenance.get('per_slide_class_cap') is not None or provenance['partition_seed']!=conf['partition_seed']:raise ValueError('Full original cohort provenance differs')
    if provenance['manifest_sha256']!=sha256(manifest):raise ValueError('Full manifest SHA differs from original provenance')
    frame=pd.read_csv(manifest,usecols=['sample_id','path','center','patient','slide','label','partition'])
    if len(frame)!=expected_images or frame.isna().any().any() or frame.sample_id.duplicated().any() or frame.path.duplicated().any():raise ValueError('Full manifest coverage/null/duplicate problem')
    if set(frame.center)!=set(conf['sources']) or set(frame.partition)!={'train','val','test'} or set(frame.label)!={0,1}:raise ValueError('Full manifest domain/label/partition coverage differs')
    if any((frame.groupby(['center',key]).partition.nunique()>1).any() for key in ['patient','slide']):raise ValueError('Patient/slide leakage')
    counts=frame.groupby(['center','partition','label']).size()
    if len(counts)!=30 or (counts<=0).any():raise ValueError('Every source partition must have both labels')


def partition_review_path_mapping(source,target,relative_manifest):
    """Allow only the relocation of this review's absolute runtime manifest path."""
    source,target=Path(source),Path(target)
    relative=Path(relative_manifest)
    if relative.is_absolute() or not relative.parts or '..' in relative.parts:
        raise ValueError('Expected manifest path must be a confined relative path')
    prepared=json.loads(source.read_text());original=json.loads(target.read_text())
    paths=[]
    for document in [original,prepared]:
        value=document.get('manifest_path') if isinstance(document,dict) else None
        if not isinstance(value,str):raise ValueError('Review manifest_path must be an absolute runtime path')
        path=Path(value)
        if not path.is_absolute() or '..' in path.parts or tuple(path.parts[-len(relative.parts):])!=relative.parts:
            raise ValueError('Review runtime path does not end with the exact configured relative manifest path')
        paths.append(value)
    original_fields={key:value for key,value in original.items() if key!='manifest_path'}
    prepared_fields={key:value for key,value in prepared.items() if key!='manifest_path'}
    canonical=lambda value:json.dumps(value,sort_keys=True,allow_nan=False)
    if paths[0]==paths[1] or canonical(original_fields)!=canonical(prepared_fields):
        raise ValueError('Partition review differs beyond runtime manifest_path relocation')
    return {'relative_manifest_path':str(relative),'imported_manifest_path':paths[0],'prepared_manifest_path':paths[1],
            'imported_file_sha256':sha256(target),'prepared_file_sha256':sha256(source),
            'policy':'Only absolute runtime manifest_path normalized; imported review bytes preserved'}


def initialize(root,config,out,workers=2):
    """Only call with coordinator exclusive and phase exclusive; workers never write shared inputs."""
    from full_run import split_config
    validate_plan(config)
    root,out=Path(root).resolve(),Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    for path,value in [(out/'full_plan.json',config),(out/'.ece/execution_plan.json',{'workers':workers,'tasks':[asdict(t) for t in ordered_tasks(config)]})]:
        if path.exists() and json.loads(path.read_text())!=value:raise ValueError(f'Existing plan mismatch: {path}')
        if not path.exists():write_json(path,value)
    for split in config['partition_seeds']:
        conf=split_config(config,split);folder=out/f'split_{split}';folder.mkdir(exist_ok=True)
        path=folder/'config.json'
        if path.exists() and json.loads(path.read_text())!=conf:raise ValueError(f'Existing split config differs: {path}')
        if not path.exists():write_json(path,conf)
        manifest=root/conf['manifest_path']
        if not manifest.is_file():raise FileNotFoundError(f'Prepared original full manifest missing: {manifest}')
        validate_inputs(root,conf,config['expected_images'])
        for name in ['manifest.csv','provenance.json','patient_partitions.csv','partition_review.json']:
            source=manifest.parent/name;target=folder/name
            if not source.is_file():raise FileNotFoundError(source)
            if target.exists():
                if sha256(target)!=sha256(source):
                    if name!='partition_review.json':raise ValueError(f'Imported split input differs: {target}')
                    mapping=partition_review_path_mapping(source,target,conf['manifest_path'])
                    mapping.update(partition_seed=split,recorded_at=now())
                    write_json(out/'.ece/runtime_path_mappings'/f'split_{split}_{mapping["prepared_file_sha256"]}.json',mapping)
            else:
                tmp=target.with_name(target.name+'.ece-import.tmp');shutil.copy2(source,tmp);os.replace(tmp,target)
    write_json(out/'.ece/initialized.json',{'plan_sha256':sha256(out/'full_plan.json'),'workers':workers,'expected_runs':len(ordered_tasks(config))})
    return out


def execute_task(root,config,out,task,context,environment,*,origin='',executor=None):
    from camelyon import train
    from full_run import split_config
    out=Path(out);folder=task.folder(out)
    with FileLock(out/'.ece/task_locks'/f'{task.key}.lock',blocking=True):
        signature=context.signature(task)
        saved=verified_completed(folder,signature)
        if saved is not None:
            action='verified_completed_skip';result=saved
        else:
            existing=(folder/'run.json').exists();last=folder/'last.pt'
            import_record={'recorded_at':now(),'task':asdict(task),'expected_signature':signature,'current_environment':environment,
                           'origin_hint':origin or 'Pre-existing run origin not specified',
                           'preexisting_run':existing,'last_checkpoint_present':last.is_file(),
                           'last_checkpoint_sha256':sha256(last) if last.is_file() else None,
                           'signature_policy':'Original config/dataset/engine/Torch signature preserved; core checks CUDA device type and checkpoint integrity'}
            if (folder/'metrics.json').exists():import_record['previous_metrics_environment']=json.loads((folder/'metrics.json').read_text()).get('environment')
            token=f'{time.time_ns()}_{os.getpid()}'
            write_json(out/'.ece/imports'/task.key/f'{token}.json',import_record)
            result=(executor or train.run_experiment)(Path(root),split_config(config,task.partition_seed),task.source,task.depth,task.model_seed,folder)
            action='resumed' if result.get('resume',{}).get('resumed_from_epoch',0)>0 else 'trained'
            # Do not publish success until the original hash/signature verifier accepts the new artifacts.
            checked=verified_completed(folder,signature)
            if checked is None:raise RuntimeError('Executor returned without original completion proof')
            result=checked
        receipt={'verified_at':now(),'task':asdict(task),'action':action,'signature':signature,
                 'completion_sha256':sha256(folder/'completion.json'),'worker_environment':environment,
                 'resumed_from_epoch':result.get('resume',{}).get('resumed_from_epoch',0)}
        write_json(out/'.ece/verified_tasks'/f'{task.key}.json',receipt)
        return receipt


def worker(root,config,out,index,workers=2,*,expected_environment=None,origin=''):
    out=Path(out).resolve();root=Path(root).resolve();validate_plan(config)
    initialized=json.loads((out/'.ece/initialized.json').read_text())
    if initialized!={'plan_sha256':sha256(out/'full_plan.json'),'workers':workers,'expected_runs':len(ordered_tasks(config))}:raise ValueError('Initialization is incomplete or a different plan was installed')
    execution=json.loads((out/'.ece/execution_plan.json').read_text())
    if execution!={'workers':workers,'tasks':[asdict(t) for t in ordered_tasks(config)]}:raise ValueError('Initialize the exact execution plan before workers')
    tasks=assigned_tasks(config,index,workers)
    state_path=out/'.ece/workers'/f'worker_{index}.json'
    with FileLock(out/'.ece/worker_locks'/f'worker_{index}.lock'),FileLock(out/'.ece/phase.lock',shared=True):
        environment=runtime_environment(expected_environment)
        write_json(out/'.ece/environments'/f'worker_{index}_{time.time_ns()}.json',environment)
        state={'worker_index':index,'workers':workers,'hostname':socket.gethostname(),'pid':os.getpid(),
               'started_at':now(),'assigned_tasks':len(tasks),'finished_tasks':0,'status':'running','environment':environment}
        context=SignatureContext(root,config,out=out)
        try:
            for task in tasks:
                state.update(current_task=asdict(task),updated_at=now());write_json(state_path,state)
                print(f'[worker {index}] {task.key}',flush=True)
                receipt=execute_task(root,config,out,task,context,environment,origin=origin)
                state.update(finished_tasks=state['finished_tasks']+1,last_action=receipt['action'],updated_at=now());write_json(state_path,state)
            state.update(status='complete',current_task=None,updated_at=now());write_json(state_path,state)
            return state
        except BaseException as error:
            state.update(status='interrupted',error=f'{type(error).__name__}: {error}',updated_at=now());write_json(state_path,state)
            raise


def finalize(root,config,out,*,expected_environment=None):
    """Caller holds coordinator/phase EX locks; verify all runs before any analysis/report writes."""
    out=Path(out);validate_plan(config);environment=runtime_environment(expected_environment)
    context=SignatureContext(root,config,out=out);count=0
    for task in ordered_tasks(config):
        if verified_completed(task.folder(out),context.signature(task)) is None:raise RuntimeError(f'Incomplete original run: {task.key}')
        count+=1
    write_json(out/'.ece/finalization.json',{'stage':'analysis','verified_original_runs':count,'environment':environment,'at':now()})
    from full_run import split_config,summarize,verify,status
    from full_analysis import extract_full_reference,compute_full_shift
    first=split_config(config,config['partition_seeds'][0])
    features=extract_full_reference(Path(root),first,out/'reference_cache')
    for split in config['partition_seeds']:
        conf=split_config(config,split)
        compute_full_shift(*features,Path(root)/conf['manifest_path'],out/f'split_{split}'/'analysis')
    summarize(out,config)
    result=verify(Path(root),config,out)
    status(out,config,'complete')
    write_json(out/'.ece/finalization.json',{'stage':'complete','verified_original_runs':count,'environment':environment,'at':now(),
             'report_complete':result['complete'],'heterogeneous_gpu_execution_recorded_separately':True})
    return result


def observe(out,config):
    out=Path(out);states=[]
    for path in sorted((out/'.ece/workers').glob('worker_*.json')):
        item=json.loads(path.read_text());lock=out/'.ece/worker_locks'/f'worker_{item["worker_index"]}.lock'
        try:
            with FileLock(lock):item['worker_lock_held']=False
        except LockBusy:item['worker_lock_held']=True
        task=item.get('current_task')
        if task:
            history=Task(**task).folder(out)/'history.csv'
            if history.exists():
                with history.open() as f:rows=list(csv.DictReader(f))
                item['last_saved_history_row']=rows[-1] if rows else None
        states.append(item)
    tasks=ordered_tasks(config)
    return {'expected_runs':len(tasks),'completion_records_present':sum((t.folder(out)/'completion.json').exists() for t in tasks),
            'worker_verified_receipts_present':sum((out/'.ece/verified_tasks'/f'{t.key}.json').exists() for t in tasks),
            'verification_note':'Counts are presence/previous worker receipts, not a fresh all-checkpoint audit; finalize performs that audit',
            'workers':states,'finalization':json.loads((out/'.ece/finalization.json').read_text()) if (out/'.ece/finalization.json').exists() else None}


class SupervisorStopped(RuntimeError):pass


def stop_worker_groups(processes,grace_seconds=30):
    """Each worker starts its own session; terminate its DataLoader descendants too."""
    def send(process,sig):
        try:os.killpg(process.pid,sig);return True
        except ProcessLookupError:return False
    for process in processes:send(process,signal.SIGTERM)
    deadline=time.monotonic()+grace_seconds
    while True:
        for process in processes:process.poll()  # Reap exited leaders before probing their groups.
        alive=[process for process in processes if send(process,0)]
        if not alive or time.monotonic()>=deadline:break
        time.sleep(.1)
    for process in alive:send(process,signal.SIGKILL)
    for process in processes:process.wait(timeout=10)


def supervise(root,config,out,gpus,*,config_path,expected_environment,origin=''):
    if len(gpus)!=2 or len(set(gpus))!=2:raise ValueError('Exactly two distinct GPU UUIDs/indices required')
    out=Path(out);processes=[];control={'signal':None,'phase':'starting'}
    with FileLock(out/'.ece/coordinator.lock'):
        state_path=out/'.ece/supervisor.json'
        state={'pid':os.getpid(),'hostname':socket.gethostname(),'gpus':list(gpus),'started_at':now(),'status':'starting','workers':[]}
        def publish(status,**fields):
            state.update(status=status,updated_at=now(),**fields);write_json(state_path,state)
        def handle_signal(signum,frame):
            control['signal']=signum
            # During Popen only set a flag, so the child is always registered before cleanup.
            if control['phase']=='finalizing':raise SupervisorStopped(f'Received {signal.Signals(signum).name}')
        def check_stop():
            if control['signal'] is not None:raise SupervisorStopped(f'Received {signal.Signals(control["signal"]).name}')
        old_handlers={sig:signal.getsignal(sig) for sig in [signal.SIGINT,signal.SIGTERM,signal.SIGHUP]}
        try:
            for sig in old_handlers:signal.signal(sig,handle_signal)
            publish('initializing')
            with FileLock(out/'.ece/phase.lock'):initialize(root,config,out,2)
            check_stop()
            for index,gpu in enumerate(gpus):
                check_stop()
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu
                command=[sys.executable,str(Path(__file__).resolve()),'worker','--root',str(root),'--config',str(config_path),
                         '--output',str(out),'--index',str(index),'--workers','2','--expected-environment',str(expected_environment),'--origin',origin]
                processes.append(subprocess.Popen(command,env=env,start_new_session=True))
                state['workers'].append({'index':index,'gpu':gpu,'pid':processes[-1].pid})
                publish('running')
            while True:
                check_stop();codes=[process.poll() for process in processes]
                if any(code is not None and code!=0 for code in codes):
                    raise RuntimeError(f'Workers exited {codes}; preserve task checkpoints and restart matching supervisor')
                if all(code is not None for code in codes):break
                time.sleep(.2)
            check_stop();control['phase']='finalizing';publish('finalizing',exit_codes=codes)
            os.environ['CUDA_VISIBLE_DEVICES']=gpus[0]
            with FileLock(out/'.ece/phase.lock'):result=finalize(root,config,out,expected_environment=expected_environment)
            control['phase']='finished';check_stop();publish('complete');return result
        except BaseException as error:
            control['phase']='cleanup'
            publish('stopping',error=f'{type(error).__name__}: {error}',received_signal=control['signal'])
            try:stop_worker_groups(processes)
            except BaseException as cleanup_error:
                publish('cleanup_failed',cleanup_error=f'{type(cleanup_error).__name__}: {cleanup_error}')
                raise cleanup_error from error
            publish('interrupted',exit_codes=[process.poll() for process in processes])
            raise
        finally:
            for sig,handler in old_handlers.items():signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['init','worker','supervise','finalize','observe'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--config',type=Path,default=ROOT/'configs/full.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--index',type=int,default=0);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--gpus',nargs=2);p.add_argument('--gpu');p.add_argument('--expected-environment',type=Path);p.add_argument('--origin',default='')
    args=p.parse_args();root,out=args.root.resolve(),args.output.resolve();conf=json.loads(args.config.read_text());validate_plan(conf)
    if args.gpu:
        if ',' in args.gpu:p.error('--gpu must select exactly one GPU')
        os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    if args.stage in ['worker','supervise','finalize'] and args.expected_environment is None:p.error('--expected-environment is required')
    if args.stage=='observe':result=observe(out,conf)
    elif args.stage=='init':
        with FileLock(out/'.ece/coordinator.lock'),FileLock(out/'.ece/phase.lock'):initialize(root,conf,out,args.workers)
        result={'initialized':True,'expected_runs':len(ordered_tasks(conf))}
    elif args.stage=='worker':
        log=out/'.ece/logs'/f'worker_{args.index}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a',buffering=1) as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
            try:result=worker(root,conf,out,args.index,args.workers,expected_environment=args.expected_environment,origin=args.origin)
            except BaseException:traceback.print_exc();raise
    elif args.stage=='supervise':
        if not args.gpus:p.error('supervise requires --gpus GPU0 GPU1')
        result=supervise(root,conf,out,args.gpus,config_path=args.config.resolve(),expected_environment=args.expected_environment.resolve(),origin=args.origin)
    else:
        with FileLock(out/'.ece/coordinator.lock'),FileLock(out/'.ece/phase.lock'):result=finalize(root,conf,out,expected_environment=args.expected_environment)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
