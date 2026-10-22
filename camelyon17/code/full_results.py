#!/usr/bin/env python3
"""汇总全量、重复患者划分的主实验；仅读取已有结果，不训练、不含域适应。

API: generate(results, split_seeds=(42,43,44), sources=(0,1,2,3,4),
              depths=DEPTHS, model_seeds=(42,43,44), expected_images=455954,
              output_dir=None) -> JSON-compatible status dict.
统计单位明确分开：同一患者划分内的模型种子，以及患者划分之间的变化。
没有基于 patch 数量的置信区间或独立性显著性检验。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

DEPTHS = ('head', 'layer4', 'layer3_layer4', 'full')
SCORES = ('roc_auc', 'accuracy', 'loss', 'balanced_accuracy', 'sensitivity', 'specificity')
SHIFTS = ('shift_raw', 'shift_balanced', 'shift_classconditional')


def _csv(path, **kwargs):
    return pd.read_csv(path, float_precision='round_trip', **kwargs)


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def _score(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=float)
    assert len(y) and set(np.unique(y)).issubset({0,1}), 'Empty/nonbinary labels'
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all(), 'Invalid probability'
    pos, neg = int((y==1).sum()), int((y==0).sum())
    sensitivity = float(((p>=.5)&(y==1)).sum()/pos) if pos else np.nan
    specificity = float(((p<.5)&(y==0)).sum()/neg) if neg else np.nan
    return {'n':len(y), 'positive_n':pos, 'negative_n':neg,
            'majority_accuracy_baseline':max(pos,neg)/len(y),
            'accuracy':float(((p>=.5)==y).mean()),
            'roc_auc':float(roc_auc_score(y,p)) if pos and neg else np.nan,
            'sensitivity':sensitivity, 'specificity':specificity,
            'balanced_accuracy':(sensitivity+specificity)/2}


def _same(a,b):
    if pd.isna(a) and (b is None or pd.isna(b)):return
    assert b is not None and np.isclose(float(a),float(b),atol=1e-6,rtol=0), f'Metric mismatch {a} vs {b}'


def _summarize(frame, keys, scores, split_count, seed_count):
    """每划分先平均模型种子，再等权平均划分；不把9次运行当9位患者。"""
    within_rows=[]
    for key, group in frame.groupby(['split_seed']+list(keys),dropna=False,sort=True):
        key = key if isinstance(key,tuple) else (key,)
        row=dict(zip(['split_seed']+list(keys),key))
        row['model_seed_count']=int(group['model_seed'].nunique())
        row['model_seed_block_complete']=row['model_seed_count']==seed_count
        for score in scores:
            row[score+'_mean']=group[score].mean()
            row[score+'_model_seed_std']=group[score].std(ddof=1)
            row[score+'_model_seed_variance']=group[score].var(ddof=1)
        within_rows.append(row)
    within=pd.DataFrame(within_rows)
    across_rows=[]
    if within.empty:return within,pd.DataFrame()
    for key,group in within.loc[within.model_seed_block_complete].groupby(list(keys),dropna=False,sort=True):
        key=key if isinstance(key,tuple) else (key,)
        row=dict(zip(keys,key))
        row['patient_split_count']=int(group.split_seed.nunique())
        row['all_patient_splits_complete']=row['patient_split_count']==split_count
        for score in scores:
            row[score+'_mean']=group[score+'_mean'].mean()
            row[score+'_between_patient_split_std']=group[score+'_mean'].std(ddof=1)
            row[score+'_between_patient_split_variance']=group[score+'_mean'].var(ddof=1)
            row[score+'_mean_within_split_model_seed_std']=group[score+'_model_seed_std'].mean()
            row[score+'_mean_within_split_model_seed_variance']=group[score+'_model_seed_variance'].mean()
        across_rows.append(row)
    return within,pd.DataFrame(across_rows)


def _table(frame):
    return '<p>尚无完整比较块。</p>' if frame.empty else frame.to_html(index=False,float_format=lambda x:f'{x:.5f}',border=0,na_rep='—',escape=True)


def _plot_matrices(out, summary, sources, depths):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if summary.empty:return []
    images=[]
    for score in ['roc_auc','loss','balanced_accuracy']:
        fig,axes=plt.subplots(1,len(depths),figsize=(4.1*len(depths),4),squeeze=False)
        for ax,depth in zip(axes.flat,depths):
            selected=summary.loc[summary.depth.eq(depth)]
            matrix=selected.pivot(index='source',columns='target',values=score+'_mean').reindex(index=sources,columns=sources)
            im=ax.imshow(matrix.to_numpy(dtype=float),cmap='viridis')
            ax.set_title(depth);ax.set_xlabel('Target hospital');ax.set_ylabel('Source hospital')
            ax.set_xticks(range(len(sources)),sources);ax.set_yticks(range(len(sources)),sources)
            for i in range(len(sources)):
                for j in range(len(sources)):
                    value=matrix.iloc[i,j]
                    ax.text(j,i,f'{value:.3f}' if pd.notna(value) else 'missing',ha='center',va='center',fontsize=8,color='white',bbox={'facecolor':'black','alpha':.25,'pad':1,'edgecolor':'none'})
            fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle(score+': equal-weight patient-split means; model seeds averaged within split')
        fig.tight_layout();name=f'full_{score}_matrices.png';fig.savefig(out/name,dpi=150);plt.close(fig);images.append(name)
    return images


def generate(results, *, split_seeds=(42,43,44), sources=(0,1,2,3,4), depths=DEPTHS,
             model_seeds=(42,43,44), expected_images=455954, output_dir=None):
    results=Path(results).resolve();out=Path(output_dir).resolve() if output_dir else results
    out.mkdir(parents=True,exist_ok=True)
    split_seeds,sources,depths,model_seeds=map(tuple,(split_seeds,sources,depths,model_seeds))
    expected=list(itertools.product(split_seeds,sources,depths,model_seeds))
    # Remove only this generator's derived outputs before rebuilding a partial snapshot;
    # otherwise a later incomplete run could accidentally link an old complete CSV.
    managed={'full_metrics_all.csv','full_within_patient_split_summary.csv','full_across_patient_split_summary.csv',
             'full_depth_minus_head_per_run.csv','full_depth_minus_head_within_split.csv','full_depth_minus_head_across_splits.csv',
             'full_observed_winners.csv','full_winner_frequency_by_pair.csv','full_winner_frequency_by_target.csv',
             'full_loss_curves_by_split_epoch.csv','full_representation_shift_all.csv','full_shift_performance_observations.csv',
             'full_shift_associations.csv','full_roc_auc_matrices.png','full_loss_matrices.png','full_balanced_accuracy_matrices.png'}
    managed.update(f'full_{score}_{depth}_{stat}_matrix.csv' for score in SCORES for depth in depths for stat in ['mean','between_patient_split_std'])
    managed.update(f'full_{field}_{stat}_matrix.csv' for field in SHIFTS for stat in ['mean','std'])
    managed.update(f'full_{field}_split_{split}_matrix.csv' for field in SHIFTS for split in split_seeds)
    for name in managed:
        if (out/name).is_file():(out/name).unlink()
    assert all(len(values)==len(set(values)) and len(values)>0 for values in [split_seeds,sources,depths,model_seeds])
    issues=[];inventory=[];rows=[];histories=[];records=[];shift_frames=[];coverage=[]
    references={};initializations={};inputs={};test_baselines={};configs={}
    expected_paths={results/f'split_{s}'/'runs'/f'source_{c}'/d/f'seed_{z}'/'metrics.json' for s,c,d,z in expected}
    unexpected=set(results.glob('split_*/runs/source_*/*/seed_*/metrics.json'))-expected_paths
    issues.extend({'path':str(p),'error':'Unexpected run outside prespecified matrix'} for p in sorted(unexpected))
    for split in split_seeds:
        folder=results/f'split_{split}'
        try:
            config=json.loads((folder/'config.json').read_text())
            assert set(config['sources'])==set(sources) and set(config['depths'])==set(depths) and set(config['seeds'])==set(model_seeds),'Config matrix differs from prespecified plan'
            assert config['epochs']==20 and config['early_stopping_patience']==5,'Expected max epochs20/patience5'
            actual_split=config.get('partition_seed',config.get('patient_split_seed',config.get('split_seed',config.get('sampling_seed'))))
            assert actual_split==split,'Patient split seed differs from folder'
            assert config.get('per_slide_class') is None and all(config.get(k) is None for k in ['max_train_per_class','max_val_per_class','max_test_per_class']),'Sampling cap in full-data config'
            configs[split]=config
        except Exception as error:issues.append({'path':str(folder/'config.json'),'error':str(error)})
        try:
            shift=_csv(folder/'analysis/representation_shift.csv')
            assert set(SHIFTS).issubset(shift.columns) and len(shift)==len(sources)*(len(sources)-1)
            assert set(zip(shift.source,shift.target))=={(s,t) for s in sources for t in sources if s!=t}
            assert not shift.duplicated(['source','target']).any()
            for field in SHIFTS:
                assert np.isfinite(shift[field]).all() and (shift[field]>=0).all()
                matrix=shift.pivot(index='source',columns='target',values=field).reindex(index=sources,columns=sources)
                for center in sources:matrix.loc[center,center]=0.
                np.testing.assert_allclose(matrix.to_numpy(),matrix.to_numpy().T,atol=1e-12,rtol=0)
                matrix.to_csv(out/f'full_{field}_split_{split}_matrix.csv')
            shift['split_seed']=split;shift_frames.append(shift)
        except Exception as error:issues.append({'path':str(folder/'analysis/representation_shift.csv'),'error':str(error)})
    for split,source,depth,seed in expected:
        path=results/f'split_{split}'/'runs'/f'source_{source}'/depth/f'seed_{seed}'
        item={'split_seed':split,'source':source,'depth':depth,'model_seed':seed,'path':str(path),'status':'missing'}
        try:
            metrics=json.loads((path/'metrics.json').read_text())
            assert metrics['status']=='complete' and metrics['reload_verified'] is True,'Run incomplete or reload not verified'
            assert metrics['config']==configs.get(split),'Run config mismatch'
            assert str(metrics['source'])==str(source) and metrics['depth']==depth and metrics['seed']==seed,'Wrong run identity'
            selection=metrics['selection']
            assert selection['split']=='source_val' and str(selection['center'])==str(source) and selection['target_test_used_for_selection'] is False,'Selection leakage/role mismatch'
            history=_csv(path/'history.csv')
            assert len(history)>0 and history.epoch.tolist()==list(range(1,len(history)+1)) and len(history)<=20,'Invalid epoch history'
            assert np.isfinite(history[['train_loss','val_loss']]).all().all(),'Nonfinite loss history'
            best=history.iloc[int(history.val_loss.argmin())]
            assert int(best.epoch)==selection['best_epoch'] and selection['epochs_completed']==len(history),'Checkpoint selection differs from first minimum source-val loss'
            _same(float(best.val_loss),selection['best_val_loss']);_same(float(best.val_loss),metrics['reloaded_validation']['loss'])
            if len(history)<20:
                assert len(history)-int(best.epoch)>=5,'Early stop before patience5 exhausted'
            pred=_csv(path/'predictions.csv',dtype={'sample_id':str,'center':str,'partition':str})
            assert len(pred)==metrics['test_n'] and not pred.sample_id.duplicated().any(),'Prediction count/duplicate IDs'
            assert pred.sample_id.tolist()==sorted(pred.sample_id),'Unstable prediction order'
            assert pred.partition.eq('test').all() and pred.source.astype(str).eq(str(source)).all(),'Wrong prediction roles'
            assert set(pred.center)==set(map(str,sources)),'Missing target hospital'
            digest=hashlib.sha256(pred[['sample_id','center','label','partition']].to_csv(index=False).encode()).hexdigest()
            assert metrics['sampling']['test_sample_ids_digest']==hashlib.sha256('\n'.join(pred.sample_id).encode()).hexdigest(),'Test IDs differ from saved input digest'
            if split in references:assert references[split]==digest,'Test members/labels differ across sources/depths/seeds'
            init_key=seed
            if init_key in initializations:assert initializations[init_key]==metrics['initial_state_digest'],'Initial model differs at fixed model seed'
            input_key=(split,source)
            identity=tuple(metrics['sampling'][f'{p}_sample_ids_digest'] for p in ['train','val','test'])+(metrics['dataset_digest'],metrics['train_n'],metrics['val_n'],metrics['test_n'])
            if input_key in inputs:assert inputs[input_key]==identity,'Input sampling differs across model seeds/depths'
            run_rows=[];run_baselines={}
            assert set(metrics['test_by_center'])==set(map(str,sources)),'Wrong per-center metric coverage'
            for center,group in pred.groupby('center',sort=True):
                target=int(center);computed=_score(group.label.to_numpy(),group.probability.to_numpy())
                recorded=metrics['test_by_center'][center]
                for field in ['n','accuracy','roc_auc']:_same(computed[field],recorded[field])
                assert np.isfinite(recorded['loss']) and recorded['loss']>=0,'Invalid stored cross-entropy'
                assert recorded['role']==('ID' if target==source else 'OOD'),'Wrong ID/OOD role'
                baseline={k:computed[k] for k in ['n','positive_n','negative_n','majority_accuracy_baseline']}
                key=(split,target)
                if key in test_baselines:assert test_baselines[key]==baseline,'Class counts differ between runs'
                run_baselines[key]=baseline
                run_rows.append({'split_seed':split,'source':source,'depth':depth,'model_seed':seed,
                                 'target':target,'role':recorded['role'],**computed,'loss':recorded['loss'],
                                 'best_epoch':selection['best_epoch'],'epochs_completed':len(history)})
            references[split]=digest;initializations[init_key]=metrics['initial_state_digest'];inputs[input_key]=identity;test_baselines.update(run_baselines)
            rows.extend(run_rows);records.append((split,source,metrics))
            histories.append(history.assign(split_seed=split,source=source,depth=depth,model_seed=seed))
            item.update(status='complete',best_epoch=selection['best_epoch'],epochs_completed=len(history),train_n=metrics['train_n'],val_n=metrics['val_n'],test_n=metrics['test_n'])
        except FileNotFoundError as error:item['error']=str(error)
        except Exception as error:
            item.update(status='invalid',error=f'{type(error).__name__}: {error}')
            issues.append({'path':str(path),'error':item['error']})
        inventory.append(item)
    inv=pd.DataFrame(inventory);inv.to_csv(out/'full_run_inventory.csv',index=False)
    for split in split_seeds:
        c={'split_seed':split,'expected_images':expected_images,'counted_images':None,'provenance_images':None,'verified':False}
        try:
            representatives={source:m for s,source,m in records if s==split}
            assert set(representatives)==set(sources),'Not all source domains have a valid completed run'
            count=sum(m['train_n']+m['val_n']+m['test_by_center'][str(source)]['n'] for source,m in representatives.items())
            c['counted_images']=count
            provenance=json.loads((results/f'split_{split}'/'provenance.json').read_text())
            c['provenance_images']=provenance['selected_images']
            assert count==expected_images==provenance['selected_images'],'Dataset coverage count differs from full cohort'
            assert all(provenance.get(k) is None for k in ['per_slide_class','per_slide_class_cap','fixture_per_center']),'Capped/fixture provenance'
            c['verified']=True
        except Exception as error:c['error']=str(error)
        coverage.append(c)
    _json(out/'full_dataset_coverage.json',coverage)
    baselines=pd.DataFrame([{'split_seed':split,'target':target,**values} for (split,target),values in sorted(test_baselines.items())])
    baselines.to_csv(out/'full_test_class_baselines.csv',index=False)
    data=pd.DataFrame(rows)
    within=pd.DataFrame();across=pd.DataFrame();differences=pd.DataFrame();delta_within=pd.DataFrame();delta_across=pd.DataFrame();winners=pd.DataFrame();frequency=pd.DataFrame();target_frequency=pd.DataFrame();relationships=pd.DataFrame();images=[]
    if not data.empty:
        data.to_csv(out/'full_metrics_all.csv',index=False)
        within,across=_summarize(data,['source','depth','target','role'],SCORES,len(split_seeds),len(model_seeds))
        within.to_csv(out/'full_within_patient_split_summary.csv',index=False)
        across.to_csv(out/'full_across_patient_split_summary.csv',index=False)
        if not across.empty:
            for depth,group in across.groupby('depth'):
                for score in SCORES:
                    for stat in ['mean','between_patient_split_std']:
                        group.pivot(index='source',columns='target',values=score+'_'+stat).reindex(index=sources,columns=sources).to_csv(out/f'full_{score}_{depth}_{stat}_matrix.csv')
            images=_plot_matrices(out,across,sources,depths)
        head=data.loc[data.depth.eq('head')]
        other=data.loc[~data.depth.eq('head')]
        paired=other.merge(head,on=['split_seed','source','target','model_seed'],suffixes=('','_head'),validate='many_to_one')
        if not paired.empty:
            differences=paired[['split_seed','source','target','depth','model_seed','role']].copy()
            for score in SCORES:differences[score+'_delta']=paired[score]-paired[score+'_head']
            differences.to_csv(out/'full_depth_minus_head_per_run.csv',index=False)
            delta_within,delta_across=_summarize(differences,['source','depth','target','role'],[s+'_delta' for s in SCORES],len(split_seeds),len(model_seeds))
            delta_within.to_csv(out/'full_depth_minus_head_within_split.csv',index=False)
            delta_across.to_csv(out/'full_depth_minus_head_across_splits.csv',index=False)
        win_rows=[]
        for key,group in within.loc[within.model_seed_block_complete].groupby(['split_seed','source','target']):
            if set(group.depth)!=set(depths):continue
            for metric in SCORES:
                values=group.set_index('depth')[metric+'_mean'].reindex(depths)
                if values.isna().any():continue
                best=values.min() if metric=='loss' else values.max()
                selected=values.index[np.isclose(values,best,atol=1e-12,rtol=0)].tolist()
                for depth in selected:win_rows.append({'split_seed':key[0],'source':key[1],'target':key[2],'metric':metric,'depth':depth,'winner_weight':1/len(selected),'tied_depths':len(selected)})
        winners=pd.DataFrame(win_rows)
        if not winners.empty:
            winners.to_csv(out/'full_observed_winners.csv',index=False)
            totals=winners.groupby(['source','target','metric']).winner_weight.sum().rename('complete_split_comparisons')
            frequency=winners.groupby(['source','target','metric','depth']).winner_weight.sum().rename('weighted_wins').reset_index().merge(totals.reset_index())
            grid=pd.DataFrame([{'source':s,'target':t,'metric':m,'depth':d} for s,t,m in totals.index for d in depths])
            frequency=grid.merge(frequency,on=['source','target','metric','depth'],how='left').drop(columns='complete_split_comparisons').merge(totals.reset_index())
            frequency['weighted_wins']=frequency.weighted_wins.fillna(0.)
            frequency['observed_frequency']=frequency.weighted_wins/frequency.complete_split_comparisons
            frequency.to_csv(out/'full_winner_frequency_by_pair.csv',index=False)
            ood=winners.loc[winners.source.ne(winners.target)]
            if not ood.empty:
                totals=ood.groupby(['target','metric']).winner_weight.sum().rename('complete_source_split_comparisons')
                target_frequency=ood.groupby(['target','metric','depth']).winner_weight.sum().rename('weighted_wins').reset_index().merge(totals.reset_index())
                grid=pd.DataFrame([{'target':t,'metric':m,'depth':d} for t,m in totals.index for d in depths])
                target_frequency=grid.merge(target_frequency,on=['target','metric','depth'],how='left').drop(columns='complete_source_split_comparisons').merge(totals.reset_index())
                target_frequency['weighted_wins']=target_frequency.weighted_wins.fillna(0.)
                target_frequency['observed_frequency']=target_frequency.weighted_wins/target_frequency.complete_source_split_comparisons
                target_frequency.to_csv(out/'full_winner_frequency_by_target.csv',index=False)
    if histories:
        history=pd.concat(histories,ignore_index=True)
        curve=history.groupby(['split_seed','source','depth','epoch'])[['train_loss','val_loss']].agg(['mean','std','count'])
        curve.columns=['_'.join(c) for c in curve.columns];curve.reset_index().to_csv(out/'full_loss_curves_by_split_epoch.csv',index=False)
    shifts=pd.concat(shift_frames,ignore_index=True) if shift_frames else pd.DataFrame()
    if not shifts.empty:
        shifts.to_csv(out/'full_representation_shift_all.csv',index=False)
        for field in SHIFTS:
            for stat in ['mean','std']:
                group=shifts.groupby(['source','target'])[field].agg(stat).reset_index()
                matrix=group.pivot(index='source',columns='target',values=field).reindex(index=sources,columns=sources)
                for center in sources:matrix.loc[center,center]=0.
                matrix.to_csv(out/f'full_{field}_{stat}_matrix.csv')
        observations=[]
        if not within.empty:
            complete=within.loc[within.model_seed_block_complete & within.source.ne(within.target)]
            for depth,group in complete.groupby('depth'):
                for score in ['roc_auc','accuracy','loss','balanced_accuracy']:
                    observations.append(group[['split_seed','source','target',score+'_mean']].rename(columns={score+'_mean':'outcome'}).assign(outcome_name=f'{depth}:{score}'))
        if not delta_within.empty:
            for depth,group in delta_within.loc[delta_within.model_seed_block_complete & delta_within.source.ne(delta_within.target)].groupby('depth'):
                observations.append(group[['split_seed','source','target','roc_auc_delta_mean']].rename(columns={'roc_auc_delta_mean':'outcome'}).assign(outcome_name=f'{depth}-head:roc_auc'))
        if not winners.empty:
            ordinal=winners.loc[winners.metric.eq('roc_auc') & winners.source.ne(winners.target)].copy()
            ordinal['outcome']=ordinal.depth.map({depth:i for i,depth in enumerate(depths)})*ordinal.winner_weight
            ordinal=ordinal.groupby(['split_seed','source','target']).outcome.sum().reset_index().assign(outcome_name='observed_winner_depth:roc_auc')
            observations.append(ordinal)
        rel=[]
        if observations:
            joined=pd.concat(observations,ignore_index=True).merge(shifts[['split_seed','source','target',*SHIFTS]],on=['split_seed','source','target'],validate='many_to_one')
            joined.to_csv(out/'full_shift_performance_observations.csv',index=False)
            for outcome,group in joined.groupby('outcome_name'):
                scopes=[(f'patient_split_{split}',g) for split,g in group.groupby('split_seed')]
                # Do not pool 60 related pairs and pretend they are independent observations.
                means=group.groupby(['source','target'])[['outcome',*SHIFTS]].mean().reset_index()
                counts=group.groupby(['source','target']).split_seed.nunique()
                valid_pairs=set(counts[counts.eq(len(split_seeds))].index)
                means=means.loc[[tuple(pair) in valid_pairs for pair in means[['source','target']].itertuples(index=False,name=None)]]
                scopes.append(('pair_mean_across_all_patient_splits',means))
                for scope,values in scopes:
                    for field in SHIFTS:
                        valid=values[[field,'outcome']].dropna()
                        rho=float(spearmanr(valid[field],valid.outcome).statistic) if len(valid)>=3 and valid[field].nunique()>1 and valid.outcome.nunique()>1 else np.nan
                        rel.append({'scope':scope,'outcome':outcome,'shift':field,'directed_hospital_pairs':len(valid),'spearman_rho':rho})
            relationships=pd.DataFrame(rel);relationships.to_csv(out/'full_shift_associations.csv',index=False)
    n_complete=int(inv.status.eq('complete').sum());n_invalid=int(inv.status.eq('invalid').sum())
    complete=(n_complete==len(expected) and not issues and all(c['verified'] for c in coverage))
    summary={'status':'complete' if complete else 'partial','complete':complete,'expected_runs':len(expected),'complete_runs':n_complete,'invalid_runs':n_invalid,
             'missing_runs':int(inv.status.eq('missing').sum()),'patient_split_seeds':list(split_seeds),'model_seeds':list(model_seeds),
             'expected_images_per_split':expected_images,'dataset_coverage':coverage,'issues':issues,'hospital_metric_rows':len(data),
             'winner_frequency_scope':'post-hoc target-test observed winners of within-split model-seed means; fractional tie credit; not a deployment selection rule',
             'variability_scope':'sample SD across model seeds within each fixed patient split; sample SD of split means across patient splits; no confidence intervals',
             'dependence_warning':'Patient splits reuse the same patients; ordered hospital pairs share hospitals; patches are not independent patients; no p-values or independent-patch CI',
             'loss_scope':'stored logit-based cross-entropy retained; not reconstructed from potentially saturated exported probabilities',
             'distance_scope':'fixed ImageNet avgpool; mean of marginal 1D W1, not joint 2048D Wasserstein/W2; raw label-free, balanced/classconditional use labels descriptively',
             'coral_included':False,'checkpoint_bytes_read':False}
    _json(out/'full_completion.json',summary)
    status='完整' if complete else '未完整（partial）'
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>全量主项目稳定性结果</title><style>body{{max-width:1250px;margin:auto;padding:28px;font:16px/1.65 system-ui,-apple-system,"PingFang SC",sans-serif;color:#20354b;background:#f4f7fb}}section{{background:white;padding:24px;margin:22px 0;border:1px solid #dce5ef;border-radius:12px}}table{{border-collapse:collapse;display:block;overflow:auto;font-size:13px}}td,th{{padding:7px 9px;border-bottom:1px solid #e2e8f0;text-align:right;white-space:nowrap}}img{{max-width:100%}}a{{color:#175bad}}.warn{{color:#9b4715}}</style><h1>全量主项目：重复患者划分与模型种子的稳定性</h1><section><h2>{status}：{n_complete} / {len(expected)} 组有效完成</h2><p>预定 {len(split_seeds)} 个患者划分 × {len(model_seeds)} 个模型随机种子 × {len(sources)} 家来源医院 × {len(depths)} 档微调。每组最多20 epochs，来源医院验证集选最小损失模型，耐心值5；目标医院不参与模型训练或选点；其train分区仅用于固定参考距离分析。每个划分预定使用全部 {expected_images:,} 张图像，无每类/切片抽样上限。</p><p>缺失 {summary['missing_runs']} 组，无效 {n_invalid} 组。患者划分覆盖记录：{sum(c['verified'] for c in coverage)} / {len(split_seeds)} 已核对。partial 页面只汇总已保存的有效实验，不能代替完整稳定性结论。模型文件仍位于各运行目录的 best.pt；本汇总不读取权重文件，也不冒称完成独立图像字节/训练成员清单审查。</p><p><a href="full_completion.json">完成情况与问题</a> · <a href="full_run_inventory.csv">全部预定实验清单</a> · <a href="full_dataset_coverage.json">全量覆盖计数与provenance</a></p></section><section><h2>类别比例与多数类基线</h2><p>保留该导出数据及患者划分的原有类别构成，不代表临床患病率。Accuracy 必须对照“永远预测本测试集多数类”的基线；balanced accuracy 为敏感度与特异度的平均。敏感度衡量阳性识别，特异度衡量阴性识别，阈值均固定0.5。AUC 衡量排序，交叉熵 loss 同时受置信程度影响；一个指标领先不意味着全面领先。</p>{_table(baselines)}<p><a href="full_test_class_baselines.csv">下载每医院/患者划分类别组成</a></p></section><section><h2>医院矩阵与两种波动</h2><p>先在每个固定患者划分内平均 {len(model_seeds)} 个模型种子，再对模型种子齐全的患者划分等权平均。各单元实际划分数见 patient_split_count；partial 下可能不足全部划分。表中分别保留模型种子标准差与患者划分均值之间的标准差；不把 patch 当独立患者计算置信区间。不同划分仍复用同一批患者，不能把它们说成新增患者。</p>'''
    body+=''.join(f'<img src="{name}" alt="{name}">' for name in images)
    a=across.loc[across.source.eq(sources[0])] if not across.empty else across
    cols=['source','depth','target','patient_split_count','roc_auc_mean','roc_auc_between_patient_split_std','roc_auc_mean_within_split_model_seed_std','accuracy_mean','balanced_accuracy_mean','sensitivity_mean','specificity_mean','loss_mean']
    body+='<details><summary>展开首家来源医院的各目标结果与波动</summary>'+_table(a[[c for c in cols if c in a]])+'</details><p><a href="full_across_patient_split_summary.csv">跨患者划分汇总</a> · <a href="full_within_patient_split_summary.csv">每个划分内模型种子均值/标准差</a> · <a href="full_metrics_all.csv">全部有效逐运行指标</a></p></section><section><h2>相对于只训练分类头的配对差值</h2><p>每个差值严格匹配同一患者划分、来源、目标与模型种子。AUC/Accuracy/敏感度等差值为正表示提升；loss差值为负表示提升。缺少任一对照的实验不生成配对差值。</p><p><a href="full_depth_minus_head_per_run.csv">逐运行配对差值</a> · <a href="full_depth_minus_head_across_splits.csv">差值的模型种子/患者划分稳定性</a></p></section><section><h2>每个目标医院的观察赢家频率</h2><p>每次比较先平均固定患者划分内的全部模型种子，只比较四档都齐全的组合。并列赢家各分1/k票。频率按有效来源×患者划分比较数计算；缺失组合不算作落败，也不补成0票。这里使用目标测试结果事后描述赢家，不能当作部署时的自动深度选择规则。</p>'
    body+=_table(target_frequency.loc[target_frequency.metric.eq('roc_auc')] if not target_frequency.empty else target_frequency)
    body+='<p><a href="full_winner_frequency_by_target.csv">每目标频率（仅跨医院）</a> · <a href="full_winner_frequency_by_pair.csv">每来源→目标频率</a> · <a href="full_observed_winners.csv">逐患者划分的完整比较与并列记录</a></p></section><section><h2>表征距离与性能／最佳深度</h2><p>使用各医院train分区的固定 ImageNet ResNet50 avgpool 表征，逐维计算一维 Wasserstein-1，再对2048维取平均；不是2048维联合 Wasserstein，也不是W2。原始距离不读取类别标签；均衡/类别条件距离使用类别控制，只用于事后描述。对角线按定义为0，其他元素来自保存的距离CSV。</p><p>分别报告每个患者划分的有向医院对相关，以及先对同一医院对跨全部患者划分平均的相关；不将所有重复划分的60对混在一起当独立样本。医院对共享医院，最佳深度的序号只是有序编码。仅报告描述性 Spearman rho，不给独立patch置信区间、显著性或因果结论。</p>'
    selected=relationships.loc[relationships.outcome.isin(['full-head:roc_auc','observed_winner_depth:roc_auc'])] if not relationships.empty else relationships
    body+=_table(selected)+'<p><a href="full_shift_associations.csv">全部描述性关联</a> · <a href="full_shift_performance_observations.csv">关联的逐医院对输入</a> · <a href="full_representation_shift_all.csv">三种距离原值</a></p></section><section><h2>Loss 与完成边界</h2><p>测试loss使用模型保存的logit交叉熵；不从可能饱和为0/1的概率重算loss。逐epoch曲线统计保留每个epoch实际到达的模型数；早停后没有观测的epoch不补值，因此后段均值不一定对应同一批运行。<a href="full_loss_curves_by_split_epoch.csv">查看均值/标准差/运行数量</a>。</p><p>本页仅主项目，不包含CORAL。完整训练、保存记录汇总与独立审计是不同证据；未来应与全量manifest和独立验收记录一起使用。</p></section></html>'
    # Partial reports must not contain dangling links to not-yet-created CSVs.
    import re
    for name in set(re.findall(r'href="(full_[^"]+)"',body)):
        if not (out/name).exists():
            body=re.sub(r'<a href="'+re.escape(name)+r'">(.*?)</a>',r'<span class="warn">\1（尚无有效结果）</span>',body)
    (out/'full_project_report.html').write_text(body,encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path(__file__).resolve().parent/'results/colab_full')
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args();result=generate(args.results,output_dir=args.output_dir)
    print(json.dumps({k:result[k] for k in ['status','complete_runs','expected_runs','missing_runs','invalid_runs']},ensure_ascii=False))
    raise SystemExit(0 if result['complete'] else 1)

if __name__=='__main__':main()
