"""CPU-only orchestration checks; no GPU initialization, inference or training."""
import hashlib
import json
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch

import torch
import ece_run as e

ROOT=Path(e.__file__).resolve().parent
CONFIG=json.loads((ROOT/'configs/full.json').read_text())


def completed_fixture(folder,signature='expected',resumed=0):
    folder.mkdir(parents=True,exist_ok=True)
    e.write_json(folder/'run.json',{'signature':signature,'spec':{}})
    torch.save({'signature':signature,'state_digest':'weights-digest','state_dict':{'tiny':torch.tensor([1.])}},folder/'best.pt')
    (folder/'history.csv').write_text('epoch,val_loss\n1,0.5\n')
    (folder/'predictions.csv').write_text('sample_id,label,probability\nx,1,0.6\n')
    result={'signature':signature,'status':'complete','checkpoint_state_digest':'weights-digest','resume':{'resumed_from_epoch':resumed}}
    e.write_json(folder/'metrics.json',result)
    e.write_json(folder/'completion.json',{'signature':signature,'status':'complete','artifacts':{name:e.sha256(folder/name) for name in ['run.json','best.pt','history.csv','predictions.csv','metrics.json']}})
    return result


class EceRunTests(unittest.TestCase):
    def test_disjoint_full_plan_shards_and_source_A_first(self):
        e.validate_plan(CONFIG)
        tasks=e.ordered_tasks(CONFIG);a=e.assigned_tasks(CONFIG,0,2);b=e.assigned_tasks(CONFIG,1,2)
        self.assertEqual(len(tasks),180);self.assertEqual((len(a),len(b)),(90,90))
        self.assertFalse(set(a)&set(b));self.assertEqual(set(a)|set(b),set(tasks))
        self.assertTrue(all(t.source==0 for t in tasks[:36]));self.assertTrue(all(t.source!=0 for t in tasks[36:]))
        for shard in [a,b]:self.assertTrue(all(t.source==0 for t in shard[:18]))
        with self.assertRaises(ValueError):e.assigned_tasks(CONFIG,2,2)

    def test_task_lock_excludes_another_process(self):
        with tempfile.TemporaryDirectory() as td:
            lock=Path(td)/'task.lock'
            script='import sys;from ece_run import FileLock,LockBusy\ntry:\n with FileLock(sys.argv[1]):sys.exit(0)\nexcept LockBusy:sys.exit(23)'
            with e.FileLock(lock):
                result=subprocess.run([sys.executable,'-c',script,str(lock)],cwd=ROOT,check=False,capture_output=True,text=True)
                self.assertEqual(result.returncode,23,result.stderr)
            with e.FileLock(lock):pass

    def test_finalizer_exclusive_phase_waits_for_all_shared_workers(self):
        with tempfile.TemporaryDirectory() as td:
            phase=Path(td)/'phase.lock'
            with e.FileLock(phase,shared=True),e.FileLock(phase,shared=True):
                with self.assertRaises(e.LockBusy):
                    with e.FileLock(phase):pass
            with e.FileLock(phase):pass

    def test_completed_result_requires_real_core_hash_and_signature_validation(self):
        with tempfile.TemporaryDirectory() as td:
            folder=Path(td)/'run';completed_fixture(folder)
            self.assertEqual(e.verified_completed(folder,'expected')['status'],'complete')
            with self.assertRaises(ValueError):e.verified_completed(folder,'wrong')
            (folder/'predictions.csv').write_text('corrupted\n')
            with self.assertRaisesRegex(RuntimeError,'checksum mismatch'):e.verified_completed(folder,'expected')

    def test_completed_task_skips_executor_and_bad_completion_never_retrains(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);task=e.Task(42,0,'head',42);folder=task.folder(out);completed_fixture(folder)
            context=Mock();context.signature.return_value='expected';executor=Mock(side_effect=AssertionError('Must not train'))
            receipt=e.execute_task(out,CONFIG,out,task,context,{'gpu':'fixture'},executor=executor)
            executor.assert_not_called();self.assertEqual(receipt['action'],'verified_completed_skip')
            self.assertTrue((out/'.ece/verified_tasks'/f'{task.key}.json').exists())
            (folder/'history.csv').write_text('corrupted')
            with self.assertRaisesRegex(RuntimeError,'checksum mismatch'):
                e.execute_task(out,CONFIG,out,task,context,{'gpu':'fixture'},executor=executor)
            executor.assert_not_called()

    def test_partial_checkpoint_is_forwarded_unchanged_with_original_config_and_origin(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);task=e.Task(42,0,'layer4',42);folder=task.folder(out);folder.mkdir(parents=True)
            e.write_json(folder/'run.json',{'signature':'expected','spec':{}})
            payload=b'opaque original CUDA epoch checkpoint';(folder/'last.pt').write_bytes(payload)
            context=Mock();context.signature.return_value='expected'
            def executor(root,config,source,depth,seed,destination):
                from full_run import split_config
                self.assertEqual(config,split_config(CONFIG,42));self.assertEqual((source,depth,seed),(0,'layer4',42))
                self.assertEqual((destination/'last.pt').read_bytes(),payload)
                return completed_fixture(destination,resumed=7)
            receipt=e.execute_task(out,CONFIG,out,task,context,{'gpu':'RTX3060'},origin='Imported Colab T4',executor=executor)
            self.assertEqual(receipt['action'],'resumed');self.assertEqual(receipt['resumed_from_epoch'],7)
            origin=json.loads(next((out/'.ece/imports'/task.key).glob('*.json')).read_text())
            self.assertEqual(origin['origin_hint'],'Imported Colab T4')
            self.assertEqual(origin['last_checkpoint_sha256'],hashlib.sha256(payload).hexdigest())
            self.assertEqual(origin['current_environment']['gpu'],'RTX3060')

    def test_duplicate_supervisor_does_not_spawn_workers(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)
            with e.FileLock(out/'.ece/coordinator.lock'),patch('ece_run.subprocess.Popen') as popen:
                with self.assertRaises(e.LockBusy):e.supervise(out,CONFIG,out,['GPU0','GPU1'],config_path=ROOT/'configs/full.json',expected_environment=out/'versions.json')
                popen.assert_not_called()

    def test_original_software_versions_are_enforced_before_GPU_work(self):
        with tempfile.TemporaryDirectory() as td:
            versions={'python':'incorrect','torch':torch.__version__,'torchvision':'x','numpy':'x','pandas':'x','scipy':'x','sklearn':'x'}
            path=Path(td)/'versions.json';e.write_json(path,versions)
            with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.device_count',return_value=1),patch('torch.cuda.get_device_properties') as props:
                with self.assertRaisesRegex(RuntimeError,'software environment mismatch'):e.runtime_environment(path)
                props.assert_not_called()

    def test_second_worker_failure_is_detected_without_waiting_for_first(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);first=Mock(pid=10001);second=Mock(pid=10002)
            first.poll.return_value=None;second.poll.return_value=17
            with patch('ece_run.initialize'),patch('ece_run.subprocess.Popen',side_effect=[first,second]) as spawn,patch('ece_run.stop_worker_groups') as cleanup,patch('ece_run.finalize') as finalize:
                with self.assertRaisesRegex(RuntimeError,r'Workers exited \[None, 17\]'):
                    e.supervise(out,CONFIG,out,['GPU0','GPU1'],config_path=ROOT/'configs/full.json',expected_environment=out/'versions.json')
                cleanup.assert_called_once_with([first,second]);first.wait.assert_not_called();finalize.assert_not_called()
                self.assertTrue(all(call.kwargs['start_new_session'] for call in spawn.call_args_list))
            state=json.loads((out/'.ece/supervisor.json').read_text())
            self.assertEqual(state['status'],'interrupted');self.assertEqual(state['exit_codes'],[None,17])

    def test_supervisor_term_and_hup_cleanup_and_restore_signal_handlers(self):
        for sig in [signal.SIGTERM,signal.SIGHUP,signal.SIGINT]:
            with self.subTest(signal=sig),tempfile.TemporaryDirectory() as td:
                out=Path(td);first=Mock(pid=10001);second=Mock(pid=10002);sent=[]
                old=signal.getsignal(sig)
                def poll():
                    if not sent:sent.append(sig);signal.raise_signal(sig)
                    return None
                first.poll.side_effect=poll;second.poll.return_value=None
                with patch('ece_run.initialize'),patch('ece_run.subprocess.Popen',side_effect=[first,second]),patch('ece_run.stop_worker_groups') as cleanup,patch('ece_run.finalize') as finalize:
                    with self.assertRaises(e.SupervisorStopped):
                        e.supervise(out,CONFIG,out,['GPU0','GPU1'],config_path=ROOT/'configs/full.json',expected_environment=out/'versions.json')
                    cleanup.assert_called_once_with([first,second]);finalize.assert_not_called()
                self.assertIs(signal.getsignal(sig),old)
                state=json.loads((out/'.ece/supervisor.json').read_text())
                self.assertEqual(state['status'],'interrupted');self.assertEqual(state['received_signal'],sig)

    def test_cleanup_signals_entire_worker_groups_including_exited_leaders(self):
        first=Mock(pid=10001);second=Mock(pid=10002)
        first.poll.return_value=0;second.poll.return_value=None
        with patch('ece_run.os.killpg') as killpg:
            e.stop_worker_groups([first,second],grace_seconds=0)
        for process in [first,second]:
            self.assertIn(((process.pid,signal.SIGTERM),),[(c.args,) for c in killpg.call_args_list])
            self.assertIn(((process.pid,signal.SIGKILL),),[(c.args,) for c in killpg.call_args_list])
            process.wait.assert_called_once_with(timeout=10)

    def test_success_waits_for_both_workers_then_finalizes_once(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);first=Mock(pid=10001);second=Mock(pid=10002)
            first.poll.return_value=0;second.poll.side_effect=[None,0]
            with patch('ece_run.initialize'),patch('ece_run.subprocess.Popen',side_effect=[first,second]),patch('ece_run.time.sleep'),patch('ece_run.stop_worker_groups') as cleanup,patch('ece_run.finalize',return_value={'complete':True}) as finalize,patch.dict('os.environ',{},clear=False):
                result=e.supervise(out,CONFIG,out,['GPU0','GPU1'],config_path=ROOT/'configs/full.json',expected_environment=out/'versions.json')
                self.assertTrue(result['complete']);finalize.assert_called_once();cleanup.assert_not_called()
            self.assertEqual(json.loads((out/'.ece/supervisor.json').read_text())['status'],'complete')

    def test_initialize_preserves_imported_review_bytes_and_records_only_path_relocation(self):
        from full_run import split_config
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'prepared';out=Path(td)/'persistent'
            for split in CONFIG['partition_seeds']:
                relative=split_config(CONFIG,split)['manifest_path'];prepared=(root/relative).parent
                imported=out/f'split_{split}';prepared.mkdir(parents=True);imported.mkdir(parents=True)
                for name in ['manifest.csv','provenance.json','patient_partitions.csv']:
                    (prepared/name).write_text('identical original bytes\n');(imported/name).write_bytes((prepared/name).read_bytes())
                original={'manifest_path':f'/content/camelyon17_zhang/{relative}','partition_seed':split,'checks':{'patient_leakage':False}}
                e.write_json(imported/'partition_review.json',original)
                original['manifest_path']=f'/tmp/ece/{relative}';e.write_json(prepared/'partition_review.json',original)
            before={split:(out/f'split_{split}'/'partition_review.json').read_bytes() for split in CONFIG['partition_seeds']}
            with patch('ece_run.validate_inputs'):e.initialize(root,CONFIG,out)
            for split in CONFIG['partition_seeds']:
                self.assertEqual((out/f'split_{split}'/'partition_review.json').read_bytes(),before[split])
                mapping=json.loads(next((out/'.ece/runtime_path_mappings').glob(f'split_{split}_*.json')).read_text())
                self.assertEqual(mapping['relative_manifest_path'],split_config(CONFIG,split)['manifest_path'])
                self.assertTrue(mapping['imported_manifest_path'].startswith('/content/'))
                self.assertTrue(mapping['prepared_manifest_path'].startswith('/tmp/ece/'))
                self.assertEqual(mapping['imported_file_sha256'],hashlib.sha256(before[split]).hexdigest())
            relative=split_config(CONFIG,43)['manifest_path'];review=(root/relative).parent/'partition_review.json'
            changed=json.loads(review.read_text());changed['checks']['patient_leakage']=True;e.write_json(review,changed)
            with patch('ece_run.validate_inputs'),self.assertRaisesRegex(ValueError,'beyond runtime manifest_path'):
                e.initialize(root,CONFIG,out)

    def test_partition_review_path_exception_rejects_wrong_split_relative_traversal_and_type_changes(self):
        with tempfile.TemporaryDirectory() as td:
            original=Path(td)/'original.json';prepared=Path(td)/'prepared.json';relative='data/full/split_43/manifest.csv'
            base={'manifest_path':f'/content/project/{relative}','count':1};e.write_json(original,base)
            for value in ['data/full/split_43/manifest.csv','/tmp/data/full/split_44/manifest.csv','/tmp/../data/full/split_43/manifest.csv','/tmp/other_manifest.csv']:
                e.write_json(prepared,{**base,'manifest_path':value})
                with self.subTest(path=value),self.assertRaises(ValueError):e.partition_review_path_mapping(prepared,original,relative)
            e.write_json(prepared,{**base,'manifest_path':f'/tmp/project/{relative}','count':True})
            with self.assertRaisesRegex(ValueError,'beyond runtime manifest_path'):e.partition_review_path_mapping(prepared,original,relative)


if __name__=='__main__':unittest.main()
