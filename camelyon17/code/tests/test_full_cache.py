"""隔离的小真实PNG验证；不接触正式图像、Drive或训练。"""
import csv
import hashlib
import io
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import full_cache


class FullCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base/'source'
        self.target = self.base/'target'
        self.cache = self.base/'private_cache'
        self.paths = []
        for i in range(3):
            relative = f'data/images/{full_cache.PINNED_REVISION}/c0_p4_n4_x{i}_y2.png'
            self.paths.append(relative)
            image = self.root/relative
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (96,96), (20+i*40,30+i,50)).save(image)
        self.original = {p:(self.root/p).read_bytes() for p in self.paths}
        self.manifest = self.write_manifest(self.root, self.paths)
        self.target_manifest = self.write_manifest(self.target, self.paths)

    def write_manifest(self, root, paths):
        path = root/'data/manifest.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as destination:
            writer=csv.DictWriter(destination,fieldnames=['path','partition'])
            writer.writeheader()
            for i, relative in enumerate(paths):
                writer.writerow({'path':relative,'partition':['train','val','test'][i%3]})
        return path

    def export(self):
        return full_cache.export_cache(self.root,self.manifest,self.cache,expected_images=3)

    def restore(self):
        return full_cache.restore_cache(self.target,self.target_manifest,self.cache,expected_images=3)

    def rewrite_archive(self, members):
        # 模拟tar与其SHA都被替换；必须由member安全检查拦截，而不只靠checksum。
        path=self.cache/full_cache.TAR_NAME
        with tarfile.open(path,'w',format=tarfile.USTAR_FORMAT) as archive:
            for member, data in members:
                archive.addfile(member,io.BytesIO(data) if member.isfile() else None)
        metadata_path=self.cache/full_cache.METADATA_NAME
        metadata=json.loads(metadata_path.read_text())
        metadata.update(tar_size=path.stat().st_size,
                        tar_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        metadata_path.write_text(json.dumps(metadata))

    def regular(self,name,data):
        member=tarfile.TarInfo(name)
        member.size=len(data)
        return member,data

    def test_roundtrip_is_byte_exact_and_partition_independent(self):
        archive=self.export()
        metadata=json.loads((self.cache/full_cache.METADATA_NAME).read_text())
        self.assertEqual(metadata['identity']['expected_images'],3)
        self.assertEqual(metadata['tar_size'],archive.stat().st_size)
        self.assertEqual(metadata['tar_sha256'],hashlib.sha256(archive.read_bytes()).hexdigest())
        # 改变patient partition概念对应列，不改变图像集合，不应失效。
        text=self.target_manifest.read_text().replace('train','val')
        self.target_manifest.write_text(text)
        self.assertTrue(self.restore())
        for path,data in self.original.items():
            self.assertEqual((self.target/path).read_bytes(),data)
        with tarfile.open(archive,'r:') as source:
            self.assertEqual(source.getnames(),sorted(self.paths))
            self.assertTrue(all(m.isfile() for m in source.getmembers()))

    def test_identity_mismatch_never_replaces_existing_complete_cache(self):
        archive=self.export()
        checksum=hashlib.sha256(archive.read_bytes()).hexdigest()
        different=self.paths.copy()
        different[-1]=different[-1].replace('x2','x3')
        self.write_manifest(self.target,different)
        with self.assertRaisesRegex(ValueError,'identity/schema'):
            self.restore()
        with self.assertRaisesRegex(ValueError,'identity/schema'):
            full_cache.export_cache(self.target,self.target_manifest,self.cache,expected_images=3)
        self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(),checksum)

    def test_truncated_complete_cache_raises_before_any_png_restore(self):
        archive=self.export()
        archive.write_bytes(archive.read_bytes()[:-1000])
        with self.assertRaises(ValueError):
            self.restore()
        self.assertFalse((self.target/self.paths[0]).exists())

    def test_malicious_members_are_rejected_before_any_extraction(self):
        self.export()
        sample=self.original[self.paths[0]]
        invalid=[]
        invalid.append(self.regular('../escape.png',sample))
        invalid.append(self.regular('/absolute.png',sample))
        invalid.append(self.regular(f'data/images/{full_cache.PINNED_REVISION}/unlisted.png',sample))
        for kind in [tarfile.SYMTYPE,tarfile.LNKTYPE,tarfile.DIRTYPE]:
            member=tarfile.TarInfo(self.paths[0]);member.type=kind;member.linkname='../../escape.png'
            invalid.append((member,b''))
        for bad in invalid:
            with self.subTest(member=bad[0].name,kind=bad[0].type):
                remaining=[self.regular(p,self.original[p]) for p in self.paths[1:]]
                self.rewrite_archive([bad,*remaining])
                with self.assertRaises(ValueError):
                    self.restore()
                self.assertFalse((self.target/self.paths[1]).exists())
        duplicate=[self.regular(self.paths[0],sample),self.regular(self.paths[0],sample),
                   self.regular(self.paths[2],self.original[self.paths[2]])]
        self.rewrite_archive(duplicate)
        with self.assertRaises(ValueError):
            self.restore()

    def test_tar_without_complete_zero_end_marker_is_rejected(self):
        self.export()
        path=self.cache/full_cache.TAR_NAME
        with tarfile.open(path,'r:') as source:
            last=source.getmembers()[-1]
            end=last.offset_data+((last.size+511)//512)*512
        data=path.read_bytes()[:end]
        path.write_bytes(data)
        metadata_path=self.cache/full_cache.METADATA_NAME
        metadata=json.loads(metadata_path.read_text())
        metadata.update(tar_size=len(data),tar_sha256=hashlib.sha256(data).hexdigest())
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError,'end marker'):
            self.restore()

    def test_partial_only_is_not_a_complete_cache_and_export_can_finish_it(self):
        self.cache.mkdir()
        partial=self.cache/'images.tar.partial'
        partial.write_bytes(b'interrupted transfer')
        self.assertFalse(self.restore())
        # 即使主tar存在，没有最后提交metadata也不能恢复。
        (self.cache/full_cache.TAR_NAME).write_bytes(b'not committed')
        self.assertFalse(self.restore())
        self.export()
        self.assertTrue(self.restore())

    def test_interrupted_export_does_not_commit_metadata_or_become_reusable(self):
        def broken_copy(source,target):
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(b'partial')
            raise RuntimeError('planned copy interruption')
        with patch('full_cache._copy_file',side_effect=broken_copy):
            with self.assertRaisesRegex(RuntimeError,'planned copy'):
                self.export()
        self.assertTrue((self.cache/'images.tar.partial').exists())
        self.assertFalse((self.cache/full_cache.METADATA_NAME).exists())
        self.assertFalse(self.restore())
        self.export()
        self.assertTrue(self.restore())

    def test_partial_png_restore_preserves_existing_file_then_retry_succeeds(self):
        self.export()
        for path in self.paths:
            destination=self.target/path;destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(b'existing bytes')
        copy=shutil.copyfileobj
        calls=0
        def interrupted(incoming,outgoing,length=0):
            nonlocal calls
            if length==1<<20:
                calls+=1
                if calls==2:
                    outgoing.write(incoming.read(4))
                    raise RuntimeError('planned PNG interruption')
            return copy(incoming,outgoing,length)
        with patch('full_cache.shutil.copyfileobj',side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError,'planned PNG'):
                self.restore()
        second=self.target/sorted(self.paths)[1]
        self.assertEqual(second.read_bytes(),b'existing bytes')
        self.assertTrue(second.with_name(second.name+'.restore.partial').exists())
        self.assertTrue(self.restore())
        for path,data in self.original.items():
            self.assertEqual((self.target/path).read_bytes(),data)

    def test_existing_valid_cache_is_reused_without_rewriting(self):
        archive=self.export()
        before=archive.stat().st_mtime_ns
        # 清除export记录，模拟新runtime：restore先验证，然后export应复用此次结果。
        full_cache._VALIDATED_CACHES.clear()
        self.assertTrue(self.restore())
        with patch('full_cache._copy_file',side_effect=AssertionError('must not rewrite existing cache')):
            self.assertEqual(self.export(),archive)
        self.assertEqual(archive.stat().st_mtime_ns,before)
        payload=bytearray(archive.read_bytes());payload[1024]^=1
        archive.write_bytes(payload)
        # 远端stat变化后，即使长度不变也不能误用旧的验证记录。
        with self.assertRaisesRegex(ValueError,'SHA256'):
            self.export()

    def test_symlink_local_destination_is_never_followed(self):
        self.export()
        outside=self.base/'outside';outside.mkdir()
        images=self.target/'data/images';images.mkdir(parents=True)
        (images/full_cache.PINNED_REVISION).symlink_to(outside,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'escapes root|Symlink'):
            self.restore()
        self.assertEqual(list(outside.iterdir()),[])


if __name__=='__main__':
    unittest.main()
