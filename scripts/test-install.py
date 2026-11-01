#!/usr/bin/env python3
"""Exercise the real installer with local downloads; no network/root writes.

INSTALL_TEST_ARCHIVE may point to the native release archive. Otherwise a
small fixture covers immutable component paths and mtime preservation.
"""
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


@unittest.skipUnless(sys.platform == 'linux', 'Linux runtime installer')
class InstallTest(unittest.TestCase):
    def test_versioned_runtime(self):
        with tempfile.TemporaryDirectory(prefix='warmctl-install-test-') as tmp:
            root = Path(tmp)
            downloads, commands, prefix = [root / n for n in ('downloads', 'commands', 'prefix')]
            downloads.mkdir()
            commands.mkdir()
            arch = {'x86_64': 'amd64', 'aarch64': 'arm64'}[platform.machine()]
            version = 'v0.0.0-installer-test'
            name = f'warmctl-{version}'
            archive = downloads / f'{name}-linux-{arch}.tar.gz'
            supplied = os.environ.get('INSTALL_TEST_ARCHIVE')
            if supplied:
                shutil.copyfile(supplied, archive)
            else:
                fixture = root / 'fixture'
                fixture.mkdir()
                binary = fixture / 'warmctl'
                binary.write_text('#!/bin/sh\necho warmctl-test\n')
                binary.chmod(0o755)
                component = fixture / 'dmtcp' / f'linux-{arch}' / 'v5.0.0-warmctl.1'
                component.mkdir(parents=True)
                (component / 'BUILDINFO').write_text('release=test\n')
                os.utime(component / 'BUILDINFO', ns=(1700000000000000000,) * 2)
                with tarfile.open(archive, 'w:gz') as tar:
                    for child in fixture.iterdir():
                        tar.add(child, arcname=child.name)
            with tarfile.open(archive) as tar:
                markers = [m for m in tar.getmembers() if m.name.endswith('/BUILDINFO')]
                self.assertEqual(len(markers), 1)
                archive_mtime = markers[0].mtime
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            (downloads / f'{name}-checksums.txt').write_text(f'{digest}  {archive.name}\n')
            # Use actual filesystem tools and SHA-256; only HTTPS transport is
            # mocked. No cosign in this PATH, exercising its optional absence.
            for command in ('uname', 'id', 'mktemp', 'rm', 'mkdir', 'tar', 'chmod',
                            'cp', 'mv', 'ln', 'dirname', 'diff', 'sha256sum', 'cut',
                            'awk', 'sed', 'head', 'gzip'):
                os.symlink(shutil.which(command), commands / command)
            curl = commands / 'curl'
            curl.write_text(f'''#!{sys.executable}
import os,sys,shutil
from pathlib import Path
from urllib.parse import urlparse
args=sys.argv[1:]
url=next(a for a in args if a.startswith('https://'))
source=Path(os.environ['INSTALL_TEST_DOWNLOADS'])/Path(urlparse(url).path).name
if not source.is_file(): sys.exit(22)
if any(a.startswith('-') and not a.startswith('--') and 'I' in a for a in args): sys.exit(0)
out=args[args.index('-o')+1]
if out=='-': sys.stdout.buffer.write(source.read_bytes())
else: shutil.copyfile(source,out)
''')
            curl.chmod(0o755)
            env = dict(os.environ, PATH=str(commands), WARMCTL_VERSION=version,
                       WARMCTL_PREFIX=str(prefix), WARMCTL_BIN_DIR='',
                       INSTALL_TEST_DOWNLOADS=str(downloads))
            installer = Path(__file__).resolve().with_name('install.sh')

            def install():
                return subprocess.run(['/bin/sh', str(installer)], env=env,
                                      text=True, capture_output=True, timeout=60)

            first = install()
            self.assertEqual(first.returncode, 0, first.stderr)
            components = list((prefix / 'dmtcp' / f'linux-{arch}').glob('*/BUILDINFO'))
            self.assertEqual(len(components), 1)
            marker = components[0]
            self.assertAlmostEqual(marker.stat().st_mtime, archive_mtime, places=5)
            before = (marker.read_bytes(), marker.stat().st_mtime_ns, marker.stat().st_ino)
            old = marker.parent.parent / 'v0-old-snapshot'
            old.mkdir()
            (old / 'library').write_text('old snapshot library')
            again = install()
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertEqual(before, (marker.read_bytes(), marker.stat().st_mtime_ns, marker.stat().st_ino))
            self.assertEqual((old / 'library').read_text(), 'old snapshot library')
            marker.write_text('different installed build')
            refused = install()
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn('preserving it for existing snapshots', refused.stderr)
            self.assertEqual(marker.read_text(), 'different installed build')
            self.assertTrue((prefix / 'warmctl').is_file())


if __name__ == '__main__':
    unittest.main()
