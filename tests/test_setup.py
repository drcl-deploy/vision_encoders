import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/setup_onnxruntime.sh'


class RuntimeSetupTest(unittest.TestCase):
    def test_architecture_and_jetpack_selection(self):
        selections = [(['--arch', 'x86_64'], '1.22.0 (prebuilt)'),
            (['--arch', 'aarch64', '--cuda-major', '11', '--cudnn-major', '8'], '1.15.1 (source)'),
            (['--arch', 'arm64', '--cuda-major', '12', '--cudnn-major', '8'], '1.18.0 (source)'),
            (['--arch', 'aarch64', '--cuda-major', '12', '--cudnn-major', '9'], '1.22.0 (source)')]
        for args, expected in selections:
            result = subprocess.run(['bash', str(SCRIPT), '--dry-run', '--jobs', '1'] + args,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(expected, result.stdout)
            self.assertIn('source build jobs: 1', result.stdout)

    def test_unsupported_cuda_is_not_silently_treated_as_cuda12(self):
        result = subprocess.run(['bash', str(SCRIPT), '--dry-run', '--arch', 'aarch64',
            '--cuda-major', '13', '--cudnn-major', '9'], text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('unsupported CUDA/cuDNN pair', result.stderr)

    def test_source_sdk_layout_and_custom_dependency_directory(self):
        # These small native libraries only exercise SDK packaging/ldd, not CUDA inference.
        for layout in ['onnxruntime', 'onnxruntime/core/session']:
            with self.subTest(layout=layout), tempfile.TemporaryDirectory(prefix='ort fixture ') as temp:
                root = Path(temp)
                script = root / 'encoder/scripts/setup_onnxruntime.sh'
                script.parent.mkdir(parents=True)
                shutil.copyfile(SCRIPT, script)
                sdk = root / 'payload'
                headers = sdk / 'include' / layout
                headers.mkdir(parents=True)
                (headers / 'onnxruntime_cxx_api.h').write_text('// fixture\n')
                (sdk / 'lib').mkdir()
                library = sdk / 'lib/libonnxruntime.so'
                subprocess.run(['cc', '-shared', '-fPIC', '-x', 'c', '-', '-o', str(library)],
                    input='#include <stdio.h>\nvoid fixture(void) { puts("fixture"); }\n', text=True, check=True)
                for name in ['libonnxruntime_providers_cuda.so', 'libonnxruntime_providers_shared.so']:
                    shutil.copyfile(library, sdk / 'lib' / name)
                archive = root / 'sdk.tgz'
                with tarfile.open(archive, 'w:gz') as output:
                    output.add(sdk, arcname='sdk')
                checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
                env = dict(os.environ, VISION_ENCODERS_DEPS_DIR=str(root / 'custom deps'))
                args = ['bash', str(script), '--archive', str(archive), '--sha256', checksum,
                        '--cuda-major', '12', '--cudnn-major', '9']
                result = subprocess.run(args, env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                link = script.parents[1] / 'thirdparty/onnxruntime'
                self.assertTrue(link.is_symlink())
                self.assertTrue((link / 'include/onnxruntime_cxx_api.h').is_file())
                self.assertIn(str(root / 'custom deps'), str(link.resolve()))
                again = subprocess.run(args, env=env, text=True, capture_output=True)
                self.assertEqual(again.returncode, 0, again.stderr)
                self.assertIn('reusing existing validated installation', again.stdout)


if __name__ == '__main__':
    unittest.main()
