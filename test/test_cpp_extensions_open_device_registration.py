# Owner(s): ["module: cpp-extensions"]

import os
import shutil
import sys
from typing import Union
import unittest

import torch.testing._internal.common_utils as common
from torch.testing._internal.common_utils import IS_ARM64
import torch
import torch.utils.cpp_extension
from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME


TEST_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
TEST_CUDNN = False
TEST_ROCM = torch.cuda.is_available() and torch.version.hip is not None and ROCM_HOME is not None
if TEST_CUDA and torch.version.cuda is not None:  # the skip CUDNN test for ROCm
    CUDNN_HEADER_EXISTS = os.path.isfile(os.path.join(CUDA_HOME, "include/cudnn.h"))
    TEST_CUDNN = (
        TEST_CUDA and CUDNN_HEADER_EXISTS and torch.backends.cudnn.is_available()
    )


def remove_build_path():
    if sys.platform == "win32":
        # Not wiping extensions build folder because Windows
        return
    default_build_root = torch.utils.cpp_extension.get_default_build_root()
    if os.path.exists(default_build_root):
        shutil.rmtree(default_build_root, ignore_errors=True)


class DummyModule(object):

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_rng_state(device: Union[int, str, torch.device] = 'foo') -> torch.Tensor:
        # create a tensor using our custom device object.
        return torch.empty(4, 4, device="foo")

    @staticmethod
    def set_rng_state(new_state: torch.Tensor, device: Union[int, str, torch.device] = 'foo') -> None:
        pass

    @staticmethod
    def is_available():
        return True


@unittest.skipIf(IS_ARM64, "Does not work on arm")
class TestCppExtensionOpenRgistration(common.TestCase):
    """Tests Open Device Registration with C++ extensions.
    """
    module = None

    def setUp(self):
        super().setUp()
        # cpp extensions use relative paths. Those paths are relative to
        # this file, so we'll change the working directory temporarily
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        assert self.module is not None

    def tearDown(self):
        super().tearDown()
        # return the working directory (see setUp)
        os.chdir(self.old_working_dir)

    @classmethod
    def setUpClass(cls):
        remove_build_path()
        cls.module = torch.utils.cpp_extension.load(
            name="custom_device_extension",
            sources=[
                "cpp_extensions/open_registration_extension.cpp",
            ],
            extra_include_paths=["cpp_extensions"],
            extra_cflags=["-g"],
            verbose=True,
        )

    @classmethod
    def tearDownClass(cls):
        remove_build_path()

    def test_open_device_registration(self):
        def test_base_device_registration():
            torch.utils.rename_privateuse1_backend('foo')
            self.assertFalse(self.module.custom_add_called())
            # create a tensor using our custom device object
            device = self.module.custom_device()
            x = torch.empty(4, 4, device=device)
            y = torch.empty(4, 4, device=device)
            # Check that our device is correct.
            self.assertTrue(x.device == device)
            self.assertFalse(x.is_cpu)
            self.assertFalse(self.module.custom_add_called())
            # calls out custom add kernel, registered to the dispatcher
            z = x + y
            # check that it was called
            self.assertTrue(self.module.custom_add_called())
            z_cpu = z.to(device='cpu')
            # Check that our cross-device copy correctly copied the data to cpu
            self.assertTrue(z_cpu.is_cpu)
            self.assertFalse(z.is_cpu)
            self.assertTrue(z.device == device)
            self.assertEqual(z, z_cpu)
            z2 = z_cpu + z_cpu

        # check whether the error can be reported correctly
        def test_before_common_registration():
            # check that register module name should be the same as custom backend
            with self.assertRaisesRegex(RuntimeError, "Expected one of cpu"):
                torch._register_device_module('xxx', DummyModule)
            # check generator registered before using
            with self.assertRaisesRegex(RuntimeError, "torch has no module of"):
                with torch.random.fork_rng(device_type="foo"):
                    pass
            # check attributes before registered
            self.assertFalse(hasattr(torch.Tensor, 'is_foo'))
            self.assertFalse(hasattr(torch.Tensor, 'foo'))
            self.assertFalse(hasattr(torch.TypedStorage, 'is_foo'))
            self.assertFalse(hasattr(torch.TypedStorage, 'foo'))
            self.assertFalse(hasattr(torch.UntypedStorage, 'is_foo'))
            self.assertFalse(hasattr(torch.UntypedStorage, 'foo'))
            self.assertFalse(hasattr(torch.nn.Module, 'foo'))

        def test_after_common_registration():
            # check attributes after registered
            self.assertTrue(hasattr(torch.Tensor, 'is_foo'))
            self.assertTrue(hasattr(torch.Tensor, 'foo'))
            self.assertTrue(hasattr(torch.TypedStorage, 'is_foo'))
            self.assertTrue(hasattr(torch.TypedStorage, 'foo'))
            self.assertTrue(hasattr(torch.UntypedStorage, 'is_foo'))
            self.assertTrue(hasattr(torch.UntypedStorage, 'foo'))
            self.assertTrue(hasattr(torch.nn.Module, 'foo'))

        def test_common_registration():
            # first rename custom backend
            torch.utils.rename_privateuse1_backend('foo')
            # backend name can only rename once
            with self.assertRaisesRegex(RuntimeError, "torch.register_privateuse1_backend()"):
                torch.utils.rename_privateuse1_backend('xxx')
            # register foo module, torch.foo
            torch._register_device_module('foo', DummyModule)
            # default set for_tensor and for_module are True, so only set for_storage is True
            torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
            # generator tensor and module can be registered only once
            with self.assertRaisesRegex(RuntimeError, "The custom device module of"):
                torch.utils.generate_methods_for_privateuse1_backend()

        def test_generator_registration():
            device = self.module.custom_device()
            # None of our CPU operations should call the custom add function.
            self.assertFalse(self.module.custom_add_called())
            # check generator registered before using
            with self.assertRaisesRegex(RuntimeError,
                                        "Please register a generator to the PrivateUse1 dispatch key"):
                gen_ = torch.Generator(device=device)
            self.module.register_generator()
            gen = torch.Generator(device=device)
            self.assertTrue(gen.device == device)
            # generator can be registered only once
            with self.assertRaisesRegex(RuntimeError,
                                        "Only can register a generator to the PrivateUse1 dispatch key once"):
                self.module.register_generator()

        def test_open_device_random():
            with torch.random.fork_rng(device_type="foo"):
                pass

        def test_open_device_tensor():
            device = self.module.custom_device()
            # check whether print tensor.type() meets the expectation
            dtypes = {
                torch.bool: 'torch.foo.BoolTensor',
                torch.double: 'torch.foo.DoubleTensor',
                torch.float32: 'torch.foo.FloatTensor',
                torch.half: 'torch.foo.HalfTensor',
                torch.int32: 'torch.foo.IntTensor',
                torch.int64: 'torch.foo.LongTensor',
                torch.int8: 'torch.foo.CharTensor',
                torch.short: 'torch.foo.ShortTensor',
                torch.uint8: 'torch.foo.ByteTensor',
            }
            for tt, dt in dtypes.items():
                test_tensor = torch.empty(4, 4, dtype=tt, device=device)
                self.assertTrue(test_tensor.type() == dt)
            # check whether the attributes and methods of the corresponding custom backend are generated correctly
            x = torch.empty(4, 4)
            self.assertFalse(x.is_foo)
            x = x.foo(torch.device("foo"))
            self.assertFalse(self.module.custom_add_called())
            self.assertTrue(x.is_foo)
            # test different device type input
            y = torch.empty(4, 4)
            self.assertFalse(y.is_foo)
            y = y.foo(torch.device("foo:0"))
            self.assertFalse(self.module.custom_add_called())
            self.assertTrue(y.is_foo)
            # test different device type input
            z = torch.empty(4, 4)
            self.assertFalse(z.is_foo)
            z = z.foo(0)
            self.assertFalse(self.module.custom_add_called())
            self.assertTrue(z.is_foo)

        def test_open_device_storage():
            # check whether the attributes and methods for storage of the corresponding custom backend are generated correctly
            x = torch.empty(4, 4)
            z1 = x.storage()
            self.assertFalse(z1.is_foo)
            z1 = z1.foo()
            self.assertFalse(self.module.custom_add_called())
            self.assertTrue(z1.is_foo)
            with self.assertRaisesRegex(RuntimeError, "Invalid device"):
                z1.foo(torch.device("cpu"))
            z1 = z1.cpu()
            self.assertFalse(self.module.custom_add_called())
            self.assertFalse(z1.is_foo)
            # check UntypedStorage
            y = torch.empty(4, 4)
            z2 = y.untyped_storage()
            self.assertFalse(z2.is_foo)
            z2 = z2.foo()
            self.assertFalse(self.module.custom_add_called())
            self.assertTrue(z2.is_foo)

        def test_open_device_storage_pin_memory():
            torch.utils.rename_privateuse1_backend('foo')
            with self.assertRaisesRegex(RuntimeError, "The custom device module of"):
                torch.utils.generate_methods_for_privateuse1_backend(for_tensor=False, for_module=False, for_storage=True)
            # Check if the pin_memory is functioning properly on custom device
            cpu_tensor = torch.empty(3)
            self.assertFalse(cpu_tensor.is_foo)
            self.assertFalse(cpu_tensor.is_pinned("foo"))
            cpu_tensor_pin = cpu_tensor.pin_memory("foo")
            self.assertTrue(cpu_tensor_pin.is_pinned("foo"))
            # Test storage pin_memory on custom device
            cpu_storage = cpu_tensor.storage()
            self.assertFalse(cpu_storage.is_pinned("foo"))
            cpu_storage_pin = cpu_storage.pin_memory("foo")
            self.assertFalse(cpu_storage.is_pinned("foo"))
            self.assertTrue(cpu_storage_pin.is_pinned("foo"))
            cpu_storage_pin_already = cpu_storage_pin.pin_memory("foo")
            self.assertTrue(cpu_storage_pin.is_pinned("foo"))
            self.assertTrue(cpu_storage_pin_already.is_pinned("foo"))
            # Test storage pin_memory on error device
            self.assertFalse(cpu_storage.is_pinned("foo"))
            with self.assertRaisesRegex(TypeError, "cannot pin CPU memory to hpu device, please check the target device."):
                cpu_storage_pin = cpu_storage.pin_memory("hpu")
            self.assertFalse(cpu_storage.is_pinned("hpu"))
            self.assertFalse(cpu_storage.is_pinned())

        def test_open_device_serialization():
            storage = torch.UntypedStorage(4, device=torch.device('foo'))
            self.assertEqual(torch.serialization.location_tag(storage), 'foo')
            cpu_storage = torch.empty(4, 4).storage()
            foo_storage = torch.serialization.default_restore_location(cpu_storage, 'foo:0')
            self.assertTrue(foo_storage.is_foo)

        test_base_device_registration()
        test_before_common_registration()
        test_common_registration()
        test_after_common_registration()
        test_generator_registration()
        test_open_device_random()
        test_open_device_tensor()
        test_open_device_storage()
        test_open_device_storage_pin_memory()
        test_open_device_serialization()

if __name__ == "__main__":
    common.run_tests()
