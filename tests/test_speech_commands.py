import copy
import gc
import os
import pickle
import re
import sys
import tempfile
import unittest

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score


try:
    from speech_commands.speech_commands import MobilenetRecognizer, DTWRecognizer
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from speech_commands.speech_commands import MobilenetRecognizer, DTWRecognizer


class TestMobilenetRecognizer(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, 'cls'):
            del self.cls
        if hasattr(self, 'another_cls'):
            del self.another_cls
        if hasattr(self, 'temp_file_name'):
            if os.path.isfile(self.temp_file_name):
                os.remove(self.temp_file_name)

    def test_creation(self):
        self.cls = MobilenetRecognizer(cache_dir='abc')
        self.assertIsInstance(self.cls, MobilenetRecognizer)
        self.assertTrue(hasattr(self.cls, 'batch_size'))
        self.assertTrue(hasattr(self.cls, 'max_epochs'))
        self.assertTrue(hasattr(self.cls, 'patience'))
        self.assertTrue(hasattr(self.cls, 'verbose'))
        self.assertTrue(hasattr(self.cls, 'warm_start'))
        self.assertTrue(hasattr(self.cls, 'window_size'))
        self.assertTrue(hasattr(self.cls, 'shift_size'))
        self.assertTrue(hasattr(self.cls, 'sampling_frequency'))
        self.assertTrue(hasattr(self.cls, 'cache_dir'))
        self.assertIsInstance(self.cls.batch_size, int)
        self.assertIsInstance(self.cls.max_epochs, int)
        self.assertIsInstance(self.cls.patience, int)
        self.assertIsInstance(self.cls.verbose, bool)
        self.assertIsInstance(self.cls.warm_start, bool)
        self.assertIsInstance(self.cls.sampling_frequency, int)
        self.assertIsInstance(self.cls.window_size, float)
        self.assertIsInstance(self.cls.shift_size, float)
        self.assertIsInstance(self.cls.cache_dir, str)
        self.assertIsNone(self.cls.random_seed)

    def test_check_params_positive(self):
        MobilenetRecognizer.check_params(
            sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
            verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3, hidden_layers=(300, 100)
        )
        self.assertTrue(True)

    def test_check_params_negative01(self):
        true_err_msg = re.escape('`sampling_frequency` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5, hidden_layers=(300, 100),
                verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3
            )

    def test_check_params_negative02(self):
        true_err_msg = re.escape('`sampling_frequency` is wrong! Expected `{0}`, got `{1}`.'.format(type(3), type(3.5)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=3.5, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative03(self):
        true_err_msg = re.escape('`sampling_frequency` is wrong! Expected a positive integer value, but -3 is not '
                                 'positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=-3, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative04(self):
        true_err_msg = re.escape('`sampling_frequency` is wrong! Minimal admissible value is 16000 Hz.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=6000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative05(self):
        true_err_msg = re.escape('`window_size` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative06(self):
        true_err_msg = re.escape('`window_size` is wrong! Expected a positive floating-point value, but {0} is not '
                                 'positive.'.format(-2.5))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=-2.5, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative07(self):
        true_err_msg = re.escape('`window_size` is wrong! Expected `{0}`, got `{1}`.'.format(type(3.5), type(3)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=2, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative08(self):
        true_err_msg = re.escape('`window_size` is wrong! {0:.6f} is too small value for `window_size`.'.format(1e-4))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=1e-4, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative09(self):
        true_err_msg = re.escape('`shift_size` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative10(self):
        true_err_msg = re.escape('`shift_size` is wrong! Expected a positive floating-point value, '
                                 'but {0} is not positive.'.format(-0.01))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=-0.01, batch_size=32, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative11(self):
        true_err_msg = re.escape('`shift_size` is wrong! Expected `{0}`, got `{1}`.'.format(type(3.5), type(3)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=1, batch_size=32, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative12(self):
        true_err_msg = re.escape('`shift_size` is wrong! {0:.6f} is too small value for `shift_size`.'.format(1e-5))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=1e-5, batch_size=32, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative13(self):
        true_err_msg = re.escape('`batch_size` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative14(self):
        true_err_msg = re.escape('`batch_size` is wrong! Expected `{0}`, got `{1}`.'.format(type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size='3', max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative15(self):
        true_err_msg = re.escape('`batch_size` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=-3, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative16(self):
        true_err_msg = re.escape('`max_epochs` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, patience=5,
                verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative17(self):
        true_err_msg = re.escape('`max_epochs` is wrong! Expected `{0}`, got `{1}`.'.format(type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs='100',
                patience=5, verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative18(self):
        true_err_msg = re.escape('`max_epochs` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=-3,
                patience=5, verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative19(self):
        true_err_msg = re.escape('`patience` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100,
                verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative20(self):
        true_err_msg = re.escape('`patience` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100,
                patience='5', verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative21(self):
        true_err_msg = re.escape('`patience` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100,
                patience=-3, verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative22(self):
        true_err_msg = re.escape('`verbose` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                warm_start=False, random_seed=None, cache_dir=None, layer_level=3, hidden_layers=(300, 100)
            )

    def test_check_params_negative23(self):
        true_err_msg = re.escape('`random_seed` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                warm_start=False, verbose=True, cache_dir=None, layer_level=3, hidden_layers=(300, 100)
            )

    def test_check_params_negative24(self):
        true_err_msg = re.escape('`random_seed` is wrong! Expected `{0}`, got `{1}`.'.format(type(3), type(3.5)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                warm_start=False, verbose=True, random_seed=-3.5, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative25(self):
        true_err_msg = re.escape('`window_size` is too small for specified sampling frequency!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.00625, shift_size=0.01, batch_size=32, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative26(self):
        true_err_msg = re.escape('`cache_dir` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                warm_start=False, verbose=True, random_seed=None, layer_level=3, hidden_layers=(300, 100)
            )

    def test_check_params_negative27(self):
        true_err_msg = re.escape('`cache_dir` is wrong! Expected `{0}`, got `{1}`.'.format(type('3s'), type(3.5)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                warm_start=False, verbose=True, random_seed=None, cache_dir=3.5, layer_level=3, hidden_layers=(300, 100)
            )

    def test_check_params_negative28(self):
        true_err_msg = re.escape('`warm_start` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, random_seed=None, cache_dir=None, layer_level=3, hidden_layers=(300, 100)
            )

    def test_check_params_negative29(self):
        true_err_msg = re.escape('`layer_level` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, cache_dir=None, hidden_layers=(300, 100)
            )

    def test_check_params_negative30(self):
        true_err_msg = re.escape('`layer_level` is wrong! Expected `{0}`, got `{1}`.'.format(type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level='3',
                hidden_layers=(300, 100)
            )

    def test_check_params_negative31(self):
        true_err_msg = re.escape('`layer_level` is wrong! Expected a positive integer value in the range from 1 to 13, '
                                 'but 0 is less than 1.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=0,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative32(self):
        true_err_msg = re.escape('`layer_level` is wrong! Expected a positive integer value in the range from 1 to 13, '
                                 'but 15 is greater than 13.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=15,
                hidden_layers=(300, 100)
            )

    def test_check_params_negative33(self):
        true_err_msg = re.escape('`hidden_layers` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3
            )

    def test_check_params_negative34(self):
        true_err_msg = re.escape('`hidden_layers` is wrong! Expected a 1-D array, got 2-D one.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=np.array([[300, 100], [100, 50]], dtype=np.int32)
            )

    def test_check_params_negative35(self):
        true_err_msg = re.escape('`hidden_layers` is wrong! Expected `{0}`, got `{1}`.'.format(
            type((3, 4, 5)), type({1, 2, 3})))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers={300, 100}
            )

    def test_check_params_negative36(self):
        true_err_msg = re.escape('`hidden_layers` is wrong! 0 is wrong size of layer 3.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=[300, 100, 0]
            )

    def test_check_params_negative37(self):
        true_err_msg = re.escape('`hidden_layers` is wrong! {0} is wrong size of layer 2.'.format(150.5))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, cache_dir=None, layer_level=3,
                hidden_layers=[300, 150.5, 100]
            )

    def test_check_X_positive01(self):
        MobilenetRecognizer.check_X(
            [np.random.uniform(-1.0, 1.0, (n,)) for n in np.random.randint(300, 5000, (10,))], 'X_train'
        )
        self.assertTrue(True)

    def test_check_X_positive02(self):
        MobilenetRecognizer.check_X(
            np.random.uniform(-1.0, 1.0, (10, 5000)), 'X_train'
        )
        self.assertTrue(True)

    def test_check_X_negative01(self):
        true_err_msg = re.escape('{0} is wrong type for `X_train`!'.format(type({1, 2, 3})))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_X({10, 15, 20}, 'X_train')

    def test_check_X_negative02(self):
        true_err_msg = re.escape('`X_test` is wrong! Expected a 2-D array, but got a 3-D one!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_X(np.random.uniform(-1.0, 1.0, (10, 5000, 5)), 'X_test')

    def test_check_X_negative03(self):
        true_err_msg = re.escape('`X_val`[1] is wrong! Expected a {0}, but got a {1}!'.format(
            type(np.array([1, 2], dtype=np.float32)), type([1, 2])))
        X_val = [np.random.uniform(-1.0, 1.0, (n,)) for n in np.random.randint(300, 5000, size=(10,)).tolist()]
        X_val[1] = X_val[1].tolist()
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_X(X_val, 'X_val')

    def test_check_X_negative04(self):
        true_err_msg = re.escape('`X`[0] is wrong! Expected a 1-D array, but got a 2-D one!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_X(
                [np.random.uniform(-1.0, 1.0, (n, 35)) for n in np.random.randint(300, 5000, size=(10,)).tolist()],
                'X'
            )

    def test_check_Xy_positive01(self):
        res = MobilenetRecognizer.check_Xy(
            [np.random.uniform(-1.0, 1.0, (n,)) for n in np.random.randint(300, 5000, (10,))], 'X_train',
            ['sound' for _ in range(5)] + ['silence' for _ in range(3)] + [2 for _ in range(2)], 'y_train'
        )
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], dict)
        self.assertIsInstance(res[1], list)
        self.assertEqual(res[0], {'sound': 0, 'silence': 1, 2: 2})
        self.assertEqual(res[1], ['sound', 'silence', 2])

    def test_check_Xy_positive02(self):
        res = MobilenetRecognizer.check_Xy(
            [np.random.uniform(-1.0, 1.0, (n,)) for n in np.random.randint(300, 5000, (10,))], 'X_train',
            ['sound' for _ in range(5)] + ['silence' for _ in range(3)] + [-1 for _ in range(2)], 'y_train'
        )
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        self.assertIsInstance(res[0], dict)
        self.assertIsInstance(res[1], list)
        self.assertEqual(res[0], {'sound': 0, 'silence': 1})
        self.assertEqual(res[1], ['sound', 'silence'])

    def test_check_Xy_negative01(self):
        true_err_msg = re.escape('Size of `X_train` does not correspond to size of `y_train`. 10 != 11')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _, _ = MobilenetRecognizer.check_Xy(
                [np.random.uniform(-1.0, 1.0, (n,)) for n in np.random.randint(300, 5000, (10,))], 'X_train',
                ['sound' for _ in range(5)] + ['silence' for _ in range(3)] + [2 for _ in range(3)], 'y_train'
            )

    def test_check_Xy_negative02(self):
        true_err_msg = re.escape('There are too few classes in the `y_val`!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _, _ = MobilenetRecognizer.check_Xy(
                [np.random.uniform(-1.0, 1.0, (n,)) for n in np.random.randint(300, 5000, (10,))], 'X_val',
                ['sound' for _ in range(10)], 'y_val'
            )

    def test_check_Xy_negative03(self):
        true_err_msg = re.escape('`y_test` is wrong! Expected a 1-D array, but got a 2-D one!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _, _ = MobilenetRecognizer.check_Xy(
                [np.random.uniform(-1.0, 1.0, (n,)) for n in np.random.randint(300, 5000, (10,))], 'X_test',
                np.random.randint(0, 3, (10, 2)), 'y_test'
            )

    def test_check_Xy_negative04(self):
        true_err_msg = re.escape('-2 is inadmissible value for `y_train`[8]!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            _, _ = MobilenetRecognizer.check_Xy(
                [np.random.uniform(-1.0, 1.0, (n,)) for n in np.random.randint(300, 5000, (10,))], 'X_train',
                ['sound' for _ in range(5)] + ['silence' for _ in range(3)] + [-2 for _ in range(2)], 'y_train'
            )


class TestDTWRecognizer(unittest.TestCase):
    def tearDown(self):
        if hasattr(self, 'cls'):
            del self.cls
        if hasattr(self, 'another_cls'):
            del self.another_cls
        if hasattr(self, 'temp_file_name'):
            if os.path.isfile(self.temp_file_name):
                os.remove(self.temp_file_name)

    def test_creation(self):
        self.cls = DTWRecognizer()
        self.assertIsInstance(self.cls, DTWRecognizer)
        self.assertTrue(hasattr(self.cls, 'verbose'))
        self.assertTrue(hasattr(self.cls, 'warm_start'))
        self.assertTrue(hasattr(self.cls, 'window_size'))
        self.assertTrue(hasattr(self.cls, 'shift_size'))
        self.assertTrue(hasattr(self.cls, 'sampling_frequency'))
        self.assertTrue(hasattr(self.cls, 'k'))
        self.assertIsInstance(self.cls.k, int)
        self.assertIsInstance(self.cls.verbose, bool)
        self.assertIsInstance(self.cls.warm_start, bool)
        self.assertIsInstance(self.cls.sampling_frequency, int)
        self.assertIsInstance(self.cls.window_size, float)
        self.assertIsInstance(self.cls.shift_size, float)

    def test_check_params_positive(self):
        DTWRecognizer.check_params(
            sampling_frequency=16000, window_size=0.025, shift_size=0.01, k=7, verbose=False, warm_start=False
        )
        self.assertTrue(True)

    def test_check_params_negative01(self):
        true_err_msg = re.escape('`sampling_frequency` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            DTWRecognizer.check_params(
                window_size=0.025, shift_size=0.01, k=7, verbose=False, warm_start=False
            )

    def test_check_params_negative02(self):
        true_err_msg = re.escape('`sampling_frequency` is wrong! Expected `{0}`, got `{1}`.'.format(type(3), type(3.5)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            DTWRecognizer.check_params(
                sampling_frequency=3.5, window_size=0.025, shift_size=0.01, k=7, verbose=False, warm_start=False
            )

    def test_check_params_negative03(self):
        true_err_msg = re.escape('`sampling_frequency` is wrong! Expected a positive integer value, but -3 is not '
                                 'positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            DTWRecognizer.check_params(
                sampling_frequency=-3, window_size=0.025, shift_size=0.01, k=7, verbose=False, warm_start=False
            )

    def test_check_params_negative04(self):
        true_err_msg = re.escape('`sampling_frequency` is wrong! Minimal admissible value is 16000 Hz.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            DTWRecognizer.check_params(
                sampling_frequency=6000, window_size=0.025, shift_size=0.01, k=7, verbose=False, warm_start=False
            )

    def test_check_params_negative05(self):
        true_err_msg = re.escape('`window_size` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            DTWRecognizer.check_params(
                sampling_frequency=16000, shift_size=0.01, k=7, verbose=False, warm_start=False
            )

    def test_check_params_negative06(self):
        true_err_msg = re.escape('`window_size` is wrong! Expected a positive floating-point value, but {0} is not '
                                 'positive.'.format(-2.5))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            DTWRecognizer.check_params(
                sampling_frequency=16000, window_size=-2.5, shift_size=0.01, k=7, verbose=False, warm_start=False
            )

    def test_check_params_negative07(self):
        true_err_msg = re.escape('`window_size` is wrong! Expected `{0}`, got `{1}`.'.format(type(3.5), type(3)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            DTWRecognizer.check_params(
                sampling_frequency=16000, window_size=2, shift_size=0.01, k=7, verbose=False, warm_start=False
            )

    def test_check_params_negative08(self):
        true_err_msg = re.escape('`window_size` is wrong! {0:.6f} is too small value for `window_size`.'.format(1e-4))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            DTWRecognizer.check_params(
                sampling_frequency=16000, window_size=1e-4, shift_size=0.01, k=7, verbose=False, warm_start=False
            )

    def test_check_params_negative09(self):
        true_err_msg = re.escape('`shift_size` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            DTWRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, k=7, verbose=False, warm_start=False
            )

    def test_check_params_negative10(self):
        true_err_msg = re.escape('`shift_size` is wrong! Expected a positive floating-point value, '
                                 'but {0} is not positive.'.format(-0.01))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            DTWRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=-0.01, k=7, verbose=False, warm_start=False
            )

    def test_check_params_negative11(self):
        true_err_msg = re.escape('`shift_size` is wrong! Expected `{0}`, got `{1}`.'.format(type(3.5), type(3)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            DTWRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=1, k=7, verbose=False, warm_start=False
            )

    def test_check_params_negative12(self):
        true_err_msg = re.escape('`shift_size` is wrong! {0:.6f} is too small value for `shift_size`.'.format(1e-5))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            DTWRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=1e-5, k=7, verbose=False, warm_start=False
            )

    def test_check_params_negative13(self):
        true_err_msg = re.escape('`k` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            DTWRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, verbose=False, warm_start=False
            )

    def test_check_params_negative14(self):
        true_err_msg = re.escape('`k` is wrong! Expected `{0}`, got `{1}`.'.format(type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            DTWRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, k='3', verbose=False, warm_start=False
            )

    def test_check_params_negative15(self):
        true_err_msg = re.escape('`k` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            DTWRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, k=-3, verbose=False, warm_start=False
            )

    def test_check_params_negative16(self):
        true_err_msg = re.escape('`verbose` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            DTWRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, k=7, warm_start=False
            )

    def test_check_params_negative17(self):
        true_err_msg = re.escape('`warm_start` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            DTWRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, k=7, verbose=False
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
