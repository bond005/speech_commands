import copy
import gc
import os
import pickle
import re
import sys
import tempfile
import unittest

import keras
import librosa
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import f1_score


try:
    from speech_commands.speech_commands import MobilenetRecognizer, DTWRecognizer
    from speech_commands.utils import read_tensorflow_speech_recognition_challenge
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from speech_commands.speech_commands import MobilenetRecognizer, DTWRecognizer
    from speech_commands.utils import read_tensorflow_speech_recognition_challenge


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
        self.cls = MobilenetRecognizer(hidden_layers=(300, 100))
        self.assertIsInstance(self.cls, MobilenetRecognizer)
        self.assertTrue(hasattr(self.cls, 'batch_size'))
        self.assertTrue(hasattr(self.cls, 'max_epochs'))
        self.assertTrue(hasattr(self.cls, 'patience'))
        self.assertTrue(hasattr(self.cls, 'verbose'))
        self.assertTrue(hasattr(self.cls, 'warm_start'))
        self.assertTrue(hasattr(self.cls, 'window_size'))
        self.assertTrue(hasattr(self.cls, 'shift_size'))
        self.assertTrue(hasattr(self.cls, 'sampling_frequency'))
        self.assertTrue(hasattr(self.cls, 'hidden_layers'))
        self.assertTrue(hasattr(self.cls, 'layer_level'))
        self.assertTrue(hasattr(self.cls, 'use_augmentation'))
        self.assertIsInstance(self.cls.batch_size, int)
        self.assertIsInstance(self.cls.max_epochs, int)
        self.assertIsInstance(self.cls.patience, int)
        self.assertIsInstance(self.cls.verbose, bool)
        self.assertIsInstance(self.cls.warm_start, bool)
        self.assertIsInstance(self.cls.sampling_frequency, int)
        self.assertIsInstance(self.cls.window_size, float)
        self.assertIsInstance(self.cls.shift_size, float)
        self.assertIsInstance(self.cls.hidden_layers, tuple)
        self.assertIsInstance(self.cls.layer_level, int)
        self.assertIsInstance(self.cls.use_augmentation, bool)
        self.assertIsNone(self.cls.random_seed)

    def test_check_params_positive(self):
        MobilenetRecognizer.check_params(
            sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
            verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
            use_augmentation=False, l2_reg=0.001
        )
        self.assertTrue(True)

    def test_check_params_negative01(self):
        true_err_msg = re.escape('`sampling_frequency` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5, hidden_layers=(300, 100),
                verbose=False, warm_start=False, random_seed=None, layer_level=3, use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative02(self):
        true_err_msg = re.escape('`sampling_frequency` is wrong! Expected `{0}`, got `{1}`.'.format(type(3), type(3.5)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=3.5, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative03(self):
        true_err_msg = re.escape('`sampling_frequency` is wrong! Expected a positive integer value, but -3 is not '
                                 'positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=-3, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative04(self):
        true_err_msg = re.escape('`sampling_frequency` is wrong! Minimal admissible value is 16000 Hz.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=6000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative05(self):
        true_err_msg = re.escape('`window_size` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative06(self):
        true_err_msg = re.escape('`window_size` is wrong! Expected a positive floating-point value, but {0} is not '
                                 'positive.'.format(-2.5))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=-2.5, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative07(self):
        true_err_msg = re.escape('`window_size` is wrong! Expected `{0}`, got `{1}`.'.format(type(3.5), type(3)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=2, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative08(self):
        true_err_msg = re.escape('`window_size` is wrong! {0:.6f} is too small value for `window_size`.'.format(1e-4))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=1e-4, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative09(self):
        true_err_msg = re.escape('`shift_size` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative10(self):
        true_err_msg = re.escape('`shift_size` is wrong! Expected a positive floating-point value, '
                                 'but {0} is not positive.'.format(-0.01))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=-0.01, batch_size=32, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative11(self):
        true_err_msg = re.escape('`shift_size` is wrong! Expected `{0}`, got `{1}`.'.format(type(3.5), type(3)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=1, batch_size=32, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative12(self):
        true_err_msg = re.escape('`shift_size` is wrong! {0:.6f} is too small value for `shift_size`.'.format(1e-5))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=1e-5, batch_size=32, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                l2_reg=0.001
            )

    def test_check_params_negative13(self):
        true_err_msg = re.escape('`batch_size` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative14(self):
        true_err_msg = re.escape('`batch_size` is wrong! Expected `{0}`, got `{1}`.'.format(type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size='3', max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative15(self):
        true_err_msg = re.escape('`batch_size` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=-3, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative16(self):
        true_err_msg = re.escape('`max_epochs` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, patience=5,
                verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative17(self):
        true_err_msg = re.escape('`max_epochs` is wrong! Expected `{0}`, got `{1}`.'.format(type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs='100',
                patience=5, verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative18(self):
        true_err_msg = re.escape('`max_epochs` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=-3,
                patience=5, verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative19(self):
        true_err_msg = re.escape('`patience` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100,
                verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative20(self):
        true_err_msg = re.escape('`patience` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100,
                patience='5', verbose=False, warm_start=False, random_seed=None, layer_level=3,
                hidden_layers=(300, 100), use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative21(self):
        true_err_msg = re.escape('`patience` is wrong! Expected a positive integer value, but -3 is not positive.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100,
                patience=-3, verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative22(self):
        true_err_msg = re.escape('`verbose` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100), use_augmentation=False,
                l2_reg=0.001
            )

    def test_check_params_negative23(self):
        true_err_msg = re.escape('`random_seed` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                warm_start=False, verbose=True, layer_level=3, hidden_layers=(300, 100), use_augmentation=False,
                l2_reg=0.001
            )

    def test_check_params_negative24(self):
        true_err_msg = re.escape('`random_seed` is wrong! Expected `{0}`, got `{1}`.'.format(type(3), type(3.5)))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                warm_start=False, verbose=True, random_seed=-3.5, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative25(self):
        true_err_msg = re.escape('`window_size` is too small for specified sampling frequency!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.00625, shift_size=0.01, batch_size=32, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative28(self):
        true_err_msg = re.escape('`warm_start` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, random_seed=None, layer_level=3, hidden_layers=(300, 100), use_augmentation=False,
                l2_reg=0.001
            )

    def test_check_params_negative29(self):
        true_err_msg = re.escape('`layer_level` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, max_epochs=100, patience=5, l2_reg=0.001,
                verbose=False, warm_start=False, random_seed=None, hidden_layers=(300, 100), use_augmentation=False
            )

    def test_check_params_negative30(self):
        true_err_msg = re.escape('`layer_level` is wrong! Expected `{0}`, got `{1}`.'.format(type(3), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, layer_level='3',
                hidden_layers=(300, 100), use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative31(self):
        true_err_msg = re.escape('`layer_level` is wrong! Expected a positive integer value in the range from 1 to 13, '
                                 'but 0 is less than 1.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, layer_level=0, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative32(self):
        true_err_msg = re.escape('`layer_level` is wrong! Expected a positive integer value in the range from 1 to 13, '
                                 'but 15 is greater than 13.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, layer_level=15, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative33(self):
        true_err_msg = re.escape('`hidden_layers` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, layer_level=3, use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative34(self):
        true_err_msg = re.escape('`hidden_layers` is wrong! Expected a 1-D array, got 2-D one.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, layer_level=3, use_augmentation=False,
                hidden_layers=np.array([[300, 100], [100, 50]], dtype=np.int32), l2_reg=0.001
            )

    def test_check_params_negative35(self):
        true_err_msg = re.escape('`hidden_layers` is wrong! Expected `{0}`, got `{1}`.'.format(
            type((3, 4, 5)), type({1, 2, 3})))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers={300, 100},
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative36(self):
        true_err_msg = re.escape('`hidden_layers` is wrong! 0 is wrong size of layer 3.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=[300, 100, 0],
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative37(self):
        true_err_msg = re.escape('`hidden_layers` is wrong! {0} is wrong size of layer 2.'.format(150.5))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=[300, 150.5, 100],
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative38(self):
        true_err_msg = re.escape('`hidden_layers` is wrong! It is empty.')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=[],
                use_augmentation=False, l2_reg=0.001
            )

    def test_check_params_negative39(self):
        true_err_msg = re.escape('`use_augmentation` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, random_seed=None, layer_level=3, hidden_layers=(300, 100), warm_start=False, l2_reg=0.001
            )

    def test_check_params_negative40(self):
        true_err_msg = re.escape('`use_augmentation` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(True), type('True')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100, patience=5,
                verbose=False, random_seed=None, layer_level=3, hidden_layers=(300, 100), warm_start=False,
                use_augmentation='False', l2_reg=0.001
            )

    def test_check_params_negative41(self):
        true_err_msg = re.escape('`l2_reg` is not specified!')
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False
            )

    def test_check_params_negative42(self):
        true_err_msg = re.escape('`l2_reg` is wrong! Expected `{0}`, got `{1}`.'.format(
            type(3.2), type('3')))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, layer_level=3,
                hidden_layers=(300, 100), use_augmentation=False, l2_reg='0.001'
            )

    def test_check_params_negative43(self):
        true_err_msg = re.escape('`l2_reg` is wrong! Expected a positive floating-point value, but {0} is not '
                                 'positive.'.format(0.0))
        with self.assertRaisesRegex(ValueError, true_err_msg):
            MobilenetRecognizer.check_params(
                sampling_frequency=16000, window_size=0.025, shift_size=0.01, batch_size=32, max_epochs=100,
                patience=5, verbose=False, warm_start=False, random_seed=None, layer_level=3, hidden_layers=(300, 100),
                use_augmentation=False, l2_reg=0.0
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

    def test_select_labeled_samples(self):
        y = [
            -1, -1, '-1', -1, -1, 'down', 'down', 'down', 'eight', 'eight', 'five', 'four', 'four', 'go', 'go', -1, -1,
            'left', 'left', 'left', -1, -1, 'nine', 'nine', 'no', 'no', 'off', 'on', 'on', 'on', 'on', 'one', 'one',
            'one', 'one', 'one', 'right', 'right', 'seven', 'seven', -1, 'six', 'stop', 'stop', 'three', '-1', 'two',
            'two', 'up', 'up', 'up', -1, 'yes', 'yes', 'yes', 'zero', 'zero'
        ]
        true_indices = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                        34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56)
        self.assertEqual(true_indices, MobilenetRecognizer.select_labeled_samples(y))

    def test_sound_to_melspectrogram(self):
        sound_name = os.path.join(os.path.dirname(__file__), 'testdata', 'tensorflow_data', 'audio',
                                  '_background_noise_', 'doing_the_dishes.wav')
        sound, fs = librosa.core.load(path=sound_name, sr=None, mono=True)
        self.cls = MobilenetRecognizer(hidden_layers=(300, 100), sampling_frequency=fs)
        self.cls.update_triangle_filters()
        spectrogram = MobilenetRecognizer.sound_to_melspectrogram(sound, self.cls.window_size, self.cls.shift_size, fs,
                                                                  self.cls.melfb_)
        self.assertIsInstance(spectrogram, np.ndarray)
        self.assertEqual(2, len(spectrogram.shape))
        self.assertGreater(spectrogram.shape[0], 1)
        self.assertNotEqual(spectrogram.shape[0], MobilenetRecognizer.IMAGESIZE[0])
        self.assertEqual(spectrogram.shape[1], MobilenetRecognizer.IMAGESIZE[1] // 2)
        self.assertGreaterEqual(spectrogram.min(), 0.0)
        self.assertGreater(spectrogram.max(), spectrogram.min())

    def test_normalize_melspectrogram(self):
        sound_name = os.path.join(os.path.dirname(__file__), 'testdata', 'tensorflow_data', 'audio',
                                  '_background_noise_', 'doing_the_dishes.wav')
        sound, fs = librosa.core.load(path=sound_name, sr=None, mono=True)
        self.cls = MobilenetRecognizer(hidden_layers=(300, 100), sampling_frequency=fs)
        self.cls.update_triangle_filters()
        spectrogram = MobilenetRecognizer.sound_to_melspectrogram(sound, self.cls.window_size, self.cls.shift_size, fs,
                                                                  self.cls.melfb_)
        normalized = MobilenetRecognizer.normalize_melspectrogram(spectrogram)
        del spectrogram
        self.assertIsInstance(normalized, np.ndarray)
        self.assertEqual(2, len(normalized.shape))
        self.assertEqual(normalized.shape[0], MobilenetRecognizer.IMAGESIZE[0])
        self.assertEqual(normalized.shape[1], MobilenetRecognizer.IMAGESIZE[1] // 2)
        self.assertAlmostEqual(normalized.min(), 0.0, places=5)
        self.assertAlmostEqual(normalized.max(), 1.0, places=5)
        values = sorted(normalized.reshape((normalized.shape[0] * normalized.shape[1],)).tolist())
        self.assertGreater(values[len(values) // 2], 1e-1)
        self.assertLess(values[len(values) // 2], 1.0 - 1e-1)

    def test_spectrograms_to_images(self):
        sound_name = os.path.join(os.path.dirname(__file__), 'testdata', 'tensorflow_data', 'audio',
                                  '_background_noise_', 'doing_the_dishes.wav')
        sound, fs = librosa.core.load(path=sound_name, sr=None, mono=True)
        self.cls = MobilenetRecognizer(hidden_layers=(300, 100), sampling_frequency=fs)
        self.cls.update_triangle_filters()
        spectrogram = MobilenetRecognizer.sound_to_melspectrogram(sound, self.cls.window_size, self.cls.shift_size, fs,
                                                                  self.cls.melfb_)
        normalized = np.expand_dims(MobilenetRecognizer.normalize_melspectrogram(spectrogram), axis=0)
        del spectrogram
        images = MobilenetRecognizer.spectrograms_to_images(normalized)
        del normalized
        self.assertIsInstance(images, np.ndarray)
        self.assertEqual((1, MobilenetRecognizer.IMAGESIZE[0], MobilenetRecognizer.IMAGESIZE[1], 3), images.shape)
        self.assertAlmostEqual(images[0, :, :, 0].min(), 0.0, places=5)
        self.assertAlmostEqual(images[0, :, :, 0].max(), 255.0, places=5)
        self.assertAlmostEqual(images[0, :, :, 1].min(), 0.0, places=5)
        self.assertAlmostEqual(images[0, :, :, 1].max(), 255.0, places=5)
        self.assertAlmostEqual(images[0, :, :, 2].min(), 0.0, places=5)
        self.assertAlmostEqual(images[0, :, :, 2].max(), 255.0, places=5)

    def test_class_label_to_vector_positive01(self):
        classes_dict = {'a': 0, 'b': 1, 'c': 2}
        class_label = 'b'
        true_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        calc_vector = MobilenetRecognizer.class_label_to_vector(class_label, classes_dict)
        self.assertIsInstance(calc_vector, np.ndarray)
        self.assertEqual(true_vector.shape, calc_vector.shape)
        for sample_idx in range(true_vector.shape[0]):
            self.assertAlmostEqual(true_vector[sample_idx], calc_vector[sample_idx], places=5)

    def test_class_label_to_vector_positive02(self):
        classes_dict = {'a': 0, 'b': 1, 'c': 2}
        class_label = 'e'
        true_vector = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        calc_vector = MobilenetRecognizer.class_label_to_vector(class_label, classes_dict)
        self.assertIsInstance(calc_vector, np.ndarray)
        self.assertEqual(true_vector.shape, calc_vector.shape)
        for sample_idx in range(true_vector.shape[0]):
            self.assertAlmostEqual(true_vector[sample_idx], calc_vector[sample_idx], places=5)

    def test_class_label_to_vector_positive03(self):
        classes_dict = {'a': 0, 'b': 1, 'c': 2}
        class_label = -1
        true_vector = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        calc_vector = MobilenetRecognizer.class_label_to_vector(class_label, classes_dict)
        self.assertIsInstance(calc_vector, np.ndarray)
        self.assertEqual(true_vector.shape, calc_vector.shape)
        for sample_idx in range(true_vector.shape[0]):
            self.assertAlmostEqual(true_vector[sample_idx], calc_vector[sample_idx], places=5)

    def test_fit_predict_positive01(self):
        self.cls = MobilenetRecognizer(verbose=True)
        data_dir = os.path.join(os.path.dirname(__file__), 'testdata', 'tensorflow_data')
        data_for_training, data_for_validation, data_for_testing, background_sounds, fs = \
            read_tensorflow_speech_recognition_challenge(data_dir)
        res = self.cls.fit(X=data_for_training[0], y=data_for_training[1],
                           validation_data=(data_for_validation[0], data_for_validation[1]),
                           background=background_sounds[0])
        true_set_of_classes = {'down', 'eight', 'five', 'four', 'go', 'left', 'nine', 'no', 'off', 'on', 'one', 'right',
                               'seven', 'six', 'stop', 'three', 'two', 'up', 'yes', 'zero'}
        self.assertIsInstance(res, MobilenetRecognizer)
        self.assertTrue(hasattr(res, 'melfb_'))
        self.assertTrue(hasattr(res, 'recognizer_'))
        self.assertTrue(hasattr(res, 'classes_'))
        self.assertTrue(hasattr(res, 'classes_reverse_'))
        self.assertTrue(hasattr(res, 'threshold_'))
        self.assertTrue(hasattr(res, 'min_amplitude_'))
        self.assertTrue(hasattr(res, 'max_amplitude_'))
        self.assertIsInstance(res.melfb_, np.ndarray)
        self.assertEqual(len(res.melfb_.shape), 2)
        self.assertIsInstance(res.recognizer_, keras.models.Model)
        self.assertIsInstance(res.classes_, dict)
        self.assertIsInstance(res.classes_reverse_, list)
        self.assertIsInstance(res.threshold_, float)
        self.assertIsInstance(res.min_amplitude_, float)
        self.assertIsInstance(res.max_amplitude_, float)
        self.assertGreaterEqual(res.min_amplitude_, 0.0)
        self.assertLess(res.min_amplitude_, res.max_amplitude_)
        self.assertGreater(res.threshold_, 0.0)
        self.assertLess(res.threshold_, 1.0)
        self.assertEqual(set(res.classes_.keys()), true_set_of_classes)
        self.assertEqual(len(res.classes_), len(res.classes_reverse_))
        self.assertEqual(len(res.classes_), len(true_set_of_classes))
        class_indices = []
        for class_name in true_set_of_classes:
            self.assertEqual(res.classes_reverse_[res.classes_[class_name]], class_name)
            class_indices.append(res.classes_[class_name])
        self.assertEqual(np.min(class_indices), 0)
        self.assertEqual(np.max(class_indices), len(true_set_of_classes) - 1)
        self.assertEqual(len(class_indices), len(set(class_indices)))
        probabilities = res.predict_proba(data_for_testing[0])
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(2, len(probabilities.shape))
        self.assertEqual(probabilities.shape[0], len(data_for_testing[0]))
        self.assertEqual(probabilities.shape[1], len(true_set_of_classes))
        y_pred = res.predict(data_for_testing[0])
        self.assertIsInstance(y_pred, list)
        for sample_idx in range(len(y_pred)):
            if probabilities[sample_idx].max() >= res.threshold_:
                self.assertEqual(y_pred[sample_idx], res.classes_reverse_[probabilities[sample_idx].argmax()])
            else:
                self.assertEqual(y_pred[sample_idx], -1)
        score = res.score(data_for_testing[0], data_for_testing[1])
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertAlmostEqual(score, f1_score(data_for_testing[1], y_pred, average='macro'), places=4)

    def test_fit_predict_positive02(self):
        self.cls = MobilenetRecognizer(verbose=True)
        data_dir = os.path.join(os.path.dirname(__file__), 'testdata', 'tensorflow_data')
        data_for_training, data_for_validation, data_for_testing, background_sounds, fs = \
            read_tensorflow_speech_recognition_challenge(data_dir)
        res = self.cls.fit(X=data_for_training[0], y=data_for_training[1],
                           validation_data=(data_for_validation[0], data_for_validation[1]),
                           background=background_sounds[0])
        res.warm_start = True
        res.fit(X=data_for_validation[0], y=data_for_validation[1],
                validation_data=(data_for_testing[0], data_for_testing[1]))
        true_set_of_classes = {'down', 'eight', 'five', 'four', 'go', 'left', 'nine', 'no', 'off', 'on', 'one', 'right',
                               'seven', 'six', 'stop', 'three', 'two', 'up', 'yes', 'zero'}
        self.assertIsInstance(res, MobilenetRecognizer)
        self.assertTrue(hasattr(res, 'melfb_'))
        self.assertTrue(hasattr(res, 'recognizer_'))
        self.assertTrue(hasattr(res, 'classes_'))
        self.assertTrue(hasattr(res, 'classes_reverse_'))
        self.assertTrue(hasattr(res, 'threshold_'))
        self.assertTrue(hasattr(res, 'min_amplitude_'))
        self.assertTrue(hasattr(res, 'max_amplitude_'))
        self.assertIsInstance(res.melfb_, np.ndarray)
        self.assertEqual(len(res.melfb_.shape), 2)
        self.assertIsInstance(res.recognizer_, keras.models.Model)
        self.assertIsInstance(res.classes_, dict)
        self.assertIsInstance(res.classes_reverse_, list)
        self.assertIsInstance(res.threshold_, float)
        self.assertIsInstance(res.min_amplitude_, float)
        self.assertIsInstance(res.max_amplitude_, float)
        self.assertGreaterEqual(res.min_amplitude_, 0.0)
        self.assertLess(res.min_amplitude_, res.max_amplitude_)
        self.assertGreater(res.threshold_, 0.0)
        self.assertLess(res.threshold_, 1.0)
        self.assertEqual(set(res.classes_.keys()), true_set_of_classes)
        self.assertEqual(len(res.classes_), len(res.classes_reverse_))
        self.assertEqual(len(res.classes_), len(true_set_of_classes))
        for class_name in true_set_of_classes:
            self.assertEqual(res.classes_reverse_[res.classes_[class_name]], class_name)
        probabilities = res.predict_proba(data_for_testing[0])
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(2, len(probabilities.shape))
        self.assertEqual(probabilities.shape[0], len(data_for_testing[0]))
        self.assertEqual(probabilities.shape[1], len(true_set_of_classes))
        y_pred = res.predict(data_for_testing[0])
        self.assertIsInstance(y_pred, list)
        for sample_idx in range(len(y_pred)):
            if probabilities[sample_idx].max() >= res.threshold_:
                self.assertEqual(y_pred[sample_idx], res.classes_reverse_[probabilities[sample_idx].argmax()])
            else:
                self.assertEqual(y_pred[sample_idx], -1)
        score = res.score(data_for_testing[0], data_for_testing[1])
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertAlmostEqual(score, f1_score(data_for_testing[1], y_pred, average='macro'), places=4)

    def test_fit_negative01(self):
        self.cls = MobilenetRecognizer(verbose=True, warm_start=True)
        data_dir = os.path.join(os.path.dirname(__file__), 'testdata', 'tensorflow_data')
        data_for_training, data_for_validation, data_for_testing, background_sounds, fs = \
            read_tensorflow_speech_recognition_challenge(data_dir)
        with self.assertRaises(NotFittedError):
            _ = self.cls.fit(X=data_for_training[0], y=data_for_training[1],
                             validation_data=(data_for_validation[0], data_for_validation[1]),
                             background=background_sounds[0])

    def test_predict_negative01(self):
        self.cls = MobilenetRecognizer(verbose=True)
        data_dir = os.path.join(os.path.dirname(__file__), 'testdata', 'tensorflow_data')
        data_for_training, data_for_validation, data_for_testing, background_sounds, fs = \
            read_tensorflow_speech_recognition_challenge(data_dir)
        with self.assertRaises(NotFittedError):
            _ = self.cls.predict(data_for_testing[0])

    def test_serialize_positive01(self):
        self.cls = MobilenetRecognizer(verbose=True, warm_start=True)
        self.temp_file_name = tempfile.NamedTemporaryFile(mode='w', suffix='sound_recognizer.pkl').name
        with open(self.temp_file_name, 'wb') as fp:
            pickle.dump(self.cls, fp)
        with open(self.temp_file_name, 'rb') as fp:
            self.another_cls = pickle.load(fp)
        self.assertIsInstance(self.another_cls, MobilenetRecognizer)
        self.assertTrue(hasattr(self.another_cls, 'batch_size'))
        self.assertTrue(hasattr(self.another_cls, 'max_epochs'))
        self.assertTrue(hasattr(self.another_cls, 'patience'))
        self.assertTrue(hasattr(self.another_cls, 'verbose'))
        self.assertTrue(hasattr(self.another_cls, 'warm_start'))
        self.assertTrue(hasattr(self.another_cls, 'window_size'))
        self.assertTrue(hasattr(self.another_cls, 'shift_size'))
        self.assertTrue(hasattr(self.another_cls, 'sampling_frequency'))
        self.assertTrue(hasattr(self.another_cls, 'hidden_layers'))
        self.assertTrue(hasattr(self.another_cls, 'layer_level'))
        self.assertTrue(hasattr(self.another_cls, 'use_augmentation'))
        self.assertIsInstance(self.another_cls.batch_size, int)
        self.assertIsInstance(self.another_cls.max_epochs, int)
        self.assertIsInstance(self.another_cls.patience, int)
        self.assertIsInstance(self.another_cls.verbose, bool)
        self.assertIsInstance(self.another_cls.warm_start, bool)
        self.assertIsInstance(self.another_cls.use_augmentation, bool)
        self.assertIsInstance(self.another_cls.sampling_frequency, int)
        self.assertIsInstance(self.another_cls.window_size, float)
        self.assertIsInstance(self.another_cls.shift_size, float)
        self.assertIsInstance(self.another_cls.hidden_layers, tuple)
        self.assertIsInstance(self.another_cls.layer_level, int)
        self.assertIsNone(self.another_cls.random_seed)
        self.assertEqual(self.cls.batch_size, self.another_cls.batch_size)
        self.assertEqual(self.cls.max_epochs, self.another_cls.max_epochs)
        self.assertEqual(self.cls.patience, self.another_cls.patience)
        self.assertAlmostEqual(self.cls.window_size, self.another_cls.window_size)
        self.assertAlmostEqual(self.cls.shift_size, self.another_cls.shift_size)
        self.assertEqual(self.cls.sampling_frequency, self.another_cls.sampling_frequency)
        self.assertEqual(self.cls.hidden_layers, self.another_cls.hidden_layers)
        self.assertEqual(self.cls.layer_level, self.another_cls.layer_level)
        self.assertTrue(self.another_cls.warm_start)
        self.assertFalse(self.another_cls.use_augmentation)
        self.assertTrue(self.another_cls.verbose)

    def test_serialize_positive02(self):
        true_set_of_classes = {'down', 'eight', 'five', 'four', 'go', 'left', 'nine', 'no', 'off', 'on', 'one', 'right',
                               'seven', 'six', 'stop', 'three', 'two', 'up', 'yes', 'zero'}
        self.cls = MobilenetRecognizer(verbose=True)
        data_dir = os.path.join(os.path.dirname(__file__), 'testdata', 'tensorflow_data')
        data_for_training, data_for_validation, data_for_testing, background_sounds, fs = \
            read_tensorflow_speech_recognition_challenge(data_dir)
        self.cls.fit(X=data_for_training[0], y=data_for_training[1],
                     validation_data=(data_for_validation[0], data_for_validation[1]),
                     background=background_sounds[0])
        y_pred = self.cls.predict(data_for_testing[0])
        self.temp_file_name = tempfile.NamedTemporaryFile(mode='w', suffix='sound_recognizer.pkl').name
        with open(self.temp_file_name, 'wb') as fp:
            pickle.dump(self.cls, fp)
        del self.cls
        with open(self.temp_file_name, 'rb') as fp:
            self.another_cls = pickle.load(fp)
        self.assertIsInstance(self.another_cls, MobilenetRecognizer)
        self.assertTrue(hasattr(self.another_cls, 'batch_size'))
        self.assertTrue(hasattr(self.another_cls, 'max_epochs'))
        self.assertTrue(hasattr(self.another_cls, 'patience'))
        self.assertTrue(hasattr(self.another_cls, 'verbose'))
        self.assertTrue(hasattr(self.another_cls, 'warm_start'))
        self.assertTrue(hasattr(self.another_cls, 'use_augmentation'))
        self.assertTrue(hasattr(self.another_cls, 'window_size'))
        self.assertTrue(hasattr(self.another_cls, 'shift_size'))
        self.assertTrue(hasattr(self.another_cls, 'sampling_frequency'))
        self.assertTrue(hasattr(self.another_cls, 'hidden_layers'))
        self.assertTrue(hasattr(self.another_cls, 'layer_level'))
        self.assertIsInstance(self.another_cls.batch_size, int)
        self.assertIsInstance(self.another_cls.max_epochs, int)
        self.assertIsInstance(self.another_cls.patience, int)
        self.assertIsInstance(self.another_cls.verbose, bool)
        self.assertIsInstance(self.another_cls.warm_start, bool)
        self.assertIsInstance(self.another_cls.use_augmentation, bool)
        self.assertIsInstance(self.another_cls.sampling_frequency, int)
        self.assertIsInstance(self.another_cls.window_size, float)
        self.assertIsInstance(self.another_cls.shift_size, float)
        self.assertIsInstance(self.another_cls.hidden_layers, tuple)
        self.assertIsInstance(self.another_cls.layer_level, int)
        self.assertIsNotNone(self.another_cls.random_seed)
        self.assertTrue(hasattr(self.another_cls, 'recognizer_'))
        self.assertTrue(hasattr(self.another_cls, 'classes_'))
        self.assertTrue(hasattr(self.another_cls, 'classes_reverse_'))
        self.assertTrue(hasattr(self.another_cls, 'threshold_'))
        self.assertTrue(hasattr(self.another_cls, 'min_amplitude_'))
        self.assertTrue(hasattr(self.another_cls, 'max_amplitude_'))
        self.assertIsInstance(self.another_cls.recognizer_, keras.models.Model)
        self.assertIsInstance(self.another_cls.classes_, dict)
        self.assertIsInstance(self.another_cls.classes_reverse_, list)
        self.assertIsInstance(self.another_cls.threshold_, float)
        self.assertIsInstance(self.another_cls.min_amplitude_, float)
        self.assertIsInstance(self.another_cls.max_amplitude_, float)
        self.assertGreaterEqual(self.another_cls.min_amplitude_, 0.0)
        self.assertLess(self.another_cls.min_amplitude_, self.another_cls.max_amplitude_)
        self.assertGreater(self.another_cls.threshold_, 0.0)
        self.assertLess(self.another_cls.threshold_, 1.0)
        self.assertEqual(set(self.another_cls.classes_.keys()), true_set_of_classes)
        self.assertEqual(len(self.another_cls.classes_), len(self.another_cls.classes_reverse_))
        self.assertEqual(len(self.another_cls.classes_), len(true_set_of_classes))
        for class_name in true_set_of_classes:
            self.assertEqual(self.another_cls.classes_reverse_[self.another_cls.classes_[class_name]], class_name)
        self.assertEqual(y_pred, self.another_cls.predict(data_for_testing[0]))

    def test_strip_sound_positive01(self):
        sound = np.concatenate((np.random.uniform(1e-3, 1.0, (100,)), np.random.uniform(-1.0, -1e-3, (100,))))
        np.random.shuffle(sound)
        sound = np.concatenate((sound, np.zeros((50,), dtype=sound.dtype)))
        true_length = 200
        self.assertEqual(MobilenetRecognizer.strip_sound(sound), true_length)

    def test_strip_sound_positive02(self):
        sound = np.zeros((50,), dtype=np.float32)
        true_length = 0
        self.assertEqual(MobilenetRecognizer.strip_sound(sound), true_length)

    def test_strip_sound_positive03(self):
        sound = np.concatenate((np.random.uniform(1e-3, 1.0, (100,)), np.random.uniform(-1.0, -1e-3, (100,))))
        np.random.shuffle(sound)
        another_sound = np.concatenate((np.random.uniform(1e-3, 1.0, (10,)), np.random.uniform(-1.0, -1e-3, (10,))))
        np.random.shuffle(another_sound)
        sound = np.concatenate((sound, np.zeros((50,), dtype=sound.dtype), another_sound,
                                np.zeros((15,), dtype=sound.dtype)))
        true_length = 270
        self.assertEqual(MobilenetRecognizer.strip_sound(sound), true_length)


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
