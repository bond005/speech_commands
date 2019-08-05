import os
import sys
import unittest

import librosa
import numpy as np

try:
    from speech_commands.utils import parse_layers, read_custom_data, read_tensorflow_speech_recognition_challenge
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from speech_commands.utils import parse_layers, read_custom_data, read_tensorflow_speech_recognition_challenge


class TestUtils(unittest.TestCase):
    def test_parse_layers_positive01(self):
        s = '300-100'
        true_layers = (300, 100)
        self.assertEqual(true_layers, parse_layers(s))

    def test_parse_layers_positive02(self):
        s = '200'
        true_layers = (200,)
        self.assertEqual(true_layers, parse_layers(s))

    def test_parse_layers_positive03(self):
        s = ' '
        self.assertEqual(tuple(), parse_layers(s))

    def test_parse_layers_positive04(self):
        s = None
        self.assertEqual(tuple(), parse_layers(s))

    def test_parse_layers_positive05(self):
        s = '300--100-'
        true_layers = (300, 100)
        self.assertEqual(true_layers, parse_layers(s))

    def test_read_tensorflow_speech_recognition_challenge(self):
        true_labels_for_training = [
            -1, -1, -1, -1, 'down', 'eight', 'five', 'four', 'go', -1, -1, 'left', -1, 'nine', 'no', 'off', 'on', 'one',
            'right', 'seven', -1, 'six', 'stop', 'three', -1, 'two', 'up', -1, 'yes', 'zero'
        ]
        true_names_for_training = [
            'bed/eb67fcbc_nohash_0.wav', 'bird/f953e1af_nohash_0.wav', 'cat/3bc21161_nohash_0.wav',
            'dog/da1d320c_nohash_1.wav', 'down/74551073_nohash_1.wav', 'eight/3d53244b_nohash_3.wav',
            'five/a1533da4_nohash_2.wav', 'four/a996cf66_nohash_0.wav', 'go/80f056c0_nohash_0.wav',
            'happy/fc28c8d8_nohash_0.wav', 'house/6f2c6f7e_nohash_1.wav', 'left/1eddce1d_nohash_1.wav',
            'marvin/cae62f38_nohash_2.wav', 'nine/e41a903b_nohash_3.wav', 'no/190821dc_nohash_4.wav',
            'off/b7a6f709_nohash_0.wav', 'on/229978fd_nohash_4.wav', 'one/8830e17f_nohash_1.wav',
            'right/7c1d8533_nohash_3.wav', 'seven/1816b768_nohash_0.wav', 'sheila/1df483c0_nohash_0.wav',
            'six/5588c7e6_nohash_0.wav', 'stop/eb3f7d82_nohash_5.wav', 'three/e900b652_nohash_0.wav',
            'tree/638685f2_nohash_0.wav', 'two/a3502f15_nohash_0.wav', 'up/caedb73a_nohash_1.wav',
            'wow/1e4064b8_nohash_0.wav', 'yes/05b2db80_nohash_0.wav', 'zero/32561e9e_nohash_0.wav'
        ]
        true_labels_for_validation = [
            -1, -1, -1, -1, 'down', 'down', 'eight', 'eight', 'five', 'four', 'go', 'go', -1, -1, 'left', 'left', -1,
            -1, 'nine', 'nine', 'no', 'no', 'no', 'no', 'off', 'off', 'off', 'on', 'on', 'on', 'one', 'one', 'right',
            'right', 'seven', 'seven', -1, 'six', 'stop', 'three', 'three', -1, 'two', 'two', 'up', -1, 'yes', 'zero'
        ]
        true_names_for_validation = [
            'bed/fde2dee7_nohash_0.wav', 'bird/099d52ad_nohash_0.wav', 'cat/66cff190_nohash_0.wav',
            'dog/541120c7_nohash_0.wav', 'down/364c0853_nohash_0.wav', 'down/364c0853_nohash_1.wav',
            'eight/099d52ad_nohash_1.wav', 'eight/099d52ad_nohash_2.wav', 'five/099d52ad_nohash_3.wav',
            'four/099d52ad_nohash_0.wav', 'go/0ab3b47d_nohash_0.wav', 'go/0ab3b47d_nohash_1.wav',
            'happy/4abefdf5_nohash_0.wav', 'house/65ec06e5_nohash_0.wav', 'left/44bc77f7_nohash_0.wav',
            'left/62641b88_nohash_0.wav', 'marvin/50f55535_nohash_1.wav', 'marvin/97ae8b25_nohash_0.wav',
            'nine/1657c9fa_nohash_1.wav', 'nine/258f4559_nohash_2.wav', 'no/3ca784ec_nohash_1.wav',
            'no/3e2ba5f7_nohash_0.wav', 'no/409c962a_nohash_0.wav', 'no/41285056_nohash_0.wav',
            'off/605ed0ff_nohash_0.wav', 'off/65ec06e5_nohash_0.wav', 'off/7eee5973_nohash_1.wav',
            'on/1657c9fa_nohash_0.wav', 'on/1657c9fa_nohash_1.wav', 'on/fde2dee7_nohash_0.wav',
            'one/c50225fa_nohash_0.wav', 'one/c6389ab0_nohash_0.wav', 'right/060cd039_nohash_0.wav',
            'right/099d52ad_nohash_0.wav', 'seven/471a0925_nohash_2.wav', 'seven/471a0925_nohash_3.wav',
            'sheila/44c201dd_nohash_0.wav', 'six/2e0d80f7_nohash_0.wav', 'stop/3aa6f4e2_nohash_1.wav',
            'three/471a0925_nohash_3.wav', 'three/471a0925_nohash_4.wav', 'tree/4abefdf5_nohash_0.wav',
            'two/7c1d8533_nohash_4.wav', 'two/7c1d8533_nohash_5.wav', 'up/364c0853_nohash_0.wav',
            'wow/6a27a9bf_nohash_0.wav', 'yes/56eb74ae_nohash_4.wav', 'zero/7c1d8533_nohash_1.wav'
        ]
        true_labels_for_testing = [
            -1, -1, -1, -1, -1, 'down', 'down', 'down', 'eight', 'eight', 'five', 'four', 'four', 'go', 'go', -1, -1,
            'left', 'left', 'left', -1, -1, 'nine', 'nine', 'no', 'no', 'off', 'on', 'on', 'on', 'on', 'one', 'one',
            'one', 'one', 'one', 'right', 'right', 'seven', 'seven', -1, 'six', 'stop', 'stop', 'three', -1, 'two',
            'two', 'up', 'up', 'up', -1, 'yes', 'yes', 'yes', 'zero', 'zero'
        ]
        true_names_for_testing = [
            'bed/0c40e715_nohash_0.wav', 'bed/0ea0e2f4_nohash_0.wav', 'bird/42beb5eb_nohash_0.wav',
            'bird/44260689_nohash_0.wav', 'bird/44715c1c_nohash_0.wav', 'down/022cd682_nohash_0.wav',
            'down/096456f9_nohash_0.wav', 'down/0c40e715_nohash_0.wav', 'eight/096456f9_nohash_0.wav',
            'eight/0c40e715_nohash_0.wav', 'five/022cd682_nohash_0.wav', 'four/0c40e715_nohash_0.wav',
            'four/0d53e045_nohash_0.wav', 'go/022cd682_nohash_0.wav', 'go/ffa76c4a_nohash_0.wav',
            'happy/0c40e715_nohash_0.wav', 'house/ffb86d3c_nohash_1.wav', 'left/022cd682_nohash_0.wav',
            'left/022cd682_nohash_1.wav', 'left/0487ba9b_nohash_0.wav', 'marvin/47d01978_nohash_0.wav',
            'marvin/48a8a69d_nohash_0.wav', 'nine/283d7a53_nohash_0.wav', 'nine/283d7a53_nohash_1.wav',
            'no/3efef882_nohash_0.wav', 'no/3efef882_nohash_1.wav', 'off/096456f9_nohash_1.wav',
            'on/0fa1e7a9_nohash_0.wav', 'on/105a0eea_nohash_0.wav', 'on/e41a903b_nohash_2.wav',
            'on/e5e54cee_nohash_0.wav', 'one/1b4c9b89_nohash_1.wav', 'one/1b4c9b89_nohash_2.wav',
            'one/5e3dde6b_nohash_4.wav', 'one/5f814c23_nohash_0.wav', 'one/adebe223_nohash_1.wav',
            'right/0c40e715_nohash_0.wav', 'right/0c40e715_nohash_1.wav', 'seven/1f653d27_nohash_0.wav',
            'seven/1f653d27_nohash_1.wav', 'sheila/1cb788bc_nohash_1.wav', 'six/105a0eea_nohash_0.wav',
            'stop/0c40e715_nohash_0.wav', 'stop/0c40e715_nohash_1.wav', 'three/3c165869_nohash_2.wav',
            'tree/283d7a53_nohash_0.wav', 'two/1093c8e7_nohash_0.wav', 'two/1b4c9b89_nohash_0.wav',
            'up/210f3aa9_nohash_1.wav', 'up/f6af2457_nohash_0.wav', 'up/f6af2457_nohash_1.wav',
            'wow/0d53e045_nohash_0.wav', 'yes/0fa1e7a9_nohash_0.wav', 'yes/105a0eea_nohash_0.wav',
            'yes/105a0eea_nohash_1.wav', 'zero/37dca74f_nohash_3.wav', 'zero/37dca74f_nohash_4.wav'
        ]
        true_names_for_background = ['_background_noise_/doing_the_dishes.wav', '_background_noise_/pink_noise.wav']
        data_dir = os.path.join(os.path.dirname(__file__), 'testdata', 'tensorflow_data')
        data_for_training, data_for_validation, data_for_testing, background_sounds, fs = \
            read_tensorflow_speech_recognition_challenge(data_dir)
        self.assertIsInstance(data_for_training, tuple)
        self.assertIsInstance(data_for_validation, tuple)
        self.assertIsInstance(data_for_testing, tuple)
        self.assertIsInstance(background_sounds, tuple)
        self.assertIsInstance(fs, int)
        self.assertEqual(fs, 16000)
        self.assertEqual(len(data_for_training), 3)
        self.assertEqual(len(data_for_validation), 3)
        self.assertEqual(len(data_for_testing), 3)
        self.assertEqual(len(background_sounds), 2)
        self.assertIsInstance(data_for_training[0], list)
        self.assertIsInstance(data_for_training[1], list)
        self.assertIsInstance(data_for_training[2], list)
        self.assertEqual(len(data_for_training[0]), len(data_for_training[1]))
        self.assertEqual(len(data_for_training[0]), len(data_for_training[2]))
        self.assertIsInstance(data_for_validation[0], list)
        self.assertIsInstance(data_for_validation[1], list)
        self.assertIsInstance(data_for_validation[2], list)
        self.assertEqual(len(data_for_validation[0]), len(data_for_validation[1]))
        self.assertEqual(len(data_for_validation[0]), len(data_for_validation[2]))
        self.assertIsInstance(data_for_testing[0], list)
        self.assertIsInstance(data_for_testing[1], list)
        self.assertIsInstance(data_for_testing[2], list)
        self.assertEqual(len(data_for_testing[0]), len(data_for_testing[1]))
        self.assertEqual(len(data_for_testing[0]), len(data_for_testing[2]))
        self.assertIsInstance(background_sounds[0], list)
        self.assertIsInstance(background_sounds[1], list)
        self.assertEqual(len(background_sounds[0]), len(background_sounds[1]))
        for sample_idx in range(len(data_for_training[0])):
            self.assertIsInstance(data_for_training[0][sample_idx], np.ndarray)
            self.assertEqual(len(data_for_training[0][sample_idx].shape), 1)
            self.assertGreater(data_for_training[0][sample_idx].shape[0], 0)
        for sample_idx in range(len(data_for_validation[0])):
            self.assertIsInstance(data_for_validation[0][sample_idx], np.ndarray)
            self.assertEqual(len(data_for_validation[0][sample_idx].shape), 1)
            self.assertGreater(data_for_validation[0][sample_idx].shape[0], 0)
        for sample_idx in range(len(data_for_testing[0])):
            self.assertIsInstance(data_for_testing[0][sample_idx], np.ndarray)
            self.assertEqual(len(data_for_testing[0][sample_idx].shape), 1)
            self.assertGreater(data_for_testing[0][sample_idx].shape[0], 0)
        self.assertEqual(data_for_training[1], true_labels_for_training)
        self.assertEqual(data_for_training[2], true_names_for_training)
        self.assertEqual(data_for_validation[1], true_labels_for_validation)
        self.assertEqual(data_for_validation[2], true_names_for_validation)
        self.assertEqual(data_for_testing[1], true_labels_for_testing)
        self.assertEqual(data_for_testing[2], true_names_for_testing)
        self.assertEqual(background_sounds[1], true_names_for_background)
        for sample_idx in range(len(data_for_training[0])):
            new_sound, new_sampling_frequency = librosa.core.load(
                path=os.path.join(data_dir, 'audio', data_for_training[2][sample_idx]),
                sr=None, mono=True
            )
            self.assertEqual(new_sampling_frequency, fs)
            self.assertEqual(new_sound.tolist(), data_for_training[0][sample_idx].tolist())
        for sample_idx in range(len(data_for_validation[0])):
            new_sound, new_sampling_frequency = librosa.core.load(
                path=os.path.join(data_dir, 'audio', data_for_validation[2][sample_idx]),
                sr=None, mono=True
            )
            self.assertEqual(new_sampling_frequency, fs)
            self.assertEqual(new_sound.tolist(), data_for_validation[0][sample_idx].tolist())
        for sample_idx in range(len(data_for_testing[0])):
            new_sound, new_sampling_frequency = librosa.core.load(
                path=os.path.join(data_dir, 'audio', data_for_testing[2][sample_idx]),
                sr=None, mono=True
            )
            self.assertEqual(new_sampling_frequency, fs)
            self.assertEqual(new_sound.tolist(), data_for_testing[0][sample_idx].tolist())

    def test_read_custom_data(self):
        data_dir = os.path.join(os.path.dirname(__file__), 'testdata', 'custom_data')
        true_classes = ['NO', 'NO', 'NO', 'NO', 'NO', 'YES', 'NO', 'NO', 'YES', 'YES', -1, 'YES', 'YES', -1, -1, -1, -1]
        true_names = ['sounds/no_01.wav', 'sounds/no_03.wav', 'sounds/no_02.wav', 'sounds/no_05.wav',
                      'sounds/no_04.wav', 'sounds/yes_02.wav', 'sounds/no_06.wav', 'sounds/no_07.wav',
                      'sounds/yes_01.wav', 'sounds/yes_03.wav', 'sounds/unknown_01.wav', 'sounds/yes_05.wav',
                      'sounds/yes_04.wav', 'sounds/unknown_02.wav', 'sounds/unknown_03.wav', 'sounds/unknown_04.wav',
                      'sounds/unknown_05.wav']
        true_fs = 16000
        sounds, classes, names, fs = read_custom_data(os.path.join(data_dir, 'data_for_training.csv'), data_dir)
        self.assertIsInstance(sounds, list)
        self.assertIsInstance(classes, list)
        self.assertIsInstance(names, list)
        self.assertIsInstance(fs, int)
        self.assertEqual(true_fs, fs)
        self.assertEqual(len(sounds), len(classes))
        self.assertEqual(len(sounds), len(names))
        self.assertEqual(true_classes, classes)
        self.assertEqual(true_names, names)
        for sample_idx in range(len(sounds)):
            self.assertIsInstance(sounds[sample_idx], np.ndarray)
            self.assertEqual(1, len(sounds[sample_idx].shape))
            new_sound, new_sampling_frequency = librosa.core.load(
                path=os.path.join(data_dir, names[sample_idx]),
                sr=None, mono=True
            )
            self.assertEqual(new_sampling_frequency, fs)
            self.assertEqual(new_sound.tolist(), sounds[sample_idx].tolist())


if __name__ == '__main__':
    unittest.main(verbosity=2)
