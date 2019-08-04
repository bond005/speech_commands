from argparse import ArgumentParser
import codecs
import os
import pickle
import sys
from typing import List, Tuple, Union

import librosa
import numpy as np
from sklearn.metrics import classification_report


try:
    from speech_commands.speech_commands import MobilenetRecognizer
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from speech_commands.speech_commands import MobilenetRecognizer


def read_data(dir_name: str) -> Tuple[Tuple[List[np.ndarray], List[int]], Tuple[List[np.ndarray], List[int]],
                                      Tuple[List[np.ndarray], List[int]], List[np.ndarray], int]:
    core_words = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two", "three",
                  "four", "five", "six", "seven", "eight", "nine"}
    if not os.path.isdir(os.path.normpath(dir_name)):
        raise ValueError('The directory `{0}` does not exist!'.format(dir_name))
    if not os.path.isdir(os.path.join(os.path.normpath(dir_name), 'audio')):
        raise ValueError('The directory `{0}` does not exist!'.format(
            os.path.join(os.path.normpath(dir_name), 'audio')))
    if not os.path.isdir(os.path.join(os.path.normpath(dir_name), 'audio', '_background_noise_')):
        raise ValueError('The directory `{0}` does not exist!'.format(
            os.path.join(os.path.normpath(dir_name), 'audio', '_background_noise_')))
    if not os.path.isfile(os.path.join(os.path.normpath(dir_name), 'testing_list.txt')):
        raise ValueError('The file `{0}` does not exist!'.format(
            os.path.join(os.path.normpath(dir_name), 'testing_list.txt')))
    if not os.path.isfile(os.path.join(os.path.normpath(dir_name), 'validation_list.txt')):
        raise ValueError('The file `{0}` does not exist!'.format(
            os.path.join(os.path.normpath(dir_name), 'validation_list.txt')))
    all_words = set(filter(lambda it: (not it.startswith('_')) and (not it.endswith('_')) and (it not in {'.', '..'}),
                           os.listdir(os.path.join(os.path.normpath(dir_name), 'audio'))))
    if len(all_words) != len(set(map(lambda it: it.lower(), all_words))):
        raise ValueError('The directory `{0}` contains wrong data!'.format(
            os.path.join(os.path.normpath(dir_name), 'audio')))
    if not (core_words <= all_words):
        raise ValueError('The directory `{0}` contains wrong data!'.format(
            os.path.join(os.path.normpath(dir_name), 'audio')))
    all_sounds = dict()
    for cur_word in sorted(list(all_words)):
        sound_subdir = os.path.join(os.path.normpath(dir_name), 'audio', cur_word)
        for cur_filename in filter(lambda it: it.endswith('.wav'), os.listdir(sound_subdir)):
            all_sounds[cur_word + '/' + cur_filename] = os.path.join(sound_subdir, cur_filename)
    with codecs.open(os.path.join(os.path.normpath(dir_name), 'testing_list.txt'), mode='r', encoding='utf-8') as fp:
        sounds_for_testing = set(map(lambda it: it.strip(), fp.readlines()))
    with codecs.open(os.path.join(os.path.normpath(dir_name), 'validation_list.txt'), mode='r', encoding='utf-8') as fp:
        sounds_for_validation = set(map(lambda it: it.strip(), fp.readlines()))
    if not (sounds_for_testing <= set(all_sounds.keys())):
        raise ValueError('The file `{0}` contains wrong data!'.format(
            os.path.join(os.path.normpath(dir_name), 'testing_list.txt')))
    if not (sounds_for_validation <= set(all_sounds.keys())):
        raise ValueError('The file `{0}` contains wrong data!'.format(
            os.path.join(os.path.normpath(dir_name), 'validation_list.txt')))
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []
    sampling_frequency = None
    for cur_sound in sorted(list(all_sounds.keys())):
        parts_of_name = cur_sound.split('/')
        cur_word = parts_of_name[0]
        if cur_word in core_words:
            class_name = cur_word
        else:
            class_name = -1
        new_sound, new_sampling_frequency = librosa.core.load(path=all_sounds[cur_sound], sr=None, mono=True)
        if sampling_frequency is None:
            sampling_frequency = new_sampling_frequency
        elif sampling_frequency != new_sampling_frequency:
            raise ValueError('Sampling frequency of the sound `{0}` is {1}, but target sampling frequency '
                             'is {2}!'.format(all_sounds[cur_sound], new_sampling_frequency, sampling_frequency))
        if cur_sound in sounds_for_validation:
            X_val.append(new_sound)
            y_val.append(class_name)
        elif cur_sound in sounds_for_testing:
            X_test.append(new_sound)
            y_test.append(class_name)
        else:
            X_train.append(new_sound)
            y_train.append(class_name)
    background_sounds = []
    names_of_background_sounds = list(filter(
        lambda it: it.lower().endswith('.wav'),
        os.listdir(os.path.join(os.path.normpath(dir_name), 'audio', '_background_noise_'))
    ))
    if len(names_of_background_sounds) == 0:
        raise ValueError('There are no sounds in the `{0}`.'.format(
            os.path.join(os.path.normpath(dir_name), 'audio', '_background_noise_')))
    for cur_sound_name in names_of_background_sounds:
        full_sound_name = os.path.join(os.path.normpath(dir_name), 'audio', '_background_noise_', cur_sound_name)
        new_sound, new_sampling_frequency = librosa.core.load(path=full_sound_name, sr=None, mono=True)
        if sampling_frequency != new_sampling_frequency:
            raise ValueError('Sampling frequency of the sound `{0}` is {1}, but target sampling frequency '
                             'is {2}!'.format(full_sound_name, new_sampling_frequency, sampling_frequency))
        background_sounds.append(new_sound)
        del new_sound
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), background_sounds, sampling_frequency


def filter_data(X_train: List[np.ndarray], y_train: Union[List[int], None]=None) -> \
        Union[Tuple[List[np.ndarray], List[int]], List[np.ndarray]]:
    nonempty_indices = []
    for sound_idx in range(len(X_train)):
        sound_length = MobilenetRecognizer.strip_sound(X_train[sound_idx])
        if sound_length > 0:
            nonempty_indices.append(sound_idx)
    if y_train is None:
        return [X_train[idx] for idx in nonempty_indices]
    return [X_train[idx] for idx in nonempty_indices], [y_train[idx] for idx in nonempty_indices]


def parse_layers(src: Union[str, None]) -> tuple:
    if src is None:
        return tuple()
    prep = src.strip()
    if len(prep) == 0:
        return tuple()
    return tuple(map(
        lambda it3: int(it3),
        filter(
            lambda it2: len(it2) > 0,
            map(lambda it1: it1.strip(), prep.split('-'))
        )
    ))


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The binary file with the speech classifier.')
    parser.add_argument('-d', '--data', dest='data_dir_name', type=str, required=True,
                        help='Path to the directory with labeled data of the TensorFlow Speech Recognition Challenge.')
    parser.add_argument('--deep', dest='deep_of_mobilenet', type=int, required=False, default=6,
                        help='Number of pre-trained layers from MobileNet which will be used in out speech recognizer '
                             '(in the range from 1 to 13).')
    parser.add_argument('--layers', dest='hidden_layers', type=str, required=False, default=None,
                        help='Sizes of hidden layers which will be added after MobileNet (these sizes must be '
                             'splitted by `-`).')
    cmd_args = parser.parse_args()

    data_for_training, data_for_validation, data_for_testing, background_sounds, fs = read_data(
        os.path.normpath(cmd_args.data_dir_name)
    )
    print('Number of sounds for training is {0}.'.format(len(data_for_training[0])))
    print('Number of sounds for validation is {0}.'.format(len(data_for_validation[0])))
    print('Number of sounds for final testing is {0}.'.format(len(data_for_testing[0])))
    print('Number of background sounds is {0}.'.format(len(background_sounds)))
    print('Sampling frequency is {0} Hz.'.format(fs))
    print('')
    data_for_training = filter_data(data_for_training[0], data_for_training[1])
    data_for_validation = filter_data(data_for_validation[0], data_for_validation[1])
    data_for_testing = filter_data(data_for_testing[0], data_for_testing[1])
    background_sounds = filter_data(background_sounds)
    print('Number of sounds for training after filtering is {0}.'.format(len(data_for_training[0])))
    print('Number of sounds for validation after filtering is {0}.'.format(len(data_for_validation[0])))
    print('Number of sounds for final testing after filtering is {0}.'.format(len(data_for_testing[0])))
    print('Number of background sounds after filtering is {0}.'.format(len(background_sounds)))
    print('')

    layers = parse_layers(cmd_args.hidden_layers)
    model_name = os.path.normpath(cmd_args.model_name)
    if os.path.isfile(model_name):
        with open(model_name, 'rb') as fp:
            recognizer = pickle.load(fp)
    else:
        recognizer = MobilenetRecognizer(sampling_frequency=fs, window_size=0.025, shift_size=0.01,
                                         batch_size=64, max_epochs=100, patience=7, verbose=True, warm_start=False,
                                         random_seed=42, hidden_layers=layers, layer_level=cmd_args.deep_of_mobilenet)
        recognizer.fit(data_for_training[0], data_for_training[1], validation_data=data_for_validation,
                       background=background_sounds)
        with open(model_name, 'wb') as fp:
            pickle.dump(recognizer, fp)
    print('')
    print('Report for data for training:')
    y_pred = recognizer.predict(data_for_training[0])
    print(classification_report(
        list(map(lambda it1: '__UNKNOWN__' if it1 == -1 else it1, data_for_training[1])),
        list(map(lambda it2: '__UNKNOWN__' if it2 == -1 else it2, y_pred))
    ))
    print('')
    print('')
    print('Report for validation data:')
    y_pred = recognizer.predict(data_for_validation[0])
    print(classification_report(
        list(map(lambda it1: '__UNKNOWN__' if it1 == -1 else it1, data_for_validation[1])),
        list(map(lambda it2: '__UNKNOWN__' if it2 == -1 else it2, y_pred))
    ))
    print('')
    print('')
    print('Report for data for testing:')
    y_pred = recognizer.predict(data_for_testing[0])
    print(classification_report(
        list(map(lambda it1: '__UNKNOWN__' if it1 == -1 else it1, data_for_testing[1])),
        list(map(lambda it2: '__UNKNOWN__' if it2 == -1 else it2, y_pred))
    ))


if __name__ == '__main__':
    main()
