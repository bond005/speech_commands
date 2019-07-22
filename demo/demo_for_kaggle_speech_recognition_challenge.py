from argparse import ArgumentParser
import codecs
import os
import pickle
import sys
from typing import List, Tuple

import librosa
import numpy as np
from sklearn.metrics import classification_report


try:
    from speech_commands.speech_commands import SoundRecognizer
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from speech_commands.speech_commands import SoundRecognizer


def read_data(dir_name: str) -> Tuple[Tuple[List[np.ndarray], List[int]], Tuple[List[np.ndarray], List[int]],
                                      Tuple[List[np.ndarray], List[int]], int]:
    core_words = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two", "three",
                  "four", "five", "six", "seven", "eight", "nine"}
    if not os.path.isdir(os.path.normpath(dir_name)):
        raise ValueError('The directory `{0}` does not exist!'.format(dir_name))
    if not os.path.isdir(os.path.join(os.path.normpath(dir_name), 'audio')):
        raise ValueError('The directory `{0}` does not exist!'.format(
            os.path.join(os.path.normpath(dir_name), 'audio')))
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
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), sampling_frequency


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The binary file with the speech classifier.')
    parser.add_argument('-d', '--data', dest='data_dir_name', type=str, required=True,
                        help='Path to the directory with labeled data of the TensorFlow Speech Recognition Challenge.')
    cmd_args = parser.parse_args()

    data_for_training, data_for_validation, data_for_testing, fs = read_data(os.path.normpath(cmd_args.data_dir_name))
    print('Number of sounds for training is {0}.'.format(len(data_for_training[0])))
    print('Number of sounds for validation is {0}.'.format(len(data_for_validation[0])))
    print('Number of sounds for final testing is {0}.'.format(len(data_for_testing[0])))
    print('Sampling frequency is {0} Hz.'.format(fs))
    print('')

    model_name = os.path.normpath(cmd_args.model_name)
    if os.path.isfile(model_name):
        with open(model_name, 'rb') as fp:
            recognizer = pickle.load(fp)
    else:
        recognizer = SoundRecognizer(sampling_frequency=fs, window_size=0.025, shift_size=0.01,
                                     batch_size=32, max_epochs=100, patience=3, verbose=True, warm_start=False,
                                     random_seed=42)
        recognizer.fit(data_for_training[0], data_for_training[1], validation_data=data_for_validation)
        with open(model_name, 'wb') as fp:
            pickle.dump(recognizer, fp)
    y_pred = recognizer.predict(data_for_testing[0])
    print(classification_report(
        list(map(lambda it1: 'UNKNOWN' if it1 == -1 else it1, data_for_testing[1])),
        list(map(lambda it2: 'UNKNOWN' if it2 == -1 else it2, y_pred))
    ))


if __name__ == '__main__':
    main()
