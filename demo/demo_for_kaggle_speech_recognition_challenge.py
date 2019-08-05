from argparse import ArgumentParser
import os
import pickle
import sys
from typing import List, Tuple, Union

import numpy as np
from sklearn.metrics import classification_report


try:
    from speech_commands.speech_commands import MobilenetRecognizer
    from speech_commands.utils import read_tensorflow_speech_recognition_challenge, parse_layers
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from speech_commands.speech_commands import MobilenetRecognizer
    from speech_commands.utils import read_tensorflow_speech_recognition_challenge, parse_layers


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

    data_for_training, data_for_validation, data_for_testing, background_sounds, fs = \
        read_tensorflow_speech_recognition_challenge(os.path.normpath(cmd_args.data_dir_name))
    print('Number of sounds for training is {0}.'.format(len(data_for_training[0])))
    print('Number of sounds for validation is {0}.'.format(len(data_for_validation[0])))
    print('Number of sounds for final testing is {0}.'.format(len(data_for_testing[0])))
    print('Number of background sounds is {0}.'.format(len(background_sounds)))
    print('Sampling frequency is {0} Hz.'.format(fs))
    print('')
    data_for_training = filter_data(data_for_training[0], data_for_training[1])
    data_for_validation = filter_data(data_for_validation[0], data_for_validation[1])
    data_for_testing = filter_data(data_for_testing[0], data_for_testing[1])
    background_sounds = filter_data(background_sounds[0])
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
                                         batch_size=128, max_epochs=100, patience=7, verbose=True, warm_start=False,
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
