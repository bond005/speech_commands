from argparse import ArgumentParser
import codecs
import csv
import os
import pickle
import sys
from typing import List, Tuple, Union

import librosa
import numpy as np
from sklearn.metrics import classification_report


try:
    from speech_commands.speech_commands import SoundRecognizer
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from speech_commands.speech_commands import SoundRecognizer


def read_data_for_training(annotation_file_name: str,
                           sound_base_dir: Union[str, None]=None) -> Tuple[List[np.ndarray], List[str], int]:
    if not os.path.isfile(os.path.normpath(annotation_file_name)):
        raise ValueError('The file `{0}` does not exist!'.format(annotation_file_name))
    if sound_base_dir is None:
        sound_base_dir_ = os.path.dirname(os.path.normpath(annotation_file_name))
    else:
        sound_base_dir_ = os.path.normpath(sound_base_dir)
    if not os.path.isdir(sound_base_dir_):
        raise ValueError('The directory `{0}` does not exist!'.format(sound_base_dir_))
    with codecs.open(os.path.normpath(annotation_file_name), mode='r', encoding='utf-8', errors='ignore') as fp:
        reader = csv.reader(fp, quotechar='"', delimiter=',')
        rows = list(filter(lambda it: len(it) > 0, reader))
    if len(rows) < 3:
        raise ValueError('The file `{0}` is empty!'.format(annotation_file_name))
    if rows[0] != ['sound_name', 'class_name']:
        raise ValueError('The CSV file `{0}` is wrong! Its header does not correspond to target one!'.format(
            annotation_file_name))
    sounds = []
    classes_of_sounds = []
    sampling_frequency = None
    for cur in rows[1:]:
        sound_name = os.path.join(sound_base_dir_, os.path.normpath(cur[0].strip()))
        if not os.path.isfile(sound_name):
            raise ValueError('The file `{0}` does not exist!'.format(sound_name))
        class_name = cur[1].strip().upper()
        if len(class_name) > 0:
            try:
                _ = int(class_name)
                ok = False
            except:
                try:
                    _ = float(class_name)
                    ok = False
                except:
                    ok = class_name.isalnum()
        else:
            ok = True
        if not ok:
            raise ValueError('`{0}` is wrong name for sound class!'.format(class_name))
        new_sound, new_sampling_frequency = librosa.core.load(path=sound_name, sr=None, mono=True)
        if sampling_frequency is None:
            sampling_frequency = new_sampling_frequency
        elif sampling_frequency != new_sampling_frequency:
            raise ValueError('Sampling frequency of the sound `{0}` is {1}, but target sampling frequency '
                             'is {2}!'.format(sound_name, new_sampling_frequency, sampling_frequency))
        sounds.append(new_sound)
        classes_of_sounds.append(class_name if len(class_name) > 0 else -1)
    return sounds, classes_of_sounds, sampling_frequency


def main():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_name', type=str, required=True,
                        help='The binary file with the speech classifier.')
    parser.add_argument('-t', '--train', dest='train_file_name', type=str, required=True,
                        help='Path to the CSV file with data for training.')
    parser.add_argument('-a', '--val', dest='val_file_name', type=str, required=True,
                        help='Path to the CSV file with data for validation.')
    parser.add_argument('-e', '--test', dest='test_file_name', type=str, required=True,
                        help='Path to the CSV file with data for final evaluation.')
    parser.add_argument('-c', '--cache', dest='cache_dir_name', type=str, required=False,
                        help='Path to the directory with cached data.')
    cmd_args = parser.parse_args()

    model_name = os.path.normpath(cmd_args.model_name)
    train_data_name = os.path.normpath(cmd_args.train_file_name)
    validation_data_name = os.path.normpath(cmd_args.val_file_name)
    test_data_name = os.path.normpath(cmd_args.test_file_name)
    cache_dir_name = None if cmd_args.cache_dir_name is None else os.path.normpath(cmd_args.cache_dir_name)

    sounds_for_training, labels_for_training, sampling_frequency = read_data_for_training(
        train_data_name, os.path.dirname(train_data_name)
    )
    sounds_for_testing, labels_for_testing, sampling_frequency_ = read_data_for_training(
        test_data_name, os.path.dirname(test_data_name)
    )
    if sampling_frequency != sampling_frequency_:
        raise ValueError('Sampling frequency for train sounds does not equal to sampling frequency for test sounds! '
                         '{0} != {1}'.format(sampling_frequency, sampling_frequency_))
    sounds_for_validation, labels_for_validation, sampling_frequency_ = read_data_for_training(
        validation_data_name, os.path.dirname(validation_data_name)
    )
    if sampling_frequency != sampling_frequency_:
        raise ValueError('Sampling frequency for train sounds does not equal to sampling frequency for validation '
                         'sounds! {0} != {1}'.format(sampling_frequency, sampling_frequency_))
    if os.path.isfile(model_name):
        with open(model_name, 'rb') as fp:
            recognizer = pickle.load(fp)
    else:
        recognizer = SoundRecognizer(sampling_frequency=sampling_frequency, window_size=0.025, shift_size=0.01,
                                     batch_size=4, max_epochs=100, patience=5, verbose=True, warm_start=False,
                                     random_seed=42, cache_dir=cache_dir_name)
        recognizer.fit(sounds_for_training, labels_for_training,
                       validation_data=(sounds_for_validation, labels_for_validation), sample_weight='balanced')
        with open(model_name, 'wb') as fp:
            pickle.dump(recognizer, fp)
    y_pred = recognizer.predict(sounds_for_testing)
    print(classification_report(
        list(map(lambda it1: 'UNKNOWN' if it1 == -1 else it1, labels_for_testing)),
        list(map(lambda it2: 'UNKNOWN' if it2 == -1 else it2, y_pred))
    ))


if __name__ == '__main__':
    main()
