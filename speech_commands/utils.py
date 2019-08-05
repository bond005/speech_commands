import codecs
import csv
import os
from typing import List, Tuple, Union

import librosa
import numpy as np


def read_custom_data(annotation_file_name: str,
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


def read_tensorflow_speech_recognition_challenge(dir_name: str) -> Tuple[
    Tuple[List[np.ndarray], List[int], List[str]], Tuple[List[np.ndarray], List[int], List[str]],
    Tuple[List[np.ndarray], List[int], List[str]], Tuple[List[np.ndarray], List[str]], int
]:
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
    filenames_for_training = []
    X_val = []
    y_val = []
    filenames_for_validation = []
    X_test = []
    y_test = []
    filenames_for_testing = []
    sampling_frequency = None
    for cur_sound in sorted(list(all_sounds.keys()), key=lambda it: it.split()):
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
            filenames_for_validation.append(cur_sound)
        elif cur_sound in sounds_for_testing:
            X_test.append(new_sound)
            y_test.append(class_name)
            filenames_for_testing.append(cur_sound)
        else:
            X_train.append(new_sound)
            y_train.append(class_name)
            filenames_for_training.append(cur_sound)
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
    return (X_train, y_train, filenames_for_training), (X_val, y_val, filenames_for_validation), \
           (X_test, y_test, filenames_for_testing), \
           (background_sounds, list(map(lambda it: '_background_noise_/' + it, names_of_background_sounds))), \
           sampling_frequency


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
