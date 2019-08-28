from argparse import ArgumentParser
import os
import pickle
import sys

from sklearn.metrics import classification_report


try:
    from speech_commands.speech_commands import MobilenetRecognizer, DTWRecognizer
    from speech_commands.utils import read_custom_data, parse_layers
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from speech_commands.speech_commands import MobilenetRecognizer, DTWRecognizer
    from speech_commands.utils import read_custom_data, parse_layers


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
    parser.add_argument('-k', '--kind', dest='kind_of_model', type=str, required=False, default='ann',
                        help='Kind of used model (DTW or ANN).')
    parser.add_argument('--deep', dest='deep_of_mobilenet', type=int, required=False, default=6,
                        help='Number of pre-trained layers from MobileNet which will be used in out speech recognizer '
                             '(in the range from 1 to 13).')
    parser.add_argument('--layers', dest='hidden_layers', type=str, required=False, default=None,
                        help='Sizes of hidden layers which will be added after MobileNet (these sizes must be '
                             'splitted by `-`).')
    parser.add_argument('--with_augmentation', dest='with_augmentation', action='store_true',
                        help='Need to augmentate data during training.')
    cmd_args = parser.parse_args()

    model_name = os.path.normpath(cmd_args.model_name)
    train_data_name = os.path.normpath(cmd_args.train_file_name)
    validation_data_name = os.path.normpath(cmd_args.val_file_name)
    test_data_name = os.path.normpath(cmd_args.test_file_name)
    model_kind = cmd_args.kind_of_model.strip().lower()
    if model_kind not in {'ann', 'dtw'}:
        raise ValueError('{0} is unknown kind of model!'.format(cmd_args.kind_of_model))

    sounds_for_training, labels_for_training, _, sampling_frequency = read_custom_data(
        train_data_name, os.path.dirname(train_data_name)
    )
    sounds_for_testing, labels_for_testing, _, sampling_frequency_ = read_custom_data(
        test_data_name, os.path.dirname(test_data_name)
    )
    if sampling_frequency != sampling_frequency_:
        raise ValueError('Sampling frequency for train sounds does not equal to sampling frequency for test sounds! '
                         '{0} != {1}'.format(sampling_frequency, sampling_frequency_))
    sounds_for_validation, labels_for_validation, _, sampling_frequency_ = read_custom_data(
        validation_data_name, os.path.dirname(validation_data_name)
    )
    if sampling_frequency != sampling_frequency_:
        raise ValueError('Sampling frequency for train sounds does not equal to sampling frequency for validation '
                         'sounds! {0} != {1}'.format(sampling_frequency, sampling_frequency_))
    if os.path.isfile(model_name):
        with open(model_name, 'rb') as fp:
            recognizer = pickle.load(fp)
    else:
        if model_kind == 'ann':
            layers = parse_layers(cmd_args.hidden_layers)
            recognizer = MobilenetRecognizer(sampling_frequency=sampling_frequency, window_size=0.025, shift_size=0.01,
                                             batch_size=8, max_epochs=200, patience=7, verbose=True, warm_start=False,
                                             random_seed=42, hidden_layers=layers,
                                             layer_level=cmd_args.deep_of_mobilenet,
                                             use_augmentation=cmd_args.with_augmentation)
            recognizer.fit(sounds_for_training, labels_for_training,
                           validation_data=(sounds_for_validation, labels_for_validation))
        else:
            recognizer = DTWRecognizer(sampling_frequency=sampling_frequency, window_size=0.025, shift_size=0.01,
                                       verbose=True, warm_start=False, k=3)
            recognizer.fit(sounds_for_training, labels_for_training)
            recognizer.update_threshold(sounds=sounds_for_validation, labels=labels_for_validation)
        with open(model_name, 'wb') as fp:
            pickle.dump(recognizer, fp)
    print('')
    print('Report for data for training:')
    y_pred = recognizer.predict(sounds_for_training)
    print(classification_report(
        list(map(lambda it1: '__UNKNOWN__' if it1 == -1 else it1, labels_for_training)),
        list(map(lambda it2: '__UNKNOWN__' if it2 == -1 else it2, y_pred))
    ))
    print('')
    print('')
    print('Report for validation data:')
    y_pred = recognizer.predict(sounds_for_validation)
    print(classification_report(
        list(map(lambda it1: '__UNKNOWN__' if it1 == -1 else it1, labels_for_validation)),
        list(map(lambda it2: '__UNKNOWN__' if it2 == -1 else it2, y_pred))
    ))
    print('')
    print('')
    print('Report for data for testing:')
    y_pred = recognizer.predict(sounds_for_testing)
    print(classification_report(
        list(map(lambda it1: '__UNKNOWN__' if it1 == -1 else it1, labels_for_testing)),
        list(map(lambda it2: '__UNKNOWN__' if it2 == -1 else it2, y_pred))
    ))


if __name__ == '__main__':
    main()
