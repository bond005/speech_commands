import copy
import os
import pickle
import random
import tempfile
import time
from typing import List, Tuple, Union

import keras
import keras.backend as K
import librosa
from matplotlib.pyplot import cm
import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import f1_score
from sklearn.utils.validation import check_is_fitted


class SoundRecognizer(ClassifierMixin, BaseEstimator):
    COLORMAP = cm.get_cmap('jet')
    IMAGESIZE = (224, 224)

    def __init__(self, sampling_frequency: int=16000, window_size: float=0.025, shift_size: float=0.01,
                 batch_size: int=32, max_epochs: int=100, patience: int=5, verbose: bool=False, warm_start: bool=False,
                 random_seed=None, cache_dir: Union[str, None]=None):
        self.sampling_frequency = sampling_frequency
        self.window_size = window_size
        self.shift_size = shift_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_seed = random_seed
        self.warm_start = warm_start
        self.verbose = verbose
        self.cache_dir = cache_dir

    def fit(self, X: Union[np.ndarray, List[np.ndarray]], y: Union[np.ndarray, List[int], List[str]], **kwargs):
        self.check_params(sampling_frequency=self.sampling_frequency, window_size=self.window_size,
                          shift_size=self.shift_size, batch_size=self.batch_size, max_epochs=self.max_epochs,
                          patience=self.patience, warm_start=self.warm_start, verbose=self.verbose,
                          random_seed=self.random_seed, cache_dir=self.cache_dir)
        classes_dict, classes_dict_reverse = self.check_Xy(X, 'X', y, 'y')
        n_train = len(y)
        if self.verbose:
            print('Train set size is {0}.'.format(n_train))
            print('Classes number is {0}.'.format(len(classes_dict)))
        if 'validation_data' in kwargs:
            if (not isinstance(kwargs['validation_data'], list)) and (not isinstance(kwargs['validation_data'], tuple)):
                raise ValueError('`validation_data` is wrong! Expected a {0} or {1}, but got a {2}!'.format(
                    type([1, 2]), type((1, 2)), type(kwargs['validation_data'])))
            if len(kwargs['validation_data']) != 2:
                raise ValueError('`validation_data` is wrong! Expected a 2-element sequence (`X_val` and `y_val`), '
                                 'but got a {0}-element one.'.format(len(kwargs['validation_data'])))
            classes_dict_for_validation, _ = self.check_Xy(kwargs['validation_data'][0], 'X_val',
                                                           kwargs['validation_data'][1], 'y_val')
            if not set(classes_dict_for_validation.keys()) <= set(classes_dict.keys()):
                raise ValueError('Classes in validation set do not correspond to classes in train set!')
            X_val = kwargs['validation_data'][0]
            y_val = kwargs['validation_data'][1]
            n_val = len(y_val)
            if self.verbose:
                print('Train set size is {0}.'.format(n_val))
                print('Classes number is {0}.'.format(len(classes_dict_for_validation)))
        else:
            X_val = None
            y_val = None
        self.update_random_seed()
        if self.warm_start:
            self.check_is_fitted()
            if hasattr(self, 'recognizer_'):
                del self.recognizer_
            input_data = keras.layers.Input(shape=(self.IMAGESIZE[0], self.IMAGESIZE[1], 3), name='InputSpectrogram')
            output_layer = keras.layers.Dense(
                units=len(self.classes_), activation='softmax',
                kernel_initializer=keras.initializers.glorot_normal(seed=self.random_seed), name='OutputLayer'
            )(self.recognizer_.get_layer('ResNet_Base')(input_data))
            self.recognizer_ = keras.models.Model(input_data, output_layer)
            self.recognizer_.compile(optimizer=keras.optimizers.RMSprop(lr=1e-4), loss='categorical_crossentropy',
                                     metrics=['categorical_accuracy'])
        else:
            self.finalize_model()
            self.classes_ = classes_dict
            self.classes_reverse_ = classes_dict_reverse
            input_data = keras.layers.Input(shape=(self.IMAGESIZE[0], self.IMAGESIZE[1], 3), name='InputSpectrogram')
            resnet = keras.applications.resnet50.ResNet50(
                input_shape=(self.IMAGESIZE[0], self.IMAGESIZE[1], 3), include_top=False, weights='imagenet',
                input_tensor=input_data, pooling='avg')
            resnet.name = 'RESNet_Base'
            output_layer = keras.layers.Dense(
                units=len(self.classes_), activation='softmax',
                kernel_initializer=keras.initializers.glorot_normal(seed=self.random_seed), name='OutputLayer'
            )(resnet(input_data))
            self.recognizer_ = keras.models.Model(input_data, output_layer)
            self.recognizer_.compile(optimizer=keras.optimizers.RMSprop(lr=1e-4), loss='categorical_crossentropy',
                                     metrics=['categorical_accuracy'])
        if self.verbose:
            if self.cache_dir is not None:
                print('Cache directory is `{0}`.'.format(self.cache_dir))
            keras.utils.print_summary(self.recognizer_, line_length=120)
        if not hasattr(self, 'melfb_'):
            self.update_triangle_filters()
        class_freq = dict()
        freq_sum = 0
        for cur in y:
            if cur != -1:
                class_freq[cur] = class_freq.get(cur, 0) + 1
                freq_sum += 1
        if self.verbose:
            print('Class weights:')
            for cur in sorted(list(class_freq.keys())):
                print('  - {0}: {1:.6f}'.format(cur, float(2 * freq_sum - class_freq.get(cur, 0)) / float(freq_sum)))
            print('Sampling frequency is {0} Hz.'.format(self.sampling_frequency))
        if 'sample_weight' in kwargs:
            if kwargs['sample_weight'] == 'balanced':
                sample_weight = np.array([float(2 * freq_sum - class_freq.get(cur, 0)) / float(freq_sum) for cur in y],
                                         dtype=np.float32)
            else:
                sample_weight = kwargs['sample_weight']
        else:
            sample_weight = None
        trainset_generator = TrainsetGenerator(
            X=X, y=y, batch_size=self.batch_size, melfb=self.melfb_,
            window_size=self.window_size, shift_size=self.shift_size, sampling_frequency=self.sampling_frequency,
            classes=self.classes_, sample_weight=sample_weight, cache_dir_name=self.cache_dir, suffix='train'
        )
        if (X_val is None) or (y_val is None):
            self.recognizer_.fit_generator(trainset_generator, shuffle=True, epochs=self.max_epochs,
                                           verbose=2 if self.verbose else 0)
            indices_of_unknown = list(filter(lambda it: y[it] == -1, range(len(y))))
            if len(indices_of_unknown) > 0:
                indices_of_known = list(filter(lambda it: y[it] != -1, range(len(y))))
                max_probabilities_for_known = self.recognizer_.predict_generator(
                    DatasetGenerator(X=X, batch_size=self.batch_size,
                                     melfb=self.melfb_, window_size=self.window_size, shift_size=self.shift_size,
                                     sampling_frequency=self.sampling_frequency, indices=indices_of_known)
                ).max(axis=1)
                max_probabilities_for_unknown = self.recognizer_.predict_generator(
                    DatasetGenerator(X=X, batch_size=self.batch_size,
                                     melfb=self.melfb_, window_size=self.window_size, shift_size=self.shift_size,
                                     sampling_frequency=self.sampling_frequency, indices=indices_of_unknown)
                ).max(axis=1)
                self.threshold_ = self.find_optimal_threshold(max_probabilities_for_known,
                                                              max_probabilities_for_unknown)
            else:
                self.threshold_ = self.recognizer_.predict_generator(
                    DatasetGenerator(X=X, batch_size=self.batch_size,
                                     melfb=self.melfb_, window_size=self.window_size, shift_size=self.shift_size,
                                     sampling_frequency=self.sampling_frequency)
                ).max(axis=1).min()
        else:
            if 'sample_weight' in kwargs:
                if kwargs['sample_weight'] == 'balanced':
                    sample_weight_for_validation = np.array(
                        [float(2 * freq_sum - class_freq.get(cur, 0)) / float(freq_sum) for cur in y_val],
                        dtype=np.float32
                    )
                else:
                    sample_weight_for_validation = kwargs['sample_weight']
            else:
                sample_weight_for_validation = None
            validset_generator = TrainsetGenerator(
                X=X_val, y=y_val, batch_size=self.batch_size, melfb=self.melfb_, window_size=self.window_size,
                shift_size=self.shift_size, sampling_frequency=self.sampling_frequency, classes=self.classes_,
                sample_weight=sample_weight_for_validation, cache_dir_name=self.cache_dir, suffix='valid'
            )
            early_stopping_callback = keras.callbacks.EarlyStopping(
                patience=self.patience, verbose=self.verbose, restore_best_weights=True,
                monitor='val_categorical_accuracy', mode='max'
            )
            self.recognizer_.fit_generator(trainset_generator, validation_data=validset_generator, shuffle=True,
                                           epochs=self.max_epochs, verbose=2 if self.verbose else 0,
                                           callbacks=[early_stopping_callback])
            indices_of_unknown_for_training = list(filter(lambda it: y[it] == -1, range(len(y))))
            indices_of_unknown_for_validation = list(filter(lambda it: y_val[it] == -1, range(len(y_val))))
            if (len(indices_of_unknown_for_training) + len(indices_of_unknown_for_validation)) > 0:
                indices_of_known = list(filter(lambda it: y_val[it] != -1, range(len(y_val))))
                max_probabilities_for_known = self.recognizer_.predict_generator(
                    DatasetGenerator(X=X_val, batch_size=self.batch_size, melfb=self.melfb_,
                                     window_size=self.window_size, shift_size=self.shift_size,
                                     sampling_frequency=self.sampling_frequency, indices=indices_of_known)
                ).max(axis=1)
                if (len(indices_of_unknown_for_training) > 0) and (len(indices_of_unknown_for_validation) > 0):
                    max_probabilities_for_unknown = np.concatenate(
                        (
                            self.recognizer_.predict_generator(
                                DatasetGenerator(X=X, batch_size=self.batch_size, melfb=self.melfb_,
                                                 window_size=self.window_size, shift_size=self.shift_size,
                                                 sampling_frequency=self.sampling_frequency,
                                                 indices=indices_of_unknown_for_training)
                            ).max(axis=1),
                            self.recognizer_.predict_generator(
                                DatasetGenerator(X=X_val, batch_size=self.batch_size, melfb=self.melfb_,
                                                 window_size=self.window_size, shift_size=self.shift_size,
                                                 sampling_frequency=self.sampling_frequency,
                                                 indices=indices_of_unknown_for_validation)
                            ).max(axis=1)
                        )
                    )
                elif len(indices_of_unknown_for_training) > 0:
                    max_probabilities_for_unknown = self.recognizer_.predict_generator(
                        DatasetGenerator(X=X, batch_size=self.batch_size, melfb=self.melfb_,
                                         window_size=self.window_size, shift_size=self.shift_size,
                                         sampling_frequency=self.sampling_frequency,
                                         indices=indices_of_unknown_for_training)
                    ).max(axis=1)
                else:
                    max_probabilities_for_unknown = self.recognizer_.predict_generator(
                        DatasetGenerator(X=X_val, batch_size=self.batch_size, melfb=self.melfb_,
                                         window_size=self.window_size, shift_size=self.shift_size,
                                         sampling_frequency=self.sampling_frequency,
                                         indices=indices_of_unknown_for_validation)
                    ).max(axis=1)
                self.threshold_ = self.find_optimal_threshold(max_probabilities_for_known,
                                                              max_probabilities_for_unknown)
            else:
                self.threshold_ = self.recognizer_.predict_generator(
                    DatasetGenerator(X=X_val, batch_size=self.batch_size, melfb=self.melfb_,
                                     window_size=self.window_size, shift_size=self.shift_size,
                                     sampling_frequency=self.sampling_frequency)
                ).max(axis=1).min()
        return self

    def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        probabilities = self.predict_proba(X)
        indices_of_classes = probabilities.argmax(axis=1)
        max_probabilities = probabilities.max(axis=1)
        res = []
        for idx in range(probabilities.shape[0]):
            if max_probabilities[idx] < self.threshold_:
                res.append(-1)
            else:
                res.append(self.classes_reverse_[indices_of_classes[idx]])
        return np.array(res, dtype=object)

    def predict_proba(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        self.check_params(sampling_frequency=self.sampling_frequency, window_size=self.window_size,
                          shift_size=self.shift_size, batch_size=self.batch_size, max_epochs=self.max_epochs,
                          patience=self.patience, warm_start=self.warm_start, verbose=self.verbose,
                          random_seed=self.random_seed, cache_dir=self.cache_dir)
        self.check_X(X, 'X')
        self.check_is_fitted()
        if not hasattr(self, 'melfb_'):
            self.update_triangle_filters()
        return self.recognizer_.predict_generator(
            DatasetGenerator(X=X, batch_size=self.batch_size, melfb=self.melfb_, window_size=self.window_size,
                             shift_size=self.shift_size, sampling_frequency=self.sampling_frequency)
        )

    def predict_log_proba(self, X: Union[list, tuple, np.ndarray]) -> np.ndarray:
        return np.log(np.asarray(self.predict_proba(X), dtype=np.float64) + 1e-9)

    def score(self, X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray],
              sample_weight: Union[list, tuple, np.ndarray, None] = None) -> float:
        y_pred = self.predict(X)
        return f1_score(y, y_pred, average='macro', sample_weight=sample_weight)

    def fit_predict(self, X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray],
                    sample_weight: Union[list, tuple, np.ndarray, None]=None, **kwargs):
        if 'validation_data' in kwargs:
            return self.fit(X, y, sample_weight=sample_weight, validation_data=kwargs['validation_data']).predict(X)
        return self.fit(X, y, sample_weight=sample_weight).predict(X)

    def finalize_model(self):
        if hasattr(self, 'recognizer_'):
            del self.recognizer_
        K.clear_session()

    def update_random_seed(self):
        if self.random_seed is None:
            self.random_seed = int(round(time.time()))
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        if K.backend() == 'tensorflow':
            import tensorflow as tf
            tf.random.set_random_seed(self.random_seed)

    def update_triangle_filters(self):
        n_fft = self.get_n_fft(self.sampling_frequency, self.window_size)
        if hasattr(self, 'melfb_'):
            del self.melfb_
        self.melfb_ = librosa.filters.mel(sr=self.sampling_frequency, n_fft=n_fft, n_mels=self.IMAGESIZE[1])

    def check_is_fitted(self):
        check_is_fitted(self, ['recognizer_', 'classes_', 'classes_reverse_', 'threshold_'])

    def get_spectrogram_length(self, sound: np.ndarray) -> int:
        n_window = int(round(self.sampling_frequency * self.window_size))
        n_shift = int(round(self.sampling_frequency * self.shift_size))
        return int(np.ceil((sound.shape[0] - n_window) / float(n_shift)))

    def get_params(self, deep=True):
        return {'sampling_frequency': self.sampling_frequency, 'window_size': self.window_size,
                'shift_size': self.shift_size, 'batch_size': self.batch_size, 'max_epochs': self.max_epochs,
                'patience': self.patience, 'verbose': self.verbose, 'warm_start': self.warm_start,
                'random_seed': self.random_seed, 'cache_dir': self.cache_dir}

    def set_params(self, **params):
        for parameter, value in params.items():
            self.__setattr__(parameter, value)
        return self

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.set_params(
            sampling_frequency=self.sampling_frequency, window_size=self.window_size, shift_size=self.shift_size,
            batch_size=self.batch_size, max_epochs=self.max_epochs, patience=self.patience, warm_start=self.warm_start,
            verbose=self.verbose, random_seed=self.random_seed
        )
        try:
            self.check_is_fitted()
            is_fitted = True
        except:
            is_fitted = False
        if is_fitted:
            result.classes_ = copy.copy(self.classes_)
            result.classes_reverse_ = copy.copy(self.classes_reverse_)
            result.recognizer_ = self.recognizer_
        return result

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.set_params(
            sampling_frequency=self.sampling_frequency, window_size=self.window_size, shift_size=self.shift_size,
            batch_size=self.batch_size, max_epochs=self.max_epochs, patience=self.patience, warm_start=self.warm_start,
            verbose=self.verbose, random_seed=self.random_seed
        )
        try:
            self.check_is_fitted()
            is_fitted = True
        except:
            is_fitted = False
        if is_fitted:
            result.classes_ = copy.deepcopy(self.classes_)
            result.classes_reverse_ = copy.deepcopy(self.classes_reverse_)
            result.recognizer_ = self.recognizer_
        return result

    def __getstate__(self):
        return self.dump_all()

    def __setstate__(self, state: dict):
        self.load_all(state)

    def dump_all(self):
        try:
            self.check_is_fitted()
            is_fitted = True
        except:
            is_fitted = False
        params = self.get_params(True)
        if is_fitted:
            params['classes_'] = copy.deepcopy(self.classes_)
            params['classes_reverse_'] = copy.deepcopy(self.classes_reverse_)
            params['threshold_'] = self.threshold_
            model_file_name = self.get_temp_model_name()
            try:
                self.recognizer_.save(model_file_name)
                with open(model_file_name, 'rb') as fp:
                    model_data = fp.read()
                params['model_data_'] = model_data
                del model_data
            finally:
                if os.path.isfile(model_file_name):
                    os.remove(model_file_name)
        return params

    def load_all(self, new_params: dict):
        if not isinstance(new_params, dict):
            raise ValueError('`new_params` is wrong! Expected `{0}`, got `{1}`.'.format(type({0: 1}), type(new_params)))
        self.check_params(**new_params)
        self.finalize_model()
        is_fitted = ('classes_' in new_params) and ('classes_reverse_' in new_params) and \
                    ('threshold_' in new_params) and ('model_data_' in new_params)
        if is_fitted:
            self.set_params(**new_params)
            self.classes_reverse_ = copy.deepcopy(new_params['classes_reverse_'])
            self.classes_ = copy.deepcopy(new_params['classes_'])
            self.threshold_ = new_params['threshold_']
            model_file_name = self.get_temp_model_name()
            try:
                with open(model_file_name, 'wb') as fp:
                    fp.write(new_params['model_data_'])
                self.recognizer_ = keras.models.load_model(model_file_name)
            finally:
                if os.path.isfile(model_file_name):
                    os.remove(model_file_name)
        else:
            self.set_params(**new_params)
        return self

    @staticmethod
    def get_temp_model_name() -> str:
        return tempfile.NamedTemporaryFile(mode='w', suffix='sound_recognizer.h5y').name

    @staticmethod
    def sound_to_melspectrogram(sound: np.ndarray, window_size: float, shift_size: float, sampling_frequency: int,
                                melfb: np.ndarray) -> np.ndarray:
        n_window = int(round(sampling_frequency * window_size))
        n_shift = int(round(sampling_frequency * shift_size))
        n_fft = SoundRecognizer.get_n_fft(sampling_frequency, window_size)
        specgram = librosa.core.stft(y=sound, n_fft=n_fft, hop_length=n_shift, win_length=n_window, window='hamming')
        specgram = np.asarray(np.absolute(specgram), dtype=np.float64)
        return np.dot(melfb, specgram).transpose()

    @staticmethod
    def melspectrogram_to_image(spectrogram: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.sort(spectrogram.reshape((spectrogram.shape[0] * spectrogram.shape[1],)))
        n = int(round(0.01 * values.shape[0]))
        max_value = values[-n - 1]
        min_value = values[n]
        del values
        normalized = spectrogram - min_value
        if max_value > min_value:
            normalized /= (max_value - min_value)
        if normalized.shape[0] < SoundRecognizer.IMAGESIZE[0]:
            normalized = np.vstack(
                (
                    normalized,
                    np.zeros((SoundRecognizer.IMAGESIZE[0] - normalized.shape[0], normalized.shape[1]),
                             dtype=normalized.dtype)
                )
            )
        elif normalized.shape[0] > SoundRecognizer.IMAGESIZE[0]:
            normalized = normalized[0:SoundRecognizer.IMAGESIZE[0]]
        r = np.zeros(shape=normalized.shape, dtype=np.float32)
        g = np.zeros(shape=normalized.shape, dtype=np.float32)
        b = np.zeros(shape=normalized.shape, dtype=np.float32)
        for row_idx in range(normalized.shape[0]):
            for col_idx in range(normalized.shape[1]):
                cur_value = normalized[row_idx][col_idx]
                if cur_value < 0.0:
                    cur_value = 0.0
                elif cur_value > 1.0:
                    cur_value = 1.0
                r_, g_, b_, _ = SoundRecognizer.COLORMAP(normalized[row_idx][col_idx])
                r[row_idx][col_idx] = r_
                g[row_idx][col_idx] = g_
                b[row_idx][col_idx] = b_
        return r, g, b

    @staticmethod
    def check_params(**kwargs):
        if 'batch_size' not in kwargs:
            raise ValueError('`batch_size` is not specified!')
        if (not isinstance(kwargs['batch_size'], int)) and (not isinstance(kwargs['batch_size'], np.int32)) and \
                (not isinstance(kwargs['batch_size'], np.uint32)):
            raise ValueError('`batch_size` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['batch_size'])))
        if kwargs['batch_size'] < 1:
            raise ValueError('`batch_size` is wrong! Expected a positive integer value, '
                             'but {0} is not positive.'.format(kwargs['batch_size']))
        if 'max_epochs' not in kwargs:
            raise ValueError('`max_epochs` is not specified!')
        if (not isinstance(kwargs['max_epochs'], int)) and (not isinstance(kwargs['max_epochs'], np.int32)) and \
                (not isinstance(kwargs['max_epochs'], np.uint32)):
            raise ValueError('`max_epochs` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['max_epochs'])))
        if kwargs['max_epochs'] < 1:
            raise ValueError('`max_epochs` is wrong! Expected a positive integer value, '
                             'but {0} is not positive.'.format(kwargs['max_epochs']))
        if 'patience' not in kwargs:
            raise ValueError('`patience` is not specified!')
        if (not isinstance(kwargs['patience'], int)) and (not isinstance(kwargs['patience'], np.int32)) and \
                (not isinstance(kwargs['patience'], np.uint32)):
            raise ValueError('`patience` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['patience'])))
        if kwargs['patience'] < 1:
            raise ValueError('`patience` is wrong! Expected a positive integer value, '
                             'but {0} is not positive.'.format(kwargs['patience']))
        if 'random_seed' not in kwargs:
            raise ValueError('`random_seed` is not specified!')
        if kwargs['random_seed'] is not None:
            if (not isinstance(kwargs['random_seed'], int)) and (not isinstance(kwargs['random_seed'], np.int32)) and \
                    (not isinstance(kwargs['random_seed'], np.uint32)):
                raise ValueError('`random_seed` is wrong! Expected `{0}`, got `{1}`.'.format(
                    type(3), type(kwargs['random_seed'])))
        if 'sampling_frequency' not in kwargs:
            raise ValueError('`sampling_frequency` is not specified!')
        if (not isinstance(kwargs['sampling_frequency'], int)) and \
                (not isinstance(kwargs['sampling_frequency'], np.int32)) and \
                (not isinstance(kwargs['sampling_frequency'], np.uint32)):
            raise ValueError('`sampling_frequency` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['sampling_frequency'])))
        if kwargs['sampling_frequency'] < 1:
            raise ValueError('`sampling_frequency` is wrong! Expected a positive integer value, '
                             'but {0} is not positive.'.format(kwargs['sampling_frequency']))
        if kwargs['sampling_frequency'] < 8000:
            raise ValueError('`sampling_frequency` is wrong! Minimal admissible value is 8000 Hz.')
        if 'window_size' not in kwargs:
            raise ValueError('`window_size` is not specified!')
        if (not isinstance(kwargs['window_size'], float)) and \
                (not isinstance(kwargs['window_size'], np.float32)) and \
                (not isinstance(kwargs['window_size'], np.float64)):
            raise ValueError('`window_size` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3.5), type(kwargs['window_size'])))
        if kwargs['window_size'] <= 0.0:
            raise ValueError('`window_size` is wrong! Expected a positive floating-point value, '
                             'but {0} is not positive.'.format(kwargs['window_size']))
        n_window = int(round(kwargs['sampling_frequency'] * kwargs['window_size']))
        if n_window < 10:
            raise ValueError('`window_size` is wrong! {0:.6f} is too small value for `window_size`.'.format(
                kwargs['window_size']))
        if 'shift_size' not in kwargs:
            raise ValueError('`shift_size` is not specified!')
        if (not isinstance(kwargs['shift_size'], float)) and \
                (not isinstance(kwargs['shift_size'], np.float32)) and \
                (not isinstance(kwargs['shift_size'], np.float64)):
            raise ValueError('`shift_size` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3.5), type(kwargs['shift_size'])))
        if kwargs['shift_size'] <= 0.0:
            raise ValueError('`shift_size` is wrong! Expected a positive floating-point value, '
                             'but {0} is not positive.'.format(kwargs['shift_size']))
        n_shift = int(round(kwargs['sampling_frequency'] * kwargs['shift_size']))
        if n_shift < 5:
            raise ValueError('`shift_size` is wrong! {0:.6f} is too small value for `shift_size`.'.format(
                kwargs['shift_size']))
        if 'verbose' not in kwargs:
            raise ValueError('`verbose` is not specified!')
        if (not isinstance(kwargs['verbose'], int)) and (not isinstance(kwargs['verbose'], np.int32)) and \
                (not isinstance(kwargs['verbose'], np.uint32)) and \
                (not isinstance(kwargs['verbose'], bool)) and (not isinstance(kwargs['verbose'], np.bool)):
            raise ValueError('`verbose` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(True), type(kwargs['verbose'])))
        if 'warm_start' not in kwargs:
            raise ValueError('`warm_start` is not specified!')
        if (not isinstance(kwargs['warm_start'], int)) and (not isinstance(kwargs['warm_start'], np.int32)) and \
                (not isinstance(kwargs['warm_start'], np.uint32)) and \
                (not isinstance(kwargs['warm_start'], bool)) and (not isinstance(kwargs['warm_start'], np.bool)):
            raise ValueError('`warm_start` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(True), type(kwargs['warm_start'])))
        if 'random_seed' not in kwargs:
            raise ValueError('`random_seed` is not specified!')
        if kwargs['random_seed'] is not None:
            if (not isinstance(kwargs['random_seed'], int)) and (not isinstance(kwargs['random_seed'], np.int32)) and \
                    (not isinstance(kwargs['random_seed'], np.uint32)):
                raise ValueError('`random_seed` is wrong! Expected `{0}`, got `{1}`.'.format(
                    type(3), type(kwargs['random_seed'])))
        if 'cache_dir' not in kwargs:
            raise ValueError('`cache_dir` is not specified!')
        if kwargs['cache_dir'] is not None:
            if (not hasattr(kwargs['cache_dir'], 'split')) or (not hasattr(kwargs['cache_dir'], 'strip')):
                raise ValueError('`cache_dir` is wrong! Expected `{0}`, got `{1}`.'.format(
                    type('3s'), type(kwargs['cache_dir'])))
        n_fft = SoundRecognizer.get_n_fft(kwargs['sampling_frequency'], kwargs['window_size'])
        if SoundRecognizer.IMAGESIZE[1] >= (n_fft // 2):
            raise ValueError('`window_size` is too small for specified sampling frequency!')

    @staticmethod
    def check_X(X: Union[np.ndarray, List[np.ndarray]], X_name: str):
        if (not isinstance(X, list)) and (not isinstance(X, tuple)) and (not isinstance(X, np.ndarray)):
            raise ValueError('{0} is wrong type for `{1}`!'.format(type(X), X_name))
        if isinstance(X, np.ndarray):
            if len(X.shape) != 2:
                raise ValueError('`{0}` is wrong! Expected a 2-D array, but got a {1}-D one!'.format(
                    X_name, len(X.shape)))
        else:
            for idx in range(len(X)):
                if not isinstance(X[idx], np.ndarray):
                    raise ValueError('`{0}`[{1}] is wrong! Expected a {2}, but got a {3}!'.format(
                        X_name, idx, type(np.array([1, 2], dtype=np.float32)), type(X[idx])))
                if len(X[idx].shape) != 1:
                    raise ValueError('`{0}`[{1}] is wrong! Expected a 1-D array, but got a {2}-D one!'.format(
                        X_name, idx, len(X[idx].shape)))

    @staticmethod
    def check_Xy(X: Union[np.ndarray, List[np.ndarray]], X_name: str,
                 y: Union[np.ndarray, List[int], List[str]], y_name: str) -> Tuple[dict, list]:
        SoundRecognizer.check_X(X, X_name)
        n = X.shape[0] if isinstance(X, np.ndarray) else len(X)
        if (not isinstance(y, list)) and (not isinstance(y, tuple)) and (not isinstance(y, np.ndarray)):
            raise ValueError('{0} is wrong type for `{1}`!'.format(type(y), y_name))
        if isinstance(y, np.ndarray):
            if len(y.shape) != 1:
                raise ValueError('`{0}` is wrong! Expected a 1-D array, but got a {1}-D one!'.format(
                    y_name, len(y.shape)))
        if len(y) != n:
            raise ValueError('Size of `{0}` does not correspond to size of `{1}`. {2} != {3}'.format(
                X_name, y_name, n, len(y)
            ))
        classed_dict = dict()
        classes_dict_reverse = list()
        n_classes = 0
        for sample_idx, cur in enumerate(y):
            if cur not in classed_dict:
                if isinstance(cur, int) or isinstance(cur, np.int32) or isinstance(cur, np.int16) or \
                        isinstance(cur, np.int8) or isinstance(cur, np.int64):
                    if cur < 0:
                        if cur < -1:
                            raise ValueError('{0} is inadmissible value for `{1}`[{2}]!'.format(cur, y_name,
                                                                                                sample_idx))
                    else:
                        classed_dict[cur] = n_classes
                        classes_dict_reverse.append(cur)
                        n_classes += 1
                else:
                    classed_dict[cur] = n_classes
                    classes_dict_reverse.append(cur)
                    n_classes += 1
        if n_classes < 2:
            raise ValueError('There are too few classes in the `{0}`!'.format(y_name))
        return classed_dict, classes_dict_reverse

    @staticmethod
    def get_n_fft(sampling_freq: int, window_size: float) -> int:
        n_window = int(round(sampling_freq * window_size))
        n_fft = 2
        while n_fft < n_window:
            n_fft *= 2
        return n_fft

    @staticmethod
    def find_optimal_threshold(probabilities_for_known: np.ndarray, probabilities_for_unknown: np.ndarray) -> float:
        best_threshold = 1e-2
        y_true = np.array(
            [1 for _ in range(probabilities_for_known.shape[0])] +
            [0 for _ in range(probabilities_for_unknown.shape[0])],
            dtype=np.int32
        )
        y_pred = np.concatenate(
            (
                np.asarray(probabilities_for_known >= best_threshold, dtype=np.int32),
                np.asarray(probabilities_for_unknown >= best_threshold, dtype=np.int32)
            )
        )
        best_f1 = f1_score(y_true, y_pred, average='binary')
        threshold = best_threshold + 1e-2
        del y_pred
        while threshold < 1.0:
            y_pred = np.concatenate(
                (
                    np.asarray(probabilities_for_known >= threshold, dtype=np.int32),
                    np.asarray(probabilities_for_unknown >= threshold, dtype=np.int32)
                )
            )
            new_f1 = f1_score(y_true, y_pred, average='binary')
            if new_f1 > best_f1:
                best_f1 = new_f1
                best_threshold = threshold
            threshold += 1e-2
            del y_pred
        return best_threshold


class TrainsetGenerator(keras.utils.Sequence):
    def __init__(self, X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray],
                 batch_size: int, window_size: float, shift_size: float, sampling_frequency: int, melfb: np.ndarray,
                 classes: dict, sample_weight: Union[list, tuple, np.ndarray, None]=None,
                 cache_dir_name: Union[str, None]=None, suffix: str=''):
        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self.batch_size = batch_size
        self.window_size = window_size
        self.shift_size = shift_size
        self.sampling_frequency = sampling_frequency
        self.melfb = melfb
        self.classes = classes
        self.indices = list(filter(lambda it: y[it] != -1, range(len(y))))
        assert len(self.indices) > 2
        assert min(self.indices) >= 0
        assert max(self.indices) < len(y)
        assert len(self.indices) == len(set(self.indices))
        self.cache_dir_name = None if cache_dir_name is None else os.path.normpath(cache_dir_name)
        self.suffix = suffix.strip()
        if len(self.suffix) == 0:
            raise ValueError('Suffix for cached files must be non-empty!')

    def __len__(self):
        return int(np.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = min((idx + 1) * self.batch_size, len(self.indices))
        batch_size = batch_end - batch_start
        if (self.cache_dir_name is None) or \
                (not os.path.isfile(os.path.join(self.cache_dir_name, 'batch_{0}_{1}.pkl'.format(self.suffix, idx)))):
            spectrograms_as_images = np.empty(
                (batch_size, SoundRecognizer.IMAGESIZE[0], SoundRecognizer.IMAGESIZE[1], 3),
                dtype=np.float16
            )
            targets = np.zeros(shape=(batch_size, len(self.classes)), dtype=np.float32)
            for sample_idx in range(batch_start, batch_end):
                spectrogram = SoundRecognizer.sound_to_melspectrogram(
                    sound=self.X[self.indices[sample_idx]], sampling_frequency=self.sampling_frequency,
                    melfb=self.melfb,
                    window_size=self.window_size, shift_size=self.shift_size
                )
                r, g, b = SoundRecognizer.melspectrogram_to_image(spectrogram=spectrogram)
                spectrograms_as_images[sample_idx - batch_start, :, :, 0] = np.asarray(r, dtype=np.float16)
                spectrograms_as_images[sample_idx - batch_start, :, :, 1] = np.asarray(g, dtype=np.float16)
                spectrograms_as_images[sample_idx - batch_start, :, :, 2] = np.asarray(b, dtype=np.float16)
                targets[sample_idx - batch_start][self.classes[self.y[self.indices[sample_idx]]]] = 1.0
            if self.cache_dir_name is not None:
                with open(os.path.join(self.cache_dir_name, 'batch_{0}_{1}.pkl'.format(self.suffix, idx)), 'wb') as fp:
                    pickle.dump((spectrograms_as_images, targets), fp)
        else:
            with open(os.path.join(self.cache_dir_name, 'batch_{0}_{1}.pkl'.format(self.suffix, idx)), 'rb') as fp:
                spectrograms_as_images, targets = pickle.load(fp)
        if self.sample_weight is None:
            return np.asarray(spectrograms_as_images, dtype=np.float32), targets
        sample_weights = np.zeros(shape=(batch_size,), dtype=np.float32)
        for sample_idx in range(batch_start, batch_end):
            sample_weights[sample_idx - batch_start] = self.sample_weight[self.indices[sample_idx]]
        return np.asarray(spectrograms_as_images, dtype=np.float32), targets, sample_weights


class DatasetGenerator(keras.utils.Sequence):
    def __init__(self, X: Union[list, tuple, np.ndarray], batch_size: int, window_size: float, shift_size: float,
                 sampling_frequency: int, melfb: np.ndarray, indices: Union[np.ndarray, List[int], None]=None):
        self.X = X
        self.batch_size = batch_size
        self.window_size = window_size
        self.shift_size = shift_size
        self.sampling_frequency = sampling_frequency
        self.melfb = melfb
        self.indices = indices
        if indices is not None:
            if len(indices) != len(set(indices.tolist()) if isinstance(indices, np.ndarray) else set(indices)):
                raise ValueError('Indices of used data samples are repeated!')
            if (np.min(indices) < 0) or (np.max(indices) >= (X.shape[0] if isinstance(X, np.ndarray) else len(X))):
                raise ValueError('Indices of used data samples are wrong!')

    def __len__(self):
        return int(np.ceil(self.get_number_of_samples() / float(self.batch_size)))

    def __getitem__(self, idx):
        n_samples = self.get_number_of_samples()
        batch_start = idx * self.batch_size
        batch_end = min((idx + 1) * self.batch_size, n_samples)
        batch_size = batch_end - batch_start
        spectrograms_as_images = np.empty((batch_size, SoundRecognizer.IMAGESIZE[0], SoundRecognizer.IMAGESIZE[1], 3),
                                          dtype=np.float16)
        for sample_idx in range(batch_start, batch_end):
            spectrogram = SoundRecognizer.sound_to_melspectrogram(
                sound=self.X[sample_idx if self.indices is None else self.indices[sample_idx]],
                sampling_frequency=self.sampling_frequency, melfb=self.melfb,
                window_size=self.window_size, shift_size=self.shift_size
            )
            r, g, b = SoundRecognizer.melspectrogram_to_image(spectrogram=spectrogram)
            spectrograms_as_images[sample_idx - batch_start, :, :, 0] = np.asarray(r, dtype=np.float16)
            spectrograms_as_images[sample_idx - batch_start, :, :, 1] = np.asarray(g, dtype=np.float16)
            spectrograms_as_images[sample_idx - batch_start, :, :, 2] = np.asarray(b, dtype=np.float16)
        return np.asarray(spectrograms_as_images, dtype=np.float32)

    def get_number_of_samples(self):
        if self.indices is None:
            n_samples = self.X.shape[0] if isinstance(self.X, np.ndarray) else len(self.X)
        else:
            n_samples = len(self.indices)
        return n_samples
