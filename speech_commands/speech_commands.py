import copy
import os
import random
import tempfile
import time
from typing import List, Tuple, Union
import warnings

import keras
import keras.backend as K
import librosa
from matplotlib.pyplot import cm
import numpy as np
from scipy.signal import resample
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import f1_score
from sklearn.utils.validation import check_is_fitted


class MobilenetRecognizer(ClassifierMixin, BaseEstimator):
    COLORMAP = cm.get_cmap('jet')
    IMAGESIZE = (224, 224)

    def __init__(self, sampling_frequency: int=16000, window_size: float=0.025, shift_size: float=0.01,
                 layer_level: int=3, hidden_layers: tuple=(100,), batch_size: int=32, max_epochs: int=100,
                 patience: int=5, verbose: bool=False, warm_start: bool=False, random_seed=None):
        self.sampling_frequency = sampling_frequency
        self.window_size = window_size
        self.shift_size = shift_size
        self.layer_level = layer_level
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_seed = random_seed
        self.warm_start = warm_start
        self.verbose = verbose
        self.hidden_layers = hidden_layers

    def fit(self, X: Union[np.ndarray, List[np.ndarray]], y: Union[np.ndarray, List[int], List[str]], **kwargs):
        self.check_params(sampling_frequency=self.sampling_frequency, window_size=self.window_size,
                          shift_size=self.shift_size, batch_size=self.batch_size, max_epochs=self.max_epochs,
                          patience=self.patience, warm_start=self.warm_start, verbose=self.verbose,
                          random_seed=self.random_seed, layer_level=self.layer_level,
                          hidden_layers=self.hidden_layers)
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
        if 'background' in kwargs:
            self.check_X(kwargs['background'], 'background')
            background_sounds = kwargs['background']
        else:
            background_sounds = None
        self.update_random_seed()
        if not hasattr(self, 'melfb_'):
            self.update_triangle_filters()
        if self.warm_start:
            self.check_is_fitted()
            self.classes_ = classes_dict
            self.classes_reverse_ = classes_dict_reverse
            input_data = keras.layers.Input(shape=(self.IMAGESIZE[0], self.IMAGESIZE[1], 3), name='InputSpectrogram')
            neural_network = self.recognizer_.get_layer('conv1_pad')(input_data)
            neural_network = self.recognizer_.get_layer('conv1')(neural_network)
            neural_network = self.recognizer_.get_layer('conv1_bn')(neural_network)
            neural_network = self.recognizer_.get_layer('conv1_relu')(neural_network)
            for layer_index in range(1, self.layer_level + 1):
                if layer_index in {2, 4, 6, 12}:
                    neural_network = self.recognizer_.get_layer('conv_pad_{0}'.format(layer_index))(neural_network)
                neural_network = self.recognizer_.get_layer('conv_dw_{0}'.format(layer_index))(neural_network)
                neural_network = self.recognizer_.get_layer('conv_dw_{0}_bn'.format(layer_index))(neural_network)
                neural_network = self.recognizer_.get_layer('conv_dw_{0}_relu'.format(layer_index))(neural_network)
                neural_network = self.recognizer_.get_layer('conv_pw_{0}'.format(layer_index))(neural_network)
                neural_network = self.recognizer_.get_layer('conv_pw_{0}_bn'.format(layer_index))(neural_network)
                neural_network = self.recognizer_.get_layer('conv_pw_{0}_relu'.format(layer_index))(neural_network)
            neural_network = self.recognizer_.get_layer('PoolingLayer')(neural_network)
            if len(self.hidden_layers) > 0:
                hidden_layer = self.recognizer_.get_layer('Dropout1')(neural_network)
                hidden_layer = self.recognizer_.get_layer(name='HiddenLayer1')(hidden_layer)
                for layer_index in range(1, len(self.hidden_layers)):
                    hidden_layer = self.recognizer_.get_layer('Dropout{0}')(hidden_layer)
                    hidden_layer = self.recognizer_.get_layer(
                        name='HiddenLayer{0}'.format(layer_index + 1)
                    )(hidden_layer)
                output_layer = keras.layers.Dense(
                    units=len(self.classes_), activation='softmax',
                    kernel_initializer=keras.initializers.glorot_normal(seed=self.random_seed)
                )(self.recognizer_.get_layer('Dropout{0}'.format(len(self.hidden_layers) + 1))(hidden_layer))
            else:
                output_layer = keras.layers.Dense(
                    units=len(self.classes_), activation='softmax',
                    kernel_initializer=keras.initializers.glorot_normal(seed=self.random_seed)
                )(self.recognizer_.get_layer('Dropout1')(neural_network))
            self.recognizer_ = keras.models.Model(input_data, output_layer)
        else:
            self.finalize_model()
            self.classes_ = classes_dict
            self.classes_reverse_ = classes_dict_reverse
            self.min_amplitude_, self.max_amplitude_ = self.calculate_bounds_of_amplitude(
                X, self.window_size, self.shift_size, self.sampling_frequency, self.melfb_
            )
            if self.verbose:
                print('Sampling frequency is {0} Hz.'.format(self.sampling_frequency))
                print('Minimal amplitude value of spectrogram is {0:.6f}.'.format(self.min_amplitude_))
                print('Maximal amplitude value of spectrogram is {0:.6f}.'.format(self.max_amplitude_))
                print('')
            input_data = keras.layers.Input(shape=(self.IMAGESIZE[0], self.IMAGESIZE[1], 3), name='InputSpectrogram')
            mobilenet = keras.applications.mobilenet.MobileNet(
                input_shape=(self.IMAGESIZE[0], self.IMAGESIZE[1], 3), include_top=False, weights='imagenet',
                input_tensor=input_data, pooling='avg')
            neural_network = mobilenet.get_layer('conv1_pad')(input_data)
            neural_network = mobilenet.get_layer('conv1')(neural_network)
            neural_network = mobilenet.get_layer('conv1_bn')(neural_network)
            neural_network = mobilenet.get_layer('conv1_relu')(neural_network)
            for layer_index in range(1, self.layer_level + 1):
                if layer_index in {2, 4, 6, 12}:
                    neural_network = mobilenet.get_layer('conv_pad_{0}'.format(layer_index))(neural_network)
                neural_network = mobilenet.get_layer('conv_dw_{0}'.format(layer_index))(neural_network)
                neural_network = mobilenet.get_layer('conv_dw_{0}_bn'.format(layer_index))(neural_network)
                neural_network = mobilenet.get_layer('conv_dw_{0}_relu'.format(layer_index))(neural_network)
                neural_network = mobilenet.get_layer('conv_pw_{0}'.format(layer_index))(neural_network)
                neural_network = mobilenet.get_layer('conv_pw_{0}_bn'.format(layer_index))(neural_network)
                neural_network = mobilenet.get_layer('conv_pw_{0}_relu'.format(layer_index))(neural_network)
            neural_network = keras.layers.GlobalMaxPooling2D(name='PoolingLayer')(neural_network)
            if len(self.hidden_layers) > 0:
                hidden_layer = keras.layers.Dropout(name='Dropout1', rate=0.3, seed=self.random_seed)(neural_network)
                hidden_layer = keras.layers.Dense(
                    units=self.hidden_layers[0], activation='relu',
                    kernel_initializer=keras.initializers.he_normal(seed=self.random_seed),
                    name='HiddenLayer1'
                )(hidden_layer)
                for layer_index in range(1, len(self.hidden_layers)):
                    hidden_layer = keras.layers.Dropout(name='Dropout{0}'.format(layer_index + 1), rate=0.3,
                                                        seed=self.random_seed)(hidden_layer)
                    hidden_layer = keras.layers.Dense(
                        units=self.hidden_layers[layer_index], activation='relu',
                        kernel_initializer=keras.initializers.he_normal(seed=self.random_seed),
                        name='HiddenLayer{0}'.format(layer_index + 1)
                    )(hidden_layer)
                output_layer = keras.layers.Dense(
                    units=len(self.classes_), activation='softmax',
                    kernel_initializer=keras.initializers.glorot_normal(seed=self.random_seed)
                )(keras.layers.Dropout(name='Dropout{0}'.format(len(self.hidden_layers) + 1), rate=0.3,
                                       seed=self.random_seed)(hidden_layer))
            else:
                output_layer = keras.layers.Dense(
                    units=len(self.classes_), activation='softmax',
                    kernel_initializer=keras.initializers.glorot_normal(seed=self.random_seed)
                )(keras.layers.Dropout(name='Dropout1', rate=0.3, seed=self.random_seed)(neural_network))
            self.recognizer_ = keras.models.Model(input_data, output_layer)
        if 'sample_weight' in kwargs:
            sample_weight = kwargs['sample_weight']
        else:
            sample_weight = None
        trainset_generator = TrainsetGenerator(
            X=X, y=y, batch_size=self.batch_size, melfb=self.melfb_,
            window_size=self.window_size, shift_size=self.shift_size, sampling_frequency=self.sampling_frequency,
            min_amplitude=self.min_amplitude_, max_amplitude=self.max_amplitude_, classes=self.classes_,
            sample_weight=sample_weight, use_augmentation=True, background_sounds=background_sounds
        )
        if (X_val is None) or (y_val is None):
            self.set_trainability_of_model(self.recognizer_, False)
            self.recognizer_.compile(optimizer='nadam', loss='categorical_crossentropy',
                                     metrics=['categorical_accuracy'])
            if self.verbose:
                print('')
                print('Training with frozen base...')
                print('')
                keras.utils.print_summary(self.recognizer_, line_length=120)
            self.recognizer_.fit_generator(trainset_generator, shuffle=True, epochs=self.max_epochs,
                                           verbose=2 if self.verbose else 0)
            self.set_trainability_of_model(self.recognizer_, True)
            self.recognizer_.compile(optimizer='nadam', loss='categorical_crossentropy',
                                     metrics=['categorical_accuracy'])
            if self.verbose:
                print('')
                print('Training with tuned base...')
                print('')
                keras.utils.print_summary(self.recognizer_, line_length=120)
            self.recognizer_.fit_generator(trainset_generator, shuffle=True, epochs=self.max_epochs,
                                           verbose=2 if self.verbose else 0)
            if self.verbose:
                print('')
            indices_of_unknown = list(filter(lambda it: y[it] == -1, range(len(y))))
            if len(indices_of_unknown) > 0:
                indices_of_known = list(filter(lambda it: y[it] != -1, range(len(y))))
                max_probabilities_for_known = self.recognizer_.predict_generator(
                    DatasetGenerator(X=X, batch_size=self.batch_size,
                                     melfb=self.melfb_, window_size=self.window_size, shift_size=self.shift_size,
                                     min_amplitude=self.min_amplitude_, max_amplitude=self.max_amplitude_,
                                     sampling_frequency=self.sampling_frequency, indices=indices_of_known)
                ).max(axis=1)
                max_probabilities_for_unknown = self.recognizer_.predict_generator(
                    DatasetGenerator(X=X, batch_size=self.batch_size,
                                     melfb=self.melfb_, window_size=self.window_size, shift_size=self.shift_size,
                                     min_amplitude=self.min_amplitude_, max_amplitude=self.max_amplitude_,
                                     sampling_frequency=self.sampling_frequency, indices=indices_of_unknown)
                ).max(axis=1)
                self.threshold_ = self.find_optimal_threshold(max_probabilities_for_known,
                                                              max_probabilities_for_unknown)
            else:
                self.threshold_ = self.recognizer_.predict_generator(
                    DatasetGenerator(X=X, batch_size=self.batch_size,
                                     melfb=self.melfb_, window_size=self.window_size, shift_size=self.shift_size,
                                     min_amplitude=self.min_amplitude_, max_amplitude=self.max_amplitude_,
                                     sampling_frequency=self.sampling_frequency)
                ).max(axis=1).min()
        else:
            validset_generator = TrainsetGenerator(
                X=X_val, y=y_val, batch_size=self.batch_size, melfb=self.melfb_, window_size=self.window_size,
                shift_size=self.shift_size, sampling_frequency=self.sampling_frequency, classes=self.classes_,
                min_amplitude=self.min_amplitude_, max_amplitude=self.max_amplitude_, sample_weight=None
            )
            early_stopping_callback = keras.callbacks.EarlyStopping(
                patience=self.patience, verbose=self.verbose, restore_best_weights=True,
                monitor='val_loss', mode='min'
            )
            self.set_trainability_of_model(self.recognizer_, False)
            self.recognizer_.compile(optimizer='nadam', loss='categorical_crossentropy',
                                     metrics=['categorical_accuracy'])
            if self.verbose:
                print('')
                print('Training with frozen base...')
                print('')
                keras.utils.print_summary(self.recognizer_, line_length=120)
            self.recognizer_.fit_generator(trainset_generator, validation_data=validset_generator, shuffle=True,
                                           epochs=self.max_epochs, verbose=2 if self.verbose else 0,
                                           callbacks=[early_stopping_callback])
            self.set_trainability_of_model(self.recognizer_, True)
            self.recognizer_.compile(optimizer='nadam', loss='categorical_crossentropy',
                                     metrics=['categorical_accuracy'])
            if self.verbose:
                print('')
                print('Training with tuned base...')
                print('')
                keras.utils.print_summary(self.recognizer_, line_length=120)
            self.recognizer_.fit_generator(trainset_generator, validation_data=validset_generator, shuffle=True,
                                           epochs=self.max_epochs, verbose=2 if self.verbose else 0,
                                           callbacks=[early_stopping_callback])
            if self.verbose:
                print('')
            indices_of_unknown_for_training = list(filter(lambda it: y[it] == -1, range(len(y))))
            indices_of_unknown_for_validation = list(filter(lambda it: y_val[it] == -1, range(len(y_val))))
            if (len(indices_of_unknown_for_training) + len(indices_of_unknown_for_validation)) > 0:
                indices_of_known = list(filter(lambda it: y_val[it] != -1, range(len(y_val))))
                max_probabilities_for_known = self.recognizer_.predict_generator(
                    DatasetGenerator(X=X_val, batch_size=self.batch_size, melfb=self.melfb_,
                                     window_size=self.window_size, shift_size=self.shift_size,
                                     sampling_frequency=self.sampling_frequency, indices=indices_of_known,
                                     min_amplitude=self.min_amplitude_, max_amplitude=self.max_amplitude_)
                ).max(axis=1)
                if (len(indices_of_unknown_for_training) > 0) and (len(indices_of_unknown_for_validation) > 0):
                    max_probabilities_for_unknown = np.concatenate(
                        (
                            self.recognizer_.predict_generator(
                                DatasetGenerator(X=X, batch_size=self.batch_size, melfb=self.melfb_,
                                                 window_size=self.window_size, shift_size=self.shift_size,
                                                 sampling_frequency=self.sampling_frequency,
                                                 indices=indices_of_unknown_for_training,
                                                 min_amplitude=self.min_amplitude_, max_amplitude=self.max_amplitude_)
                            ).max(axis=1),
                            self.recognizer_.predict_generator(
                                DatasetGenerator(X=X_val, batch_size=self.batch_size, melfb=self.melfb_,
                                                 window_size=self.window_size, shift_size=self.shift_size,
                                                 sampling_frequency=self.sampling_frequency,
                                                 indices=indices_of_unknown_for_validation,
                                                 min_amplitude=self.min_amplitude_, max_amplitude=self.max_amplitude_)
                            ).max(axis=1)
                        )
                    )
                elif len(indices_of_unknown_for_training) > 0:
                    max_probabilities_for_unknown = self.recognizer_.predict_generator(
                        DatasetGenerator(X=X, batch_size=self.batch_size, melfb=self.melfb_,
                                         window_size=self.window_size, shift_size=self.shift_size,
                                         sampling_frequency=self.sampling_frequency,
                                         indices=indices_of_unknown_for_training,
                                         min_amplitude=self.min_amplitude_, max_amplitude=self.max_amplitude_)
                    ).max(axis=1)
                else:
                    max_probabilities_for_unknown = self.recognizer_.predict_generator(
                        DatasetGenerator(X=X_val, batch_size=self.batch_size, melfb=self.melfb_,
                                         window_size=self.window_size, shift_size=self.shift_size,
                                         sampling_frequency=self.sampling_frequency,
                                         indices=indices_of_unknown_for_validation,
                                         min_amplitude=self.min_amplitude_, max_amplitude=self.max_amplitude_)
                    ).max(axis=1)
                self.threshold_ = self.find_optimal_threshold(max_probabilities_for_known,
                                                              max_probabilities_for_unknown)
            else:
                self.threshold_ = self.recognizer_.predict_generator(
                    DatasetGenerator(X=X_val, batch_size=self.batch_size, melfb=self.melfb_,
                                     window_size=self.window_size, shift_size=self.shift_size,
                                     sampling_frequency=self.sampling_frequency,
                                     min_amplitude=self.min_amplitude_, max_amplitude=self.max_amplitude_)
                ).max(axis=1).min()
        if self.verbose:
            print('Best threshold for probability is {0:.2f}.'.format(self.threshold_))
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
        if isinstance(res, np.ndarray):
            return np.array(res, dtype=object)
        return res

    def predict_proba(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        self.check_params(sampling_frequency=self.sampling_frequency, window_size=self.window_size,
                          shift_size=self.shift_size, batch_size=self.batch_size, max_epochs=self.max_epochs,
                          patience=self.patience, warm_start=self.warm_start, verbose=self.verbose,
                          random_seed=self.random_seed, layer_level=self.layer_level, hidden_layers=self.hidden_layers)
        self.check_X(X, 'X')
        self.check_is_fitted()
        if not hasattr(self, 'melfb_'):
            self.update_triangle_filters()
        return self.recognizer_.predict_generator(
            DatasetGenerator(X=X, batch_size=self.batch_size, melfb=self.melfb_, window_size=self.window_size,
                             shift_size=self.shift_size, sampling_frequency=self.sampling_frequency,
                             min_amplitude=self.min_amplitude_, max_amplitude=self.max_amplitude_)
        )

    def predict_log_proba(self, X: Union[list, tuple, np.ndarray]) -> np.ndarray:
        return np.log(np.asarray(self.predict_proba(X), dtype=np.float64) + 1e-9)

    def score(self, X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray],
              sample_weight: Union[list, tuple, np.ndarray, None] = None) -> float:
        y_pred = self.predict(X)
        return f1_score(y, y_pred, average='macro', sample_weight=sample_weight)

    def fit_predict(self, X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray],
                    sample_weight: Union[list, tuple, np.ndarray, None]=None, **kwargs):
        if ('validation_data' in kwargs) and ('background' in kwargs):
            return self.fit(X, y, sample_weight=sample_weight, validation_data=kwargs['validation_data'],
                            background=kwargs['background']).predict(X)
        if 'validation_data' in kwargs:
            return self.fit(X, y, sample_weight=sample_weight, validation_data=kwargs['validation_data']).predict(X)
        if 'background' in kwargs:
            return self.fit(X, y, sample_weight=sample_weight, background=kwargs['background']).predict(X)
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
        self.melfb_ = librosa.filters.mel(sr=self.sampling_frequency, n_fft=n_fft, n_mels=self.IMAGESIZE[1] // 2,
                                          fmin=350.0, fmax=6000.0)

    def check_is_fitted(self):
        check_is_fitted(self, ['recognizer_', 'classes_', 'classes_reverse_', 'threshold_', 'min_amplitude_',
                               'max_amplitude_'])

    def get_params(self, deep=True):
        return {'sampling_frequency': self.sampling_frequency, 'window_size': self.window_size,
                'layer_level': self.layer_level, 'hidden_layers': copy.copy(self.hidden_layers),
                'shift_size': self.shift_size, 'batch_size': self.batch_size, 'max_epochs': self.max_epochs,
                'patience': self.patience, 'verbose': self.verbose, 'warm_start': self.warm_start,
                'random_seed': self.random_seed}

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
            verbose=self.verbose, random_seed=self.random_seed, layer_level=self.layer_level,
            hidden_layers=copy.copy(self.hidden_layers)
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
            verbose=self.verbose, random_seed=self.random_seed, layer_level=self.layer_level,
            hidden_layers=copy.deepcopy(self.hidden_layers)
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
            params['max_amplitude_'] = self.max_amplitude_
            params['min_amplitude_'] = self.min_amplitude_
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
                    ('threshold_' in new_params) and ('model_data_' in new_params) and \
                    ('max_amplitude_' in new_params) and ('min_amplitude_' in new_params)
        if is_fitted:
            self.set_params(**new_params)
            self.classes_reverse_ = copy.deepcopy(new_params['classes_reverse_'])
            self.classes_ = copy.deepcopy(new_params['classes_'])
            self.threshold_ = new_params['threshold_']
            self.max_amplitude_ = new_params['max_amplitude_']
            self.min_amplitude_ = new_params['min_amplitude_']
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
    def set_trainability_of_model(model: keras.models.Model, trainability: bool):
        for cur_layer in model.layers:
            if cur_layer.name.startswith('conv'):
                cur_layer.trainable = trainability

    @staticmethod
    def get_temp_model_name() -> str:
        return tempfile.NamedTemporaryFile(mode='w', suffix='sound_recognizer.h5y').name

    @staticmethod
    def calculate_bounds_of_amplitude(sounds: Union[np.ndarray, List[np.ndarray]], window_size: float,
                                      shift_size: float, sampling_frequency: int,
                                      melfb: np.ndarray) -> Tuple[float, float]:
        values = []
        n_sounds = sounds.shape[0] if isinstance(sounds, np.ndarray) else len(sounds)
        for sound_idx in range(n_sounds):
            new_spectrogram = MobilenetRecognizer.sound_to_melspectrogram(sounds[sound_idx], window_size, shift_size,
                                                                          sampling_frequency, melfb)
            if new_spectrogram is not None:
                values.append(
                    np.resize(
                        new_spectrogram,
                        new_shape=(new_spectrogram.shape[0] * new_spectrogram.shape[1],)
                    )
                )
                del new_spectrogram
        values = np.sort(np.concatenate(values))
        n = int(round(0.02 * (values.shape[0] - 1)))
        return values[n], values[values.shape[0] - 1 - n]

    @staticmethod
    def strip_sound(sound: np.ndarray) -> int:
        sample_idx = sound.shape[0] - 1
        while sample_idx >= 0:
            if abs(sound[sample_idx]) > 1e-6:
                break
            sample_idx -= 1
        return  sample_idx + 1

    @staticmethod
    def sound_to_melspectrogram(sound: np.ndarray, window_size: float, shift_size: float, sampling_frequency: int,
                                melfb: np.ndarray) -> Union[np.ndarray, None]:
        n_window = int(round(sampling_frequency * window_size))
        n_shift = int(round(sampling_frequency * shift_size))
        n_fft = MobilenetRecognizer.get_n_fft(sampling_frequency, window_size)
        sound_length = MobilenetRecognizer.strip_sound(sound)
        if sound_length == 0:
            warnings.warn('Sound is empty!')
            return None
        if sound_length < (n_window + n_shift):
            warnings.warn('Sound is too short!')
            return None
        specgram = librosa.core.stft(y=sound[0:sound_length], n_fft=n_fft, hop_length=n_shift, win_length=n_window,
                                     window='hamming')
        specgram = np.asarray(np.absolute(specgram), dtype=np.float64)
        return np.dot(melfb, specgram).transpose()

    @staticmethod
    def normalize_melspectrogram(spectrogram: Union[np.ndarray, None],
                                 amplitude_bounds: Union[Tuple[float, float], None]=None) -> Union[np.ndarray, None]:
        if spectrogram is None:
            return np.zeros(shape=(MobilenetRecognizer.IMAGESIZE[0], MobilenetRecognizer.IMAGESIZE[1] // 2),
                            dtype=np.float32)
        if amplitude_bounds is None:
            values = np.sort(spectrogram.reshape((spectrogram.shape[0] * spectrogram.shape[1],)))
            n = int(round(0.02 * values.shape[0]))
            max_value = values[-n - 1]
            min_value = values[n]
            del values
        else:
            max_value = amplitude_bounds[1]
            min_value = amplitude_bounds[0]
        normalized = np.asarray(spectrogram - min_value, dtype=np.float64)
        if max_value > min_value:
            normalized /= (max_value - min_value)
            np.putmask(normalized, normalized < 0.0, 0.0)
            np.putmask(normalized, normalized > 1.0, 1.0)
            if normalized.shape[0] < MobilenetRecognizer.IMAGESIZE[0]:
                normalized = np.vstack(
                    (
                        normalized,
                        np.zeros((MobilenetRecognizer.IMAGESIZE[0] - normalized.shape[0], normalized.shape[1]),
                                 dtype=normalized.dtype)
                    )
                )
            elif normalized.shape[0] > MobilenetRecognizer.IMAGESIZE[0]:
                normalized = normalized[0:MobilenetRecognizer.IMAGESIZE[0]]
        else:
            normalized = np.zeros(shape=(MobilenetRecognizer.IMAGESIZE[0], MobilenetRecognizer.IMAGESIZE[1] // 2),
                                  dtype=np.float32)
        return np.asarray(normalized, dtype=np.float32)

    @staticmethod
    def spectrograms_to_images(normalized_spectrograms: np.ndarray) -> np.ndarray:
        if len(normalized_spectrograms.shape) != 3:
            raise ValueError('Normalized spectrograms are wrong! Expected a 3-D array, but got a {0}-D one.'.format(
                len(normalized_spectrograms.shape)))
        if (normalized_spectrograms.shape[1] != MobilenetRecognizer.IMAGESIZE[0]) or \
                ((normalized_spectrograms.shape[2] != (MobilenetRecognizer.IMAGESIZE[1] // 2))):
            raise ValueError('Sizes of normalized spectrogram are wrong! Expected ({0}, {1}), got ({2}, {3}).'.format(
                MobilenetRecognizer.IMAGESIZE[0], MobilenetRecognizer.IMAGESIZE[1] // 2,
                normalized_spectrograms.shape[1], normalized_spectrograms.shape[2]
            ))
        return keras.applications.mobilenet.preprocess_input(
            MobilenetRecognizer.COLORMAP(np.repeat(normalized_spectrograms, 2, axis=2))[:, :, :, 0:3] * 255.0
        )

    @staticmethod
    def check_params(**kwargs):
        if 'layer_level' not in kwargs:
            raise ValueError('`layer_level` is not specified!')
        if (not isinstance(kwargs['layer_level'], int)) and (not isinstance(kwargs['layer_level'], np.int32)) and \
                (not isinstance(kwargs['layer_level'], np.uint32)):
            raise ValueError('`layer_level` is wrong! Expected `{0}`, got `{1}`.'.format(
                type(3), type(kwargs['layer_level'])))
        if kwargs['layer_level'] < 1:
            raise ValueError('`layer_level` is wrong! Expected a positive integer value in the range from 1 to 13, '
                             'but {0} is less than 1.'.format(kwargs['layer_level']))
        if kwargs['layer_level'] > 13:
            raise ValueError('`layer_level` is wrong! Expected a positive integer value in the range from 1 to 13, '
                             'but {0} is greater than 13.'.format(kwargs['layer_level']))
        if 'hidden_layers' not in kwargs:
            raise ValueError('`hidden_layers` is not specified!')
        if (not isinstance(kwargs['hidden_layers'], tuple)) and (not isinstance(kwargs['hidden_layers'], list)) and \
                (not isinstance(kwargs['hidden_layers'], np.ndarray)):
            raise ValueError('`hidden_layers` is wrong! Expected `{0}`, got `{1}`.'.format(
                type((3, 4, 5)), type(kwargs['hidden_layers'])))
        if isinstance(kwargs['hidden_layers'], np.ndarray):
            if len(kwargs['hidden_layers'].shape) != 1:
                raise ValueError('`hidden_layers` is wrong! Expected a 1-D array, got {0}-D one.'.format(
                    len(kwargs['hidden_layers'].shape)))
        if len(kwargs['hidden_layers']) > 0:
            for layer_idx in range(len(kwargs['hidden_layers'])):
                if kwargs['hidden_layers'][layer_idx] != int(kwargs['hidden_layers'][layer_idx]):
                    raise ValueError('`hidden_layers` is wrong! {0} is wrong size of layer {1}.'.format(
                        kwargs['hidden_layers'][layer_idx], layer_idx + 1))
                if kwargs['hidden_layers'][layer_idx] < 1:
                    raise ValueError('`hidden_layers` is wrong! {0} is wrong size of layer {1}.'.format(
                        kwargs['hidden_layers'][layer_idx], layer_idx + 1))
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
        if kwargs['sampling_frequency'] < 16000:
            raise ValueError('`sampling_frequency` is wrong! Minimal admissible value is 16000 Hz.')
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
        n_fft = MobilenetRecognizer.get_n_fft(kwargs['sampling_frequency'], kwargs['window_size'])
        if (MobilenetRecognizer.IMAGESIZE[1] // 2) >= (n_fft // 3):
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
        MobilenetRecognizer.check_X(X, X_name)
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

    @staticmethod
    def select_labeled_samples(y: Union[list, tuple, np.ndarray]) -> tuple:
        return tuple(filter(
            lambda it: ((y[it] != '-1') if (hasattr(y[it], 'split') and hasattr(y[it], 'strip')) else (y[it] != -1)),
            range(len(y))
        ))


class TrainsetGenerator(keras.utils.Sequence):
    def __init__(self, X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray], batch_size: int,
                 window_size: float, shift_size: float, sampling_frequency: int, melfb: np.ndarray,
                 min_amplitude: float, max_amplitude: float, classes: dict,
                 sample_weight: Union[list, tuple, np.ndarray, None]=None, use_augmentation: bool=False,
                 background_sounds: Union[list, tuple, np.ndarray, None]=None):
        self.X = X
        self.y = y
        self.sample_weight = sample_weight
        self.batch_size = batch_size
        self.window_size = window_size
        self.shift_size = shift_size
        self.sampling_frequency = sampling_frequency
        self.melfb = melfb
        self.classes = classes
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.indices = MobilenetRecognizer.select_labeled_samples(y)
        self.use_augmentation = use_augmentation
        self.background_sounds = background_sounds
        assert len(self.indices) > 2
        assert min(self.indices) >= 0
        assert max(self.indices) < len(y)
        assert len(self.indices) == len(set(self.indices))

    def __len__(self):
        return int(np.ceil(len(self.indices) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = min((idx + 1) * self.batch_size, len(self.indices))
        batch_size = batch_end - batch_start
        normalized_spectrograms = np.empty(
            (batch_size, MobilenetRecognizer.IMAGESIZE[0], MobilenetRecognizer.IMAGESIZE[1] // 2),
            dtype=np.float32
        )
        targets = np.zeros(shape=(batch_size, len(self.classes)), dtype=np.float32)
        for sample_idx in range(batch_start, batch_end):
            source_sound = self.X[self.indices[sample_idx]]
            sound_length = MobilenetRecognizer.strip_sound(source_sound)
            if sound_length == 0:
                normalized_spectrograms[sample_idx - batch_start] = MobilenetRecognizer.normalize_melspectrogram(
                    spectrogram=None, amplitude_bounds=(self.min_amplitude, self.max_amplitude)
                )
            else:
                if self.use_augmentation:
                    bins_per_octave = 24
                    pitch_pm = 4
                    pitch_change = pitch_pm * 2 * (np.random.uniform() - 0.5)
                    augmented_sound = librosa.effects.pitch_shift(
                        source_sound[0:sound_length].astype("float64"), self.sampling_frequency, n_steps=pitch_change,
                        bins_per_octave=bins_per_octave
                    )
                    if self.background_sounds != None:
                        number_of_background_sounds = self.background_sounds.shape[0] \
                            if isinstance(self.background_sounds, np.ndarray) else len(self.background_sounds)
                        background_sound = self.background_sounds[random.randint(0, number_of_background_sounds - 1)]
                        sound_length = MobilenetRecognizer.strip_sound(background_sound)
                        if sound_length > 0:
                            if sound_length > len(augmented_sound):
                                sound_start = random.randint(0, sound_length - len(augmented_sound))
                                background_sound = background_sound[sound_start:(sound_start + len(augmented_sound))]
                            elif sound_length < len(augmented_sound):
                                sound_start = random.randint(0, len(augmented_sound) - sound_length)
                                if (sound_start > 0) and ((sound_start + sound_length) < len(augmented_sound)):
                                    background_sound = np.concatenate(
                                        (
                                            np.zeros((sound_start,), dtype=background_sound.dtype),
                                            background_sound,
                                            np.zeros((len(augmented_sound) - (sound_start + sound_length),),
                                                     dtype=background_sound.dtype)
                                        )
                                    )
                                elif sound_start > 0:
                                    background_sound = np.concatenate(
                                        (
                                            np.zeros((sound_start,), dtype=background_sound.dtype),
                                            background_sound,
                                        )
                                    )
                                else:
                                    background_sound = np.concatenate(
                                        (
                                            background_sound,
                                            np.zeros((len(augmented_sound) - (sound_start + sound_length),),
                                                     dtype=background_sound.dtype)
                                        )
                                    )
                            augmented_sound = augmented_sound * np.random.uniform(0.8, 1.2) + \
                                              background_sound.astype("float64") * np.random.uniform(0, 0.1)
                    spectrogram = MobilenetRecognizer.sound_to_melspectrogram(
                        sound=augmented_sound, sampling_frequency=self.sampling_frequency,
                        melfb=self.melfb, window_size=self.window_size, shift_size=self.shift_size,
                    )
                else:
                    spectrogram = MobilenetRecognizer.sound_to_melspectrogram(
                        sound=source_sound[0:sound_length], sampling_frequency=self.sampling_frequency,
                        melfb=self.melfb, window_size=self.window_size, shift_size=self.shift_size,
                    )
                if self.use_augmentation:
                    speed_rate = np.random.uniform(0.9, 1.1)
                    new_length = int(round(speed_rate * spectrogram.shape[0]))
                    if (new_length != spectrogram.shape[0]) and (new_length > 2):
                        spectrogram = resample(spectrogram, num=new_length, axis=0)
                normalized_spectrograms[sample_idx - batch_start] = MobilenetRecognizer.normalize_melspectrogram(
                    spectrogram=spectrogram, amplitude_bounds=(self.min_amplitude, self.max_amplitude)
                )
            targets[sample_idx - batch_start][self.classes[self.y[self.indices[sample_idx]]]] = 1.0
        spectrograms_as_images = MobilenetRecognizer.spectrograms_to_images(normalized_spectrograms)
        del normalized_spectrograms
        if self.sample_weight is None:
            return spectrograms_as_images, targets
        sample_weights = np.zeros(shape=(batch_size,), dtype=np.float32)
        for sample_idx in range(batch_start, batch_end):
            sample_weights[sample_idx - batch_start] = self.sample_weight[self.indices[sample_idx]]
        return spectrograms_as_images, targets, sample_weights


class DatasetGenerator(keras.utils.Sequence):
    def __init__(self, X: Union[list, tuple, np.ndarray], batch_size: int, window_size: float, shift_size: float,
                 sampling_frequency: int, melfb: np.ndarray, min_amplitude: float, max_amplitude: float,
                 indices: Union[np.ndarray, List[int], None]=None):
        self.X = X
        self.batch_size = batch_size
        self.window_size = window_size
        self.shift_size = shift_size
        self.sampling_frequency = sampling_frequency
        self.melfb = melfb
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
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
        normalized_spectrograms = np.empty(
            (batch_size, MobilenetRecognizer.IMAGESIZE[0], MobilenetRecognizer.IMAGESIZE[1] // 2,),
            dtype=np.float32
        )
        for sample_idx in range(batch_start, batch_end):
            spectrogram = MobilenetRecognizer.sound_to_melspectrogram(
                sound=self.X[sample_idx if self.indices is None else self.indices[sample_idx]],
                sampling_frequency=self.sampling_frequency, melfb=self.melfb,
                window_size=self.window_size, shift_size=self.shift_size
            )
            normalized_spectrograms[sample_idx - batch_start] = MobilenetRecognizer.normalize_melspectrogram(
                spectrogram, amplitude_bounds=(self.min_amplitude, self.max_amplitude)
            )
        return MobilenetRecognizer.spectrograms_to_images(normalized_spectrograms)

    def get_number_of_samples(self):
        if self.indices is None:
            n_samples = self.X.shape[0] if isinstance(self.X, np.ndarray) else len(self.X)
        else:
            n_samples = len(self.indices)
        return n_samples


class DTWRecognizer(ClassifierMixin, BaseEstimator):
    N_MELS = 40

    def __init__(self, k: int=3, sampling_frequency: int=16000, window_size: float=0.025, shift_size: float=0.01,
                 warm_start: bool=False, verbose: bool=False):
        self.sampling_frequency = sampling_frequency
        self.window_size = window_size
        self.shift_size = shift_size
        self.verbose = verbose
        self.warm_start = warm_start
        self.k = k

    def fit(self, X: Union[np.ndarray, List[np.ndarray]], y: Union[np.ndarray, List[int], List[str]], **kwargs):
        self.check_params(sampling_frequency=self.sampling_frequency, window_size=self.window_size,
                          shift_size=self.shift_size,warm_start=self.warm_start, verbose=self.verbose, k=self.k)
        classes_dict, classes_dict_reverse = MobilenetRecognizer.check_Xy(X, 'X', y, 'y')
        if self.warm_start:
            self.check_is_fitted()
            class_idx = len(self.classes_)
            for cur in classes_dict:
                if cur not in self.classes_:
                    self.classes_[cur] = class_idx
                    class_idx += 1
                    self.classes_reverse_.append(cur)
        else:
            self.classes_ = classes_dict
            self.classes_reverse_ = classes_dict_reverse
            if hasattr(self, 'patterns_'):
                del self.patterns_
            self.patterns_ = dict()
        melspecgrams = self.sounds_to_melspecgrams(X)
        for sample_idx in range(len(melspecgrams)):
            if (y[sample_idx] != -1) and (y[sample_idx] != '-1'):
                if y[sample_idx] in self.patterns_:
                    self.patterns_[y[sample_idx]].append(melspecgrams[sample_idx])
                else:
                    self.patterns_[y[sample_idx]] = [melspecgrams[sample_idx]]
        for class_name in self.patterns_.keys():
            if len(self.patterns_[class_name]) < self.k:
                raise ValueError('There are too few sounds for the class `{0}!` Minimal number of sounds is {1}, '
                                 'but now there are only {2} sounds.'.format(class_name, self.k,
                                                                             len(self.patterns_[class_name])))
        if self.verbose:
            print('Classes for recognition:')
            for class_name in sorted(list(self.patterns_.keys())):
                print('  - {0} ({1} sounds);'.format(class_name, len(self.patterns_[class_name])))
        self.update_threshold(melspecgrams=melspecgrams, labels=y)
        return self

    def predict(self, X: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        probabilities = self.predict_proba(X=X)
        indices_of_classes = probabilities.argmax(axis=1)
        max_probabilities = probabilities.max(axis=1)
        res = []
        for idx in range(probabilities.shape[0]):
            if max_probabilities[idx] < self.threshold_:
                res.append(-1)
            else:
                res.append(self.classes_reverse_[indices_of_classes[idx]])
        return np.array(res, dtype=object)

    def predict_proba(self, X: Union[np.ndarray, List[np.ndarray], None]=None, **kwargs) -> np.ndarray:
        self.check_params(sampling_frequency=self.sampling_frequency, window_size=self.window_size,
                          shift_size=self.shift_size, warm_start=self.warm_start, verbose=self.verbose, k=self.k)
        self.check_is_fitted()
        if 'melspecgrams' in kwargs:
            if not isinstance(kwargs['melspecgrams'], list):
                raise ValueError('Mel-spectrograms are wrong! Expected a list of mel-spectrograms, but got a '
                                 '`{0}!`'.format(type(kwargs['melspecgrams'])))
            for sound_idx in range(len(kwargs['melspecgrams'])):
                if not isinstance(kwargs['melspecgrams'][sound_idx], np.ndarray):
                    raise ValueError('Item {0} of the mel-spectrograms list is wrong! Expected a `{1}`, but got a '
                                     '`{2}`!'.format(sound_idx, type(np.array([1, 2])),
                                                     type(kwargs['melspecgrams'][sound_idx])))
                if len(kwargs['melspecgrams'][sound_idx].shape) != 2:
                    raise ValueError('Item {0} of the mel-spectrograms list is wrong! Expected a 2-D array, but got a '
                                     '{1}-D one!`'.format(sound_idx, len(kwargs['melspecgrams'][sound_idx].shape)))
                if kwargs['melspecgrams'][sound_idx].shape[0] != self.N_MELS:
                    raise ValueError('Item {0} of the melspecgrams list is wrong, because number of mel-spectrograms '
                                     'does not equal to {1}!'.format(sound_idx, self.N_MELS))
            input_melspecgrams = kwargs['melspecgrams']
        else:
            if X is None:
                raise ValueError('Input data are not specified!')
            MobilenetRecognizer.check_X(X, 'X')
            input_melspecgrams = self.sounds_to_melspecgrams(X)
        res = np.empty((len(input_melspecgrams), len(self.classes_)))
        for sound_idx in range(res.shape[0]):
            for class_name in self.classes_:
                class_idx = self.classes_[class_name]
                distances = []
                D, wp = librosa.sequence.dtw(input_melspecgrams[sound_idx], self.patterns_[class_name][0],
                                             metric='cosine')
                distances.append(D[len(input_melspecgrams[sound_idx]) - 1][len(self.patterns_[class_name][0]) - 1])
                del D, wp
                for pattern in self.patterns_[class_name][1:]:
                    D, wp = librosa.sequence.dtw(input_melspecgrams[sound_idx], pattern, metric='cosine')
                    distances.append(D[len(input_melspecgrams[sound_idx]) - 1][len(self.patterns_[class_name][0]) - 1])
                distances.sort()
                res[sound_idx][class_idx] = sum(map(lambda it: 1.0 / (it * it + 1e-9), distances[:self.k]))
            res[sound_idx] /= res[sound_idx].sum()
        return res

    def predict_log_proba(self, X: Union[list, tuple, np.ndarray]) -> np.ndarray:
        return np.log(np.asarray(self.predict_proba(X), dtype=np.float64) + 1e-9)

    def score(self, X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray],
              sample_weight: Union[list, tuple, np.ndarray, None] = None) -> float:
        y_pred = self.predict(X)
        return f1_score(y, y_pred, average='macro', sample_weight=sample_weight)

    def fit_predict(self, X: Union[list, tuple, np.ndarray], y: Union[list, tuple, np.ndarray],
                    sample_weight: Union[list, tuple, np.ndarray, None]=None, **kwargs):
        return self.fit(X, y)

    def check_is_fitted(self):
        check_is_fitted(self, ['patterns_', 'classes_', 'classes_reverse_', 'threshold_'])

    def update_threshold(self, labels: Union[np.ndarray, List[int], List[str]],
                         sounds: Union[np.ndarray, List[np.ndarray], None]=None,
                         melspecgrams: Union[List[np.ndarray], None]=None):
        if (sounds is None) and (melspecgrams is None):
            raise ValueError('Neither sounds nor melspecgrams are specified!')
        if melspecgrams is None:
            melspecgrams_ = self.sounds_to_melspecgrams(sounds)
        else:
            melspecgrams_ = melspecgrams
        self.threshold_ = 0.0
        self.check_is_fitted()
        max_probabilities = self.predict_proba(melspecgrams=melspecgrams_).max(axis=1)
        y_true = np.zeros(shape=max_probabilities.shape, dtype=np.int32)
        for sample_idx in range(len(labels)):
            if (labels[sample_idx] != -1) and (labels[sample_idx] != '-1'):
                y_true[sample_idx] = 1
        if y_true.min() == 1:
            self.threshold_ = max_probabilities.min()
        else:
            best_threshold = 1e-2
            y_pred = np.asarray(max_probabilities >= best_threshold, dtype=np.int32)
            best_f1 = f1_score(y_true, y_pred)
            cur_threshold = best_threshold + 1e-2
            del y_pred
            while cur_threshold < 1.0:
                y_pred = np.asarray(max_probabilities >= cur_threshold, dtype=np.int32)
                cur_f1 = f1_score(y_true, y_pred)
                if cur_f1 > best_f1:
                    best_f1 = cur_f1
                    best_threshold = cur_threshold
                cur_threshold += 1e-2
            self.threshold_ = best_threshold
        if self.verbose:
            print('Best threshold of probability is {0:.2f}.'.format(self.threshold_))

    def sounds_to_melspecgrams(self, sounds: Union[np.ndarray, List[np.ndarray]]) -> List[np.ndarray]:
        list_of_melspecgrams = []
        n_window = int(round(self.sampling_frequency * self.window_size))
        n_shift = int(round(self.sampling_frequency * self.shift_size))
        n_fft = MobilenetRecognizer.get_n_fft(self.sampling_frequency, self.window_size)
        if not hasattr(self, 'melfb_'):
            self.update_triangle_filters()
        for sound_idx in range(sounds.shape[0] if isinstance(sounds, np.ndarray) else len(sounds)):
            specgram = librosa.core.stft(y=sounds[sound_idx], n_fft=n_fft, hop_length=n_shift, win_length=n_window,
                                         window='hamming')
            specgram = np.asarray(np.absolute(specgram), dtype=np.float64)
            mel_specgram = np.dot(self.melfb_, specgram)
            values = mel_specgram.reshape((mel_specgram.shape[0] * mel_specgram.shape[1],))
            values = np.sort(values)
            n = int(round(0.02 * (len(values) - 1)))
            min_value = values[n]
            max_value = values[-n]
            del values
            if max_value > min_value:
                mel_specgram = (mel_specgram - min_value) / (max_value - min_value)
                np.putmask(mel_specgram, mel_specgram < 0.0, 0.0)
                np.putmask(mel_specgram, mel_specgram > 1.0, 1.0)
            else:
                mel_specgram = np.zeros(shape=mel_specgram.shape, dtype=mel_specgram.dtype)
            list_of_melspecgrams.append(mel_specgram)
        return list_of_melspecgrams

    def update_triangle_filters(self):
        n_fft = MobilenetRecognizer.get_n_fft(self.sampling_frequency, self.window_size)
        if hasattr(self, 'melfb_'):
            del self.melfb_
        self.melfb_ = librosa.filters.mel(sr=self.sampling_frequency, n_fft=n_fft, n_mels=self.N_MELS,
                                          fmin=350.0, fmax=6000.0)

    @staticmethod
    def check_params(**kwargs):
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
        if kwargs['sampling_frequency'] < 16000:
            raise ValueError('`sampling_frequency` is wrong! Minimal admissible value is 16000 Hz.')
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
        if 'k' not in kwargs:
            raise ValueError('`k` is not specified!')
        if (not isinstance(kwargs['k'], int)) and \
                (not isinstance(kwargs['k'], np.int32)) and \
                (not isinstance(kwargs['k'], np.uint32)):
            raise ValueError('`k` is wrong! Expected `{0}`, got `{1}`.'.format(type(3), type(kwargs['k'])))
        if kwargs['k'] < 1:
            raise ValueError('`k` is wrong! Expected a positive integer value, '
                             'but {0} is not positive.'.format(kwargs['k']))

    def get_params(self, deep=True):
        return {'sampling_frequency': self.sampling_frequency, 'window_size': self.window_size,
                'shift_size': self.shift_size, 'k': self.k, 'verbose': self.verbose, 'warm_start': self.warm_start}

    def set_params(self, **params):
        for parameter, value in params.items():
            self.__setattr__(parameter, value)
        return self

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.set_params(
            sampling_frequency=self.sampling_frequency, window_size=self.window_size, shift_size=self.shift_size,
            k=self.k, warm_start=self.warm_start, verbose=self.verbose
        )
        try:
            self.check_is_fitted()
            is_fitted = True
        except:
            is_fitted = False
        if is_fitted:
            result.classes_ = copy.copy(self.classes_)
            result.classes_reverse_ = copy.copy(self.classes_reverse_)
            result.patterns_ = copy.copy(self.patterns_)
        return result

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        result.set_params(
            sampling_frequency=self.sampling_frequency, window_size=self.window_size, shift_size=self.shift_size,
            k=self.k, warm_start=self.warm_start, verbose=self.verbose
        )
        try:
            self.check_is_fitted()
            is_fitted = True
        except:
            is_fitted = False
        if is_fitted:
            result.classes_ = copy.deepcopy(self.classes_)
            result.classes_reverse_ = copy.deepcopy(self.classes_reverse_)
            result.patterns_ = copy.deepcopy(self.patterns_)
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
            params['patterns_'] = copy.deepcopy(self.patterns_)
        return params

    def load_all(self, new_params: dict):
        if not isinstance(new_params, dict):
            raise ValueError('`new_params` is wrong! Expected `{0}`, got `{1}`.'.format(type({0: 1}), type(new_params)))
        self.check_params(**new_params)
        is_fitted = ('classes_' in new_params) and ('classes_reverse_' in new_params) and \
                    ('threshold_' in new_params) and ('patterns_' in new_params)
        if is_fitted:
            self.set_params(**new_params)
            self.classes_reverse_ = copy.deepcopy(new_params['classes_reverse_'])
            self.classes_ = copy.deepcopy(new_params['classes_'])
            self.threshold_ = new_params['threshold_']
            self.patterns_ = new_params['patterns_']
        else:
            self.set_params(**new_params)
            if hasattr(self, 'patterns_'):
                del self.patterns_
            if hasattr(self, 'classes_reverse_'):
                del self.classes_reverse_
            if hasattr(self, 'classes_'):
                del self.classes_
            if hasattr(self, 'threshold_'):
                del self.threshold_
        return self
