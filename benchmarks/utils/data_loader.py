import os

import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp
from jax import Array
import tensorflow as tf

# sklearn
from sklearn import datasets as skl_datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# libsvm
import libsvmdata

# TFDS
# import tensorflow as tf
import tensorflow_datasets as tfds

# HuggingFace Datasets
import datasets as hf_datasets


def load_data(
        dataset_id: str,
        test_size: float,
        seed: int = 1337,
) -> tuple[tuple[Array, Array, Array, Array], bool, int | None]:
    # folder for cached datasets
    data_dir = os.path.join('../..', 'artifacts', 'data')

    # split is always the same for reproducibility, think, a hold-out set
    # dataset is then shuffled with the provided seed
    split_seed = 1337
    Y_train_labels, Y_test_labels = None, None  # for classification
    n_classes = None  # filled only for classification

    # ------------- SKLEARN DATASETS -------------
    if dataset_id == 'diabetes':
        # already scaled
        X, Y = skl_datasets.load_diabetes(return_X_y=True, scaled=True)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=split_seed)

    elif dataset_id == 'california_housing':
        X, Y = skl_datasets.fetch_california_housing(return_X_y=True)

        X_scaled = StandardScaler(copy=False).fit_transform(X)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X_scaled, Y, test_size=test_size, random_state=split_seed)

    elif dataset_id == 'breast_cancer':
        X, test_labels = skl_datasets.load_breast_cancer(return_X_y=True)

        X_scaled = StandardScaler(copy=False).fit_transform(X)

        X_train, X_test, Y_train_labels, Y_test_labels = train_test_split(
            X_scaled, test_labels, test_size=test_size, random_state=split_seed)

        n_classes = 1

    elif dataset_id == 'iris':
        X, test_labels = skl_datasets.load_iris(return_X_y=True)

        X_scaled = StandardScaler(copy=False).fit_transform(X)

        X_train, X_test, Y_train_labels, Y_test_labels = train_test_split(
            X_scaled, test_labels, test_size=test_size, random_state=split_seed)

        n_classes = 3

    # ------------- HUGGINGFACE DATASETS -------------
    elif dataset_id == 'superconduct':
        hf_dataset = hf_datasets.load_dataset(
            'inria-soda/tabular-benchmark', data_files='reg_num/superconduct.csv', cache_dir=data_dir)
        df = hf_dataset['train'].to_pandas()

        Y = df['criticaltemp'].to_numpy()
        X = df.drop(columns=['criticaltemp']).to_numpy()

        X_scaled = StandardScaler(copy=False).fit_transform(X)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X_scaled, Y, test_size=test_size, random_state=split_seed)


    # ------------- TFDS DATASETS -------------
    elif dataset_id == 'diamonds':
        data = _load_from_tfds(dataset_id, data_dir, is_clf=False)

        feature_map, Y = data['train']

        # OHE encoding or Scaling
        cat_features = ['clarity', 'color', 'cut', ]
        skip_cols = []
        X = _ohe_or_scale_columns(feature_map, cat_features, skip_cols)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=split_seed)

    elif dataset_id == 'wine_quality':
        data = _load_from_tfds(dataset_id, data_dir, is_clf=True)

        feature_map, Y = data['train']['features'], data['train']['quality']

        # OHE encoding or Scaling
        cat_features = []
        skip_cols = []
        X = _ohe_or_scale_columns(feature_map, cat_features, skip_cols)

        X_train, X_test, Y_train_labels, Y_test_labels = train_test_split(
            X, Y, test_size=test_size, random_state=split_seed)

        n_classes = 11

    elif dataset_id in ['mnist', 'fashion_mnist', 'cifar10']:
        data = _load_from_tfds(dataset_id, data_dir, is_clf=True)

        train_data, test_data = data['train'], data['test']

        X_train = train_data['image']
        Y_train_labels = train_data['label']

        X_test = test_data['image']
        Y_test_labels = test_data['label']

        n_classes = 10


    elif dataset_id == 'imdb_reviews':
        # load directly pre-saved embedding vectors for train and test
        embeddings_dir = os.path.join('../..', 'artifacts', 'data', dataset_id)

        try:
            X_train = np.load(os.path.join(embeddings_dir, 'train_vectors.npy'))
            X_test = np.load(os.path.join(embeddings_dir, 'test_vectors.npy'))
            Y_train_labels = np.load(os.path.join(embeddings_dir, 'train_labels.npy'))
            Y_test_labels = np.load(os.path.join(embeddings_dir, 'test_labels.npy'))
        except Exception as e:
            print(f"Error loading embeddings for {dataset_id}. "
                  f"Ensure the processed files are saved in {embeddings_dir}.")
            raise e

        n_classes = 1

    # ------------- LIBSVM -------------
    # last resort - libsvm
    else:
        try:
            # naming tricks for shorter dataset names
            if dataset_id == 'covtype':
                dataset_id_full = 'covtype.multiclass'
            else:
                dataset_id_full = dataset_id

            # load_dotenv()
            X, Y = libsvmdata.fetch_libsvm(dataset_id_full, normalize=False, verbose=True, )

            # scale features
            X = StandardScaler(copy=False).fit_transform(X.toarray() if sp.issparse(X) else X)

            # encode labels since libsvm returns smth like [-1, 1] or [1, 2, 3, ...]
            Y = LabelEncoder().fit_transform(Y)

            n_classes = len(np.unique(Y))
            n_classes = n_classes if n_classes > 2 else 1  # special case for binary classification

            X_train, X_test, Y_train_labels, Y_test_labels = train_test_split(
                X, Y, test_size=test_size, random_state=split_seed)
        except Exception as e:
            print(f"Error loading dataset {dataset_id} from libsvm.")
            raise e

    # ------------- POST-PROCESSING -------------

    # unique shuffle of the train set for each seed
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(X_train.shape[0])
    X_train = X_train[permutation]

    if Y_train_labels is not None:
        Y_train_labels = Y_train_labels[permutation]
    else:
        Y_train = Y_train[permutation]

    is_clf = n_classes is not None

    # !! TODO re-work the loader for large datasets (esp. images)
    # !! TODO i.e. make it more memory-efficient, e.g. a generator returning batches

    # transfer data to JAX (Device) Array
    X_train = jnp.array(X_train)
    # for cls: OHE for CE loss
    if not is_clf:
        Y_train = jnp.array(Y_train)
    elif n_classes == 1:
        Y_train = jnp.array(Y_train_labels)
    else:
        Y_train = jax.nn.one_hot(Y_train_labels, n_classes)

    X_test = jnp.array(X_test)
    # for cls: no OHE for test since reporting Accuracy instead of CE loss
    Y_test = jnp.array(Y_test) if not is_clf else jnp.array(Y_test_labels)

    return (X_train, X_test, Y_train, Y_test), is_clf, n_classes


def _load_from_tfds(dataset_id: str, data_dir: str, is_clf: bool):
    # restrict TF from grabbing resources
    # TODO requires TF, consider smth else
    tf.config.set_visible_devices([], device_type='GPU')

    # load
    data, info = tfds.load(
        name=dataset_id,
        batch_size=-1,
        data_dir=data_dir,
        shuffle_files=False,
        as_supervised=not is_clf,
        with_info=True,
    )

    # pre-process generic
    data = tfds.as_numpy(data)

    return data


def _ohe_or_scale_columns(feature_map, cat_features, skip_cols):
    features = []
    for f_name, f_vals in feature_map.items():
        if f_name in cat_features:
            f_vals_proc = OneHotEncoder().fit_transform(f_vals.reshape(-1, 1)).toarray()
        elif f_name in skip_cols:
            continue
        else:
            f_vals_proc = StandardScaler(copy=False).fit_transform(f_vals.reshape(-1, 1))
        features.append(f_vals_proc)

    X = np.concatenate(features, axis=1)

    return X
