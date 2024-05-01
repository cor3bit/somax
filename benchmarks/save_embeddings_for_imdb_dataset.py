import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import spacy
from tqdm import tqdm


def load_from_tfds(dataset_id: str, data_dir: str, is_clf: bool):
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


if __name__ == '__main__':
    def process_text(text: bytes):
        return nlp(str(text.decode('utf-8'))).vector


    dataset_id = 'imdb_reviews'
    data_dir = os.path.join('..', 'artifacts', 'data')
    embedding_dir = os.path.join(data_dir, dataset_id)

    # ensure folders exist
    os.makedirs(os.path.join(data_dir), exist_ok=True)
    os.makedirs(os.path.join(embedding_dir), exist_ok=True)

    # prepare spacy
    print('Loading spacy model')
    model_id = 'en_core_web_lg'  # en_core_web_md en_core_web_lg
    nlp = spacy.load(model_id)

    # load raw data
    print('Loading data from TFDS')
    data = load_from_tfds(dataset_id, data_dir, is_clf=True)
    train_data, test_data = data['train'], data['test']
    train_texts = train_data['text']
    test_texts = test_data['text']
    Y_train_labels = train_data['label']
    Y_test_labels = test_data['label']

    # save labels
    print('Saving labels')
    np.save(os.path.join(embedding_dir, 'train_labels.npy'), Y_train_labels)
    np.save(os.path.join(embedding_dir, 'test_labels.npy'), Y_test_labels)

    # save features
    print('Saving features')

    total_length = len(train_data['text']) + len(test_data['text'])

    train_vectors = []
    test_vectors = []

    with tqdm(total=total_length) as pbar:
        for train_text in train_texts:
            train_vectors.append(process_text(train_text))
            pbar.update(1)

        for test_text in test_texts:
            test_vectors.append(process_text(test_text))
            pbar.update(1)

    # Save the embeddings
    np.save(os.path.join(embedding_dir, 'train_vectors.npy'), np.array(train_vectors))
    np.save(os.path.join(embedding_dir, 'test_vectors.npy'), np.array(test_vectors))
