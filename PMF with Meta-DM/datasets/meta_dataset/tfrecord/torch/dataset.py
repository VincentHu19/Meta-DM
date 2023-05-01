"""Load tfrecord files into torch datasets."""

import typing
import numpy as np

import torch.utils.data

from .. import reader
from .. import iterator_utils
import warnings


class TFRecordDataset(torch.utils.data.IterableDataset):
    """Parse (generic) TFRecords dataset into `IterableDataset` object,
    which contain `np.ndarrays`s. By default (when `sequence_description`
    is None), it treats the TFRecords as containing `tf.Example`.
    Otherwise, it assumes it is a `tf.SequenceExample`.

    Params:
    -------
    data_path: str
        The path to the tfrecords file.

    index_path: str or None
        The path to the index file.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    shuffle: bool, optional, default=None
        Whether to shuffle the dataset

    transform : a callable, default = None
        A function that takes in the input `features` i.e the dict
        provided in the description, transforms it and returns a
        desirable output.

    sequence_description: list or dict of str, optional, default=None
        Similar to `description`, but refers to the sequence features
        within a `SequenceExample`. When this field is `None`, then it
        is assumed that an `Example` is being read otherwise, a
        `SequenceExample` is read. If an empty list or dictionary is
        passed, then all features contained in the file are extracted.

    """

    def __init__(self,
                 data_path: str,
                 index_path: typing.Union[str, None],
                 description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                 shuffle: typing.Optional[bool] = None,
                 sequence_description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                 ) -> None:
        super(TFRecordDataset, self).__init__()
        self.data_path = data_path
        self.index_path = index_path
        self.description = description
        self.sequence_description = sequence_description
        self.shuffle = shuffle

    def __iter__(self):
        #worker_info = torch.utils.data.get_worker_info()
        #if worker_info is not None:
        #    shard = worker_info.id, worker_info.num_workers
        #    np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        #else:
        #    shard = None
        shard = None
        it = reader.tfrecord_loader(data_path=self.data_path,
                                    index_path=self.index_path,
                                    description=self.description,
                                    shard=shard,
                                    shuffle=self.shuffle,
                                    sequence_description=self.sequence_description)
        return it


class MultiTFRecordDataset(torch.utils.data.IterableDataset):
    """Parse multiple (generic) TFRecords datasets into an `IterableDataset`
    object, which contain `np.ndarrays`s.

    Params:
    -------
    data_pattern: str
        Input data path pattern.

    index_pattern: str or None
        Input index path pattern.

    splits: dict
        Dictionary of (key, value) pairs, where the key is used to
        construct the data and index path(s) and the value determines
        the contribution of each split to the batch.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    shuffle: int, optional, default=None
        Length of buffer. Determines how many records are queued to
        sample from.

    transform : a callable, default = None
        A function that takes in the input `features` i.e the dict
        provided in the description, transforms it and returns a
        desirable output.

    sequence_description: list or dict of str, optional, default=None
        Similar to `description`, but refers to the sequence features
        within a `SequenceExample`. When this field is `None`, then it
        is assumed that an `Example` is being read otherwise, a
        `SequenceExample` is read. If an empty list or dictionary is
        passed, then all features contained in the file are extracted.

    """

    def __init__(self,
                 data_pattern: str,
                 index_pattern: typing.Union[str, None],
                 splits: typing.Dict[str, float],
                 description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                 queue_size: typing.Optional[int] = None,
                 transform: typing.Callable[[dict], typing.Any] = None,
                 sequence_description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                 ) -> None:
        super(MultiTFRecordDataset, self).__init__()
        self.data_pattern = data_pattern
        self.index_pattern = index_pattern
        self.splits = splits
        self.description = description
        self.sequence_description = sequence_description
        self.queue_size = queue_size
        self.transform = transform

    def __iter__(self):
        #worker_info = torch.utils.data.get_worker_info()
        #if worker_info is not None:
        #    np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        it = reader.multi_tfrecord_loader(data_pattern=self.data_pattern,
                                          index_pattern=self.index_pattern,
                                          splits=self.splits,
                                          description=self.description,
                                          sequence_description=self.sequence_description)
        if self.queue_size:
            it = iterator_utils.shuffle_iterator(it, self.queue_size)
        if self.transform:
            it = map(self.transform, it)
        return it
