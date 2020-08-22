import torch


class MixDataLoader(object):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs a index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __init__(self, dataloader_main, dataloader_aux):
        self.dataloader_main = dataloader_main
        self.dataloader_aux = dataloader_aux
        assert len(dataloader_main) == len(dataloader_aux)
        self.len = len(dataloader_main)

    def __len__(self):
        return self.len

    def __iter__(self):
        self.dataloader_main_iter = self.dataloader_main.__iter__()
        self.dataloader_aux_iter = self.dataloader_aux.__iter__()
        return self

    def __next__(self):
        inputs_main, target_main = next(self.dataloader_main_iter)
        inputs_aux, target_aux = next(self.dataloader_aux_iter)
        return (torch.cat([inputs_main, inputs_aux]), torch.cat([target_main, target_aux]))
