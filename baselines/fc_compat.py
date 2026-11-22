# The official FinDKG code indexes CPU tensors with GPU indices, which torch >= 1.9 rejects. Move such index tensors to the indexed tensor's device; values are unchanged.
import torch
_get, _set = torch.Tensor.__getitem__, torch.Tensor.__setitem__
def _mv(self, i):
    if isinstance(i, torch.Tensor):
        return i.to(self.device) if i.device != self.device else i
    if isinstance(i, tuple):
        return tuple(_mv(self, x) if isinstance(x, torch.Tensor) else x for x in i)
    return i
torch.Tensor.__getitem__ = lambda self, i: _get(self, _mv(self, i))
torch.Tensor.__setitem__ = lambda self, i, v: _set(self, _mv(self, i), v)
