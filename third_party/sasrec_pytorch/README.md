# SASRec PyTorch model

`model.py` is an unmodified copy from
[pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch), revision
`b253407b3c5d6ec3201a39767a572f44fd5ef6d0`, file `python/model.py`.

The original authors' TensorFlow implementation links to this PyTorch port:
[kang205/SASRec](https://github.com/kang205/SASRec).
The Apache 2.0 license is included. Upstream code retains its original comments.

Our training adapter samples complete training sequences, computes the
positive/negative BCE at every nonpadding position, and selects checkpoints
using validation only. We evaluate the full training catalog on our frozen
chronological splits instead of upstream's sampled-negative evaluator. The
configuration records normalization mode and evaluation policy. The pinned
PyTorch revision is described by its implementation revision rather than as an
exact reproduction of the 2018 TensorFlow code.
