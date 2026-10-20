"""Read-only contiguous truth HDF I/O without data-sieve amplification.

Caller retains admission, path validation, allocation checks and handle ownership.
"""
import os

import h5py


def open_truth_readonly(path):
    """Open the caller-validated path with sec2 and a zero-byte sieve buffer."""
    access = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
    try:
        access.set_fapl_sec2()
        access.set_sieve_buf_size(0)
        identifier = h5py.h5f.open(os.fsencode(path), h5py.h5f.ACC_RDONLY, fapl=access)
    finally:
        access.close()
    try:
        return h5py.File(identifier)
    except BaseException:
        identifier.close()
        raise
