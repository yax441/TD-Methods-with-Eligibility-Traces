import multiprocessing.pool as mpp
from sys import version_info
import contextlib
import joblib

def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap"""
    if version_info[0]>3 or (version_info[0]==3 and version_info[1]>=8):#python 3.8+
        self._check_running()
        if chunksize < 1:
            raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))
        task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
        result = mpp.IMapIterator(self)
        self._taskqueue.put((self._guarded_task_generation(result._job, mpp.starmapstar, task_batches), result._set_length))
        return (item for chunk in result for item in chunk)
    else:
        if self._state != mpp.RUN:
            raise ValueError("Pool not running")
        if chunksize < 1:
            raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))
        task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
        result = mpp.IMapIterator(self._cache)
        self._taskqueue.put((self._guarded_task_generation(result._job, mpp.starmapstar, task_batches), result._set_length))
        return (item for chunk in result for item in chunk)
mpp.Pool.istarmap = istarmap

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()