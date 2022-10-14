import numpy as np
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
from utils.tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed, cpu_count

def concurrent_runs(runs, algorithm, *args, seed=None, processes=None, show_progress_bar=True, bar_desc=None, leave_bar=False):
    errors = None
    processes = cpu_count() if processes is None else processes
    with tqdm_joblib(tqdm(total=runs, desc=bar_desc, leave=leave_bar, disable=not show_progress_bar)) as bar:
        for run,result in enumerate(Parallel(n_jobs=processes)(delayed(algorithm)(*args, _seed) for _seed in np.random.SeedSequence(seed).generate_state(runs).tolist())):
            if run==0:
                errors = np.array(result)
            else:
                errors += np.array(result)
    return (errors/runs).tolist()