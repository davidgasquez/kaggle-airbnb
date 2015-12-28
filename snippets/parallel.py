from joblib import Parallel, delayed
import multiprocessing


def processInput(i):
    return i * i

inputs = range(20)

num_cores = multiprocessing.cpu_count()

results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)

print results
