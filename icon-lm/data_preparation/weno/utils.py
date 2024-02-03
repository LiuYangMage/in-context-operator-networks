import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds', flush = True)
        return result
    return timeit_wrapper

class TicToc:
  def __init__(self):
    self.start_time = {}
    self.end_time = {}
  def tic(self, name):
    self.start_time[name] = time.perf_counter()
  def toc(self, name):
    self.end_time[name] = time.perf_counter()
    total_time = self.end_time[name] - self.start_time[name]
    print(f'{name} Took {total_time:.4f} seconds', flush = True)


timer = TicToc()