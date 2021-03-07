import multiprocessing as mp
from time import time


class BaseNode(mp.Process):
    def __init__(self):
        super().__init__()
        self.exit = mp.Event()
        self.pause = mp.Event()

    def run(self):
        self.setup()
        times = []
        while not self.exit.is_set():
            start = time()
            if self.pause.is_set():
                self.pause.wait()
            self.task()
            times.append(time() - start)
        self.teardown()
        mean = sum(times) / len(times)
        minim, maxim = min(times), max(times)
        print(f'{self.__class__.__name__} processing times:\n'
              f'mean: {mean}, max: {maxim}, min: {minim}')

    def setup(self):
        pass

    def teardown(self):
        pass

    def task(self):
        raise NotImplementedError

    def join(self, **kwargs) -> None:
        self.exit.set()
        super(BaseNode, self).join()
