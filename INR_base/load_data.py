import numpy as np
import torch
import threading
import queue
import tqdm
from utils import BatchIndex

class LoadData:
    def preload_data(self):
        if self.loader_queue.full():
            return  
        self.loader_queue.put(self._get_data())

    def get_data(self):
        if self.loader_queue.empty():
            print("DataLoader is not ready yet! Waiting...")
        while self.loader_queue.empty():
            pass
        current_data = self.loader_queue.get()
        self.executor.submit(self.preload_data)
        return current_data

    @torch.no_grad()
    def _get_testing_data(self, idx):
        t = idx - 1
        t = t / max((self.total_samples-1), 1)
        t = 2.0 * t - 1.0
        testing_data_inputs = self.testing_data_inputs.clone()
        testing_data_inputs[:,0] = t
        batchidxgenerator = BatchIndex(testing_data_inputs.shape[0], self.batch_size, False)
        return testing_data_inputs, batchidxgenerator

    def load_volume_data(self, idx):
        file_path = self.data_path / f"{self.samples[idx]:04d}.raw"
        d = np.fromfile(file_path, dtype='<f')
        d = 2. * (d - np.min(d)) / (np.max(d) - np.min(d)) - 1.  
        d = d.reshape(self.dim[2],self.dim[1],self.dim[0])  
        d = d.transpose(2,1,0)  
        return d

    def _preload_worker(self, data_list, load_func, q, lock, idx_tqdm):
        while True:
            idx = q.get()
            data_list[idx] = load_func(idx)
            with lock:
                idx_tqdm.update()
            q.task_done()

    def preload_with_multi_threads(self, load_func, num_workers, data_str='images'):
        data_list = [None] * len(self.samples)

        q = queue.Queue(maxsize=len(self.samples))
        idx_tqdm = tqdm.tqdm(range(len(self.samples)), desc=f"Loading {data_str}", leave=False)
        for i in range(len(self.samples)):
            q.put(i)
        lock = threading.Lock()
        for ti in range(num_workers):
            t = threading.Thread(target=self._preload_worker,
                                    args=(data_list, load_func, q, lock, idx_tqdm), daemon=True)
            t.start()
        q.join()
        idx_tqdm.close()
        assert all(map(lambda x: x is not None, data_list))

        return data_list  