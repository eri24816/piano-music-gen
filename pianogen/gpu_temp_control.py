import time
import torch


class GPUTempControl:
    def __init__(self, max_temp: float, margin: float):
        self.max_temp = max_temp
        self.margin = margin
        self.sample_interval = 0.5
        self.last_sample_time = time.time()

    def get_temp(self) -> int:
        return torch.cuda.temperature()

    def cooldown(self):
        if time.time() - self.last_sample_time < self.sample_interval:
            return
        self.last_sample_time = time.time()
        temp = self.get_temp()
        if temp > self.max_temp:
            while temp > self.max_temp - self.margin:
                time.sleep(5)
                temp = self.get_temp()
        if temp > self.max_temp - self.margin:
            while temp > self.max_temp - self.margin:
                time.sleep(0.2)
                temp = self.get_temp()
