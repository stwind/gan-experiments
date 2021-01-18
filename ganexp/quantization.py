import numpy as np


class RangeQuantizer(object):
    def __init__(self, alpha, beta, dtype=np.int8):
        self.dtype = dtype
        if dtype == np.uint8:
            self.alpha_q, self.beta_q = 0, 255
        elif dtype == np.int8:
            self.alpha_q, self.beta_q = -128, 127
        elif dtype == np.int16:
            self.alpha_q, self.beta_q = -32768, 65535
        self.s = (beta - alpha) / (self.beta_q - self.alpha_q)
        self.z = round((beta * self.alpha_q - alpha * self.beta_q) / (beta - alpha))

    def quantize(self, x):
        x_q = np.round(1 / self.s * x + self.z)
        return np.clip(x_q, self.alpha_q, self.beta_q).astype(self.dtype)

    def dequantize(self, x_q):
        return self.s * (x_q.astype(np.float32) - self.z)
