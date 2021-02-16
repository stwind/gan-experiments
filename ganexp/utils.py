import io
import requests
import PIL
import numpy as np


def slerp(p0, p1, t):
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1


def make_analogy_grid(anchors):
    n, dim = len(anchors), anchors.shape[1]
    rows = n // 2 + 1
    cols = n - rows + 1
    grid = np.zeros((rows, cols, dim), dtype=np.float32)
    for y in range(rows):
        for x in range(cols):
            if x == 0 or y == 0:
                if x == 0 and y == 0:
                    idx = 0
                elif x == 0:
                    idx = y * 2 - 1
                else:
                    idx = x * 2
                grid[y, x, :] = anchors[idx]
            else:
                anal_vec = grid[y, x - 1, :] + grid[y - 1, x, :] - grid[y - 1, x - 1, :]
                anal_unit_vec = np.nan_to_num(anal_vec / np.linalg.norm(anal_vec))
                avg_len = (
                    np.linalg.norm(grid[y, x - 1, :])
                    + np.linalg.norm(grid[y - 1, x, :])
                    + np.linalg.norm(grid[y - 1, x - 1, :])
                ) / 3.0
                grid[y, x, :] = avg_len * anal_unit_vec
    return grid.reshape(-1, dim)


def make_mine_grid(anchors, nrow, space):
    dim = anchors.shape[1]
    ncol = len(anchors) // nrow
    rows = (nrow - 1) * space + 1
    cols = (ncol - 1) * space + 1

    grid = np.zeros((rows, cols, dim), dtype=np.float32)

    for y in range(rows):
        for x in range(cols):
            if y % space == 0 and x % space == 0:
                idx = (y // space) * ncol + (x // space)
                grid[y, x, :] = anchors[idx]

    for y in range(rows):
        for x in range(cols):
            if y % space == 0 and x % space != 0:
                prev = space * (x // space)
                cur = prev + space
                t = (x - prev) / space
                grid[y, x, :] = slerp(grid[y, prev, :], grid[y, cur, :], t)

    for y in range(rows):
        for x in range(cols):
            if y % space != 0:
                prev = space * (y // space)
                cur = prev + space
                t = (y - prev) / space
                grid[y, x, :] = slerp(grid[prev, x, :], grid[cur, x, :], t)

    return grid.reshape(-1, dim)


def fetch_image(url):
    resp = requests.get(url)
    return PIL.Image.open(io.BytesIO(resp.content))


def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    N, C, H, W = x.shape
    K, _, R, S = weight.shape
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (padding, padding) if isinstance(padding, int) else padding
    dh, dw = (dilation, dilation) if isinstance(dilation, int) else dilation

    x = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")

    H += 2 * ph
    W += 2 * pw
    P = int((H - dh * (R - 1) - 1) / sh + 1)
    Q = int((W - dw * (S - 1) - 1) / sw + 1)

    # im2col
    shape = (C, R, S, N, P, Q)
    strides = np.array([H * W, W, 1, C * H * W, sh * W, sw]) * x.itemsize
    x = np.lib.stride_tricks.as_strided(
        x, shape=shape, strides=strides, writeable=False
    )
    x_cols = np.ascontiguousarray(x).reshape(C * R * S, N * P * Q)

    res = weight.reshape(K, C * R * S).dot(x_cols)
    if bias is not None:
        res += bias.reshape(-1, 1)

    out = res.reshape(K, N, P, Q).swapaxes(0, 1)
    return np.ascontiguousarray(out)


def dilate(x, stride=1):
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    if sh == sw == 1:
        return x
    N, C, H, W = x.shape
    out = np.zeros((N, C, (H - 1) * sh + 1, (W - 1) * sw + 1), dtype=x.dtype)
    out[..., ::sh, ::sw] = x
    return out


def conv_transpose2d(x, weight, bias=None, stride=1, padding=0, dilation=1):
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ph, pw = (padding, padding) if isinstance(padding, int) else padding
    _, _, R, S = weight.shape

    return conv2d(
        dilate(x, (sh, sw)),
        np.flip(weight.swapaxes(0, 1), (-1, -2)),
        bias,
        padding=(R - 1 - ph, S - 1 - pw),
        dilation=dilation,
    )


def batch_norm2d(x, weight, bias, mean, sigma, epsilon=1e-9):
    shape = (1, x.shape[1], 1, 1)
    norm = (x - mean.reshape(shape)) / np.sqrt(sigma + epsilon).reshape(shape)
    return norm * weight.reshape(shape) + bias.reshape(shape)