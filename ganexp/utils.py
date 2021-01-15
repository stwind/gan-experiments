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