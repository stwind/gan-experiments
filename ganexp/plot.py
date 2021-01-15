import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


def get_transform(ax):
    pts2pixels = 72.0 / ax.figure.dpi
    scale_x = pts2pixels * ax.bbox.width / ax.viewLim.width
    scale_y = pts2pixels * ax.bbox.height / ax.viewLim.height
    return mpl.transforms.Affine2D().scale(scale_x, scale_y)


class SquareCollection(mpl.collections.RegularPolyCollection):
    def __init__(self, **kwargs):
        super(SquareCollection, self).__init__(4, rotation=np.pi / 4.0, **kwargs)

    def get_transform(self):
        return get_transform(self.axes)


class CircleCollection(mpl.collections.CircleCollection):
    def get_transform(self):
        return get_transform(self.axes)


class IndexLocator(mpl.ticker.Locator):
    def __init__(self, max_ticks=10):
        self.max_ticks = max_ticks

    def __call__(self):
        dmin, dmax = self.axis.get_data_interval()
        step = 1 if dmax < self.max_ticks else np.ceil(dmax / self.max_ticks)
        return self.raise_if_exceeds(np.arange(0, dmax, step))


def hinton(
    mat,
    max_value=None,
    use_default_ticks=True,
    ax=None,
    colors=["red", "white"],
    shape="circle",
):
    ax = ax or plt.gca()

    height, width = mat.shape
    rows, cols = np.mgrid[:height, :width]
    if max_value is None:
        max_value = 2 ** np.ceil(np.log(np.max(np.abs(mat))) / np.log(2))
    values = np.clip(mat / max_value, -1, 1)
    pos, neg = np.where(values > 0), np.where(values < 0)
    for idx, color in zip([pos, neg], colors):
        if len(idx[0]) == 0:
            continue
        xy = list(zip(cols[idx], rows[idx]))
        circle_areas = np.pi / 2 * np.abs(values[idx])
        if shape == "circle":
            patches = CircleCollection(
                sizes=circle_areas / 2,
                offsets=xy,
                transOffset=ax.transData,
                facecolor=color,
                edgecolor=color,
            )
        else:
            patches = SquareCollection(
                sizes=circle_areas,
                offsets=xy,
                transOffset=ax.transData,
                facecolor=color,
                edgecolor=color,
            )
        ax.add_collection(patches, autolim=True)

    ax.set_aspect("equal", "box")
    ax.set_facecolor("black")
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    ax.autoscale_view()
    ax.grid(False)
    sns.despine(left=True, bottom=True)

    if use_default_ticks:
        ax.xaxis.set_major_locator(IndexLocator())
        ax.yaxis.set_major_locator(IndexLocator())