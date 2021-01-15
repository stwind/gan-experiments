import torch
import torch.nn.functional as F


def padding_same(in_size, kernel_size, stride=1, dilation=1):
    filter_size = (kernel_size - 1) * dilation + 1
    out_size = (in_size + stride - 1) // stride
    return max(0, (out_size - 1) * stride + filter_size - in_size)


def pad_same(x, kernel):
    """same behavior as padding='SAME' of tensorflow"""
    _, _, h, w = x.size()
    _, _, kh, kw = kernel.size()
    ph, pw = padding_same(h, kh, 2), padding_same(w, kw, 2)
    hph, hpw = ph // 2, pw // 2
    return F.pad(x, [hpw, hpw + (pw % 2), hph, hph + (ph % 2)])


def _reflect(x, minx, maxx):
    rng = maxx - minx
    double_rng = 2 * rng
    mod = torch.fmod(x - minx, double_rng)
    normed_mod = torch.where(mod < 0, mod + double_rng, mod)
    out = torch.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
    return out.type_as(x)


def pad_symmetric(im, padding):
    """same behavior as tf.pad(mode="SYMMETRIC")"""
    h, w = im.shape[-2:]
    left, right, top, bottom = padding

    x_idx = torch.arange(-left, w + right)
    y_idx = torch.arange(-top, h + bottom)

    x_pad = _reflect(x_idx, -0.5, w - 0.5)
    y_pad = _reflect(y_idx, -0.5, h - 0.5)
    yy, xx = torch.meshgrid(y_pad, x_pad)
    return im[..., yy, xx]


def depthwise_conv2d(x, kernel, stride=1, dilation=1):
    """same behavior as tf.nn.depthwise_conv2d"""
    cout, cin, kh, kw = kernel.size()
    kernel = kernel.permute(1, 0, 2, 3).reshape(cout * cin, 1, kh, kw)
    return F.conv2d(x, kernel, groups=cin, stride=stride, dilation=dilation)


def _fspecial_gauss(size, sigma):
    coords = torch.arange(size, dtype=torch.float32)
    coords -= (size - 1.0) / 2.0

    g = coords.square()
    g *= -0.5 / (sigma * sigma)

    g = g.view(1, -1) + g.view(-1, 1)
    g = F.softmax(g.view(1, -1), dim=1)
    return g.view(size, size)


def _ssim_per_channel(img1, img2, kernel, max_val=1.0, k1=0.01, k2=0.03):
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    compensation = 1.0

    mean0 = depthwise_conv2d(img1, kernel)
    mean1 = depthwise_conv2d(img2, kernel)
    num0 = mean0 * mean1 * 2.0
    den0 = mean0.square() + mean1.square()
    luminance = (num0 + c1) / (den0 + c1)

    num1 = depthwise_conv2d(img1 * img2, kernel) * 2.0
    den1 = depthwise_conv2d(img1.square() + img2.square(), kernel)
    c2 *= compensation
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    ssim_val = (luminance * cs).mean((-1, -2))
    cs = cs.mean((-1, -2))

    return ssim_val, cs


def make_ssim_kernel(size, sigma, channel):
    kernel = _fspecial_gauss(size, sigma)
    return kernel.view(1, 1, size, size).expand(1, channel, -1, -1)


def ssim(img1, img2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    assert img1.shape == img2.shape, "img1 and img2 must have same shape"

    kernel = make_ssim_kernel(filter_size, filter_sigma, img1.size(1))

    ssim_val, _ = _ssim_per_channel(img1, img2, kernel, max_val=max_val, k1=k1, k2=k2)
    return ssim_val.mean(-1)


MSSSIM_POWER_FACTORS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


def _msssim(
    img1,
    img2,
    kernel,
    max_val=1.0,
    k1=0.01,
    k2=0.03,
    power_factors=MSSSIM_POWER_FACTORS,
):
    mcs = []
    for k in range(len(power_factors)):
        if k > 0:
            _, _, h, w = img1.size()
            if h % 2 or w % 2:
                img1 = pad_symmetric(img1, (0, 1, 0, 1))
            _, _, h, w = img2.size()
            if h % 2 or w % 2:
                img2 = pad_symmetric(img2, (0, 1, 0, 1))

            img1 = F.avg_pool2d(img1, 2, 2)
            img2 = F.avg_pool2d(img2, 2, 2)

        ssim_val, cs = _ssim_per_channel(
            img1, img2, kernel, max_val=max_val, k1=k1, k2=k2
        )
        mcs.append(F.relu(cs))

    mcs.pop()
    mcs.append(F.relu(ssim_val))
    mcs_ssim = torch.stack(mcs, dim=-1)
    ms_ssim = torch.pow(mcs_ssim, torch.tensor(power_factors)).prod(-1)
    return ms_ssim.mean(-1)


def ssim_multiscale(
    img1,
    img2,
    max_val=1.0,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    power_factors=MSSSIM_POWER_FACTORS,
):
    assert img1.shape == img2.shape, "img1 and img2 must have same shape"
    kernel = make_ssim_kernel(filter_size, filter_sigma, img1.size(1))
    return _msssim(
        img1, img2, kernel, max_val=max_val, k1=k1, k2=k2, power_factors=power_factors
    )


def cov(m, y=None):
    """np.cov for pytorch"""
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov