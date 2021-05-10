
import random
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from itertools import chain
import imgaug as ia
import cv2
import imageio as imgio
import numpy as np



numMustLessMsg = "the number must less"

def _arithmetic(num):
    supported_ops = [
        iaa.Add((-40, 40)),
        iaa.AddElementwise((-40, 40), per_channel=0.5),
        iaa.AdditiveGaussianNoise((0, 0.2 * 255), per_channel=0.5),
        iaa.AdditiveLaplaceNoise(scale=0.2*255, per_channel=True),
        iaa.AdditivePoissonNoise((0, 40), per_channel=True),
        iaa.Multiply((0.5, 1.5), per_channel=0.5),
        iaa.MultiplyElementwise((0.5, 1.5), per_channel=0.5),
        iaa.Dropout(p=(0, 0.2), per_channel=0.5),
        iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
        iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
        iaa.ReplaceElementwise(0.1, iap.Normal(128, 0.4*128), per_channel=0.5),
        iaa.ReplaceElementwise(
            iap.FromLowerResolution(iap.Binomial(0.1), size_px=8),
            iap.Normal(128, 0.4*128),
            per_channel=0.5),
        iaa.ImpulseNoise(0.1),
        iaa.SaltAndPepper(0.1, per_channel=True),
        iaa.CoarseSaltAndPepper(0.05, size_percent=(0.01, 0.1)),
        iaa.CoarseSaltAndPepper(
            0.05, size_percent=(0.01, 0.1), per_channel=True),
        iaa.Salt(0.1),
        iaa.CoarseSalt(0.05, size_percent=(0.01, 0.1)),
        iaa.Pepper(0.1),
        iaa.CoarsePepper(0.05, size_percent=(0.01, 0.1)),
        iaa.Invert(0.5),
        iaa.Invert(0.25, per_channel=0.5),
        iaa.contrast.LinearContrast((0.5, 1.5)),
        iaa.contrast.LinearContrast((0.5, 1.5), per_channel=0.5),
        iaa.JpegCompression(compression=(70, 99))
    ]
    if num > len(supported_ops):
        print(f"{numMustLessMsg} %s" % len(supported_ops))
        return supported_ops
    return random.sample(supported_ops, num)

def _blend(num):
    supported_ops = [
        iaa.BlendAlpha(0.5, iaa.Grayscale(1.0)),
        iaa.BlendAlpha((0.0, 1.0), iaa.Grayscale(1.0)),
        iaa.BlendAlpha(
            (0.0, 1.0),
            iaa.Affine(rotate=(-20, 20)),
            per_channel=0.5),
        iaa.BlendAlpha(
            (0.0, 1.0),
            foreground=iaa.Add(100),
            background=iaa.Multiply(0.2)),
        iaa.BlendAlpha([0.25, 0.75], iaa.MedianBlur(13)),
        iaa.BlendAlphaElementwise (0.5, iaa.Grayscale(1.0)),
        iaa.BlendAlphaElementwise ((0, 1.0), iaa.Grayscale(1.0)),
        iaa.BlendAlphaElementwise (
            (0.0, 1.0),
            iaa.Affine(rotate=(-20, 20)),
            per_channel=0.5),
        iaa.BlendAlphaElementwise (
            (0.0, 1.0),
            foreground=iaa.Add(100),
            background=iaa.Multiply(0.2)),
        iaa.BlendAlphaElementwise ([0.25, 0.75], iaa.MedianBlur(13)),
        iaa.BlendAlphaSimplexNoise(
            iaa.EdgeDetect(1.0),
            upscale_method="nearest"),
        iaa.BlendAlphaSimplexNoise(
            iaa.EdgeDetect(1.0),
            sigmoid_thresh=iap.Normal(10.0, 5.0)),
        iaa.BlendAlphaFrequencyNoise(foreground=iaa.EdgeDetect(1.0)),
        iaa.BlendAlphaFrequencyNoise(
            foreground=iaa.EdgeDetect(1.0),
            upscale_method="nearest"),
        iaa.BlendAlphaFrequencyNoise(
            foreground=iaa.EdgeDetect(1.0),
            upscale_method="linear"),
        iaa.BlendAlphaFrequencyNoise(
            foreground=iaa.EdgeDetect(1.0),
            upscale_method="linear",
            exponent=-2,
            sigmoid=False),
        iaa.BlendAlphaFrequencyNoise(
            foreground=iaa.EdgeDetect(1.0),
            sigmoid_thresh=iap.Normal(10.0, 5.0))
    ]
    if num > len(supported_ops):
        print(f"{numMustLessMsg} %s" % len(supported_ops))
        return supported_ops
    return random.sample(supported_ops, num)

def _blur(num):
    supported_ops = [
        iaa.GaussianBlur(sigma=(0.0, 3.0)),
        iaa.AverageBlur(k=(2, 11)),
        iaa.AverageBlur(k=((5, 11), (1, 3))),
        iaa.MedianBlur(k=(3, 11)),
        iaa.BilateralBlur(
            d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),
        iaa.MotionBlur(k=15),
        iaa.MotionBlur(k=15, angle=[-45, 45]),
    ]
    if num > len(supported_ops):
        print(f"{numMustLessMsg} %s" % len(supported_ops))
        return supported_ops
    return random.sample(supported_ops, num)

def _color(num):
    supported_ops = [
        iaa.WithColorspace(
            to_colorspace="HSV",
            from_colorspace="RGB",
            children=iaa.WithChannels(
                0,
                iaa.Add((0, 50))
            )
        ),
        iaa.WithHueAndSaturation(
            iaa.WithChannels(0, iaa.Add((0, 50)))
        ),
        iaa.WithHueAndSaturation([
            iaa.WithChannels(0, iaa.Add((-30, 10))),
            iaa.WithChannels(1, [
                iaa.Multiply((0.5, 1.5)),
                iaa.LinearContrast((0.75, 1.25))
            ])
        ]),
        iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
        iaa.MultiplyHueAndSaturation(mul_hue=(0.5, 1.5)),
        iaa.MultiplyHueAndSaturation(mul_saturation=(0.5, 1.5)),
        iaa.MultiplyHue((0.5, 1.5)),
        iaa.MultiplySaturation((0.5, 1.5)),
        iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
        iaa.AddToHue((-50, 50)),
        iaa.AddToSaturation((-50, 50)),
        iaa.Sequential([
            iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
            iaa.WithChannels(0, iaa.Add((50, 100))),
            iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")
        ]),
        iaa.Grayscale(alpha=(0.0, 1.0)),
        iaa.KMeansColorQuantization(),
        iaa.KMeansColorQuantization(n_colors=(4, 16)),
        iaa.KMeansColorQuantization(
            to_colorspace=[iaa.ChangeColorspace.RGB, iaa.ChangeColorspace.HSV]),
        iaa.UniformColorQuantization(),
        iaa.UniformColorQuantization(n_colors=8),
        iaa.UniformColorQuantization(n_colors=(4, 16)),
        iaa.UniformColorQuantization(
            from_colorspace=iaa.ChangeColorspace.BGR,
            to_colorspace=[iaa.ChangeColorspace.RGB, iaa.ChangeColorspace.HSV])
    ]
    if num > len(supported_ops):
        print(f"{numMustLessMsg} %s" % len(supported_ops))
        return supported_ops
    return random.sample(supported_ops, num)

def _contrast(num):
    supported_ops = [
        iaa.GammaContrast((0.5, 2.0)),
        iaa.GammaContrast((0.5, 2.0), per_channel=True),
        iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
        iaa.SigmoidContrast(
            gain=(3, 10), cutoff=(0.4, 0.6), per_channel=True),
        iaa.LogContrast(gain=(0.6, 1.4)),
        iaa.LogContrast(gain=(0.6, 1.4), per_channel=True),
        iaa.LinearContrast((0.4, 1.6)),
        iaa.LinearContrast((0.4, 1.6), per_channel=True),
        iaa.AllChannelsCLAHE(),
        iaa.AllChannelsCLAHE(clip_limit=(1, 10)),
        iaa.AllChannelsCLAHE(clip_limit=(1, 10), per_channel=True),
        iaa.CLAHE(
            tile_grid_size_px=iap.Discretize(iap.Normal(loc=7, scale=2)),
            tile_grid_size_px_min=3),
        iaa.CLAHE(
            from_colorspace=iaa.CLAHE.BGR,
            to_colorspace=iaa.CLAHE.HSV),
        iaa.AllChannelsHistogramEqualization(),
        iaa.BlendAlpha((0.0, 1.0), iaa.AllChannelsHistogramEqualization()),
        iaa.HistogramEqualization(
            from_colorspace=iaa.HistogramEqualization.BGR,
            to_colorspace=iaa.HistogramEqualization.HSV)
    ]
    if num > len(supported_ops):
        print(f"{numMustLessMsg} %s" % len(supported_ops))
        return supported_ops
    return random.sample(supported_ops, num)

def _flip(num):
    supported_ops = [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5)
    ]
    if num > len(supported_ops):
        print(f"{numMustLessMsg} %s" % len(supported_ops))
        return supported_ops
    return random.sample(supported_ops, num)

def _geometric(num):
    supported_ops = [
        iaa.Affine(scale=(0.5, 1.5)),
        iaa.Affine(scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}),
        iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
        iaa.Affine(rotate=(-45, 45)),
        iaa.Affine(translate_percent={"x": -0.20}, mode=ia.ALL, cval=(0, 255)),
        iaa.PerspectiveTransform(scale=(0.01, 0.15)),
        iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)
    ]
    if num > len(supported_ops):
        print(f"{numMustLessMsg} %s" % len(supported_ops))
        return supported_ops
    return random.sample(supported_ops, num)

def _pooling(num):
    supported_ops = [
        # iaa.AveragePooling([2, 8]),
        # iaa.MaxPooling(2),
        # iaa.MaxPooling([2, 8]),
        # iaa.MaxPooling(((1, 7), (1, 7))),
        # iaa.MinPooling([2, 8]),
        # iaa.MinPooling(((1, 7), (1, 7)))
    ]
    if num > len(supported_ops):
        print(f"{numMustLessMsg} %s" % len(supported_ops))
        return supported_ops
    return random.sample(supported_ops, num)

def _weather(num):
    supported_ops = [
        iaa.FastSnowyLandscape(
            lightness_threshold=140,
            lightness_multiplier=2.5
        ),
        iaa.FastSnowyLandscape(
            lightness_threshold=[128, 200],
            lightness_multiplier=(1.5, 3.5)
        ),
        iaa.Clouds(),
        iaa.Snowflakes(flake_size=(0.7, 0.95), speed=(0.001, 0.03))
    ]
    if num > len(supported_ops):
        print(f"{numMustLessMsg} %s" % len(supported_ops))
        return supported_ops
    return random.sample(supported_ops, num)





class ImaugFunctions(object):

    def __init__(self, all_some_one='all', 
    use_arithmetic_num=None, 
    use_blend_num=None, 
    use_blur_num=None,
    use_color_num=None, 
    use_contrast_num=None, 
    use_flip_num=None, 
    use_geometric_num=None,                 
    use_pooling_num=None, 
    use_weather_num=None):

        self.flag = all_some_one
        self.use_arithmetic_num = use_arithmetic_num
        self.use_blend_num = use_blend_num
        self.use_blur_num = use_blur_num
        self.use_color_num = use_color_num
        self.use_contrast_num = use_contrast_num
        self.use_flip_num = use_flip_num
        self.use_geometric_num = use_geometric_num
        self.use_pooling_num = use_pooling_num
        self.use_weather_num = use_weather_num
        self.functions = []
        if self.use_arithmetic_num:
            self.functions.append(_arithmetic(self.use_arithmetic_num))
        if self.use_blend_num:
            self.functions.append(_blend(self.use_blend_num))
        if self.use_blur_num:
            self.functions.append(_blur(self.use_blur_num))
        if self.use_color_num:
            self.functions.append(_color(self.use_color_num))
        if self.use_contrast_num:
            self.functions.append(_contrast(self.use_contrast_num))
        if self.use_flip_num:
            self.functions.append(_flip(self.use_flip_num))
        if self.use_pooling_num:
            self.functions.append(_geometric(self.use_pooling_num))
        if self.use_arithmetic_num:
            self.functions.append(_pooling(self.use_arithmetic_num))
        if self.use_weather_num:
            self.functions.append(_weather(self.use_weather_num))

        self.functions = list(chain.from_iterable(self.functions))

    def __call__(self, img):
        
        if self.flag == 'all':
            return iaa.Sequential(self.functions)
        elif self.flag == 'some':
            return iaa.SomeOf((0, len(self.functions)), self.functions)
        elif self.flag == 'one':
            return iaa.OneOf(self.functions)

        # if self.flag == 'all':
        #     return iaa.Sequential(self.functions).augment_images(img)
        # elif self.flag == 'some':
        #     return iaa.SomeOf((0, len(self.functions)), self.functions).augment_images(img)
        # elif self.flag == 'one':
        #     return iaa.OneOf(self.functions).augment_images(img)


if __name__ == "__main__":
    imgfile_add = '1.jpg'
    imgfile_clear = '2.jpg'
    imgfile_push = '3.jpg'
    add = cv2.imread(imgfile_add)
    push = cv2.imread(imgfile_push)
    clear = cv2.imread(imgfile_clear)

    b, g, r = cv2.split(add)
    add = cv2.merge([r, g, b])

    b, g, r = cv2.split(push)
    push = cv2.merge([r, g, b])

    b, g, r = cv2.split(clear)
    clear = cv2.merge([r, g, b])

    imgs =[add, push, clear]
    img_aug = ImaugFunctions(all_some_one='one', use_arithmetic_num=3, use_weather_num=1, use_blur_num=2,  use_blend_num=1, use_contrast_num=0, use_flip_num=1, use_geometric_num=1)

    ia.imshow(np.hstack(img_aug(imgs)))
    # ia.imshow(np.hstack(imgs))
