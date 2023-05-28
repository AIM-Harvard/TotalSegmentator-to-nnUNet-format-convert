import SimpleITK as sitk


SITK_INTERPOLATOR_DICT = {
    'nearest': sitk.sitkNearestNeighbor,
    'linear': sitk.sitkLinear,
    'gaussian': sitk.sitkGaussian,
    'label_gaussian': sitk.sitkLabelGaussian,
    'bspline': sitk.sitkBSpline,
    'hamming_sinc': sitk.sitkHammingWindowedSinc,
    'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
    'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
    'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
}


def resample_image_to_spacing(image, new_spacing, default_value, interpolator='linear'):
    assert interpolator in SITK_INTERPOLATOR_DICT, \
        (f"Interpolator '{interpolator}' not part of SimpleITK. "
         f"Please choose one of the following {list(SITK_INTERPOLATOR_DICT.keys())}.")
    assert image.GetDimension() == len(new_spacing), \
        (f"Input is {image.GetDimension()}-dimensional while "
         f"the new spacing is {len(new_spacing)}-dimensional.")

    interpolator = SITK_INTERPOLATOR_DICT[interpolator]
    spacing = image.GetSpacing()
    size = image.GetSize()
    new_size = [int(round(siz * spac / n_spac)) for siz, spac, n_spac in zip(size, spacing, new_spacing)]
    return sitk.Resample(
        image,
        new_size,             # size
        sitk.Transform(),     # transform
        interpolator,         # interpolator
        image.GetOrigin(),    # outputOrigin
        new_spacing,          # outputSpacing
        image.GetDirection(), # outputDirection
        default_value,        # defaultPixelValue
        image.GetPixelID()    # outputPixelType
    )
