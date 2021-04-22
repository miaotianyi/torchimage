

def hdrvdp3(image_true, image_test, task, color_encoding, pixels_per_degree,
            surround=None, age=24, spectral_emission=None, mtf="hdrvdp",
            rgb_display=None, sensitivity_correction=0.0, mask_p=None, mask_q=None
            ):
    assert image_true.shape == image_test.shape, "Test and reference images must have the same shape"

    # replace the functionalities of matlabPyrTools


