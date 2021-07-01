import os
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch.nn import MSELoss, L1Loss
from torchimage.padding import Padder
from torchimage.pooling import AvgPoolNd, GaussianPoolNd
from torchimage.random import random_crop, add_gauss_noise, add_poisson_gauss_noise
from torchimage.filtering import edges, UnsharpMask
from torchimage.metrics import SSIM, MS_SSIM

nchw_axes = (2, 3)

# use constant to prevent misspelling
IDENTITY = "identity"
AVG_POOL = "avg_pool"
GAUSS_POOL = "gauss_pool"
GAUSS_NOISE = "gauss_noise"
PG_NOISE = "pg_noise"
UNSHARP_MASK = "unsharp_mask"


def format_keywords(kwargs):
    if isinstance(kwargs, str):
        return kwargs

    option = kwargs["option"]
    for ignored in ["option", "mode", "magnitude"]:
        if ignored in kwargs:
            del kwargs[ignored]  # ignore mode
    return f"{option}" + "".join(f", {key}={val}" for key, val in kwargs.items())


def get_padder(mode):
    return Padder(mode=mode)


def get_augment(option, **kwargs):
    if option == IDENTITY:
        return lambda x: x
    elif option == AVG_POOL:
        size = kwargs["size"]
        mode = kwargs["mode"]
        return lambda x: AvgPoolNd(kernel_size=size).to_filter(get_padder(mode)).forward(x, axes=nchw_axes)
    elif option == GAUSS_POOL:
        size = kwargs["size"]
        sigma = kwargs["sigma"]
        mode = kwargs["mode"]
        return lambda x: GaussianPoolNd(kernel_size=size, sigma=sigma).to_filter(get_padder(mode)).forward(
            x, axes=nchw_axes)
    elif option == GAUSS_NOISE:
        sigma = kwargs["sigma"]
        return lambda x: add_gauss_noise(x, sigma=sigma).clamp(0, 1)
    elif option == PG_NOISE:
        sigma = kwargs["sigma"]
        k = kwargs["k"]
        return lambda x: add_poisson_gauss_noise(x, k=k, sigma=sigma).clamp(0, 1)
    elif option == UNSHARP_MASK:
        sigma = kwargs["sigma"]
        size = kwargs["size"]
        mode = kwargs["mode"]
        amount = kwargs["amount"]
        return lambda x: UnsharpMask(blur=GaussianPoolNd(kernel_size=size, sigma=sigma), amount=amount, padder=Padder(mode=mode)
                                     ).forward(x, axes=nchw_axes).clamp(0, 1)
    else:
        raise ValueError(f"Unknown augmentation type {option}")


SOBEL = "sobel"
PREWITT = "prewitt"
FARID = "farid"
SCHARR = "scharr"
GAUSSIAN_GRAD = "gaussian_grad"
LAPLACE = "laplace"
LAPLACIAN_OF_GAUSSIAN = "laplacian_of_gaussian"


def get_edge(option, **kwargs):
    if option == IDENTITY:
        return lambda x: x

    mode = kwargs["mode"]

    if option == LAPLACIAN_OF_GAUSSIAN:
        size = kwargs["size"]
        sigma = kwargs["sigma"]
        return lambda x: edges.LaplacianOfGaussian(kernel_size=size, sigma=sigma, same_padder=Padder(mode=mode)
                                                   ).forward(x, axes=nchw_axes)

    magnitude = kwargs["magnitude"]

    if option == GAUSSIAN_GRAD:
        size = kwargs["size"]
        sigma = kwargs["sigma"]
        order = kwargs["order"]
        detector = edges.GaussianGrad(kernel_size=size, sigma=sigma, edge_order=order, same_padder=Padder(mode=mode))
    elif option == LAPLACE:
        detector = edges.Laplace(same_padder=Padder(mode=mode))
    elif option == SOBEL:
        normalize = kwargs["normalize"]
        detector = edges.Sobel(normalize=normalize, same_padder=Padder(mode=mode))
    elif option == PREWITT:
        normalize = kwargs["normalize"]
        detector = edges.Prewitt(normalize=normalize, same_padder=Padder(mode=mode))
    elif option == FARID:
        normalize = kwargs["normalize"]
        detector = edges.Farid(normalize=normalize, same_padder=Padder(mode=mode))
    elif option == SCHARR:
        normalize = kwargs["normalize"]
        detector = edges.Scharr(normalize=normalize, same_padder=Padder(mode=mode))

    if magnitude:
        return lambda x: detector.magnitude(x, axes=nchw_axes)
    else:
        return lambda x: torch.cat([detector.component(x, 2, 3), detector.component(x, 3, 2), ], dim=1)


def get_loss(option: str):
    # function signature: loss(y1, y2) -> float
    if option == "l1":
        return lambda y1, y2: L1Loss(reduction="mean")(y1, y2).item()
    elif option == "mse":
        return lambda y1, y2: MSELoss(reduction="mean")(y1, y2).item()
    else:
        raise ValueError(f"Unknown loss type {option}")


def all_augment():
    yield {"option": IDENTITY}

    default_mode = "reflect"

    for amount in range(1, 8, 2):
        amount = amount / 10
        for sigma in [1, 1.5, 2, 2.5]:
            size = sigma * 8 + 1
            yield {"option": UNSHARP_MASK, "sigma": sigma, "size": size, "amount": amount, "mode": default_mode}

    for size in range(3, 15, 2):
        yield {"option": AVG_POOL, "size": size, "mode": default_mode}

    for sigma in [0.5, 1, 1.5, 2, 2.5, 3]:
        size = sigma * 8 + 1
        yield {"option": GAUSS_POOL, "sigma": sigma, "size": size, "mode": default_mode}

    for sigma in [0.001, 0.003, 0.01, 0.03, 0.1]:
        yield {"option": GAUSS_NOISE, "sigma": sigma}

    for sigma in [0.001, 0.003, 0.01, 0.03, 0.1]:
        for k in [0.03, 0.1, 0.3, 1, 3]:
            yield {"option": PG_NOISE, "k": k, "sigma": sigma}


def all_edge():
    yield {"option": IDENTITY}

    default_mode = "reflect"

    for sigma in [0.5, 1, 1.5, 2, 2.5, 3]:
        size = sigma * 8 + 1
        yield {"option": LAPLACIAN_OF_GAUSSIAN, "sigma": sigma, "size": size, "mode": default_mode}

    # for magnitude in True, False:
    magnitude = False
    for order in [1, 2]:
        for sigma in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
            size = sigma * 8 + 1
            yield {"option": GAUSSIAN_GRAD, "sigma": sigma, "size": size, "mode": default_mode, "magnitude": magnitude, "order": order}

    yield {"option": LAPLACE, "mode": default_mode, "magnitude": magnitude}

    normalize = True
    for option in SOBEL, PREWITT, FARID, SCHARR:
        yield {"option": option, "normalize": normalize, "mode": default_mode,
               "magnitude": magnitude}


def all_loss():
    # for a in ["l1", "mse"]:
    for a in ["l1"]:
        yield {"option": a}


def run_experiment(image_path: str, result_path: str, crop_size: int or tuple, seed=None):
    # to be saved as JSON
    result = {}

    # reproducible
    if seed is None:
        seed = torch.seed()
    else:
        torch.manual_seed(seed)

    # read image
    image = torch.tensor(plt.imread(image_path))
    image = image.movedim(-1, 0).unsqueeze(0)
    image = random_crop(image, axes=nchw_axes, size=crop_size)

    # basic settings
    result["seed"] = seed
    result["crop_size"] = crop_size
    result["image_path"] = image_path

    # prepare results
    result["data"] = []  # 3d array (n edge options, n aug options, n loss options)
    result["aug_options"] = []  #
    result["edge_options"] = []
    result["loss_options"] = []

    for i, aug_param in enumerate(all_augment()):
        # save axis
        result["aug_options"].append(aug_param)

        result["data"].append([])

        aug_func = get_augment(**aug_param)

        noisy = aug_func(image)

        print(result["aug_options"][-1])

        for j, edge_param in enumerate(all_edge()):
            # save axis
            if i == 0:
                result["edge_options"].append(edge_param)
            result["data"][i].append([])

            edge_func = get_edge(**edge_param)

            noisy_mapped = edge_func(noisy)
            gt_mapped = edge_func(image)

            for k, loss_param in enumerate(all_loss()):
                if i == j == 0:  # save axis
                    result["loss_options"].append(format_keywords(loss_param))

                loss_func = get_loss(**loss_param)

                result["data"][i][j].append(
                    loss_func(gt_mapped, noisy_mapped)
                )

    with open(result_path, "w") as output_file:
        json.dump(result, output_file)

    return result


def run_correlation(result_path: str):
    with open(result_path, "r") as file:
        result = json.load(file)
    data = np.array(result["data"])
    edge_options = result["edge_options"]
    aug_options = result["aug_options"]

    plt.imshow(np.corrcoef(data[:, :, 0], rowvar=False), cmap="cividis")
    plt.colorbar()
    plt.yticks(range(len(edge_options)), edge_options)
    plt.xticks(range(len(edge_options)), edge_options, rotation="vertical")
    plt.show()


def visualize_experiment(result_path: str):
    with open(result_path, "r") as file:
        result = json.load(file)
    data = np.array(result["data"])
    edge_options = [format_keywords(elem) for elem in result["edge_options"]]
    aug_options = [format_keywords(elem) for elem in result["aug_options"]]

    # scale data by max score in all augmentation settings (per edge mapping)
    # data = data[:15]
    # aug_options = aug_options[:15]

    # process SSIM scores
    data[:, -3:, :] = 1 - data[:, -3:, :]

    # position of max loss when edge detector is identity (pure L1 score)
    argmax_identity = np.argmax(data[:, 0, 0])
    data = data / data[argmax_identity, :, :]

    # MSE plot
    plt.imshow(data[:, :, 0], cmap="cividis")
    plt.colorbar()

    plt.yticks(range(len(aug_options)), aug_options)
    plt.ylabel("aug_options")

    plt.xticks(range(len(edge_options)), edge_options, rotation="vertical")
    plt.xlabel("edge_options")
    plt.show()


def result_json_to_excel(result_path: str, excel_path: str):
    with open(result_path, "r") as file:
        result = json.load(file)

    os.makedirs(os.path.split(excel_path)[0], exist_ok=True)

    df = pd.DataFrame(np.array(result["data"])[:, :, 0])
    df.index = [format_keywords(kwargs) for kwargs in result["aug_options"]]
    df.columns = [format_keywords(kwargs) for kwargs in result["edge_options"]]

    df.insert(0, "Image", value=0)

    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name="Sheet1")

        # set random seed
        torch.manual_seed(result["seed"])
        # read image
        image = torch.tensor(plt.imread(result["image_path"]))
        image = image.movedim(-1, 0).unsqueeze(0)
        image = random_crop(image, axes=nchw_axes, size=result["crop_size"])

        worksheet = writer.sheets["Sheet1"]

        for i, aug_param in enumerate(all_augment()):
            print(format_keywords(aug_param))
            aug_func = get_augment(**aug_param)
            noisy = aug_func(image)
            noisy = noisy.squeeze().movedim(0, -1).numpy()
            temp_image_dir = os.path.join(os.path.split(excel_path)[0], f"noisy_{i}.png")
            plt.imsave(temp_image_dir, noisy)
            worksheet.insert_image(f"B{i+2}", temp_image_dir)


def add_new_columns(result_path: str):
    head, tail = os.path.split(result_path)
    new_path = os.path.join(head, os.path.splitext(tail)[0] + "_new.json")

    with open(result_path, "r") as file:
        result = json.load(file)

    # 3 axes: aug, edge, loss
    data = np.array(result["data"])
    result["edge_options"].extend(["SSIM", "MS_SSIM prod", "MS_SSIM sum"])
    ssim_scores = []
    multi_prod_scores = []
    multi_sum_scores = []

    # load_image
    # set random seed
    torch.manual_seed(result["seed"])
    # read image
    image = torch.tensor(plt.imread(result["image_path"]))
    image = image.movedim(-1, 0).unsqueeze(0)
    image = random_crop(image, axes=nchw_axes, size=result["crop_size"])

    for i, aug_param in enumerate(all_augment()):
        print(format_keywords(aug_param))
        aug_func = get_augment(**aug_param)
        noisy = aug_func(image)

        score_ssim = SSIM(padder=Padder(mode="reflect")).forward(noisy, image, content_axes=nchw_axes, reduce_axes=None)[0]
        ssim_scores.append(score_ssim.mean().item())
        multi_prod_scores.append(
            MS_SSIM(use_prod=True, padder=Padder(mode="reflect")).forward(noisy, image, content_axes=nchw_axes, reduce_axes=None).mean().item())
        multi_sum_scores.append(
            MS_SSIM(use_prod=False, padder=Padder(mode="reflect")).forward(noisy, image, content_axes=nchw_axes, reduce_axes=None).mean().item())
    new_data = np.concatenate([data, np.array(ssim_scores).reshape([-1, 1, 1]),
                               np.array(multi_prod_scores).reshape([-1, 1, 1]),
                               np.array(multi_sum_scores).reshape([-1, 1, 1])], axis=1)
    result["data"] = new_data.tolist()

    with open(new_path, "w") as file:
        json.dump(result, file)


def main():
    image_path = "D:/Users/miaotianyi/Downloads/images/street.png"
    json_path = "D:/Users/miaotianyi/Documents/projects/results/edge_loss_exp4.json"
    json_path_2 = "D:/Users/miaotianyi/Documents/projects/results/edge_loss_exp4_new.json"
    excel_path = "D:/Users/miaotianyi/Documents/projects/results/edge_metric_4/edge_loss_exp4.xlsx"

    # 整体思路：run experiment用JSON格式记录实验数据，visualize experiment读出JSON，作图

    # run_experiment(image_path=image_path, result_path=json_path, crop_size=512)
    visualize_experiment(result_path=json_path_2)
    # result_json_to_excel(result_path=json_path, excel_path=excel_path)
    # visualize_with_image(result_path=json_path)
    # run_correlation(result_path=json_path)
    # add_new_columns(result_path=json_path)


if __name__ == '__main__':
    main()
