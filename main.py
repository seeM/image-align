#!/usr/bin/env python
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import ORB, match_descriptors, plot_matches
from skimage.measure import ransac
from skimage.transform import resize, warp, AffineTransform


logger = logging.getLogger(__name__)

DEBUG = False
DATA_DIR = Path("data")


def preprocess(ref, x):
    logger.info("Preprocessing images")
    # Only use the red channel, since we're aligning a narrowband red image
    # It also worked best in practice
    ref = ref[:, :, 0]
    # Downscale RGB image to RED shape
    ref = resize(ref, x.shape, anti_aliasing=True)
    # Minmax scale both
    ref = ref / ref.max()
    x = x / x.max()
    return ref, x


def match_keypoints_orb(ref, x, n_keypoints):
    logger.info("Finding and matching keypoints with ORB")

    logger.info("Detecting keypoints & extracting descriptors from reference")
    ext_ref = ORB(n_keypoints=n_keypoints)
    ext_ref.detect_and_extract(ref)

    logger.info("Detecting keypoints & extracting descriptors from input image")
    ext_x = ORB(n_keypoints=n_keypoints)
    ext_x.detect_and_extract(x)

    logger.info("Matching descriptors")
    matches = match_descriptors(
        ext_ref.descriptors, ext_x.descriptors, cross_check=True
    )

    if DEBUG:
        fig, ax = plt.subplots()
        plot_matches(ax, ref, x, ext_ref.keypoints, ext_x.keypoints, matches)
        plt.show()

    src = ext_ref.keypoints[matches[:, 0]]
    dst = ext_x.keypoints[matches[:, 1]]

    return src, dst


def estimate_transform_ransac(ref, x, src, dst, **kwargs):
    logger.info("Estimating AffineTransform using RANSAC")
    transform, inliers = ransac((src, dst), **kwargs)

    if DEBUG:
        fig, ax = plt.subplots()
        inlier_idxs = np.nonzero(inliers)[0]
        plot_matches(ax, ref, x, src, dst, np.column_stack((inlier_idxs, inlier_idxs)))
        plt.show()

        outlier_idxs = np.nonzero(~inliers)[0]
        fig, ax = plt.subplots()
        plot_matches(
            ax, ref, x, src, dst, np.column_stack((outlier_idxs, outlier_idxs))
        )
        plt.show()

    return transform


def estimate_transform_basic(src, dst, transform):
    transform.estimate(src, dst)
    return transform


def mse(x, y):
    return np.mean(((x - y) ** 2))


def find_transform(row, preprocess, match_keypoints, estimate_transform):
    logger.info("Loading data")
    ref_raw = np.load(DATA_DIR / row["ref_image_path"])[0]
    x_raw = np.load(DATA_DIR / row["target_image_path"])[0]
    y_raw = np.load(DATA_DIR / row["anchor_image_path"])[0]

    ref, x = preprocess(ref_raw, x_raw)
    try:
        src, dst = match_keypoints(ref, x)
    except Exception as e:
        logger.warning(
            "Using identity transform. Keypoint matching failed with error: %s", e
        )
        transform = np.eye(3, 3)
    else:
        transform = estimate_transform(ref, x, src, dst)

    y_pred = warp(x_raw, transform)
    y = resize(y_raw, y_pred.shape, anti_aliasing=True)
    score = mse(y, y_pred)

    logger.info("MSE: %s", score)

    if DEBUG:
        images1 = np.vstack((y, y_pred))
        dummy = np.zeros_like(y_pred)
        abs_error = np.abs(y - y_pred)
        images2 = np.vstack((abs_error, dummy))
        images = np.hstack((images1, images2))
        plt.subplots(figsize=(8, 8))
        plt.imshow(images, cmap="Greys_r")
        plt.show()

    return transform, y_pred, score


def score(find_transform_impl, manifest, name):
    title = name.title()
    with ProcessPoolExecutor() as executor:
        rows = [row for _, row in manifest.iterrows()]
        start = datetime.now()
        results = list(executor.map(find_transform_impl, rows))
        dur = datetime.now() - start
        average_score = np.mean([t[-1] for t in results])
        logger.info("%s MSE: %s", title, average_score)
        logger.info("%s duration: %s", title, dur)


find_transform_orb_ransac = partial(
    find_transform,
    preprocess=preprocess,
    match_keypoints=partial(match_keypoints_orb, n_keypoints=200),
    estimate_transform=partial(
        estimate_transform_ransac,
        model_class=AffineTransform,
        min_samples=3,
        residual_threshold=2,
        max_trials=100,
    ),
)


if __name__ == "__main__":
    FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    train_manifest = pd.read_pickle("data/master_manifest_train_red.pkl")
    val_manifest = pd.read_pickle("data/master_manifest_val_red.pkl")

    score(find_transform_orb_ransac, train_manifest, "train")
    score(find_transform_orb_ransac, val_manifest, "val")
