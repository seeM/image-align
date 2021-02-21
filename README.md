# image-alignment

Run with docker:

```bash
docker build -t image-align .
docker run -v /full/path/to/local/data:/data image-align
```

This scores the approach on both the train and validation sets. Note that it
takes ~20 minutes to run and uses all CPU cores.

---

I start on problems like these by defining a metric, establishing a baseline,
then iterating with the simplest possible solutions. Given that I don't have
experience with this sort of data (aerial imagery of trees), I started with
classical methods to get a deeper understanding of the problem space.

## Problem formulation

I first considered using the `transform` array provided in the manifests as
the target variable, but found that they were defined for the full survey
data thus don't provide the best alignments for tile data.

The next simplest target is the aligned image, and a standard error metric
between it and the transformed misaligned image; I used the pixel-wise mean
squared error (MSE).

In summary, the problem is:

_Given triplets of reference images, `ref`, misaligned images, `x`, and
aligned images, `y`, determine a transformation that minimises
`mse(transform(x|ref), y)`._

## Solution: Keypoint-based alignment of grayscale images

I used keypoint-based alignment of grayscale images, specifically the [oriented
FAST and rotated BRIEF
(ORB)](https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF) detector
and extractor along with [random sample consensus
(RANSAC)](https://en.wikipedia.org/wiki/Random_sample_consensus), both
implemented in skimage and described below. This is a classical approach to
image alignment using features (keypoints and descriptors).[^tutorials]

[^tutorials]: Further reading:
    - [Image Alignment (Feature Based) using OpenCV (C++/Python)](https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/)
    - [ORB feature detector and binary descriptor](https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_orb.html)
    - [Using geometric transformations](https://scikit-image.org/docs/dev/auto_examples/transform/plot_matching.html)

The rough idea is as follows:

1. Find _keypoints_ (points of interest) in both images.
2. Extract _descriptors_ (feature vectors that describe
   each keypoint) for each keypoint.
3. Match keypoints from the first image to those in the second image, by
   comparing their descriptors.
4. Estimate a _transform_ using keypoint pairs.

### ORB

ORB, as the name suggests, uses the [Features From Accelerated Segment Test
(FAST)](https://en.wikipedia.org/wiki/Features_from_accelerated_segment_test)
feature point extractor and a modified [Binary Robust Independent Elementary
Features (BRIEF)](https://www.cs.ubc.ca/~lowe/525/papers/calonder_eccv10.pdf)
binary fature descriptor.

### FAST

FAST determines if a given pixel is a corner (or keypoint) by considering
a circle of pixels around it: If `N` contiguous pixels along the circle are
either darker (or lighter) than the center pixel within some `threshold`, its
labelled as a corner.

### BRIEF

BRIEF computes a bit string via binary tests comparing the intensities of
pairs of pixels in a given arrangement. Since these binary tests only rely on
intensity, they are noise-sensitive, therefore the image is smoothed
beforehand.[^smoothing] The spatial arrangement of point pairs was determined
through an experiment comparing random arrangements with different
distributions. The experiment used ground truth data of images undergoing
known homographies. The ability to redetect keypoints from the first image in
further images was quantified. The best arrangement was found to be an
isotropic Gaussian distribution with variance `1/25 S^2` where `S` is the
width of the square image patch.

[^smoothing]: A nice way to think of this is that the test computes the sign of a
      derivative. Smoothing is therefore useful here for the same reasons
      as in edge detection.

### Brute-force descriptor matching

I used skimage's `match_descriptors` which uses brute-force matching, i.e.,
each descriptor in the reference image is matched to the "closest" descriptor
in the misaligned image using the Hamming distance (by default, skimage uses
the Hamming distance for binary descriptors).

### RANSAC

RANSAC iteratively fits a model (in our case, an affine transformation) to a
dataset (point pairs between the reference and misaligned images) on a minimal
subset of the data, discards outliers in the dataset using the model's
likelihood, and repeats until a sufficient proportion of the data are inliers.

### Implementation details

The code is structured using a primary `find_transform` function that is
composed using `preprocess`, `match_keypoints`, and `estimate_transform`
functions. The design is to enable flexibly changing those functions in search
for a more performant method, either through hyperparameter search or by
implementing new functions at each step with the same interfaces.

The problem is [embarrassingly
parallel](https://en.wikipedia.org/wiki/Random_sample_consensus); parallel
processing is implemented using the standard library
`concurrent.futures.ProcessPoolExecutor`.

### Performance

The main performance criteria are speed (train time and inference time) and
accuracy (MSE between aligned and transformed misaligned images). Tests were
performed on a MacBook Pro (2017, 3.1 GHz Dual-Core Intel Core i5, 16 GB 2133
MHz LPDDR3, Intel Iris Plus Graphics 650 1536 MB).

- **Speed.** Train time: No training or hyperparameter tuning was performed.
  Inference time: ~8 seconds per image.
- **Accuracy.** Average train MSE: 3.998e-4, average validation MSE: 5.032e-5.

INFO:__main__:Train MSE: 3.998e-4
0.00039981827INFO:__main__:Train duratio
n: 0:18:55.040967


INFO:__main__:Val MSE: 5.031572e-05
INFO:__main__:Val duration: 0:10:46.224172


See the next section for suggested improvements.

## Future improvements

I see the following shortcomings in this solution:

- **Does not leverage the supervised dataset.** I prefer to start with
  simpler solutions, thus spent time on more classical approaches. However,
  since the problem already has a supervised dataset with a few hundred
  training examples, a method that actually learns from data would likely be
  more performant.
- **Does not make the best use of all three RGB channels.** ORB only works on
  a single channel, thus they must be aggregated or selected somehow, losing
  possibly crucial information.
- **Slow.** Detecting keypoints, extracting descriptors, and running RANSAC
  on each new image pair is slow. An end-to-end model trained to predict
  a transformation given reference and misaligned images would likely have
  slower training times but faster inference times.
- **Downscaling reference image loses information.**

With more time, I'd consider trying an end-to-end convolutional neural
network (CNN) that accepts a pair of images as input, reference and
misaligned, and predicts a transformation matrix. [This
paper](https://arxiv.org/abs/1606.03798) and [this
implementation](https://github.com/mazenmel/Deep-homography-estimation-Pytorch)
look like good starting points.

## Appendix A: Area-based alignment of non-zero pixel value masks

I excluded this solution's code because the instructions suggested that only
tile images are available, but I think it's interesting enough to mention.

If you're provided full raster image TIFF files, there's a simple solution.
Assuming that RGB and RED sensors are placed close together thus capturing
mostly the same scene, and that both images were captured in the same drone
flight, the shape of non-zero image values in the RGB image and the RED image
should be identical. We can therefore find a transform using an area-based
alignment method on non-zero pixel value masks.

I used opencv's `findTransformECC`, which is implements the algorithm in [this
paper](http://xanthippi.ceid.upatras.gr/people/evangelidis/george_files/VISAPP_2008.pdf).
It's a gradient-based iterative approach to maximising the nonlinear
optimisation criterion, the _enhanced correlation coefficient_ of intensities
of pairs of pixels in either image.
