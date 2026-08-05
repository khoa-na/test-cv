# Third-party notices

This repository combines original engineering work with third-party software,
models, datasets, and benchmark evidence. The repository is distributed under
the GNU Affero General Public License v3.0 (`AGPL-3.0-only`), but third-party
materials remain subject to their own terms.

## Ultralytics YOLO

- Component: Ultralytics Python package, YOLO26 architecture, training/export
  pipeline, and `models/pothole_yolo26n_seg.onnx`.
- License: GNU Affero General Public License v3.0 unless covered by a separate
  Ultralytics Enterprise License.
- Source: <https://github.com/ultralytics/ultralytics>
- Licensing information: <https://www.ultralytics.com/license>

The ONNX model embeds `license=AGPL-3.0` in its metadata. Commercial or
closed-source deployment may require a separate license from Ultralytics.

## Pothole-600

- Use: detector training and in-domain evaluation.
- Source: <https://sites.google.com/view/pothole-600/dataset>
- Terms observed: the publisher describes the dataset as publicly available
  for research and requests citation, but the download page does not state a
  standard open-source license.

The trained model is provided for open research and portfolio demonstration.
Users must confirm that their intended use complies with the dataset terms.

## PothRGBD

- Use: additional segmentation masks and ROI depth-regressor training.
- Dataset page: <https://www.kaggle.com/datasets/mahyeks/pothrgbd-rgb-and-depth-images-of-potholes>
- License shown by the dataset host: MIT.

No PothRGBD source image or depth file is redistributed in this repository.

## Fan stereo pothole dataset

- Use: stereo depth/area calibration and evaluation.
- Source: <https://github.com/ruirangerfan/stereo_pothole_datasets>
- License: MIT, as published with the dataset.
- Derived files: `artifacts/a3-grid/final-s03125-d112/failures/*.jpg`.

Those images are retained only as benchmark evidence and should preserve the
dataset attribution when reused.

## Pothole Videos — Mendeley Data 5bwfg4v4cd, version 2

- Use: cross-dataset evaluation and a separately hosted demo video.
- DOI: <https://doi.org/10.17632/5bwfg4v4cd.2>
- License: Creative Commons Attribution 4.0 (`CC BY 4.0`).
- Authors: Muhammad Ihsan, Agus Harjoko, and Muhammad Alfian Amrizal.

## 4Seasons

- Use: offline stereo VO, GPS/IMU replay, localization evaluation, and demo.
- Source: <https://4seasons-dataset.com/dataset>
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  (`CC BY-NC-SA 4.0`).
- Copyright: Artisense.

The dataset requires registration, is limited to non-commercial use, and is
not redistributed here. The following tracked evidence is derived from it and
is separately subject to `CC BY-NC-SA 4.0`:

- `artifacts/vo-drift-final/office_loop_1.png`
- numeric benchmark receipts under `artifacts/*localization*`,
  `artifacts/landmark-reid`, `artifacts/uturn-b3`, and
  `artifacts/system-fps-b7`
- the separately hosted Part B demo video

## Demo videos

Demo videos are hosted separately at
<https://huggingface.co/datasets/khoa-na/pothole-gps-localization-demos> and
are published under `CC BY-NC-SA 4.0`. Each video identifies its source data
on-screen.

