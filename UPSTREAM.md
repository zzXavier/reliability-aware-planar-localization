# Upstream, models and data

| Component | Pinned input | How it is used |
| --- | --- | --- |
| XFeat | [verlab/accelerated_features at e92685f](https://github.com/verlab/accelerated_features/tree/e92685f57f8318b18725c5c8c0bd28c7fe188d9a) | Unmodified sparse detector, descriptor and mutual nearest-neighbour matching; model weights come with the pinned checkout. [Upstream LICENSE](https://github.com/verlab/accelerated_features/blob/e92685f57f8318b18725c5c8c0bd28c7fe188d9a/LICENSE). |
| OpenCV | `opencv-python-headless==4.10.0.84` | ORB with mutual Hamming matching, USAC_MAGSAC homography fitting and image resizing. [OpenCV](https://github.com/opencv/opencv). |
| HPatches full sequences | [author-hosted archive at 594f211](https://huggingface.co/datasets/vbalnt/hpatches/tree/594f211e80b75081ad84a396c93960c8653235f8) | Images and homographies are downloaded for evaluation. This repository redistributes derived numeric predictions and a generated poster example. [Dataset authors and original image source citations](https://github.com/hpatches/hpatches-dataset/blob/fd79bca4dc755c9f623a5f129f3452fc663ee305/references.txt). |

The archive SHA-256 is `99cf7e1ca167896eb4ca2fe3d903beff6566243a3c0c8e07a13a2596648ad670` (1,279,545,572 bytes). The XFeat weights SHA-256 is `0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b`. The runner checks both pins before formal inference.

**Upstream references.** Guilherme Potje and coauthors, *XFeat: Accelerated Features for Lightweight Image Matching*, CVPR 2024. Vassileios Balntas, Karel Lenc, Andrea Vedaldi and Krystian Mikolajczyk, *HPatches: A Benchmark and Evaluation of Handcrafted and Learned Local Descriptors*, CVPR 2017.

**Local contribution.** Sequence-level split, fixed operating points learned on development scenes, inlier coverage/error diagnostics, formal evaluation, saved predictions, a generated-poster demo and tests. This repository evaluates pretrained features; it does not train XFeat.
