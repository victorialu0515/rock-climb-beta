# Climbing Beta — AI Move Prediction

An interactive walkthrough of a computer-vision pipeline that watches a bouldering wall, detects holds, tracks a climber's pose, and predicts the next move. Built from three models trained from scratch: a YOLOv8 hold detector, HRNet pose estimation, and a linear regression move scorer.

**[Open `index.html` in a browser to view the demo.](./index.html)**

## Folder structure

```
.
├── index.html          Main demo page (open this)
├── assets/              8 real output images from the pipeline
│   ├── diamond_raw.jpg / diamond_detection.jpg / diamond_pose.jpg / diamond_nextmove.jpg
│   └── willtopia_raw.jpg / willtopia_detection.jpg / willtopia_pose.jpg / willtopia_nextmove.jpg
├── scripts/              The actual pipeline code that generated the images
│   ├── hold_detection.py            YOLOv8 detection + K-means color/track clustering
│   ├── pose_matching_and_scoring.py Hold matching, candidate move generation, model scoring
│   └── render_assets.py             Draws the bounding boxes / skeleton / move overlays
└── README.md
```

## Source repos

- [rockDetectionLambda](https://github.com/victorialu0515/rockDetectionLambda) — hold detection, deployed as an AWS Lambda
- [HRNET](https://github.com/victorialu0515/HRNET) — pose tracking, deployed as an AWS Lambda
- [poseScore](https://github.com/victorialu0515/poseScore) — move scoring, deployed as an AWS Lambda
- [humanPoseEstimation](https://github.com/victorialu0515/humanPoseEstimation) — reference pose models and test data

## Notes on this demo vs. the live system

The live system runs on AWS Lambda + S3 with a phone camera feed. This page is a static, offline walkthrough instead — real model output, captured once rather than served live, so it costs nothing to host and never goes down.

The pose skeletons here were hand-reconstructed from saved HRNet render images (the original ONNX weights weren't available when this was built) and verified point-by-point against those renders. Everything else — detection, clustering, hold matching, and scoring — is the unmodified original code running on real data.
