Formal Verification of Object Detection
======================

<p align="center">
<a href="[[https://arxiv.org/pdf/2407.01295](https://arxiv.org/abs/2407.01295)](https://arxiv.org/pdf/2407.01295)">
<!--<img src="https://www.huan-zhang.com/images/upload/alpha-beta-crown/logo_2022.png" width="36%"></a></p>-->

This repository contains the code used to implement the paper on [Formal Verification of Object Detection](https://arxiv.org/pdf/2407.01295).
<p align="justify"> Deep Neural Networks (DNNs) are ubiquitous in real-world
applications, yet they remain vulnerable to errors and adversarial attacks. This work tackles the challenge of applying formal verification
to ensure the safety of computer vision models, extending verification
beyond image classification to object detection. We propose a general
formulation for certifying the robustness of object detection models using formal verification and outline implementation strategies compatible
with state-of-the-art verification tools. Our approach enables the application of these tools, originally designed for verifying classification models,
to object detection. We define various attacks for object detection, illustrating the diverse ways adversarial inputs can compromise neural network outputs. Our experiments, conducted on several common datasets
and networks, reveal potential errors in object detection models, highlighting system vulnerabilities and emphasizing the need for expanding
formal verification to these new domains. This work paves the way for
further research in integrating formal verification across a broader range
of computer vision applications.
</p>

## Dependencies

For verification purposes, the code uses the SOTA verification tool [Alpha-Beta-Crown](https://arxiv.org/pdf/2103.06624.pdf). Please follow their guide for installation. 

For LARD, MNIST-OD datasets and their coresponding trained networks, we used [Verification4ObjectDetection project](https://github.com/NoCohen66/Verification4ObjectDetection). We have already migrated the data and networks, so no additional installations are needed.

## Reproduce Results
For running LARD/MNIST-OD datasets, please use the relevant YAML files.


