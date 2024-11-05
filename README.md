# AdaMaskNet
A novel deep learning framework for sensor-based Human Activity Recognition

**Introduction**

With the proliferation of ubiquitous computing and the Internet of Things (IoT), sensor-based Human Activity Recognition (HAR) has become widely utilized in healthcare, sports tracking, and environmental automation. Traditional models like Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs) have struggled to balance receptive field size and computational efficiency, limiting their ability to fully capture multi-scale temporal features.

To address these issues, we propose **AdaMaskNet**, a novel deep learning framework that dynamically generates receptive fields based on the temporal length of sensor data. AdaMaskNet incorporates three key modules:
* Receptive Field Scaling
* Multi-Scale Enhancement
* Fine-Grained Refinement

These modules enable coarse-to-fine feature extraction that boosts recognition accuracy and computational efficiency.

**Features**

* **Dynamic Receptive Field Generation**: Adapts the receptive field to the temporal length of input data.
* **Multi-Scale Temporal Feature Extraction**: Captures both short-term and long-term dependencies.
* **State-of-the-Art Performance**: Achieves top accuracy on multiple HAR datasets.

**Results**

We evaluate AdaMaskNet on four popular HAR datasets, achieving state-of-the-art accuracy:

* UCI-HAR: 97.42%
* PAMAP2: 92.53%
* UNIMIB-SHAR: 76.02%
* WISDM: 99%

These results highlight the superiority of AdaMaskNet in recognizing human activities across varied time scales.

**Installation**

To be updated once the code is organized and released.

**Model Architecture**

![arch](https://github.com/user-attachments/assets/bc13f0af-bef8-4d17-a262-cae0a412b059)

**Citation**

@article{AdaMaskNet,
  title={AdaMaskNet: Adaptive Multi-scale Masked Kernels for Enhanced Sensor-based Human Activity Recognition},
  journal={Journal Name},
  year={2024}
}

**License**

This project is licensed under the MIT License - see the LICENSE file for details.

