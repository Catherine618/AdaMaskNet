# AdaMaskNet
A novel deep learning framework for sensor-based Human Activity Recognition

## **Introduction**

With the proliferation of ubiquitous computing and the Internet of Things (IoT), sensor-based Human Activity Recognition (HAR) has become widely utilized in healthcare, sports tracking, and environmental automation. Traditional models like Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs) have struggled to balance receptive field size and computational efficiency, limiting their ability to fully capture multi-scale temporal features.

To address these issues, we propose **AdaMaskNet**, a novel deep learning framework that dynamically generates receptive fields based on the temporal length of sensor data. AdaMaskNet incorporates three key modules:
* Receptive Field Scaling
* Multi-Scale Enhancement
* Fine-Grained Refinement

These modules enable coarse-to-fine feature extraction that boosts recognition accuracy and computational efficiency.

## **Features**

* **Dynamic Receptive Field Generation**: Adapts the receptive field to the temporal length of input data.
* **Multi-Scale Temporal Feature Extraction**: Captures both short-term and long-term dependencies.
* **State-of-the-Art Performance**: Achieves top accuracy on multiple HAR datasets.


## **Model Architecture**

![arch](https://github.com/user-attachments/assets/bc13f0af-bef8-4d17-a262-cae0a412b059)

## **Results**

We evaluate AdaMaskNet on four popular HAR datasets, achieving state-of-the-art accuracy:

* UCI-HAR: 97.42%
* PAMAP2: 92.53%
* UNIMIB-SHAR: 76.02%
* WISDM: 99%

These results highlight the superiority of AdaMaskNet in recognizing human activities across varied time scales.


# **Requirements**

This project requires the following environment configuration:
* Python >= 3.10
* Pytorch >= 1.12
  
To install the dependencies listed in `requirements.txt`, run the following command:
```bash
pip install -r requirements.txt
```

# **Dataset Overview and Preparation**
This work uses the following datasets for Human Activity Recognition (HAR) tasks: UCI-HAR\[1], PAMAP2\[2-3], UNIMIB-SHAR\[4], WISDM\[5]

* UCI-HAR: This dataset includes six common daily activities, providing a foundation for evaluating model performance in recognizing standard daily actions.

* PAMAP2: With a wider range of activity types and multi-modal sensor data, this dataset enables testing the model's effectiveness in handling complex, multi-modal data for diverse activity recognition.

* UNIMIB-SHAR: This dataset includes fall-related abnormal activities, allowing us to assess the model's ability to detect unusual behaviors.

* WISDM: This dataset contains noise and irregularities, making it closer to real-world scenarios. It is used to evaluate the model’s robustness and reliability in handling data with natural variability.

| Attribute          | UCI-HAR                                             | PAMAP2                                               | UNIMIB-SHAR                                  | WISDM                                            |
|--------------------|-----------------------------------------------------|------------------------------------------------------|----------------------------------------------|--------------------------------------------------|
| **Subjects**       | 30                                                  | 9                                                    | 30                                           | 29                                               |
| **Sensor Types**   | A, G, M                                             | A, G, M, HR-monitor                                  | A                                            | A                                                |
| **Sensor Placement** | Waist                                             | Hand, Chest, Ankle                                   | Front trouser pockets                        | Front pants leg pockets                          |
| **Sample Rate**    | 50Hz                                                | 33.3Hz                                               | 50Hz                                         | 50Hz                                             |
| **Activity Categories** | 6                                              | 12                                                   | 17                                           | 6                                                |
| **Activity Classes** | standing, sitting, laying down, walking, walking downstairs, upstairs | lying, sitting, standing, walking, running, cycling, etc. | walking, sitting, standing, forward falling, syncope falling, etc. | walking, jogging, ascending stairs, descending stairs, sitting, standing |
| **Window Size**    | 128                                                 | 171                                                  | 151                                          | 200                                              |
| **Overlap Rate**   | 50%                                                 | 78%                                                  | 50%                                          | 50%                                              |

*Note: A: Accelerometer, G: Gyroscope, M: Magnetometer.*

After downloading, each dataset should be placed in its respective folder under `dataset/` as shown below.
```plaintext
project_folder/
├── dataset/
│   ├── UCI_HAR/
│   │   ├── x_test.npy
│   │   ├── x_train.npy
│   │   ├── y_test.npy
│   │   └── y_train.npy
│   ├── PAMAP2/
│   │   ├── x_test.npy
│   │   ├── x_train.npy
│   │   ├── y_test.npy
│   │   └── y_train.npy
│   ├── UNIMIB_SHAR/
│   │   ├── x_test.npy
│   │   ├── x_train.npy
│   │   ├── y_test.npy
│   │   └── y_train.npy
│   └── WISDM/
│       ├── x_test.npy
│       ├── x_train.npy
│       ├── y_test.npy
│       └── y_train.npy
```

# Train and Test the Model
To train and test the model, please follow the steps below:

## Train the Model
To train the model, ensure the datasets are prepared as described above. Use the following command to start training:

```bash
python main.py --dataset WISDM --model_name AdaMaskNet --epochs 200 --batch_size 32 --learning_rate 5e-4 --output_dir /path/to/output
```
**Arguments：**

--**dataset**: Specifies the dataset to use for training. Options are UCI_HAR, PAMAP2, UNIMIB_SHAR, or WISDM. Default is WISDM.

--**model_name**: The name of the model to train. Default is AdaMaskNet.

--**epochs**: Number of training epochs. Default is 200.

--**batch_size**: Batch size for training. Default is 32.

--**learning_rate**: Learning rate for the optimizer. Default is 5e-4.

--**resume**: If you want to resume training from a checkpoint, provide the path to the checkpoint file. If not specified or set to False, training starts from scratch.

--**checkpoint_path**: Specifies where to save checkpoint files during training. Default is an empty string, which uses the default path.

--**output_dir**: Directory where outputs (logs, checkpoints) will be saved. If not specified, a default directory checkpoint/p_<timestamp> will be created.

## Test the Model

Once the model is trained, you can evaluate its performance on the test dataset using the following command:
```bash
python test.py --dataset WISDM --model_name AdaMaskNet --resume /path/to/checkpoint.pth --batch_size 32 --output_dir /path/to/output
```
**Arguments：**

--**dataset**: Specifies the dataset to use for testing. Should match the dataset used during training.

--**model_name**: The name of the model to test. Should match the model used during training.

--**resume**: Path to the trained model checkpoint file to load for testing.

--**batch_size**: Batch size for testing. Default is 32.

--**output_dir**: Directory where test outputs (logs, results) will be saved.

**Notes**

Ensure that the --resume argument points to the correct checkpoint file containing the trained model weights.


# **Reference**
\[1] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, Jorge Luis Reyes-Ortiz, et al. A public
domain dataset for human activity recognition using smartphones. In Esann, volume 3, page 3, 2013.

\[2] Attila Reiss and Didier Stricker. Introducing a new benchmarked dataset for activity monitoring.
In 2012 16th international symposium on wearable computers, pages 108–109. IEEE, 2012.

\[3] Attila Reiss and Didier Stricker. Creating and benchmarking a new dataset for physical activity
monitoring. In Proceedings of the 5th international conference on pervasive technologies related to
assistive environments, pages 1–8, 2012.

\[4] Daniela Micucci, Marco Mobilio, and Paolo Napoletano. Unimib shar: A dataset for human activity
recognition using acceleration data from smartphones. Applied Sciences, 7(10):1101, 2017.

\[5] Jennifer R Kwapisz, Gary M Weiss, and Samuel A Moore. Activity recognition using cell phone
accelerometers. ACM SigKDD Explorations Newsletter, 12(2):74–82, 2011.



# **Citation**

@article{AdaMaskNet,
  title={AdaMaskNet: Adaptive Multi-scale Masked Kernels for Enhanced Sensor-based Human Activity Recognition},
  year={2024}
}

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14056246.svg)](https://doi.org/10.5281/zenodo.14056246)


