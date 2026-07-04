# DEEPFAKE DETECTION SYSTEM

### A project submitted in partial fulfilment of the requirements for the award of the degree of

## Bachelor of Technology
### in
## Computer Science and Engineering

---

**Group Members:**

Sankalp (12311001)

Madhur Suman (12311002)

**Supervised By:**

Dr. Md. Arquam

(Assistant Professor)

---

**INDIAN INSTITUTE OF INFORMATION TECHNOLOGY, SONEPAT**

**131201, HARYANA, INDIA**

---
---

## ACKNOWLEDGEMENT

The success of this project would not have been possible without the guidance and support we received at every stage. We would like to take this opportunity to thank everyone who helped us along the way.

We are deeply grateful to Dr. Md. Arquam, Assistant Professor in the Department of Computer Science and Engineering at the Indian Institute of Information Technology, Sonepat. He gave us the opportunity to work on this project and supported us throughout, even with his busy schedule.

We are also thankful to the other faculty members of the CSE department who shared their knowledge of deep learning and multimedia forensics with us. Their feedback during review sessions helped us improve both the technical side and the written documentation of this project.

Finally, we want to thank our families and friends for their constant encouragement and support.

Sankalp

Madhur Suman

---
---

## SELF DECLARATION

We hereby state that the work contained in the project titled "Deepfake Detection System" is original. We have followed the standards of project ethics to the best of our abilities. We have acknowledged all the sources of knowledge that we have used in this project.

Sankalp (12311001)

Madhur Suman (12311002)

Department of Computer Science and Engineering,

Indian Institute of Information Technology,

Sonepat – 131201, Haryana, India

---
---

## CERTIFICATE

This is to certify that Mr. Sankalp and Mr. Madhur Suman have worked on the project entitled "Deepfake Detection System" under my supervision and guidance.

The contents of the project, being submitted to the Department of Computer Science and Engineering, IIIT Sonepat, Haryana, for the award of the degree of B.Tech in Computer Science and Engineering, are original and carried out by the candidates themselves. This project has not been submitted in full or part for the award of any other degree or diploma to this or any other university.

Dr. Md. Arquam,

Supervisor

Department of Computer Science and Engineering,

Indian Institute of Information Technology,

Sonepat, Haryana

---
---

## ABSTRACT

**Project Title:** Deepfake Detection System

**Submitted by:**

1. Sankalp (12311001)
2. Madhur Suman (12311002)

**Degree:** B.Tech

**Project Supervisor:** Dr. Md. Arquam

**Month and Year of project submission:** April, 2026

**Department of Computer Science and Engineering,**

**Indian Institute of Information Technology, Sonepat, Haryana**

Our project presents a multimodal deepfake detection framework that uses self-supervised learning to spot synthetic media by looking for inconsistencies between the audio and visual streams of a video. The system combines a Vision Transformer (ViT-B/16) for extracting visual features from video frames with a four-layer Convolutional Neural Network for processing audio. These two streams are then fused together using a Multimodal Bottleneck Transformer (MBT). Unlike most existing methods that only look at the visual side, our framework analyses both video and audio together to catch subtle mismatches like lip sync errors, temporal glitches, and spliced audio.

The training pipeline has two phases. In the first phase, the model is pretrained on the VoxCeleb dataset without any labels using three learning objectives: cross-modal contrastive learning with a MoCo-style memory bank, audio-visual synchronization prediction, and masked modality reconstruction. In the second phase, the pretrained encoders are fine-tuned on the FakeAVCeleb benchmark for binary classification of real versus fake videos. The whole system is built on PyTorch and uses mixed-precision training for efficiency. Our experiments show an accuracy of 85.2%, precision of 86.42%, recall of 75.6%, and an F1 score of 80.1% on the FakeAVCeleb dataset.

---
---

## TABLE OF CONTENTS

| Chapter | Title | Page |
|---------|-------|------|
| | Acknowledgement | i |
| | Self Declaration | ii |
| | Certificate | iii |
| | Abstract | iv |
| | Table of Contents | v |
| | List of Abbreviations | vi |
| **Chapter 1** | **Introduction** | **1-8** |
| 1.1 | Introduction | 2 |
| 1.2 | Problem Outline | 3 |
| 1.3 | Project Objectives | 4 |
| 1.4 | Project Methodology | 5-6 |
| 1.5 | Scope of Project Work | 6-7 |
| 1.6 | Limitations | 7 |
| 1.7 | Organization of Project | 8 |
| 1.8 | Summary | 8 |
| **Chapter 2** | **Application and User Interface** | **9-12** |
| 2.1 | Technologies Employed | 10-11 |
| 2.2 | User Interface Description | 11-12 |
| **Chapter 3** | **System and Pipeline Design** | **13-16** |
| 3.1 | Introduction to the System Pipeline | 14 |
| 3.2 | Project System Description | 14-16 |
| **Chapter 4** | **Model Design and Logic** | **17-28** |
| 4.1 | Deep Learning Models Used | 18-19 |
| 4.2 | Model Loading | 19-20 |
| 4.3 | Prediction Logic | 20-21 |
| 4.4 | Data Processing | 21-23 |
| 4.5 | Training and Inference | 23-26 |
| 4.6 | Error Handling | 26-27 |
| 4.7 | Additional Logic: Data Augmentation | 27-28 |
| **Chapter 5** | **Result and Conclusion** | **29-32** |
| 5.1 | Results and Observations | 30-31 |
| 5.2 | Conclusion | 31 |
| 5.3 | Future Scope | 31-32 |
| | References | 33 |
| | Appendix | 34 |

---
---

## LIST OF ABBREVIATIONS

| Abbreviation | Full Form |
|-------------|-----------|
| AI | Artificial Intelligence |
| ML | Machine Learning |
| DL | Deep Learning |
| CNN | Convolutional Neural Network |
| ViT | Vision Transformer |
| MBT | Multimodal Bottleneck Transformer |
| GAN | Generative Adversarial Network |
| SSL | Self-Supervised Learning |
| MoCo | Momentum Contrast |
| AV | Audio-Visual |
| GPU | Graphics Processing Unit |
| CPU | Central Processing Unit |
| API | Application Programming Interface |
| BCE | Binary Cross-Entropy |
| AMP | Automatic Mixed Precision |
| MLP | Multi-Layer Perceptron |
| GELU | Gaussian Error Linear Unit |
| InfoNCE | Information Noise Contrastive Estimation |
| FPS | Frames Per Second |

---
---

# Chapter 1

# Introduction

---

## 1.1 INTRODUCTION

Over the past few years, generative AI has made it possible to create extremely convincing fake videos, audio clips, and images. These are commonly known as deepfakes. They are typically produced using deep learning techniques like Generative Adversarial Networks (GANs) and autoencoders. With these tools, someone can swap faces in a video, synthesize realistic speech, or manipulate facial expressions so well that the result is nearly impossible to tell apart from real footage. What started as a research experiment has now turned into a serious problem. Deepfakes are being used for political disinformation, financial scams, identity theft, and other forms of abuse that undermine public trust in digital media.

This makes it very important to develop systems that can reliably detect deepfakes. As generative models keep getting better, the visual artifacts they leave behind become harder and harder to spot by hand. Older forensics methods that looked for things like compression artifacts or pixel-level inconsistencies are no longer enough against modern generators that produce near-photorealistic results. There is a clear need for automated detection systems that can work at scale and keep up with the evolving techniques used to create deepfakes.

Our project tackles this problem with a multimodal approach. Instead of just looking at individual video frames for visual flaws, our system analyses both the video and the audio together. The key idea is that even though deepfake generators have gotten very good at producing realistic visuals, they often introduce subtle mismatches between the audio and visual streams. For example, the lip movements might not quite match the spoken words, or the timing of facial expressions might feel slightly off compared to the speech. By learning to pick up on these cross-modal inconsistencies, our system can detect deepfakes more reliably than methods that only look at the visual side.

The framework uses a self-supervised learning approach. First, the model learns audio-visual representations from a large collection of real videos without needing any labels. Then this pretrained knowledge is transferred to the actual deepfake classification task through supervised fine-tuning. On the architecture side, we use a Vision Transformer (ViT-B/16) for extracting features from video frames, a four-layer CNN for processing audio spectrograms, and a Multimodal Bottleneck Transformer for fusing the two streams together. This combination of self-supervised pretraining and multimodal fusion gives us a solid and practical approach to the deepfake detection problem.

## 1.2 PROBLEM OUTLINE

Even though there has been a lot of research on deepfake detection, existing solutions still face several major challenges.

The biggest issue is that most current systems rely entirely on visual analysis. They look at individual video frames and try to find spatial artifacts like blurring around face boundaries, unnatural skin textures, or inconsistent lighting. These methods can work well on the specific datasets they were trained on, but they tend to fail when they encounter deepfakes made with new or different techniques. As generators improve, the visual artifacts they produce become increasingly subtle, and frame-level analysis alone is not enough anymore.

Another major challenge is the lack of large, high-quality labelled datasets. Creating ground-truth labels for deepfake videos takes a lot of time and effort, and the datasets that do exist often have limited diversity in terms of the generation methods, subjects, and conditions they cover. This means that supervised models trained on these datasets can perform well in controlled settings but struggle when deployed in the real world where they encounter a much wider variety of deepfakes.

Most detection systems also completely ignore the audio track. In many real-world deepfakes, the audio is either synthesized separately or taken from a different source, which creates timing mismatches between what you see and what you hear. These audio-visual inconsistencies are a rich source of evidence for detection, but most existing methods do not take advantage of them at all. This leaves a significant gap that our project aims to fill.

Finally, there is the practical issue of computational cost. Processing video data is expensive, and many existing approaches need powerful hardware that is not available to most users. For deepfake detection to be widely adopted, systems need to balance accuracy with efficiency, and that remains an open challenge.

## 1.3 PROJECT OBJECTIVES

The main goal of this project is to build a multimodal deepfake detection system that addresses the limitations described above. The specific objectives are as follows.

- **Cross-Modal Detection.** We want to build a system that looks at both the visual and audio streams of a video together, not just one or the other. By detecting subtle inconsistencies between the two modalities, we can achieve more robust detection than visual-only methods. The system uses a Multimodal Bottleneck Transformer to fuse information from both streams efficiently.

- **Self-Supervised Representation Learning.** We want to address the data scarcity problem by implementing a self-supervised pretraining pipeline. The model first learns useful audio-visual features from a large set of unlabelled real videos (VoxCeleb) and then fine-tunes on the smaller labelled deepfake dataset (FakeAVCeleb). The pretraining uses three objectives: contrastive learning, synchronization prediction, and masked modality reconstruction.

- **Practical Performance.** We aim to achieve competitive accuracy, precision, recall, and F1 scores on standard benchmarks while keeping the system efficient enough to run on consumer GPUs like the NVIDIA T4 or better. Mixed-precision training and an optimized architecture help us meet this goal.

- **Modular Architecture.** We want the codebase to be clean and modular so that each component (visual encoder, audio encoder, fusion engine, classification head) can be swapped out or upgraded independently as better techniques become available.

## 1.4 PROJECT METHODOLOGY

The development of this system followed a structured process with several phases.

**1. Literature Review and Technology Selection.** We started by reviewing existing work on deepfake generation and detection, multimodal learning, and self-supervised pretraining. Based on this review, we chose PyTorch as our deep learning framework, the Vision Transformer (ViT-B/16) for visual feature extraction, and the Multimodal Bottleneck Transformer for cross-modal fusion.

**2. Data Acquisition and Preprocessing.** We used two datasets for the two training phases. The VoxCeleb dataset, which contains real celebrity interview videos, was used for self-supervised pretraining. The FakeAVCeleb dataset, which contains both real and fake videos, was used for supervised fine-tuning. We built a preprocessing pipeline that handles uniform frame sampling, center cropping, ImageNet normalization for video, and mel spectrogram extraction for audio.

**3. Architecture Design and Implementation.** We designed the model with three main components. The VideoViTEncoder processes individual frames through a pretrained ViT and adds learned temporal positional embeddings. The AudioTokenEncoder compresses mel spectrograms through a four-layer CNN and projects them into the transformer embedding space. The AudioVisualFusionEngine fuses the two modalities through shared bottleneck tokens across multiple transformer layers.

**4. Self-Supervised Pretraining.** The model was pretrained on VoxCeleb using three objectives: contrastive learning with a MoCo-style momentum encoder and memory bank, audio-visual synchronization prediction with cross-sample and temporal-shift negatives, and masked modality reconstruction with block-wise masking. This phase ran for 50 epochs with gradient accumulation and mixed-precision training.

**5. Supervised Fine-Tuning.** The pretrained encoders and fusion engine were loaded into a classification model with a three-layer MLP head for binary prediction. We used differential learning rates so that the pretrained parts train slowly while the new classification head trains faster.

**6. Evaluation and Analysis.** The fine-tuned model was evaluated on a held-out validation set using accuracy, precision, recall, and F1 score. We monitored training through TensorBoard and automated metric plots to check for convergence and catch issues like overfitting or embedding collapse.

## 1.5 SCOPE OF PROJECT WORK

The scope of this project covers the following areas.

- **Multimodal Feature Extraction.** We implemented a dual-stream architecture that extracts temporal features from both video frames and audio spectrograms. The visual encoder processes 16 uniformly sampled frames through a ViT-B/16 backbone and produces 768-dimensional tokens. The audio encoder processes 128-bin mel spectrograms through a four-layer CNN and also produces 768-dimensional tokens. Both streams use learned temporal positional embeddings.

- **Cross-Modal Fusion.** We implemented the Multimodal Bottleneck Transformer with four shared bottleneck tokens across four transformer layers. This lets the two modalities exchange information without the heavy computational cost of full cross-attention. The fusion engine outputs global classification tokens for each modality.

- **Self-Supervised Pretraining Pipeline.** We developed a complete pretraining pipeline with data augmentation (flipping, color jitter, blur for video and gain adjustment plus SpecAugment for audio), dynamic block-wise masking for both modalities, contrastive learning with a 2048-sample memory bank, and synchronization prediction with two types of negative samples.

- **Fine-Tuning and Evaluation Pipeline.** We implemented the supervised fine-tuning pipeline with an 80/20 train-validation split, binary cross-entropy loss, cosine annealing learning rate schedule, and comprehensive metric tracking with automated visualization.

## 1.6 LIMITATIONS

- **Hardware Requirements.** The self-supervised pretraining phase needs a GPU with at least 16 GB of VRAM. Processing 16 frames per video through the ViT-B/16 backbone uses a lot of memory, which limits the batch size on lower-end hardware. For efficient training, we recommend an RTX 3090 or RTX 4090 with 24 GB of VRAM.

- **Dataset Dependency.** The system's performance depends heavily on the training data. VoxCeleb mostly contains celebrity interviews, which may not represent all types of real-world video. FakeAVCeleb covers a specific set of deepfake techniques, and how well the model generalizes to completely new generation methods is still an open question.

- **Audio Availability.** Our multimodal approach assumes that both audio and video are available. If the audio track is missing or heavily compressed, the system would need to fall back to visual-only analysis, which would likely reduce detection performance. The current implementation does not have a fallback mechanism for this.

- **Real-Time Processing.** The system uses mixed-precision training for efficiency, but it is currently designed for batch processing rather than real-time inference on live video. Making it work in real time would require additional optimization like model quantization or distillation.

- **Adversarial Robustness.** We have not tested the system against adversarial attacks that are specifically designed to fool deepfake detectors. A determined attacker could potentially craft deepfakes that bypass the patterns our model has learned.

## 1.7 ORGANIZATION OF THE PROJECT

The project is organized into five chapters to keep things clear and structured.

- **Chapter 1 (Introduction)** gives an overview of the deepfake detection problem, the objectives, methodology, scope, and limitations.

- **Chapter 2 (Application and User Interface)** describes the technologies we used and explains how a user interacts with the system through the command line.

- **Chapter 3 (System and Pipeline Design)** presents the overall architecture and traces the data flow from raw video input to final prediction.

- **Chapter 4 (Model Design and Logic)** goes into the technical details of each model component, including the encoders, fusion engine, pretraining objectives, and fine-tuning head. Code excerpts are included to show key implementation details.

- **Chapter 5 (Result and Conclusion)** presents the experimental results, draws conclusions, and discusses directions for future work.

## 1.8 SUMMARY

This chapter introduced the context and motivation behind our deepfake detection project. We discussed the growing threat of deepfakes and why existing detection methods that only look at visuals are not enough. The problem outline covered the main challenges: poor generalization, data scarcity, neglected audio, and computational cost. We then laid out our objectives, which focus on building a multimodal self-supervised framework that analyses both audio and visual streams. The methodology section described our six-phase development process, and we wrapped up with the scope and limitations of the work. The next chapters will go into the technical details of the implementation.

---
---

# Chapter 2

# Application and User Interface

---

## 2.1 TECHNOLOGIES EMPLOYED

The deepfake detection system is built entirely in Python and uses a set of open-source libraries that together provide everything needed for deep learning research and experimentation.

- **Python 3.8+** is the primary programming language for the entire project. Its large ecosystem of scientific computing and machine learning libraries makes it the natural choice for this kind of work. All model definitions, training scripts, preprocessing utilities, and evaluation routines are written in Python.

- **PyTorch 2.0+** is the core deep learning framework. We use it for defining neural network architectures, computing gradients, and running training loops. Its dynamic computation graph and native GPU support through CUDA make it well suited for the kind of iterative development this project requires. All model components are implemented as PyTorch nn.Module subclasses.

- **TorchVision** provides the pretrained Vision Transformer (ViT-B/16) model with ImageNet weights that serves as the backbone of our visual encoder. It also provides the video I/O functionality (torchvision.io.read_video) for decoding MP4 files into tensors, along with standard image transforms like resizing, center cropping, and normalization.

- **TorchAudio** handles all audio processing. It provides the MelSpectrogram transform for converting raw waveforms into mel-frequency spectrograms, the AmplitudeToDB transform for logarithmic scaling, and the Resample transform for converting audio to the 16 kHz sample rate our audio encoder expects.

- **Matplotlib** is used for generating training visualization plots. This includes loss curves during pretraining and a full metrics dashboard (loss, accuracy, precision, recall, F1) during fine-tuning. The plots are saved as high-resolution PNG images.

- **TensorBoard** provides real-time training monitoring during the self-supervised pretraining phase. Loss values for each objective are logged every epoch so we can track convergence and spot problems like embedding collapse or diverging losses.

- **tqdm** gives us progress bars for training and evaluation loops. It shows the current loss and accuracy alongside estimated time remaining, which is very helpful during long training runs.

- **gdown and requests** handle automated dataset downloading. gdown downloads the FakeAVCeleb dataset from Google Drive, while requests handles the VoxCeleb download from its hosting server.

## 2.2 USER INTERFACE DESCRIPTION

This system is a research and training framework, not a consumer-facing application. The user interface is entirely command-line based. Users interact with the system by running Python scripts in sequence and adjusting configuration values in the code.

**Training Pipeline Interface**

The main workflow follows a six-step pipeline. Each step has its own dedicated Python script.

1. **Dataset Download (Pretraining).** The user runs `python dataset/pretrain_datset_download.py` to download the VoxCeleb dataset. The script handles the HTTP download, shows progress, and extracts the compressed archive.

2. **Dataset Preparation (Pretraining).** The user runs `python dataset/pretrain_prepare_dataset.py` to organize the downloaded videos into a flat directory structure. This script walks through the nested VoxCeleb folder hierarchy and copies video files with unique names.

3. **Video Preprocessing.** The user runs `python dataset/preprocess_dataset.py` to convert raw MP4 videos into preprocessed PyTorch tensor files (.pt). Each file contains the sampled and normalized video frames along with the corresponding log-mel spectrogram, ready to be loaded directly during training.

4. **Self-Supervised Pretraining.** The user runs `python ssl_pretrain.py` to start pretraining. The script shows a progress bar for each epoch with running loss values and prints a diagnostic summary after each epoch showing whether each loss component is going down, staying flat, or going up. Checkpoints and loss curve plots are saved to the outputs/ directory. Training can also be monitored in real time through TensorBoard.

5. **Dataset Preparation (Fine-Tuning).** The user runs `python dataset/prepare_finetune_dataset.py` to organize the FakeAVCeleb dataset into real/ and fake/ subdirectories with balanced class counts. This script accepts command-line arguments for the source directory, output directory, number of videos per class, and whether to use symbolic links.

6. **Fine-Tuning.** The user runs `python finetune_deepfake.py` to fine-tune the pretrained model for deepfake classification. The script prints training and validation metrics after each epoch, saves the best model checkpoint based on F1 score, and generates a metrics dashboard plot at the end.

**Inference Interface**

For inference on new videos, the user loads the saved model checkpoint and passes preprocessed video and audio tensors through the model. The output is a probability between 0 and 1. Values above 0.5 mean the model thinks the video is a deepfake.

**Output Artifacts**

The system produces several output files that help the user understand how training went. These include loss curve plots in the outputs/ directory, a training metrics dashboard in the metrics/ directory, TensorBoard log files for interactive exploration, and model checkpoint files for deployment or further experiments.

The interface is designed for clarity and reproducibility rather than visual polish. This is appropriate for a research project where the users are developers and researchers who are comfortable with the command line.

---
---

# Chapter 3

# System and Pipeline Design

---

## 3.1 INTRODUCTION TO THE SYSTEM PIPELINE

A well-designed pipeline is important for any deep learning project because it defines how data flows from raw input to final prediction and makes sure each processing stage is clearly separated and easy to test on its own. For deepfake detection, the pipeline has to deal with the added complexity of multimodal data, where video and audio streams need to be extracted, preprocessed, and analysed together.

Our system follows a modular pipeline where each stage transforms the data into a progressively more abstract representation. Raw video files come in and are split into their visual and audio parts. These go through independent preprocessing to produce standardized tensors. The tensors are then fed into modality-specific encoders that produce high-dimensional feature sequences. These sequences are deeply fused through a bottleneck transformer that allows cross-modal information exchange. Finally, the fused representations go through a classification head that outputs a binary real-or-fake prediction.

## 3.2 PROJECT SYSTEM DESCRIPTION

The pipeline has five stages. Each one is implemented as a separate module in the codebase. Here we trace the path of a single video through the entire system.

**Stage 1: Data Ingestion and Splitting**

The pipeline starts with raw MP4 video files organized into real/ and fake/ subdirectories. The DeepfakeDataset class scans these directories, collects file paths with their labels (0 for real, 1 for fake), and shuffles the samples. The dataset is then split into training and validation subsets using an 80/20 ratio. A DataLoader wraps each subset and handles batching, shuffling, multi-worker loading, and pinned memory for fast GPU transfer.

**Stage 2: Preprocessing**

Each video goes through two parallel preprocessing streams.

The visual stream (implemented in utils/visual_preprocessing.py) does four things in order. First, it uniformly samples 16 frames from the full video using linearly spaced indices so that the temporal coverage is consistent no matter how long the original video is. Second, it applies a center crop at 80% of the smaller spatial dimension to focus on the face area. Third, it resizes all frames to 224 by 224 pixels to match what the Vision Transformer expects. Fourth, it normalizes pixel values using ImageNet statistics.

The audio stream (implemented in utils/audio_preprocessing.py) converts the raw waveform into a log-mel spectrogram. The audio is resampled to 16 kHz if needed and converted to mono if it is stereo. A MelSpectrogram transform with 1024-point FFT, 512-sample hop length, and 128 mel bins produces a time-frequency representation. An AmplitudeToDB transform then converts this to a logarithmic decibel scale, which better captures the perceptual qualities of audio and the artifacts introduced by deepfake generation. The spectrogram is padded or trimmed to a fixed length of 94 time steps so that all samples in a batch have the same shape.

**Stage 3: Feature Encoding**

The preprocessed visual tensor (shape B, 16, 3, 224, 224) goes to the VideoViTEncoder. This module processes each frame independently through the ViT-B/16 backbone and extracts a 768-dimensional CLS token per frame. The resulting sequence (shape B, 16, 768) then gets learned temporal positional embeddings added to it so the model knows the order of the frames.

At the same time, the preprocessed audio tensor (shape B, 1, 128, 94) goes to the AudioTokenEncoder. This module compresses the frequency dimension through four convolutional layers while keeping the time dimension intact. The output is projected to 768 dimensions and also gets temporal positional embeddings, producing a sequence of shape (B, T_a, 768).

**Stage 4: Multimodal Fusion**

The visual and audio token sequences are passed to the AudioVisualFusionEngine, which implements the Multimodal Bottleneck Transformer. A global CLS token is prepended to each sequence, and four shared bottleneck tokens are initialized. The sequences then pass through four transformer layers. In each layer, each modality attends to its own tokens plus the shared bottleneck tokens, but not directly to the other modality. After each layer, the bottleneck tokens from both streams are averaged together. This creates a synchronized representation that carries information from both sides. The design allows deep cross-modal interaction without the heavy cost of full cross-attention.

**Stage 5: Classification**

The final visual and audio CLS tokens (each 768-dimensional) are concatenated into a 1536-dimensional vector. This goes through a three-layer MLP classifier with GELU activations and dropout (1536 to 512 to 256 to 1). The output is a single logit. During inference, a sigmoid function converts this to a probability, and a threshold of 0.5 gives the final prediction of real or fake.

The whole pipeline runs inside PyTorch's automatic mixed-precision context for efficiency, and gradient clipping with a max norm of 1.0 is applied during training for stability.

---
---

# Chapter 4

# Model Design and Logic

---

## 4.1 DEEP LEARNING MODELS USED

The system uses three neural network components that work together to process, fuse, and classify multimodal video data.

**Vision Transformer (ViT-B/16) as Visual Encoder**

The visual branch is built on the Vision Transformer Base model with 16 by 16 pixel patches, commonly called ViT-B/16. This model was proposed by Dosovitskiy et al. and has shown strong performance on image classification tasks. It works by dividing an input image into a grid of non-overlapping patches, embedding each patch into a high-dimensional token, prepending a learnable CLS token, adding positional embeddings, and then processing the whole sequence through a stack of transformer encoder layers with multi-head self-attention.

In our implementation, the ViT-B/16 starts with pretrained ImageNet weights. This gives it a strong foundation of visual features learned from 1.2 million images across 1000 categories. We replace the original classification head (which outputs 1000 ImageNet classes) with an identity layer so the model just gives us the raw 768-dimensional CLS token as a summary of each frame. The VideoViTEncoder class wraps this backbone and adds something important: learned temporal positional embeddings that are added to the sequence of per-frame CLS tokens. This lets the downstream fusion module understand the time order of the frames, which is essential for catching temporal inconsistencies in deepfakes.

**CNN-Based Audio Tokenizer as Audio Encoder**

The audio branch uses a custom four-layer CNN that transforms log-mel spectrograms into a sequence of tokens that can be fed into the transformer-based fusion module. The architecture progressively compresses the frequency dimension while keeping the time dimension intact. This effectively turns a 2D time-frequency representation into a 1D sequence of audio tokens.

The four convolutional layers use 3 by 3 kernels with GroupNorm and ReLU activations. The first two layers use 2 by 2 max pooling to reduce both frequency and time. The third and fourth layers use rectangular pooling with kernel size (2,1), which only compresses frequency while leaving time alone. A final adaptive average pooling layer squashes any remaining frequency bins down to one, giving us a pure temporal sequence. This is then projected from 256 dimensions up to 768 dimensions through a linear layer with layer normalization. Temporal positional embeddings are added on top, just like in the visual encoder.

**Multimodal Bottleneck Transformer (MBT) as Fusion Engine**

The fusion of visual and audio features is handled by the Multimodal Bottleneck Transformer, inspired by the work of Nagrani et al. on Attention Bottlenecks for Multimodal Fusion. The central idea is that the two modalities never attend directly to each other. Instead, they communicate through a small set of shared bottleneck tokens that act like a shared bulletin board.

The fusion engine has four MBTLayer modules. Each one contains two independent transformer encoder layers, one for visual and one for audio. At each layer, the shared bottleneck tokens are appended to both sequences before self-attention runs. After attention, the bottleneck tokens are pulled out from both streams and averaged together. This averaged bottleneck then goes into the next layer, allowing progressively deeper cross-modal interaction. The architecture uses 8 attention heads, 4 bottleneck tokens, and 0.1 dropout.

## 4.2 MODEL LOADING

Model loading happens in two different situations in our system. The first is loading pretrained ImageNet weights when the visual encoder is initialized. The second is loading the SSL-pretrained encoders when starting the fine-tuning phase.

**Pretrained ViT Loading**

The ViT backbone is loaded with ImageNet weights during initialization of the VideoViTEncoder:

```python
weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
self.vit = models.vit_b_16(weights=weights)
self.vit.heads = nn.Sequential(nn.Identity())
```

The pretrained weights give the spatial feature extraction layers a strong starting point. Replacing the classification head with an identity layer means the model outputs raw feature vectors instead of ImageNet class probabilities. The temporal positional embeddings are initialized from a truncated normal distribution with standard deviation 0.02 since these are new parameters that need to be learned from scratch.

**SSL Encoder Loading for Fine-Tuning**

When moving from pretraining to fine-tuning, the DeepfakeClassifier loads the pretrained encoder weights from a checkpoint file. The loading logic handles two possible formats:

```python
if os.path.exists(pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    if 'visual' in checkpoint:
        self.visual_encoder.load_state_dict(checkpoint['visual'], strict=False)
        self.audio_encoder.load_state_dict(checkpoint['audio'], strict=False)
        self.fusion_engine.load_state_dict(checkpoint['fusion'], strict=False)
    else:
        visual_state = {k.replace('visual_encoder.', ''): v 
                        for k, v in checkpoint.items() 
                        if k.startswith('visual_encoder.')}
        ...
```

The first format is the one saved by the get_encoder_state_dicts() method of the SSL model, which stores separate dictionaries for the visual encoder, audio encoder, and fusion engine. The second format handles full model state dictionaries where parameter names have module prefixes. Using strict=False allows partial loading in case there are minor differences between the pretraining and fine-tuning architectures. Setting map_location='cpu' makes sure checkpoints saved on GPU can still be loaded on a CPU machine.

## 4.3 PREDICTION LOGIC

The prediction logic follows a straightforward forward pass through the three components. The DeepfakeClassifier.forward method handles the whole process:

```python
def forward(self, visual: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
    v_tokens = self.visual_encoder(visual)
    a_tokens = self.audio_encoder(audio)
    fusion_out = self.fusion_engine(v_tokens, a_tokens)
    vis_cls = fusion_out["vis_cls"]
    aud_cls = fusion_out["aud_cls"]
    combined = torch.cat([vis_cls, aud_cls], dim=1)
    logits = self.classifier(combined)
    return logits
```

The method takes two inputs: a visual tensor of shape (B, 16, 3, 224, 224) with 16 preprocessed video frames and an audio tensor of shape (B, 1, 128, 94) with the log-mel spectrogram. The visual encoder runs each frame through the ViT and adds temporal embeddings, giving us 16 tokens of dimension 768. The audio encoder compresses the spectrogram through its CNN layers and projects the result to 768 dimensions.

The fusion engine prepends learnable CLS tokens to both sequences and processes them through four bottleneck transformer layers. The final CLS tokens, now enriched with cross-modal information from the shared bottleneck, are extracted as global summary vectors. These two 768-dimensional vectors are concatenated into a 1536-dimensional representation that captures the relationship between the visual and audio content.

The classifier MLP then maps this down through three layers (1536 to 512 to 256 to 1) with GELU activations and dropout. During training, the output logit goes directly into the binary cross-entropy with logits loss. During inference, we apply sigmoid to get a probability and threshold at 0.5. Anything above 0.5 is classified as a deepfake.

## 4.4 DATA PROCESSING

Data processing is handled by dedicated modules that turn raw multimedia files into the standardized tensor formats the model needs. The processing is consistent across both pretraining and fine-tuning so the model always sees data in the same format.

**Visual Preprocessing**

The VisualPreprocessor class implements a four-step pipeline for video frames. The first step is temporal sampling. It picks 16 frames evenly distributed across the full video by computing linearly spaced indices:

```python
def _sample_frames(self, frames: torch.Tensor) -> torch.Tensor:
    total_frames = frames.shape[0]
    if total_frames == self.num_frames:
        return frames
    indices = torch.linspace(0, total_frames - 1, self.num_frames).long()
    return frames[indices]
```

The second step is a center crop at 80% of the minimum spatial dimension. This focuses on the central part of the frame where the face is most likely to be. The third step resizes all frames to 224 by 224 pixels. The fourth step normalizes pixel values using ImageNet mean ([0.485, 0.456, 0.406]) and standard deviation ([0.229, 0.224, 0.225]).

**Audio Preprocessing**

The AudioPreprocessor class converts raw waveforms into log-mel spectrograms in two steps. The MelSpectrogram transform computes a spectrogram with 128 frequency bins using a 1024-point FFT and 512-sample hop length at 16 kHz. The AmplitudeToDB transform then converts this to a log decibel scale:

```python
self.mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate, n_fft=n_fft,
    hop_length=hop_length, n_mels=n_mels, normalized=True
)
self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
    stype='power', top_db=80
)
```

The log scaling is important for deepfake detection because it better represents how we perceive audio and it amplifies the subtle spectral artifacts that deepfake generators tend to introduce. The resulting spectrogram is padded or trimmed to 94 time steps so all samples in a batch have the same dimensions.

**Batch Preprocessing for SSL**

During self-supervised pretraining, the preprocess_dataset.py script preprocesses the entire VoxCeleb dataset offline. Each video goes through both the visual and audio pipelines, and the resulting tensors are saved as individual .pt files. This way each video only needs to be decoded once instead of at every epoch. The SSLDataset class then loads these preprocessed tensors directly, which significantly reduces the I/O bottleneck during training.

## 4.5 TRAINING AND INFERENCE

Training has two phases: self-supervised pretraining and supervised fine-tuning. Each phase has its own training loop, loss functions, and optimization strategy, but they share the same underlying model components.

**Phase 1: Self-Supervised Pretraining**

The pretraining phase (ssl_pretrain.py) trains the SSLPretrainModel on VoxCeleb using three learning objectives that do not need any labels.

*Cross-Modal Contrastive Learning.* This objective pulls matching audio-visual pairs together in a shared embedding space and pushes non-matching pairs apart. The visual and audio CLS tokens from the fusion engine are projected into a 256-dimensional normalized space through a two-layer MLP. The InfoNCE loss is computed using both in-batch negatives and a 2048-sample memory bank maintained by a momentum encoder with coefficient 0.999:

```python
z_visual = self.contrastive_head(vis_cls)
z_audio = self.contrastive_head(aud_cls)
loss_contrastive = self.contrastive_head.compute_loss(
    z_visual, z_audio, vis_cls, aud_cls
)
```

A variance regularization term is also added. It penalizes the model if the standard deviation of the embeddings drops below 0.1, which helps prevent embedding collapse.

*Audio-Visual Synchronization Prediction.* This objective teaches the model to tell the difference between temporally aligned and misaligned audio-visual pairs. During data loading, 50% of samples keep their original synchronized audio (label 1) and the other 50% get desynchronized audio (label 0). The desynchronization happens in two ways: 30% of negative samples use audio from a completely different video, and 70% use the same video's audio shifted by 15 to 40 frames. The concatenated CLS tokens go through a lightweight two-class classifier and the loss is standard cross-entropy.

*Masked Modality Reconstruction.* This objective makes the model reconstruct original tokens from partially masked inputs. Block-wise masking is applied to both modalities: 85% of visual tokens are masked using blocks of size 8, and 80% of audio tokens are masked using blocks of size 12. The masked positions are replaced with learnable mask tokens, and the fused output sequences go through modality-specific reconstruction heads. The loss is smooth L1 computed only at the masked positions.

The total loss combines all objectives with different weights: 0.5 for contrastive, 1.0 for synchronization, 2.0 for audio reconstruction, 3.0 for visual reconstruction, and 0.1 for variance regularization. Training runs for 50 epochs with AdamW optimizer at learning rate 1e-4, weight decay 1e-4, gradient accumulation over 4 steps (effective batch size of 32), mixed-precision training, and gradient clipping at max norm 1.0.

**Phase 2: Supervised Fine-Tuning**

The fine-tuning phase (finetune_deepfake.py) loads the pretrained encoders and fusion engine into the DeepfakeClassifier and trains for binary classification on FakeAVCeleb.

One important detail is the use of differential learning rates. The pretrained parts get a learning rate of 1e-5 (one tenth of the base rate) to preserve what they learned during pretraining. The new classification head gets the full rate of 1e-4 so it can learn the decision boundary quickly:

```python
optimizer = torch.optim.AdamW([
    {"params": pretrained_params, "lr": LEARNING_RATE * 0.1},
    {"params": classifier_params, "lr": LEARNING_RATE}
], weight_decay=1e-4)
```

Training runs for 20 epochs with batch size 4, binary cross-entropy with logits loss, cosine annealing learning rate schedule, and mixed-precision training. After each epoch the model is evaluated on the validation set, and the checkpoint with the best F1 score is saved.

**Inference**

For inference, the trained model is loaded from the checkpoint and set to eval mode. The input video is preprocessed through the same pipelines used during training, and the tensors are passed through the model without computing gradients:

```python
checkpoint = torch.load("outputs_finetune/best_deepfake_detector.pth")
model = DeepfakeClassifier()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    logits = model(video_tensor, audio_tensor)
    probability = torch.sigmoid(logits)
    is_fake = probability > 0.5
```

The sigmoid output represents the model's confidence that the video is a deepfake. Values closer to 1.0 mean higher confidence.

## 4.6 ERROR HANDLING

Error handling is implemented at several levels to make sure the system does not crash when it runs into bad or incomplete data. This matters a lot in a multimedia pipeline where video files can be corrupted, audio tracks can be missing, and tensor shapes might not always line up.

**Data Loading Errors**

The DeepfakeDataset class has try-except blocks in both the video and audio loading methods. If a video file cannot be decoded for any reason, the method catches the exception, prints a warning, and returns a zero tensor of the expected shape as a fallback:

```python
try:
    video, audio, info = torchvision.io.read_video(video_path, pts_unit='sec')
    ...
except Exception as e:
    print(f"Error loading video {video_path}: {e}")
    return torch.zeros(self.num_frames, 3, self.target_size, self.target_size)
```

The audio loading method does the same thing, returning a silence tensor if the audio cannot be extracted. Returning zero tensors is not ideal for training quality, but it prevents a single bad file from crashing the entire run.

**Dataset Validation**

Both the SSLDataset and DeepfakeDataset classes check for valid data during initialization. If no files are found, a RuntimeError is raised right away with a clear message:

```python
if len(self.samples) == 0:
    raise RuntimeError(f"No videos found in {root_dir}/real or {root_dir}/fake")
```

This stops the user from starting a training run that would fail immediately anyway.

**Model State Validation**

The visual encoder has an assertion that checks the input tensor has the right number of dimensions:

```python
assert frames.dim() == 5, f"Expected 5D input (B, T, C, H, W), got {frames.dim()}D"
```

This catches shape mismatches early in the forward pass before they cause confusing errors deeper in the model.

**Training Stability**

The training loops use several techniques to keep things numerically stable. Gradient clipping at max norm 1.0 prevents exploding gradients. Mixed-precision training through PyTorch's GradScaler handles potential underflow in float16 by dynamically scaling the loss. The variance regularization term in the contrastive loss prevents embedding collapse, which is a common failure mode in self-supervised learning where all representations collapse to a single point.

## 4.7 ADDITIONAL LOGIC: DATA AUGMENTATION

Data augmentation is important during self-supervised pretraining because it prevents the model from learning trivial shortcuts and pushes it to develop robust features. The augmentation strategy is in utils/augmentations.py and applies separate pipelines to video and audio.

**Video Augmentation**

The VideoAugmentor class applies three types of augmentations to video frames.

*Horizontal Flipping.* With 50% probability, all frames in a video are flipped horizontally. The flip is applied consistently to every frame so the model cannot use flip inconsistencies as a shortcut.

*Color Jitter.* With 80% probability, random adjustments are made to brightness (0.6 to 1.4), contrast (0.6 to 1.4), and saturation (0.6 to 1.4). This simulates natural variation in lighting and camera settings. The same parameters are used for all frames in a video to keep things temporally consistent.

*Gaussian Blur.* With a low 10% probability, a 3 by 3 Gaussian blur kernel is applied to all frames. This simulates compression artifacts and resolution loss that are common in real-world video.

**Audio Augmentation**

The AudioAugmentor class applies SpecAugment-style augmentations to the mel spectrogram.

*Gain Adjustment.* With 50% probability, the spectrogram values are scaled by a random factor between 0.7 and 1.4 to simulate volume variation.

*Time Masking.* Two random contiguous blocks of up to 10 time steps each are zeroed out. This forces the model to use the remaining temporal context instead of memorizing specific patterns.

*Frequency Masking.* Two random contiguous blocks of up to 15 frequency bins each are zeroed out. This encourages robust spectral representations that do not depend on any single frequency band.

These augmentations are only used during SSL pretraining. During fine-tuning and inference the data is processed in its original form.

---
---

# Chapter 5

# Result and Conclusion

---

## 5.1 RESULTS AND OBSERVATIONS

The system was evaluated through both the self-supervised pretraining phase and the supervised fine-tuning phase. The results show that the multimodal approach works well and that self-supervised pretraining provides a meaningful boost.

**Self-Supervised Pretraining Results**

The SSL pretraining was run over 15 epochs on VoxCeleb. We tracked the convergence of each loss component through TensorBoard and automated plots. The table below summarizes how the losses changed over the course of training.

| Metric | Initial Value | Final Value (15 Epochs) |
|--------|--------------|------------------------|
| Total Loss | 3.6 | 0.55 |
| Contrastive Loss | 4.5 | 0.4 |
| Sync Loss | 0.65 | 0.05 |
| Audio Reconstruction Loss | 0.4 | 0.2 |
| Visual Reconstruction Loss | 0.25 | 0.05 |

All loss components went down steadily, which tells us the model was learning meaningful cross-modal representations. The contrastive loss dropped from 4.5 to 0.4, meaning the model learned to align matching audio-visual pairs in the embedding space. The sync loss fell sharply from 0.65 to 0.05, showing that the model got very good at telling apart aligned and misaligned audio-visual pairs. Both reconstruction losses also decreased, confirming that the masked modality objective helped the model build rich token-level representations.

**Fine-Tuning Results**

After pretraining, the model was fine-tuned on FakeAVCeleb for binary deepfake classification. Here are the results on the validation set.

| Metric | Score |
|--------|-------|
| Accuracy | 85.2% |
| Precision | 86.42% |
| Recall | 75.6% |
| F1 Score | 80.1% |

The 85.2% accuracy means the system correctly classifies the majority of videos. The precision of 86.42% tells us that when the model says a video is fake, it is right about 86% of the time. This is important because false accusations need to be minimized in any practical deployment. The recall of 75.6% means the model catches about three quarters of all deepfakes, with the remaining quarter being missed. The F1 score of 80.1% gives a balanced view of precision and recall.

The gap between precision and recall suggests the model is somewhat conservative. It tends to classify borderline cases as real rather than fake. This is likely due to the class distribution in the training data and the 0.5 threshold. Adjusting the threshold could shift this balance depending on what the application needs.

**Key Observations**

The results support several of our design decisions. First, the multimodal approach that analyses audio and visual streams together works better than looking at either one alone. The fusion engine lets the model catch cross-modal inconsistencies that single-modality methods would miss. Second, self-supervised pretraining gives the model a meaningful head start. The pretrained encoders already know how to extract useful audio-visual features before they ever see a labelled deepfake. Third, the differential learning rate strategy during fine-tuning does a good job of preserving what was learned during pretraining while still letting the classification head adapt to the new task.

## 5.2 CONCLUSION

The deepfake detection system presented in this report shows that multimodal audio-visual analysis is an effective approach to identifying synthetic media. By combining a Vision Transformer for visual features, a CNN-based audio tokenizer, and a Multimodal Bottleneck Transformer for fusion, the system achieves 85.2% accuracy and 80.1% F1 score on the FakeAVCeleb benchmark.

The self-supervised pretraining strategy addresses one of the biggest challenges in deepfake detection, which is the lack of large labelled datasets. By learning audio-visual representations from unlabelled VoxCeleb videos through contrastive learning, synchronization prediction, and masked reconstruction, the model builds a strong foundation that transfers well to the classification task. This two-phase approach reduces the need for expensive labelled data and helps the model generalize beyond the specific deepfake techniques in the training set.

The modular architecture means individual components can be upgraded independently. The visual encoder could be swapped for a more powerful model, the audio encoder could be replaced with a pretrained audio transformer, and the fusion mechanism could be extended to handle additional modalities like optical flow or facial landmarks. This makes the framework a good starting point for continued work in this area.

Overall, this project shows that looking at audio and visual streams together through a bottleneck fusion mechanism is a promising direction for building deepfake detectors that are both robust and practical.

## 5.3 FUTURE SCOPE

There are several promising directions for extending this work.

- **Web Application.** The system currently runs from the command line. A natural next step would be to wrap the model in a web application using FastAPI or Flask so that users can upload videos and get detection results through a browser. This would make the system accessible to a much wider audience.

- **Real-Time Video Analysis.** Extending the system to handle live video streams would open up applications like real-time content moderation and video call authentication. This would require model optimization through techniques like quantization, distillation, or TensorRT.

- **Additional Modalities.** Future versions could add more signal sources like optical flow for motion inconsistencies, facial landmark trajectories for unnatural movements, or frequency-domain analysis for GAN-specific artifacts. The modular MBT architecture makes it straightforward to plug in new streams.

- **Adversarial Robustness.** As deepfake generators improve, detectors need to be hardened against adversarial attacks. Future work could use adversarial training where the detector is trained against a generator that tries to fool it, creating a feedback loop that improves both sides.

- **Cross-Dataset Generalization.** Testing the model across multiple datasets like FaceForensics++, Celeb-DF, and DFDC would give a better picture of how well it generalizes. Domain adaptation techniques could help improve transfer across datasets without needing labels from each one.

- **Explainability.** Adding attention visualization and saliency maps would help users understand why the model flagged a particular video. This is especially important for use cases in journalism and law enforcement where decisions need to be transparent.

- **Model Compression.** Applying pruning, quantization, and knowledge distillation would make the model small enough to run on smartphones and embedded devices. This would bring deepfake detection directly to end users without needing cloud infrastructure.

---
---

## REFERENCES

[1] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In *Proceedings of the International Conference on Learning Representations (ICLR)*.

[2] Nagrani, A., Yang, S., Arnab, A., Jansen, A., Schmid, C., & Sun, C. (2021). Attention Bottlenecks for Multimodal Fusion. In *Advances in Neural Information Processing Systems (NeurIPS)*, 34, 14200-14213.

[3] He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020). Momentum Contrast for Unsupervised Visual Representation Learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 9729-9738).

[4] Khalid, H., Tariq, S., Kim, M., & Woo, S. S. (2021). FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset. In *Proceedings of the NeurIPS Datasets and Benchmarks Track*.

[5] Nagrani, A., Chung, J. S., & Zisserman, A. (2017). VoxCeleb: A Large-Scale Speaker Identification Dataset. In *Proceedings of Interspeech* (pp. 2616-2620).

[6] Park, D. S., Chan, W., Zhang, Y., Chiu, C. C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. In *Proceedings of Interspeech* (pp. 2613-2617).

[7] Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. In *Proceedings of the International Conference on Learning Representations (ICLR)*.

[8] Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., & Wu, H. (2018). Mixed Precision Training. In *Proceedings of the International Conference on Learning Representations (ICLR)*.

[9] Touvron, H., Cord, M., Douze, M., Massa, F., Saber, A., & Jegou, H. (2021). Training Data-Efficient Image Transformers and Distillation Through Attention. In *Proceedings of the International Conference on Machine Learning (ICML)* (pp. 10347-10357).

[10] Zi, B., Chang, M., Chen, J., Ma, X., & Jiang, Y. G. (2020). WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection. In *Proceedings of the ACM International Conference on Multimedia* (pp. 2382-2390).

---
---

## APPENDIX

### Project Repository

**GitHub:** https://github.com/yourusername/DeepDetect2

### Project Structure

```
DeepDetect2/
├── Models/
│   ├── Visual_encoder.py        # ViT-based video encoder with temporal embeddings
│   ├── audio_encoder.py         # 4-layer CNN audio tokenizer
│   ├── Fusion_Engine.py         # Multimodal Bottleneck Transformer (MBT)
│   └── ssl_tasks.py             # SSL pretraining orchestrator with 3 objectives
│
├── utils/
│   ├── visual_preprocessing.py  # Frame sampling, cropping, normalization
│   ├── audio_preprocessing.py   # Mel spectrogram extraction
│   └── augmentations.py         # Video and audio augmentation strategies
│
├── dataset/
│   ├── pretrain_datset_download.py    # VoxCeleb dataset downloader
│   ├── pretrain_prepare_dataset.py    # VoxCeleb directory flattener
│   ├── preprocess_dataset.py          # Video-to-tensor offline conversion
│   ├── download_finetune_dataset.py   # FakeAVCeleb dataset downloader
│   └── prepare_finetune_dataset.py    # Real/fake directory organizer
│
├── ssl_pretrain.py              # Self-supervised pretraining script (Phase 1)
├── finetune_deepfake.py         # Supervised fine-tuning script (Phase 2)
│
├── outputs/                     # SSL pretrained model checkpoints and loss plots
├── outputs_finetune/            # Fine-tuned model checkpoints
├── metrics/                     # Training metrics dashboard plots
└── README.md                    # Project documentation
```

### Key Dependencies

```
Python >= 3.8
PyTorch >= 2.0
TorchVision
TorchAudio
tqdm
matplotlib
tensorboard
gdown
requests
```

### Hardware Used for Training

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 3090 (24 GB VRAM) |
| RAM | 32 GB DDR4 |
| Storage | 500 GB SSD |
| OS | Ubuntu 22.04 LTS |
