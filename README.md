# Deep Learning Project 1
For Deep Learning Class - Project 1  
Gene Eagle, John Kent and Matt Downing

## Table of Contents
1. [Introduction](#Introduction)
2. [Approach Used](#The-Approach-Used)
3. [CNN Architecture Utilized](#CNN-Architecture-Utilized)
4. [Our Experiments](#Our-Experiments)
5. [Acknowledgement](#Acknowledgement)
6. [How to Use this repo](#How-to-Use-this-repo)
7. [How to run in Colab](#How-to-run-this-in-Colab)

## Introduction  
### What is Continual Learning?  
Continual learning is a sub-topic of AI focused on techniques to enable a machine to learn adaptively when new inputs are presented over time. Traditional machine learning tasks have focused on making a machine learn by training it on a specific set of data inputs focusing on narrow task domains. If a new class or instance presents itself in the future, then the entire model needs to be completely re-trained. This is not practical in most real-world scenarios where an autonomous agent is acting in real time.

Enabling an agent to re-use and retain knowledge that it has previously learned without having to completely re-train the model from scratch is difficult. This is a hard problem to solve due to catastrophic failure. Catastrophic failure is when a model completely forgets prior learnings when trying to gradually update its memory, due to the difference between the data distributions of the batches.

## The Approach Used
Batch level Experience Replay with Review: 

Experience Replay:  
At a very high level Experience Replay stores previously encountered examples and revisits them when learning something new.

Typical implementations of Experience Replay store a subset of the samples from a past mini-batche in a buffer. When training the current mini-batch, it concatinates this current mini-batch with another mini-batch of samples retreived from the memory buffer. Then, a SGD update step is performed with this new combined batch.

However, the approach used here differs in two ways: 
- samples are concatinated at the batch level, not the mini-batch level
    - this reduces the number of retreival and update steps required
- a review step is added before the final testing to remind the model of past learnings

 In this exact case for every epoch a random batch is taken from memory with a certain replay size (this is important!, our experiments play with this replay size), concatenate it with the current batch, conduct SGD.  

Review part: after training all the batches a review batch is randomly taken from memory and SGD is conducted again to perform a final update to the weights.

Training Procedure Algorithm Overview:  

![image](https://github.com/mddown/Deep-Learning-proj1/blob/master/pics/psuedo_code.JPG)  

## CNN Architecture Utilized   
**DenseNet**  
[DenseNets](https://arxiv.org/pdf/1608.06993.pdf) (Dense Convolutional Network) are comprised of dense blocks and transitional layers between each block. Each unit in a dense block is connected to every other unit before it and after it. Between these dense blocks a Transitional layer exists which downsamples the features passing through it.  

![image](https://github.com/mddown/Deep-Learning-proj1/blob/master/pics/denseNet_arch.JPG)

This type of CNN architecture contains shorter connections between layers close to the input and close to the output. One important aspect is that DenseNet architectures are good at alleviating the vanashing gradient problem while also reducing the total number of parameters required compared to ResNets. This is accomplished by establishing direct connections from any layer to all subsequent layers - For each layer the feature maps of all preceding layers are used as inputs, and its own feature maps are used as inputs to all subsequent layers. This creates a high level of feature sharing between the layers.  

**DenseNet161_Freeze:**  
DenseNEt161_Freeze is based on the DenseNet161 model (pre-trained on ImageNet) but has the first 2 dense blocks frozen. By freezing the first 2 dense blocks the training time is decreased while also ensuring that the model can still extract features from the images.  

## ResNet18 vs DenseNet161_Freeze

The first experiment we ran was to compare the ResNet18 model to the DenseNet161_Freeze model. The results of this experiment led us to use the DenseNet161_Freeze model as our baseline model for future experiments as the DenseNet161_Freeze model was more accurate for all three scenarios.  

### Results below:  

As a note, since the dataset was distributed as part of a competition we, were limted to testing on the validation set only. We invetigated and observed that the validation set was approximately one twentieth the size of the full test set, and given that the competition concluded in early 2020, contacted the event organizers in an attempt to obtain labeled test results for analysis. Also, the original competition was based on a composite score using a weighted sum of five metrics - accuracy on the test set, average accuracy on the test set, total training and test runtime, memory usage, and disk usage. **We choose to only focus on the validation dataset accuracy metric.**  


**New Classes Scenario (NC):**  

| Architecture       | VAl. Accuracy |
|--------------------|----------|
| ResNet18           | 95%      |
| DenseNet161_Freeze | 99%      | 

<br/>

**New Instances Scenario (NI):** 

| Architecture       | Val. Accuracy |
|--------------------|----------|
| ResNet18           | 80%      |
| DenseNet161_Freeze | 94%      | 

<br/>

**New Instances and Classes (NIC):**  

| Architecture       | Val. Accuracy |
|--------------------|----------|
| ResNet18           | 89%      |
| DenseNet161_Freeze | 94%      |

<br/>

## Our Experiments
Detailed breakdown of the results can be found in the Jupyter Notebook.  
<br/>
Our experiments (using the DenseNet161_Freeze architecture) focused on playing with the number of replay examples that are randomly drawn from the memory. We wanted to see if increasing the replay size (concatinated with the current batch) would increase the models ability to not forget what it has learned previsouly.  

**We found that increasing the size of the replay samples did not have an effect on accurracy performance.**

Results of playing with the number of replay samples below:  

## Scenario 1 - Multi Task New Classes 
In this setting the 50 different classes are split into 9 different tasks: 10 classes in the first batch and 5 classes in the other 8. In this case the task label will be provided during training and test.  

The Multi Task New Classes scenario did not use the replay memory methodology. A new independent model is assigned to each batch. Inference outweighs transfer when sharing 1 model across all batches. So instead a new fresh model is assigned to each batch.

| Architecture       | Val. Accuracy |
|--------------------|----------|
| DenseNet161_Freeze | 99%      | 

<br/>

## Scenario 2 - New Instances  
In the New Instances scenario there are 8 training batches of the same 50 classes are encountered over time. Each training batch is composed of different images collected in different environmental conditions. No batch labels are provided. 

| Architecture       | Replay Samples | Val. Accuracy |
|--------------------|-------------|----------|
| DenseNet161_Freeze | 5,000        | 94%      |
| DenseNet161_Freeze | 7,500       | 88%      |
| DenseNet161_Freeze | 10,000           | 94%      |
| DenseNet161_Freeze | 12,500    | 90%      |
<br/>

## Scenario 3 - New Instances and Classes 
391 training batches containing 300 images of a single class. No task label will be provided and each batch may contain images of a class seen before as well as a completely new class.  

| Architecture       | Replay Samples | Val. Accuracy |
|--------------------|-------------|----------|
| DenseNet161_Freeze | 5,000        | 94%      |
| DenseNet161_Freeze | 7,500        | 95%      |
| DenseNet161_Freeze | 10,000          | 94%      |
| DenseNet161_Freeze | 12,500        | 94%      |

<br/>

### Acknowledgement

The starting code of this repository is from the official starting [repository](https://github.com/vlomonaco/cvpr_clvision_challenge).
And from Zheda Mai(University of Toronto), Hyunwoo Kim(LG Sciencepark), Jihwan Jeong (University of Toronto), Scott Sanner (University of Toronto, Vector Institute) which can be found [here](https://github.com/RaptorMai/CVPR20_CLVision_challenge).

## How to Use this repo
### Getting Started
Clone repo

Download the dataset and related utilities:
```bash
sh fetch_data_and_setup.sh
```
Setup the conda environment:
```bash
conda env create -f environment.yml
conda activate clvision-challenge
```

## How to run this in Colab:  
Clone/upload repo to Google Colab env. Run Jupyter Notebook - CS677-CL-project.ipynb.