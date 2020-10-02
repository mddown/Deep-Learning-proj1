# Deep-Learning-proj1
For Deep Learning Class - Project 1

# Inrtoduction: What is Continual Learning?
Continual learning is a sub-topic of AI focused on techniques to enable a machine to learn adaptively when new inputs are presented over time. Traditional machine learning tasks have focused on making a machine learn by training it on a specific set of data inputs focusing on narrow task domains. If a new class or instance presents itself in the future, then the entire model needs to be completely re-trained. This is not practical in most real-world scenarios where an autonomous agent is acting in real time.

Enabling an agent to re-use and retain knowledge that it has previously learned without having to completely re-train the model from scratch is difficult. This is a hard problem to solve due to catastrophic failure. Catastrophic failure is when a model completely forgets prior learnings when trying to gradually update its memory, due to the difference between the data distributions of the batches.

# The Apprach Used
Experience Replay: store previously encountered examples and revisit them when learning something new.

For every incoming batch (after the first batch) we need to retrieve a different batch from memory, combine it with the current batch and then update the memory with this new batch combination. 


# How to Use this repo:
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


### Reproduce the final results for all tracks

```
sh create_submission.sh
```

The parameters for the final submissions:

- `config/final/nc.yml`
- `config/final/ni.yml`
- `config/final/nic.yml`

The detailed explanation of these parameters can be found in `general_main.py`

### Acknowledgement

The starting code of this repository is from the official starting [repository](https://github.com/vlomonaco/cvpr_clvision_challenge).
And from Zheda Mai(University of Toronto), Hyunwoo Kim(LG Sciencepark), Jihwan Jeong (University of Toronto), Scott Sanner (University of Toronto, Vector Institute) which can be found [here](https://github.com/RaptorMai/CVPR20_CLVision_challenge).

### Project Structure
This repository is structured as follows:

- [`core50/`](core50): Root directory for the CORe50  benchmark, the main dataset of the challenge.
- [`utils/`](core): Directory containing a few utility methods.
- [`cl_ext_mem/`](cl_ext_mem): It will be generated after the repository setup (you need to store here eventual 
memory replay patterns and other data needed during training by your CL algorithm)
- [`submissions/`](submissions): It will be generated after the repository setup. It is where the submissions directory
will be created.
- [`fetch_data_and_setup.sh`](fetch_data_and_setup.sh): Basic bash script to download data and other utilities.
- [`create_submission.sh`](create_submission.sh): Basic bash script to run the baseline and create the zip submission
file.
- [`naive_baseline.py`](naive_baseline.py): Basic script to run a naive algorithm on the tree challenge categories. 
This script is based on PyTorch but you can use any framework you want. CORe50 utilities are framework independent.
- [`environment.yml`](environment.yml): Basic conda environment to run the baselines.
- [`Dockerfile`](Dockerfile), [`build_docker_image.sh`](build_docker_image.sh), [`create_submission_in_docker.sh`](create_submission_in_docker.sh): Essential Docker setup that can be used as a base for creating the final dockerized solution (see: [Dockerfile for Final Submission](#dockerfile-for-final-submission)).
- [`LICENSE`](LICENSE): Standard Creative Commons Attribution 4.0 International License.
- [`README.md`](README.md): This instructions file.

Your `submission.zip` file is ready to be submitted on the [Codalab platform](https://competitions.codalab.org/competitions/23317)!
