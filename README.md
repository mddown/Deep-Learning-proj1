# Deep-Learning-proj1
For Deep Learning Class - Project 1

Note: Some of this readme and code has been taken from the official starting repository for the **CVPR 2020 CLVision 
challenge:** https://github.com/vlomonaco/cvpr_clvision_challenge

# CVPR 2020 CLVision Challenge

This is the official starting repository for the **CVPR 2020 CLVision 
challenge**. Here we provide:

- Two script to setup the environment and generate of the zip submission file.
- A complete working example to: 1) load the data and setting up the continual
learning protocols; 2) collect all the metadata during training 3) evaluate the trained model on the valid and test sets. 
- Starting Dockerfile to simplify the final submission at the end of the first phase.

You just have to write your own Continual Learning strategy (even with just a couple lines of code!) and you
are ready to partecipate.

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


### Getting Started

Download dataset and related utilities:
```bash
sh fetch_data_and_setup.sh
```
Setup the conda environment:
```bash
conda env create -f environment.yml
conda activate clvision-challenge
```
Make your first submission:
```bash
sh create_submission.sh
```

Your `submission.zip` file is ready to be submitted on the [Codalab platform](https://competitions.codalab.org/competitions/23317)! 

### Create your own CL algorithm

You can start by taking a look at the `naive_baseline.py` script. It has been already prepared for you to load the
data based on the challenge category and create the submission file. 

The simplest usage is as follow:
```bash
python naive_baseline.py --scenario="ni" --sub_dir="ni"
```

You can now customize the code in the main batches/tasks loop:

```python
   for i, train_batch in enumerate(dataset):
        train_x, train_y, t = train_batch

        print("----------- batch {0} -------------".format(i))

        # TODO: CL magic here
        # Remember to add all the metadata requested for the metrics as shown in the sample script.
``

