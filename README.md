# EM-POSE: 3D Human Pose Estimation from Sparse Electromagnetic Trackers

This repository contains the code to [our paper](https//ait.ethz.ch/projects/2021/em-pose/) published at ICCV 2021. For questions feel free to open an issue or send an e-mail to [manuel.kaufmann@inf.ethz.ch](mailto:manuel.kaufmann@inf.ethz.ch).

## Installation

### Code
This code was tested on Windows 10 with Python 3.7, PyTorch 1.6 and CUDA 10.1. To manage your environment Anaconda or Miniconda is recommended.

```commandline
git clone https://github.com/facebookresearch/em-pose.git
cd em-pose
conda create -n empose python=3.7
conda activate empose
pip install -r requirements.txt
```

To run the code you need to download some additional data and define a few environment variables as outlined in the following.

### SMPL Model
This code uses the neutral SMPL-H model which is also used by AMASS. To download the model head over to the official [MANO website](https://mano.is.tue.mpg.de/) and download the `Extended SMPL+H model` on the download page. Copy the contents of this model into a folder of your choice and set the environment variable `$SMPL_MODELS`. The code expects the neutral SMPL model to be located under `$SMPL_MODELS/smplh_amass/neutral/model.npz`.

### EM-POSE Dataset
You can download our dataset from [here](https://dataset.ait.ethz.ch/downloads/kL298kEiqA/data.zip) (roughly 100 MB). Unzip the content into a directory of your choice and set the environment variable `$EM_DATA_REAL` to this directory.

The expected directory structure is:

```commandline
$EM_DATA_REAL
  |- 0402_arms_fast_M_clean.npz
  |- 0402_arms_M_clean.npz
  |- ...
  |- 0402_offsets.npz
  |- 0715_walking_M_clean.npz
```

The initial 4 digits identify the participant. There are 5 participants with IDs `0402`, `0526`, `0612`, `0714`, `0715`. Subject `0715` is the hold out subject.

### Pre-Trained Models
You can download the pre-trained models from [here](https://dataset.ait.ethz.ch/downloads/kL298kEiqA/models.zip) (roughly 500 MB). Unzip the content into a directory of your choice and set the environment variable `$EM_EXPERIMENTS` to this directory.

### AMASS (Optional)
You do not need to download AMASS for the evaluation code to work. You only need AMASS if you want to train a model from scratch. To download AMASS please visit [their official website](https://amass.is.tue.mpg.de/) and follow the instructions in the training section below to preprocess the data.

## Evaluation


## Training


## Visualization


## License
EM-POSE (c) by Facebook, ETH Zurich, Manuel Kaufmann

EM-POSE is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

You should have received a copy of the license along with this work. If not, see <http://creativecommons.org/licenses/by-nc-sa/4.0/>.

## Citation
If you use code or data from this repository please consider citing:

```commandline
@inProceedings{kaufmann2021empose,
  title={EM-POSE: 3D Human Pose Estimation from Sparse Electromagnetic Trackers},
  author={Kaufmann, Manuel and Zhao, Yi and Tang, Chengcheng and Tao, Lingling and Twigg, Christopher and Song, Jie and Wang, Robert and Hilliges, Otmar},
  booktitle={The IEEE International Conference on Computer Vision (ICCV)},
  month={Oct},
  year={2021}
}
```