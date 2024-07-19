
# UniTalker: Scaling up Audio-Driven 3D Facial Animation through A Unified Model [ECCV2024]


# Installation
## Environment
- Linux
- Python 3.10
- Pytorch 2.2.0
- CUDA 12.1
- transformers 4.39.3
- Pytorch3d 0.7.7 (Optional: just for rendering the results)

```bash
  conda create -n unitalker python==3.10
  conda activate unitalker
  conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
  pip install transformers librosa tensorboardX
```

# Inference

## Download checkpoints, PCA models and template resources

[UniTalker-B-[D0-D7]](https://drive.google.com/file/d/1PmF8I6lyo0_64-NgeN5qIQAX6Bg0yw44/view?usp=sharing): The base model in paper. Download it and place it in "./pretrained_models"

[UniTalker-L-[D0-D7]](https://drive.google.com/file/d/1sH2T7KLFNjUnTM-V1eRMM1Tytxd2sYAp/view?usp=sharing): The default model in paper. Please first try the base model  to run the pipeline through.

[PCA models](https://drive.google.com/file/d/1e0sG2vvdrtAMgwD5njctifhX0ai4eu3g/view?usp=sharing): download the pca models and unzip it in "./unitalker_data_release"

use "git lfs pull" to get "./resources.zip" and "./test_audios.zip" and unzip it in this repo

Finally, these files should be organized as follows:

```text
├── pretrained_models
│   ├── UniTalker-B-D0-D7.pt
│   ├── UniTalker-L-D0-D7.pt
├── resources
│   ├── binary_resources
│   │   ├── 02_flame_mouth_idx.npy
│   │   ├── ...
│   │   └── vocaset_FDD_wo_eyes.npy
│   └── obj_template
│       ├── 3DETF_blendshape_weight.obj
│       ├── ...
│       └── meshtalk_6172_vertices.obj
├── test_audios
│   ├── angry1.wav
│   ├── ...
│   └── ted2.wav
├── unitalker_data_release 
│   ├── BIWI
│   │   ├── pca.npz
│   ├── vocaset
│   │   ├── pca.npz
│   ├── meshtalk
│   │   ├── pca.npz
│   ├── ...
│   ├── D5_unitalker_faceforensics++
│   ├── D6_unitalker_Chinese_speech
│   ├── D7_unitalker_song
```

## Demo

```bash
  python -m main.demo --config config/unitalker.yaml test_out_path ./test_results/demo.npz
  python -m main.render ./test_results/demo.npz ./test_audios ./test_results/
```

# Train

## Download Data
[unitalker data](https://drive.google.com/file/d/1qRBPsTdOWp72ty04oD1Q_ivtwMjrACLH/view?usp=sharing).
You can train the model on D5,D6,D7 now, the datasets have been processed and grouped into train, validation and test. Please use these three datasets to try the training step.
If you want to train the model on the D0-D7, you need to download the datasets follow these links: 
[D0: BIWI](https://github.com/Doubiiu/CodeTalker/blob/main/BIWI/README.md).
[D1: VOCASET](https://voca.is.tue.mpg.de/).
[D2: meshtalk](https://github.com/facebookresearch/meshtalk?tab=readme-ov-file).
[D4,D5: 3DETF](https://github.com/psyai-net/EmoTalk_release).

## Train

```bash
python -m main.train --config config/unitalker.yaml 
```