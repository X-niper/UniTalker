
# UniTalker: Scaling up Audio-Driven 3D Facial Animation through A Unified Model [ECCV2024]

## Useful Links

<div align="center">
    <a href="https://x-niper.github.io/projects/UniTalker/" class="button"><b>[Homepage]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://arxiv.org/abs/2408.00762" class="button"><b>[arXiv]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://www.youtube.com/watch?v=oUUh67ECzig" class="button"><b>[Video]</b></a> &nbsp;&nbsp;&nbsp;&nbsp;
</div>

![Alt text](https://github.com/X-niper/X-niper.github.io/blob/master/projects/UniTalker/assets/unitalker_architecture.png?raw=True "UniTalker Architecture")

> UniTalker generates realistic facial motion with different audio inputs, including clean and noisy voices in various languages, text-to-speech-generated audios, and even noisy songs accompanied by back- ground music. 
> UniTalker can output multiple annotations for both academic and industrial use.
> For datasets with new annotations, one can simply plug new heads into UniTalker and train it with existing datasets or solely with new ones, avoiding retopology.

## Installation
### Environment
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

## Inference

### Download checkpoints, PCA models and template resources

[UniTalker-B-[D0-D7]](https://drive.google.com/file/d/1PmF8I6lyo0_64-NgeN5qIQAX6Bg0yw44/view?usp=sharing): The base model in paper. Download it and place it in "./pretrained_models"

[UniTalker-L-[D0-D7]](https://drive.google.com/file/d/1sH2T7KLFNjUnTM-V1eRMM1Tytxd2sYAp/view?usp=sharing): The default model in paper. Please first try the base model  to run the pipeline through.

[unitalker_data_release_V1](https://drive.google.com/file/d/1Un7TB0Z5A1CG6bgeqKlhnSOECFN-C6KK/view?usp=sharing): The released datasets, PCA models, data-split json files and id-template numpy array. Download and unzip it in this repo.

<!-- [PCA models](https://drive.google.com/file/d/1e0sG2vvdrtAMgwD5njctifhX0ai4eu3g/view?usp=sharing): download the pca models and unzip it in "./unitalker_data_release" -->

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
├── unitalker_data_release_V1
│   ├── D0_BIWI
│   │   ├── id_template.npy
│   │   └── pca.npz
│   ├── D1_vocaset
│   │   ├── id_template.npy
│   │   └── pca.npz
│   ├── D2_meshtalk
│   │   ├── id_template.npy
│   │   └── pca.npz
│   ├── D3D4_3DETF
│   │   ├── D3_HDTF
│   │   └── D4_RAVDESS
│   ├── D5_unitalker_faceforensics++
│   │   ├── id_template.npy
│   │   ├── test
│   │   ├── test.json
│   │   ├── train
│   │   ├── train.json
│   │   ├── val
│   │   └── val.json
│   ├── D6_unitalker_Chinese_speech
│   │   ├── id_template.npy
│   │   ├── test
│   │   ├── test.json
│   │   ├── train
│   │   ├── train.json
│   │   ├── val
│   │   └── val.json
│   └── D7_unitalker_song
│       ├── id_template.npy
│       ├── test
│       ├── test.json
│       ├── train
│       ├── train.json
│       ├── val
│       └── val.json
```

### Demo

```bash
  python -m main.demo --config config/unitalker.yaml test_out_path ./test_results/demo.npz
  python -m main.render ./test_results/demo.npz ./test_audios ./test_results/
```

## Train

### Download Data
[unitalker_data_release_V1](https://drive.google.com/file/d/1qRBPsTdOWp72ty04oD1Q_ivtwMjrACLH/view?usp=sharing) contains D5, D6 and D7. The datasets have been processed and grouped into train, validation and test. Please use these three datasets to try the training step.
If you want to train the model on the D0-D7, you need to download the datasets follow these links: 
[D0: BIWI](https://github.com/Doubiiu/CodeTalker/blob/main/BIWI/README.md).
[D1: VOCASET](https://voca.is.tue.mpg.de/).
[D2: meshtalk](https://github.com/facebookresearch/meshtalk?tab=readme-ov-file).
[D4,D5: 3DETF](https://github.com/psyai-net/EmoTalk_release).

### Modify Config and Train
Please modify "dataset" in "config/unitalker.yaml" according to the datasets you have prepared. 

```bash
python -m main.train --config config/unitalker.yaml 
```