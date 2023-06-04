<div align="center">
<h1> Variational Inference with adversarial learning for end-to-end Singing Voice Conversion based on VITS </h1>
    
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/maxmax20160403/sovits5.0)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PY1E4bDAeHbAD4r99D_oYXB46fG8nIA5?usp=sharing)
<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/PlayVoice/so-vits-svc-5.0">
<img alt="GitHub forks" src="https://img.shields.io/github/forks/PlayVoice/so-vits-svc-5.0">
<img alt="GitHub issues" src="https://img.shields.io/github/issues/PlayVoice/so-vits-svc-5.0">
<img alt="GitHub" src="https://img.shields.io/github/license/PlayVoice/so-vits-svc-5.0">
 
</div>

- 本项目的目标群体是：深度学习初学者，具备Python和PyTorch的基本操作是使用本项目的前置条件；
- 本项目旨在帮助深度学习初学者，摆脱枯燥的纯理论学习，通过与实践结合，熟练掌握深度学习基本知识；
- 本项目不支持实时变声；（支持需要换掉whisper）
- 本项目不会开发用于其他用途的一键包。（不会指没学会）

![sovits_framework](https://github.com/PlayVoice/so-vits-svc-5.0/assets/16432329/402cf58d-6d03-4d0b-9d6a-94f079898672)

- 【低 配置】6G显存可训练(HiFiGAN分支)

- 【无 泄漏】支持多发音人

- 【捏 音色】创造独有发音人

- 【带 伴奏】也能进行转换，轻度伴奏

- 【用 Excel】进行原始调教，纯手工

## 本分支，只是48K参数实例，并未正在训练过~~~

| Feature | From | Status | Function | Remarks |
| --- | --- | --- | --- | --- |
| whisper | OpenAI | ✅ | 强大的抗噪能力 | 参数修改 |
| bigvgan  | NVIDA | ✅ | 抗锯齿与蛇形激活 | GPU占用略多，主分支删除；新bigvgan分支训练，共振峰更清晰，提升音质明显 |
| natural speech | Microsoft | ✅ | 减少发音错误 | - |
| neural source-filter | NII | ✅ | 解决断音问题 | 参数优化 |
| speaker encoder | Google | ✅ | 音色编码与聚类 | - |
| GRL for speaker | Ubisoft |✅ | 防止编码器泄漏音色 | 原理类似判别器的对抗训练 |
| one shot vits |  Samsung | ✅ | VITS 一句话克隆 | - |
| SCLN |  Microsoft | ✅ | 改善克隆 | - |
| PPG perturbation | 本项目 | ✅ | 提升抗噪性和去音色 | - |

## 数据集准备

必要的前处理：
- 1 伴奏分离
- 2 频率提升
- 3 音质提升
- 4 剪切音频，whisper要求为小于30秒�

然后按下面文件结构，将数据集放入dataset_raw目录
```shell
dataset_raw
├───speaker0
│   ├───000001.wav
│   ├───...
│   └───000xxx.wav
└───speaker1
    ├───000001.wav
    ├───...
    └───000xxx.wav
```

## 安装依赖

- 1 软件依赖
  
  > apt update && sudo apt install ffmpeg
  
  > pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

- 2 下载音色编码器: [Speaker-Encoder by @mueller91](https://drive.google.com/drive/folders/15oeBYf6Qn1edONkVLXe82MzdIi3O_9m3), 把 `best_model.pth.tar`  放到目录 `speaker_pretrain/`

- 3 下载whisper模型 [multiple language medium model](https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt), 确定下载的是`medium.pt`，把它放到文件夹 `whisper_pretrain/`

## 数据预处理
- 1， 设置工作目录:heartpulse::heartpulse::heartpulse:不设置后面会报错

    > export PYTHONPATH=$PWD

- 2， 重采样

    生成采样率16000Hz音频, 存储路径为：./data_svc/waves-16k

    > python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-16k -s 16000

    生成采样率48000Hz音频, 存储路径为：./data_svc/waves-48k

    > python prepare/preprocess_a.py -w ./dataset_raw -o ./data_svc/waves-48k -s 48000

- 3， 使用16K音频，提取音高：注意f0_ceil=900，需要根据您数据的最高音进行修改
    > python prepare/preprocess_f0.py -w data_svc/waves-16k/ -p data_svc/pitch

- 4， 使用16k音频，提取内容编码
    > python prepare/preprocess_ppg.py -w data_svc/waves-16k/ -p data_svc/whisper

- 5， 使用16k音频，提取音色编码；应该将speaker改为timbre，才准确
    > python prepare/preprocess_speaker.py data_svc/waves-16k/ data_svc/speaker

- 6， 提取音色编码均值，用于推理；也可以在生成训练索引中，替换单个音频音色，作为发音人统一音色用于训练
    > python prepare/preprocess_speaker_ave.py data_svc/speaker/ data_svc/singer

- 7， 使用48k音频，提取线性谱
    > python prepare/preprocess_spec.py -w data_svc/waves-48k/ -s data_svc/specs

- 8， 使用48k音频，生成训练索引
    > python prepare/preprocess_train.py

- 9， 训练文件调试
    > python prepare/preprocess_zzz.py

```shell
data_svc/
└── waves-16k
│    └── speaker0
│    │      ├── 000001.wav
│    │      └── 000xxx.wav
│    └── speaker1
│           ├── 000001.wav
│           └── 000xxx.wav
└── waves-32k
│    └── speaker0
│    │      ├── 000001.wav
│    │      └── 000xxx.wav
│    └── speaker1
│           ├── 000001.wav
│           └── 000xxx.wav
└── pitch
│    └── speaker0
│    │      ├── 000001.pit.npy
│    │      └── 000xxx.pit.npy
│    └── speaker1
│           ├── 000001.pit.npy
│           └── 000xxx.pit.npy
└── whisper
│    └── speaker0
│    │      ├── 000001.ppg.npy
│    │      └── 000xxx.ppg.npy
│    └── speaker1
│           ├── 000001.ppg.npy
│           └── 000xxx.ppg.npy
└── speaker
│    └── speaker0
│    │      ├── 000001.spk.npy
│    │      └── 000xxx.spk.npy
│    └── speaker1
│           ├── 000001.spk.npy
│           └── 000xxx.spk.npy
└── singer
    ├── speaker0.spk.npy
    └── speaker1.spk.npy
```

## 训练
- 1， 设置工作目录:heartpulse::heartpulse::heartpulse:不设置后面会报错

    > export PYTHONPATH=$PWD

- 2， 启动训练

    > python svc_trainer.py -c configs/base.yaml -n sovits5.0

- 3， 恢复训练

    > python svc_trainer.py -c configs/base.yaml -n sovits5.0 -p chkpt/sovits5.0/***.pth

- 4， 查看日志

    > tensorboard --logdir logs/


## 推理

- 1， 设置工作目录:heartpulse::heartpulse::heartpulse:不设置后面会报错

    > export PYTHONPATH=$PWD

- 2， 导出推理模型：文本编码器，Flow网络，Decoder网络；判别器和后验编码器只在训练中使用

    > python svc_export.py --config configs/base.yaml --checkpoint_path chkpt/sovits5.0/***.pt

- 3， 使用whisper提取内容编码，没有采用一键推理，为了降低显存占用

    > python whisper/inference.py -w test.wav -p test.ppg.npy

    生成test.ppg.npy；如果下一步没有指定ppg文件，则调用程序自动生成

- 4， 提取csv文本格式F0参数，Excel打开csv文件，对照Audition或者SonicVisualiser手动修改错误的F0

    > python pitch/inference.py -w test.wav -p test.csv

![sonic visualiser](https://user-images.githubusercontent.com/16432329/237011482-51f3a45e-72c6-4d4a-b1df-f561d1df7132.png)

- 5，指定参数，推理

    > python svc_inference.py --config configs/base.yaml --model sovits5.0.pth --spk ./configs/singers/singer0001.npy --wave test.wav --ppg test.ppg.npy --pit test.csv

    当指定--ppg后，多次推理同一个音频时，可以避免重复提取音频内容编码；没有指定，也会自动提取；

    当指定--pit后，可以加载手工调教的F0参数；没有指定，也会自动提取；

    生成文件在当前目录svc_out.wav；

    | args |--config | --model | --spk | --wave | --ppg | --pit | --shift |
    | ---  | --- | --- | --- | --- | --- | --- | --- |
    | name | 配置文件 | 模型文件 | 音色文件 | 音频文件 | 音频内容 | 音高内容 | 升降调 |

## 捏音色
纯属巧合的取名：average -> ave -> eva，夏娃代表者孕育和繁衍

> python svc_eva.py

```python
eva_conf = {
    './configs/singers/singer0022.npy': 0,
    './configs/singers/singer0030.npy': 0,
    './configs/singers/singer0047.npy': 0.5,
    './configs/singers/singer0051.npy': 0.5,
}
```

生成的音色文件为：eva.spk.npy

Flow和Decoder均需要输入音色，您甚至可以给两个模块输入不同的音色参数，捏出更独特的音色。

## 数据集

| Name | URL |
| --- | --- |
|KiSing         |http://shijt.site/index.php/2021/05/16/kising-the-first-open-source-mandarin-singing-voice-synthesis-corpus/|
|PopCS          |https://github.com/MoonInTheRiver/DiffSinger/blob/master/resources/apply_form.md|
|opencpop       |https://wenet.org.cn/opencpop/download/|
|Multi-Singer   |https://github.com/Multi-Singer/Multi-Singer.github.io|
|M4Singer       |https://github.com/M4Singer/M4Singer/blob/master/apply_form.md|
|CSD            |https://zenodo.org/record/4785016#.YxqrTbaOMU4|
|KSS            |https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset|
|JVS MuSic      |https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music|
|PJS            |https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus|
|JUST Song      |https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song|
|MUSDB18        |https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems|
|DSD100         |https://sigsep.github.io/datasets/dsd100.html|
|Aishell-3      |http://www.aishelltech.com/aishell_3|
|VCTK           |https://datashare.ed.ac.uk/handle/10283/2651|

## 代码来源和参考文献

https://github.com/facebookresearch/speech-resynthesis [paper](https://arxiv.org/abs/2104.00355)

https://github.com/jaywalnut310/vits [paper](https://arxiv.org/abs/2106.06103)

https://github.com/openai/whisper/ [paper](https://arxiv.org/abs/2212.04356)

https://github.com/NVIDIA/BigVGAN [paper](https://arxiv.org/abs/2206.04658)

https://github.com/mindslab-ai/univnet [paper](https://arxiv.org/abs/2106.07889)

https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/tree/master/project/01-nsf

https://github.com/brentspell/hifi-gan-bwe

https://github.com/mozilla/TTS

https://github.com/OlaWod/FreeVC [paper](https://arxiv.org/abs/2210.15418)

[SNAC : Speaker-normalized Affine Coupling Layer in Flow-based Architecture for Zero-Shot Multi-Speaker Text-to-Speech](https://github.com/hcy71o/SNAC)

[Adapter-Based Extension of Multi-Speaker Text-to-Speech Model for New Speakers](https://arxiv.org/abs/2211.00585)

[AdaSpeech: Adaptive Text to Speech for Custom Voice](https://arxiv.org/pdf/2103.00993.pdf)

[Cross-Speaker Prosody Transfer on Any Text for Expressive Speech Synthesis](https://github.com/ubisoft/ubisoft-laforge-daft-exprt)

[Learn to Sing by Listening: Building Controllable Virtual Singer by Unsupervised Learning from Voice Recordings](https://arxiv.org/abs/2305.05401)

[Adversarial Speaker Disentanglement Using Unannotated External Data for Self-supervised Representation Based Voice Conversion](https://arxiv.org/pdf/2305.09167.pdf)

[Speaker normalization (GRL) for self-supervised speech emotion recognition](https://arxiv.org/abs/2202.01252)

## 基于数据扰动防止音色泄露的方法

https://github.com/auspicious3000/contentvec/blob/main/contentvec/data/audio/audio_utils_1.py

https://github.com/revsic/torch-nansy/blob/main/utils/augment/praat.py

https://github.com/revsic/torch-nansy/blob/main/utils/augment/peq.py

https://github.com/biggytruck/SpeechSplit2/blob/main/utils.py

https://github.com/OlaWod/FreeVC/blob/main/preprocess_sr.py

## 贡献者

<a href="https://github.com/PlayVoice/so-vits-svc/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PlayVoice/so-vits-svc" />
</a>
