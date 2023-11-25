<div align="center">

# System Report for CCL23-Eval Task 7: THU KELab (sz) - Exploring Data Augmentation and Denoising for Chinese Grammatical Error Correction

</div>

该仓库为 CCL2023-CLTC Track 1（多维度汉语学习者文本纠错比赛）队伍 THU KELab (sz) 的解决方案。如果您认为我们的工作对您的工作有帮助，请引用我们的论文：

[System Report for CCL23-Eval Task 7: THU KELab (sz) - Exploring Data Augmentation and Denoising for Chinese Grammatical Error Correction](https://aclanthology.org/2023.ccl-3.29.pdf)

```
@inproceedings{ye-etal-2023-system,
    title = "System Report for {CCL}23-Eval Task 7: {THU} {KEL}ab (sz) - Exploring Data Augmentation and Denoising for {C}hinese Grammatical Error Correction",
    author = "Ye, Jingheng  and
      Li, Yinghui  and
      Zheng, Haitao",
    booktitle = "Proceedings of the 22nd Chinese National Conference on Computational Linguistics (Volume 3: Evaluations)",
    month = aug,
    year = "2023",
    address = "Harbin, China",
    publisher = "Chinese Information Processing Society of China",
    url = "https://aclanthology.org/2023.ccl-3.29",
    pages = "262--270",
}
```

## 最新消息

- [2023.11.25] 我们的最新工作 MixEdit 被 EMNLP-2023 会议录用，[该仓库](https://github.com/THUKElab/MixEdit)开源了多种面向语法纠错的数据增强策略、数据集、模型权重。欢迎大家试用~
- [2023.11.25] 我们的最新工作 CLEME 被 EMNLP-2023 会议录用，[该工作](https://github.com/THUKElab/CLEME)提出了一种全新的语法纠错自动评估度量 CLEME，支持多种评估功能和可视化。欢迎大家试用~

## 1 简介

汉语学习者文本 (Chinses Learner Text) 指的是以汉语作为第二语言的学习者在说或写的过程中产出的文本。汉语学习者文本纠错 (Chinese Learner Text Correction, **CLTC**) 旨在通过智能纠错系统，自动检测并修改学习者文本中的标点、拼写、语法、语义等错误，从而获得符合原意的正确句子。

赛道一的数据中提供针对一个句子的多个参考答案，并且从最小改动（Minimal Edit，M）和流利提升（Fluency Edit，F）两个维度对模型结果进行评测。最小改动维度要求尽可能好地维持原句的结构，尽可能少地增删、替换句中的词语，使句子符合汉语语法规则；流利提升维度则进一步要求将句子修改得更为流利和地道，符合汉语母语者的表达习惯。

赛道一下设置开放任务和封闭任务，**本仓库开源了开放和封闭任务下的数据和模型**。

## 2 数据集

我们对训练数据进行了去噪处理，实验证明使用去噪数据训练模型可以显著提升在官方 YACLC 评估集的测试性能。[数据下载链接](https://drive.google.com/file/d/17-pFutgOyuxilKjYluCee7YByH5gJZPf/view?usp=sharing)

将上述文件解压放在 `./models/fairseq/bart/preprocess/zho` 目录下。

## 3 模型训练

### 3.1 Seq2Seq 模型

#### 环境安装

我们采用 Python 3.9 进行实验，Seq2Seq 的全部实验（包括训练和推理）完全在 `Fairseq` 上进行。

```bash
pip install fairseq==0.12.2
```

其他组件，如 `torch`，只要能够兼容上述 `fairseq` 版本即可。

#### 数据预处理

`Fairseq` 的数据预处理比较复杂，细节可参考 `./scripts/fairseq/zho/preprocess.sh`。

#### 模型训练

我们提供了模型训练的流水线脚本，包含预处理-训练-推理的流程，细节可参考 `./scripts/fairseq/zho/train_ccl_close.sh`。

```bash
# 进入到项目根目录下
cd CCL2023/

# 启动训练脚本，使用 cuda:0 训练
nohup bash scripts/fairseq/zho/train_ccl_close.sh -g 0
```

#### 模型推理和评估

本次比赛使用中文语法纠错开源评估工具 `ChERRANT` 进行评估，评估细节可参考 `./scripts/fairseq/zho/predict.sh`。

### 3.2 Seq2Edit 模型

由于我们没有对开源的 Seq2Edit 模型进行任何更改，所以此处附上[开源项目链接](https://github.com/HillZhang1999/MuCGEC)。

### 3.3 模型权重下载

> 模型权重上传需要时间，后续将附上模型下载链接。

我们提供了微调模型的 checkpoint 以供测试（下列指标均为精确度 / 召回度 / F$_{0.5}$ 值）：

| 模型                                                         | YACLC_test_minimal    | YACLC_test_fluent     |
| ------------------------------------------------------------ | --------------------- | --------------------- |
| [**Seq2Seq_close**](https://drive.google.com/file/d/1hlY1-mON3-KqONHgxPGTbOBi1U_jOM28/view?usp=drive_link) | 77.01 / 56.75 / 71.88 | 49.67 / 26.00 / 42.02 |
| [**Seq2Seq_open**](https://drive.google.com/file/d/13SvPW8f3-gSUL3HnAII9L0L5KtEeMyeb/view?usp=drive_link) | 79.27 / 58.45 / 74.00 | 50.80 / 26.50 / 42.93 |
| **Seq2Edit_close**                                           | 72.10 / 52.76 / 67.17 | 47.43 / 25.01 / 40.22 |
| **Seq2Edit_open**                                            | 74.11 / 52.16 / 68.36 | 49.48 / 23.73 / 40.65 |

下载后，解压放入 `./models/fairseq/bart/exps/zho` 和 `./models/seq2edit/exps` 下相应目录即可使用。其中，Seq2Seq 模型基于 `Chinese-BART-Large` 预训练语言模型，Seq2Edit 模型基于 `StructBERT-Large` 预训练语言模型。

### 3.4 模型集成

我们使用的模型集成策略请参考 [MuCGEC](https://github.com/HillZhang1999/MuCGEC)。

## 联系 & 反馈

如果您对本仓库有任何疑问，欢迎联系 yejh22@mails.tsinghua.edu.cn。











