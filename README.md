
# InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models
[Hao Li*](https://leolee99.github.io/), [Xiaogeng Liu*](https://sheltonliu-n.github.io/). (*Equal Contribution)

-----
<a href='https://injecguard.github.io/'><img src='https://img.shields.io/badge/Website-Page-%23f08080'></a> 
<a href='https://github.com/SaFoLab-WISC/InjecGuard'><img src='https://img.shields.io/badge/GitHub-code-green?logo=github'></a> 
<a href='https://arxiv.org/pdf/2410.22770'><img src='https://img.shields.io/badge/Paper-PDF-red?logo=open-access'></a> 
<a href="https://huggingface.co/datasets/leolee99/NotInject"><img src="https://img.shields.io/badge/Dataset-HF-orange?logo=huggingface" alt="huggingface"/></a>
<a href="https://injecguard.github.io/"><img src="https://img.shields.io/badge/Demo-Page-yellow" alt="Github license"/></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" alt="Github license"/></a>


<!-- <a href="https://huggingface.co/leolee99/InjecGuard"><img src="https://img.shields.io/badge/Demo-%F0%9F%A4%97-yellow" alt="huggingface"/></a> -->

This repository hosts the official code, data and model weights of **InjecGuard**, the first prompt guard model against prompt injection to be built with open-source training data and detailed documentation, consistently achieving remarkable performance in benign, malicious, and over-defense accuracy.

<p align="center" width="100%">
<a target="_blank"><img src="assets/figure_performance.png" alt="Perfomance Comparison" style="width: 100%; min-width: 200px; display: block; margin: auto;"></a>
</p>

## 🎉 News 

- [2024.10.31] 🎉🎉 Our paper has been released on [arXiv](https://arxiv.org/pdf/2410.22770).
- [2024.10.28] 📷 Provide an online [demo](https://injecguard.github.io/) of InjecGuard.
- [2024.10.27] 🤗Release the [NotInject](https://huggingface.co/datasets/leolee99/NotInject) dataset.
- [2024.10.27] 🛠️ Release the [code](https://github.com/SaFoLab-WISC/InjecGuard) of InjecGuard.

## Abstract
Prompt injection attacks pose a critical threat to large language models (LLMs), enabling goal hijacking and data leakage. Prompt guard models, though effective in defense, suffer from over-defense—falsely flagging benign inputs as malicious due to trigger word bias. To address this issue, we introduce ***NotInject***, an evaluation dataset that systematically measures over-defense across various prompt guard models. NotInject contains 339 benign samples enriched with trigger words common in prompt injection attacks, enabling fine-grained evaluation. Our results show that state-of-the-art models suffer from over-defense issues, with accuracy dropping close to random guessing levels (60\%). To mitigate this, we propose ***InjecGuard***, a novel prompt guard model that incorporates a new training strategy, *Mitigating Over-defense for Free* (MOF), which significantly reduces the bias on trigger words. InjecGuard demonstrates state-of-the-art performance on diverse benchmarks including NotInject, surpassing the existing best model by 30.8\%, offering a robust and open-source solution for detecting prompt injection attacks.

## Demos of InjecGuard

https://github.com/user-attachments/assets/a6b58136-a7c4-4d7c-8b85-414884d34a39

We have released an online demo, you can access it [here](InjecGuard.github.io).


## NotInject Dataset
To address the over-defense issue commonly seen in existing guard models, we introduce the [NotInject](https://huggingface.co/datasets/leolee99/NotInject) dataset, designed to evaluate the extent of over-defense in these models. We identify certain trigger words that may cause defense shortcuts in guard models and use them to construct benign sentences. The dataset is divided into three subsets, each containing sentences with one, two, or three trigger words. For each subset, we create 113 benign sentences across four topics: Common Queries, Technique Queries, Virtual Creation, and Multilingual Queries.

<p align="center" width="100%">
<a target="_blank"><img src="assets/NotInject_distribution.png" alt="Perfomance Comparison" style="width: 60%; min-width: 200px; display: block; margin: auto;"></a>
</p>


## Requirements
We recommend the following dependencies.

* Python 3.10
* [PyTorch](http://pytorch.org/) 2.4.0

Then, please install other environment dependencies through:
```bash
pip install -r requirements.txt
```

## Getting Started

## 💾 Checkpoints

You can directly download our trained checkpoints [here](https://drive.google.com/file/d/1JpiVb_wtnbBLNEjIx1KS7PHuvmARQKTu/view?usp=sharing). 

## ⚙️ Dataset Preparation

- **Train set**: We have provided our training dataset in the path of ```InjecGuard\datasets```, collected from 20 open-source datasets and several of our LLM augmented data.

- **Valid set**: We select several samples (144) from NotInject, BIPIA, Wildguard-Benign, and PINT to conduct validation, which have been provided in the path of ```InjecGuard\datasets```.

- **Test set**: We select [NotInject](https://huggingface.co/datasets/leolee99/NotInject), [BIPIA](https://github.com/microsoft/BIPIA), [Wildguard-Benign](https://github.com/allenai/wildguard), and [PINT](https://github.com/lakeraai/pint-benchmark) to evaluate the benign, malicious, and over-defense of the model. The first three are all provided in the path of ```InjecGuard\datasets```. The benchmark of PINT is not public, but you can request access to it by filling out [here](https://share-eu1.hsforms.com/1TwiBEvLXRrCjJSdnbnHpLwfdfs3). 

**Note:** Once you’ve downloaded the PINT benchmark, convert it from original YAML to JSON format by executing the following command:
```
python util.py
```


## 🔥 Train your InjecGuard

There are some of arguments you can set:
- ```--train_set```: the path to the train set file.
- ```--valid_set```: the path to the valid set file.
- ```--dataset_root```: the folder to place test sets.
- ```--batch_size```: you can modify it to fit your GPU memory size.
- ```--epochs```: the number of training iterations for each sample.
- ```--eval_batch_size```: The batch size in the evaluation process.
- ```--save_step```: the step interval to save models.
- ```--checkpoint_path```: you can modify it to fit your GPU memory size.
- ```--logs```: where to store logs.
- ```--max_length```: the maximum length of input tokens.
- ```--resume```: the model you want to load.
- ```--save_thres```: the performance threshold to save models, the model will only be saved when the performance exceeds the threshold.
- ```--resume```: the model you want to load.

Then, you can train InjecGuard by excuting the command:
```
python train.py
```


## 📋 Evaluation

You can evaluate the model on both 4 datasets ***(NotInject, PINT, Wildguard-Benign, BIPIA)*** by excuting the command:
```
python eval.py --resume ${CHECKPOINT}$
```

## 📈 Results

<p align="center" width="100%">
<a target="_blank"><img src="assets/Results.png" alt="Perfomance Comparison" style="width: 100%; min-width: 200px; display: block; margin: auto;"></a>
</p>

<p align="center" width="100%">
<a target="_blank"><img src="assets/visualization_concat.png" alt="Perfomance Comparison" style="width: 100%; min-width: 200px; display: block; margin: auto;"></a>
</p>

## Citation

If you find this work useful in your research or applications, we appreciate that if you can kindly cite:
```
@articles{InjecGuard,
  title={InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models},
  author={Hao Li and Xiaogeng Liu},
  journal = {arXiv preprint arXiv:2410.22770},
  year={2024}
}
```
