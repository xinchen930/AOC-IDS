# AOC-IDS
This is the code for the paper: ["AOC-IDS: Autonomous Online Framework with Contrastive Learning for Intrusion Detection"](link-to-my-paper) (Infocom 2024)  
Xinchen Zhang, Running Zhao, Zhihan Jiang, Zhicong Sun, Yulong Ding, Edith C.H. Ngai, Shuang-hua Yang.

## Dependencies
The project is implemented using PyTorch and has been tested on the following hardware and software configuration:

- Ubuntu 20.04 Desktop
- NVIDIA GeForce RTX 3090 GPUs
- CUDA, version = 11.7
- PyTorch, version = 1.13.1
- Anaconda3

### Installation
To install the necessary libraries and dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Experiments
We tested the effectiveness of our proposed method on the NSL-KDD and UNSW-NB15 datasets. Preprocessed versions of these datasets are provided in this repository, allowing for immediate execution. The continuous attributes have been normalized, and categorical attributes have been one-hot encoded.

Here is an example of how to start training:
```bash
python online_training.py --dataset unsw --epochs 300 --epoch_1 3 --flip_percent 0.05 --sample_interval 2784
```

## Citation
If you find this code useful in your research, please cite:
```bibtex
@inproceedings{zhang2024aocids,
  title={AOC-IDS: Autonomous Online Framework with Contrastive Learning for Intrusion Detection},
  author={Zhang, Xinchen and Zhao, Running and Jiang, Zhihan and Sun, Zhicong and Ding, Yulong and Ngai, Edith C.H. and Yang, Shuang-hua},
  booktitle={IEEE INFOCOM 2024 - IEEE Conference on Computer Communications},
  year={2024}
}
```


