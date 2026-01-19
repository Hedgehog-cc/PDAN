# Leveraging Progressive Domain Adaptation for Unsupervised Cross-Domain Oracle Bone Inscription Recognition (PDAN)


<img width="1514" height="1504" alt="image" src="https://github.com/user-attachments/assets/336d2ab1-e7ec-4fac-992e-9e17890020ba" />

# Method
<img width="2346" height="1257" alt="image" src="https://github.com/user-attachments/assets/090d917a-2d61-4d7a-8938-93c1bdeb6faa" />

We propose an unsupervised cross-domain OBI recognition framework that transfers glyph knowledge from easily accessible labeled handwritten oracle bone characters to unlabeled scanned rubbing images. Specifically, we design a Progressive Domain Adaptation Network (PDAN) based on two key components: feature-space linear interpolation, which constructs a semantically continuous representation space between domains, and a dynamically-weighted gradient inversion mechanism that enhances domain-invariant feature learning. Additionally, we adopt a structure-aware augmentation strategy for the target domain to improve robustness against real-world degradation and noise. 
# Dataset: UDA-HS-1K
To support research in this community, we propose UDA‑HS‑1K, a large-scale benchmark dataset designed to evaluate recognition algorithms under realistic conditions. It is constructed as a paired multimodal dataset, comprising both handwritten characters and scanned rubbing images. The figure below illustrates the morphological diversity of the Chinese character 'Zhen' in our constructed dataset.

<img width="1280" height="646" alt="example" src="https://github.com/user-attachments/assets/78a33060-d450-48fa-a968-cddec1eddfe3" />

# Getting Started
The dataset split files are already provided in the `data` directory. Please download the full UDA-HS-1K dataset via the link below:

Baidu Netdisk: https://pan.baidu.com/s/1g29lySwRw8oc-QG657rNmQ?pwd=0812 (Access Code: `0812`)

Setup:
Download and move the `handprint` and `scan` folders into `./data/`.

Training:
Once the data is prepared, you can train the model by running:

```bash
python PDAN.py --num_steps 250000 --batch_size 32 --num_classes 1000
```
