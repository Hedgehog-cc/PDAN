# Leveraging Progressive Domain Adaptation for Unsupervised Cross-Domain Oracle Bone Inscription Recognition (PDAN)


<img width="1514" height="1504" alt="image" src="https://github.com/user-attachments/assets/336d2ab1-e7ec-4fac-992e-9e17890020ba" />


We propose an unsupervised cross-domain OBI recognition framework that transfers glyph knowledge from easily accessible labeled handwritten oracle bone characters to unlabeled scanned rubbing images. Specifically, we design a Progressive Domain Adaptation Network (PDAN) based on two key components: feature-space linear interpolation, which constructs a semantically continuous representation space between domains, and a dynamically-weighted gradient inversion mechanism that enhances domain-invariant feature learning. Additionally, we adopt a structure-aware augmentation strategy for the target domain to improve robustness against real-world degradation and noise. 

To support research in this community, we propose UDA‑HS‑1K, a large-scale benchmark dataset designed to evaluate recognition algorithms under realistic conditions. It is constructed as a paired multimodal dataset, comprising both handwritten characters and scanned rubbing images. The figure below illustrates the morphological diversity of the Chinese character 'Zhen' in our constructed dataset.

<img width="1280" height="646" alt="example" src="https://github.com/user-attachments/assets/78a33060-d450-48fa-a968-cddec1eddfe3" />

