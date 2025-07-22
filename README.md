# GANs in Action — Code Companion

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.6+-blue.svg" alt="Python 3.6+">
  <img src="https://img.shields.io/badge/TensorFlow-1.8.0+-orange.svg" alt="TensorFlow 1.8.0+">
  <img src="https://img.shields.io/badge/Keras-2.1.6+-red.svg" alt="Keras 2.1.6+">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License MIT">
</p>

This is the official companion repository to the book **[GANs in Action: Deep Learning with Generative Adversarial Networks](https://www.manning.com/books/gans-in-action)** by Jakub Langr and Vladimir Bok, published by Manning Publications.

This repo lets you reproduce, study, and extend every hands‑on example from the book.
The notebooks walk through every major variant in the GAN family, from the original **vanilla GAN** to **CycleGAN**, using Keras/TensorFlow.  

## 📚 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Canonical GAN Papers](#canonical-gan-papers)
- [Getting Started](#getting-started)
- [Chapter Implementations](#chapter-implementations)
- [Educational Resources](#educational-resources)
- [Best Practices](#best-practices)
- [Community Resources](#community-resources)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

This repository contains practical implementations of various Generative Adversarial Network architectures discussed in the book "GANs in Action". Each chapter includes Jupyter notebooks with fully functional code examples that demonstrate key concepts and techniques in GAN development.

### What You'll Learn

- Fundamental concepts of generative modeling and adversarial training
- Implementation of various GAN architectures from scratch
- Best practices for training stable GANs
- Real-world applications of GANs
- Advanced techniques for improving GAN performance

## 📂 Repository Structure

```
gans-in-action/
├── chapter-2/          # Autoencoders
├── chapter-3/          # Vanilla GAN
├── chapter-4/          # Deep Convolutional GAN (DCGAN)
├── chapter-6/          # Progressive GAN
├── chapter-7/          # Semi-Supervised GAN
├── chapter-8/          # Conditional GAN
├── chapter-9/          # CycleGAN
├── chapter-10/         # Adversarial examples
└── requirements.txt    # Python dependencies
```

## 📄 Canonical GAN Papers

Each implementation in this repository is based on groundbreaking research. Here are the canonical papers for each GAN architecture covered:

### Original GAN (Chapter 3)
**Paper:** [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)  
**Authors:** Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio  
**Year:** 2014  
**Key Contribution:** Introduced the foundational GAN framework with adversarial training between generator and discriminator networks.

### Deep Convolutional GAN - DCGAN (Chapter 4)
**Paper:** [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434)  
**Authors:** Alec Radford, Luke Metz, Soumith Chintala  
**Year:** 2015  
**Key Contribution:** Established architectural guidelines for stable GAN training using convolutional networks.

### Progressive GAN (Chapter 6)
**Paper:** [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)  
**Authors:** Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen  
**Year:** 2017  
**Key Contribution:** Introduced progressive training methodology for generating high-resolution images.

### Semi-Supervised GAN (Chapter 7)
**Paper:** [Semi-Supervised Learning with Generative Adversarial Networks](https://arxiv.org/abs/1606.01583)  
**Authors:** Augustus Odena  
**Year:** 2016  
**Key Contribution:** Extended GANs for semi-supervised learning by modifying the discriminator to output class labels.

### Conditional GAN - CGAN (Chapter 8)
**Paper:** [Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)  
**Authors:** Mehdi Mirza, Simon Osindero  
**Year:** 2014  
**Key Contribution:** Enabled conditional generation by incorporating label information into both generator and discriminator.

### CycleGAN (Chapter 9)
**Paper:** [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)  
**Authors:** Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros  
**Year:** 2017  
**Key Contribution:** Enabled image-to-image translation without paired training data using cycle consistency loss.

## 🚀 Getting Started

### Prerequisites

- Python 3.6 or higher
- CUDA-capable GPU (recommended for faster training)
- 8GB+ RAM

### Installation

1. Clone this repository:
```bash
git clone https://github.com/GANs-in-Action/gans-in-action.git
cd gans-in-action
```

2. Create a virtual environment:
```bash
python -m venv gan_env
source gan_env/bin/activate  # On Windows: gan_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Examples

Navigate to any chapter directory and launch Jupyter Notebook:
```bash
cd chapter-3
jupyter notebook
```

Open the notebook file (e.g., `Chapter_3_GAN.ipynb`) and run the cells sequentially.

## 📖 Chapter Implementations

### Chapter 3: Your First GAN
- **Implementation:** Basic GAN for MNIST digit generation
- **Key Concepts:** Generator and discriminator networks, adversarial loss, training dynamics
- **Dataset:** MNIST handwritten digits

### Chapter 4: Deep Convolutional GAN (DCGAN)
- **Implementation:** DCGAN for generating realistic images
- **Key Concepts:** Convolutional architectures, batch normalization, architectural guidelines
- **Dataset:** MNIST, CelebA (optional)

### Chapter 5: Training and Common Challenges
- **Implementation:** Various training techniques and solutions
- **Key Concepts:** Mode collapse, vanishing gradients, training stability
- **Techniques:** Label smoothing, feature matching, minibatch discrimination

### Chapter 6: Progressive GAN
- **Implementation:** Progressive growing for high-resolution generation
- **Key Concepts:** Progressive training, smooth fade-in, minibatch standard deviation
- **Dataset:** CelebA-HQ

### Chapter 7: Semi-Supervised GAN
- **Implementation:** SGAN for improved classification with limited labels
- **Key Concepts:** Semi-supervised learning, modified discriminator architecture
- **Dataset:** MNIST with limited labels

### Chapter 8: Conditional GAN
- **Implementation:** CGAN for controlled generation
- **Key Concepts:** Conditional generation, label embedding, targeted synthesis
- **Dataset:** MNIST with class conditions

### Chapter 9: CycleGAN
- **Implementation:** Unpaired image-to-image translation
- **Key Concepts:** Cycle consistency loss, unpaired translation, domain adaptation
- **Dataset:** Horse2Zebra, Apple2Orange

## 📚 Educational Resources

### Online Courses and Tutorials

1. **[NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160)** by Ian Goodfellow
2. **[MIT Deep Learning Course](http://introtodeeplearning.com/)** - Includes comprehensive GAN coverage
3. **[Stanford CS231n](http://cs231n.stanford.edu/)** - Convolutional Neural Networks for Visual Recognition
4. **[Fast.ai Practical Deep Learning](https://www.fast.ai/)** - Practical approach to deep learning including GANs

### Video Lectures

1. **[Ian Goodfellow: Generative Adversarial Networks (NIPS 2016)](https://www.youtube.com/watch?v=HGYYEUSm-0Q)**
2. **[Two Minute Papers: GAN Series](https://www.youtube.com/user/keeroyz)** - Accessible explanations of latest GAN research
3. **[Lex Fridman Podcast with Ian Goodfellow](https://www.youtube.com/watch?v=Z6rxFNMGdn0)**

### Books and Reading Materials

1. **[Deep Learning](https://www.deeplearningbook.org/)** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
   - Chapter 20: Deep Generative Models
2. **[Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)** by Christopher Bishop
3. **[Generative Deep Learning](https://www.oreilly.com/library/view/generative-deep-learning/9781492041931/)** by David Foster

### Research Papers - Essential Reading

1. **[Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)** (2016) - Salimans et al.
2. **[Wasserstein GAN](https://arxiv.org/abs/1701.07875)** (2017) - Arjovsky et al.
3. **[Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957)** (2018) - Miyato et al.
4. **[Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)** (2018) - Zhang et al.
5. **[StyleGAN](https://arxiv.org/abs/1812.04948)** (2018) - Karras et al.

### Interactive Resources

1. **[GAN Lab](https://poloclub.github.io/ganlab/)** - Interactive visualization of GAN training
2. **[TensorFlow GAN Playground](https://www.tensorflow.org/tutorials/generative/dcgan)** - Hands-on DCGAN tutorial
3. **[PyTorch GAN Zoo](https://github.com/facebookresearch/pytorch_GAN_zoo)** - Collection of GAN implementations

## 🏆 Best Practices

### Training Tips

1. **Normalize inputs** to `[-1, 1]` range
2. **Use different learning rates** for generator and discriminator (typically `D_lr > G_lr`)
3. **Monitor training metrics** carefully - `D_loss` and `G_loss` should be balanced
4. **Use gradient penalties** (WGAN-GP) for improved stability
5. **Apply spectral normalization** in discriminator for Lipschitz constraint

### Architecture Guidelines

1. **Replace pooling with strided convolutions** (discriminator) and fractional-strided convolutions (generator)
2. **Use BatchNorm in generator** and `LayerNorm`/`InstanceNorm` in discriminator
3. **Use LeakyReLU in discriminator** (`0.2` slope) and `ReLU` in generator
4. **Avoid fully connected layers** for deeper architectures

### Common Pitfalls to Avoid

1. **Mode collapse** - Generator produces limited variety
2. **Vanishing gradients** - Discriminator becomes too strong
3. **Oscillating losses** - Unstable training dynamics
4. **Memory issues** - Use gradient checkpointing for large models

### Evaluation Metrics

1. **Inception Score (IS)** - Measures quality and diversity
2. **Fréchet Inception Distance (FID)** - Compares feature distributions
3. **Precision and Recall** - Measures quality vs diversity trade-off
4. **Human evaluation** - Still the gold standard for many applications

## 🌐 Community Resources

### Forums and Discussion

- **[r/MachineLearning](https://www.reddit.com/r/MachineLearning/)** - Active discussions on latest GAN research
- **[GAN Discord Server](https://discord.gg/gan)** - Community of GAN researchers and practitioners
- **[Stack Overflow - GAN Tag](https://stackoverflow.com/questions/tagged/gan)** - Technical Q&A

### Tools and Frameworks

- **[TensorFlow GAN (TF-GAN)](https://github.com/tensorflow/gan)** - TensorFlow GAN library
- **[PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)** - Collection of PyTorch implementations
- **[Keras-GAN](https://github.com/eriklindernoren/Keras-GAN)** - Keras implementations

### Datasets

- **[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)** - Celebrity faces dataset
- **[LSUN](https://www.yf.io/p/lsun)** - Large-scale scene understanding
- **[FFHQ](https://github.com/NVlabs/ffhq-dataset)** - Flickr-Faces-HQ dataset
- **[ImageNet](http://www.image-net.org/)** - Large-scale image database

## 📝 Citation

If you use this code in your research, please cite the book:

```bibtex
@book{langr2019gans,
  title={GANs in Action: Deep Learning with Generative Adversarial Networks},
  author={Langr, Jakub and Bok, Vladimir},
  year={2019},
  publisher={Manning Publications}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 💬 Contact

- **Book Website:** [https://www.manning.com/books/gans-in-action](https://www.manning.com/books/gans-in-action)
- **Authors:** Jakub Langr & Vladimir Bok
- **Issues:** Please use the [GitHub issue tracker](https://github.com/GANs-in-Action/gans-in-action/issues)

---

<p align="center">
  Made with ❤️ by the GANs in Action team
</p>
