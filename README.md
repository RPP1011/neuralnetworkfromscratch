![Build Status](https://github.com/RPP1011/neuralnetworkfromscratch/actions/workflows/rust.yml/badge.svg)

## Goals Of This Project

This is a horribly misguided attempt to clone TensorFlow in Rust. Its progress has been severely hampered by my complete lack of understanding of general computer science principles and ineptness, but hopefully it will get to a usable state within a few hundred hours of effort.
q
## Steps

1. **Make MLP NN capable of attaining 80% accuracy on MNIST**  
   - [MLP (Multilayer Perceptron) on MNIST](https://en.wikipedia.org/wiki/Multilayer_perceptron)
   - [MNIST Dataset Overview](http://yann.lecun.com/exdb/mnist/)  
   *(Creating a simple feedforward neural network with dense layers to classify handwritten digits.)*

2. **Make ConvNet capable of image classification**  
   - [Introduction to Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
   - [Convolutional Neural Networks with TensorFlow](https://www.tensorflow.org/tutorials/images/cnn)  
   *(Develop a CNN model for classifying images, typically for applications like CIFAR-10 or MNIST.)*

3. **Make ConvNet version of MNIST**  
   - [CNN on MNIST Tutorial](https://www.tensorflow.org/tutorials/quickstart/advanced)  
   *(Adapt a CNN specifically for MNIST, optimizing it for digit recognition.)*

4. **Make PINN Network**  
   - [Physics-Informed Neural Networks (PINNs)](https://maziarraissi.github.io/PINNs/)  
   *(Train neural networks to solve problems governed by partial differential equations.)*

5. **Make Encoder Block**  
   - [Encoder-Decoder Architecture](https://machinelearningmastery.com/what-is-the-encoder-decoder-model/)
   - [Transformers Explained](https://jalammar.github.io/illustrated-transformer/)  
   *(Implement the encoder portion for tasks requiring sequential input processing.)*

6. **Make Decoder Block**  
   - [Decoder Block in Transformers](https://huggingface.co/transformers/v3.0.2/model_doc/transformer.html)  
   *(Build the decoder, which processes encoded information for generation or transformation tasks.)*

7. **Make Transformer**  
   - [Attention is All You Need Paper (Transformers)](https://arxiv.org/abs/1706.03762)  
   *(Construct a Transformer model using self-attention mechanisms for language or image processing.)*

8. **Make Vision Transformers**  
   - [Vision Transformer (ViT)](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html)  
   *(Build a transformer model specifically adapted for image classification tasks.)*

9. **GAN Network**  
   - [Introduction to Generative Adversarial Networks (GANs)](https://arxiv.org/abs/1406.2661)  
   *(Develop a GAN for generating new data, such as images, by training two neural networks in opposition.)*

---

## Concerns

1. **Loss vs Epoch Graph**  
   - [How to Plot Loss vs. Epochs in Machine Learning](https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)  
   *(Implement a plot that shows how the model's loss changes with each epoch to evaluate performance over time.)*


