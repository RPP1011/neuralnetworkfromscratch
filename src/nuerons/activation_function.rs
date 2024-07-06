use crate::math::{tensor::Tensor, tensor_context::TensorRef};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    LeakyReLU,
    Tanh,
    Softmax,
}