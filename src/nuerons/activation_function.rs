use crate::math::tensor::Tensor;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    LeakyReLU,
    Tanh,
    Softmax,
}

impl ActivationFunction {
    pub fn apply(&self, x: Tensor) -> Tensor {
        match self {
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::ReLU => if x > 0.0 { x } else { 0.0 },
            ActivationFunction::LeakyReLU => if x > 0.0 { x } else { 0.01 * x },
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Softmax => x.exp(),
        }
    }
}