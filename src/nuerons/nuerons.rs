use crate::{math::tensor::Tensor, nuerons::activation_function::ActivationFunction};

use super::activation_function;

#[derive(Debug, Clone)]
pub struct Nueron {
    pub weights : Tensor,
    pub bias : Tensor,
    pub activation_function: ActivationFunction,
}

impl Nueron {
    pub fn new(weight_count: usize, activation_function: ActivationFunction) -> Nueron{
        let random_weights = Tensor::random(vec![weight_count]);
        Nueron {
            weights: random_weights,
            bias: Tensor::new(vec![1], vec![rand::random::<f64>()]),
            activation_function
        }
    }

    pub fn feed_forward(&self, inputs: Tensor) -> Tensor {
        let sum: Tensor = inputs.dot(&self.weights);
        Tensor::new(vec![1], vec![self.activation_function.apply(sum + self.bias)])
    }
}