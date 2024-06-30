use crate::nuerons::activation_function::ActivationFunction;

use super::activation_function;

#[derive(Debug, Clone)]
pub struct Nueron {
    pub weights : Vec<f64>,
    pub bias : f64,
    pub activation_function: ActivationFunction,
}

impl Nueron {
    pub fn new(weight_count: usize, activation_function: ActivationFunction) -> Nueron{
        let random_weights = (0..weight_count).map(|_| rand::random::<f64>()).collect();

        Nueron {
            weights: random_weights,
            bias: rand::random::<f64>(),
            activation_function
        }
    }

    pub fn feed_forward(&self, inputs: &Vec<f64>) -> f64 {
        let sum: f64 = self.weights.iter().zip(inputs.iter()).map(|(w, i)| w * i).sum();
        self.activation_function.apply(sum + self.bias)
    }
}