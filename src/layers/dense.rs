use crate::layers::layers::layers::Layer;
use crate::math::tensor::Tensor;
use crate::nuerons;
use crate::nuerons::activation_function::ActivationFunction;
use crate::nuerons::nuerons::Nueron;

pub struct Dense {
    pub neurons: Vec<Nueron>,
}

impl Layer for Dense {
    fn forward(&self, input: Tensor) -> Tensor {
        let output: Vec<f64> = self.neurons.iter().map(|neuron| neuron.feed_forward(input.clone()).data[0]).collect();
        Tensor::new(vec![self.neurons.len()], output)
    }

    fn backward(&self) {
        
        for neuron in self.neurons.iter() {
            neuron.feed_forward(&vec![1.0]);
        }
    }
}

impl Dense {
    pub fn new(n:usize, activation_function: ActivationFunction) -> Dense {
        let mut neurons = Vec::new();
        for _ in 0..n {
            neurons.push(Nueron::new(1,activation_function));
        }
        Dense {
            neurons: neurons
        }
    }
}