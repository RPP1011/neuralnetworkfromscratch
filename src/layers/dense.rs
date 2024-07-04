use std::cell::RefCell;
use std::rc::Rc;

use crate::layers::layers::layers::Layer;
use crate::math::tensor::Tensor;
use crate::math::tensor_context::{self, TensorContext, TensorRef};
use crate::nuerons;
use crate::nuerons::activation_function::ActivationFunction;
use crate::nuerons::nuerons::Nueron;

pub struct Dense {
    pub neurons: Vec<Nueron>,
}

impl Layer for Dense {
    fn forward(&self, input: TensorRef) -> TensorRef {
        // 0
        // let output: Vec<f64> = self.neurons.iter().map(|neuron| neuron.feed_forward(input.clone()).data[0]).collect();
        // Tensor::new(vec![self.neurons.len()], output)
        0
    }

    fn backward(&self, input: TensorRef) -> TensorRef {
        1
        // for neuron in self.neurons.iter() {
        //     neuron.feed_forward(&vec![1.0]);
        // }
    }
}

impl Dense {
    pub fn new(tensor_context: Rc<RefCell<TensorContext>>, n:usize, activation_function: ActivationFunction) -> Dense {
        let mut neurons = Vec::new();
        for _ in 0..n {
            neurons.push(Nueron::new(tensor_context.clone(),1,activation_function));
        }
        Dense {
            neurons: neurons
        }
    }
}