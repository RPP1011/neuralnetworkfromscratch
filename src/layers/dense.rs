use crate::layers::layers::layers::Layer;
use crate::nuerons::activation_function::ActivationFunction;
use crate::nuerons::nuerons::Nueron;

pub struct Dense {
    pub neurons: Vec<Nueron>,
}

impl Layer for Dense {
    fn forward(&self) {
        todo!()
    }

    fn backward(&self) {
        todo!()
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