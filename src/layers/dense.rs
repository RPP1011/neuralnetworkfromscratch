use std::cell::RefCell;
use std::rc::Rc;

use crate::layers::layers::layers::Layer;

use crate::math::tensor_context::{TensorContext, TensorRef};

use crate::nuerons::activation_function::ActivationFunction;
use crate::nuerons::nuerons::Nueron;

pub struct Dense {
    pub neurons: Vec<Nueron>,
    tensor_context: Rc<RefCell<TensorContext>>,
    output_tensor: Option<TensorRef>,
    input_tensor: Option<TensorRef>,
}

impl Layer for Dense {
    fn forward(&self, input: TensorRef) -> TensorRef {
        let feed_forward_results: Vec<TensorRef> = self
            .neurons
            .iter()
            .map(|neuron| neuron.feed_forward(input))
            .collect();
        self.tensor_context
            .borrow_mut()
            .concat_inplace(feed_forward_results, self.output_tensor.unwrap());
        self.output_tensor.unwrap()
    }

    fn compile(&mut self, input: TensorRef) -> TensorRef {
        self.input_tensor = Some(input);

        let initialize_results: Vec<TensorRef> = self
            .neurons
            .iter_mut()
            .map(|neuron| neuron.initialize(input))
            .collect();

        self.output_tensor = Some(self.tensor_context.borrow_mut().concat(initialize_results));
        self.output_tensor.unwrap()
    }

    fn get_parameters(&self) -> Vec<TensorRef> {
        let mut parameters = Vec::new();
        for neuron in self.neurons.iter() {
            parameters.append(&mut neuron.get_parameters());
        }
        parameters
    
    }
}

impl Dense {
    pub fn new(
        tensor_context: Rc<RefCell<TensorContext>>,
        n: usize,
        activation_function: ActivationFunction,
    ) -> Dense {
        let mut neurons = Vec::new();
        for _ in 0..n {
            neurons.push(Nueron::new(tensor_context.clone(), 1, activation_function));
        }
        Dense {
            neurons,
            tensor_context,
            output_tensor: None,
            input_tensor: None,
        }
    }
}
