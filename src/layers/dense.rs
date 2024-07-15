use std::cell::RefCell;
use std::rc::Rc;

use crate::layers::layers::layers::Layer;

use crate::math::tensor_context::{TensorContext, TensorRef};
use crate::nuerons::activation_function::ActivationFunction;
use crate::nuerons::nuerons::Neuron;

pub struct Dense {
    pub neurons: Vec<Neuron>,
    size: usize,
    activation_function: ActivationFunction,
    tensor_context: Rc<RefCell<TensorContext>>,
    output_tensor: Option<TensorRef>,
    input_tensor: Option<TensorRef>,
}

impl Layer for Dense {
    fn forward(&self, input: TensorRef) -> TensorRef {
        println!("Forwarding through Dense Layer");
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
        let input_size = self.tensor_context.borrow().get_tensor(input).shape[0];

        for _ in 0..self.size {
            self.neurons.push(Neuron::new(self.tensor_context.clone(), input_size, self.activation_function));
        }

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
        let neurons = Vec::new();

        Dense {
            neurons,
            tensor_context,
            activation_function,
            size: n,
            output_tensor: None,
            input_tensor: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::create_tensor_context;

    use super::*;

    #[test]
    fn test_compile() {
        // Create a tensor context
        let tensor_context = create_tensor_context!(1024);
        
        // Create a dense layer
        let mut dense = Dense::new(tensor_context.clone(), 3, ActivationFunction::ReLU);

        // Create a dummy input tensor
        let input = tensor_context.borrow_mut().new_tensor(vec![2], vec![1.0, 2.0]);

        // Compile the dense layer
        let output = dense.compile(input);

        // Check if the output tensor has the correct shape

        assert_eq!(tensor_context.borrow_mut().get_tensor(output).shape, vec![3]);

        // Check if the output tensor has the correct values
        let output_values = tensor_context.borrow_mut().get_tensor(output).data.clone();
        assert_ne!(output_values, vec![0.0, 0.0, 0.0]);
        println!("{:?}", output_values);

        // Check if the output tensor responds to changes in input tensor
        tensor_context.borrow_mut().set_data(input, vec![3.0, 4.0]);
        dense.forward(input);
        let new_output_values = tensor_context.borrow_mut().get_tensor(output).data.clone();
        assert_ne!(new_output_values, output_values);
        println!("{:?}", new_output_values);

        // Check if additional forward passes affect output tensor
        let old_output_values = new_output_values.clone();
        dense.forward(input);
        let new_output_values = tensor_context.borrow_mut().get_tensor(output).data.clone();
        assert_eq!(old_output_values, new_output_values);


        // Check how additional input tensor affects output tensor
        let input = tensor_context.borrow_mut().new_tensor(vec![2], vec![5.0, 6.0]);
        let output = dense.forward(input);
        let output_values = tensor_context.borrow_mut().get_tensor(output).data.clone();
        assert_ne!(output_values, old_output_values);
    }
}
