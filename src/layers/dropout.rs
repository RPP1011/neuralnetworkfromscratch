use std::{cell::RefCell, rc::Rc, vec};

use rand::distributions::{Bernoulli, Distribution};

use crate::math::{
    tensor_context::{TensorContext, TensorRef},
};

use super::layers::layers::Layer;

pub struct Dropout {
    pub rate: f64,
    pub training: bool,
    tensor_context: Rc<RefCell<TensorContext>>,
    mask_tensor: Option<TensorRef>,
    output_tensor: Option<TensorRef>,
    distribution: Bernoulli,
    layer_shape: Option<usize>,
}

impl Layer for Dropout {
    fn forward(&self, input: TensorRef) -> TensorRef {
        if self.training {
            let mut mask = vec![0.0; self.layer_shape.unwrap()];
            for i in 0..self.layer_shape.unwrap() {
                mask[i] = if self.distribution.sample(&mut rand::thread_rng()) {
                    1.0
                } else {
                    0.0
                };
            }

            self.tensor_context
                .borrow_mut()
                .set_data(self.mask_tensor.unwrap(), mask);
        
        } else {
            self.tensor_context.borrow_mut().set_data(
                self.mask_tensor.unwrap(),
                vec![1.0; self.layer_shape.unwrap()],
            );
        }

        self.tensor_context.borrow_mut().mul_inplace(
            input,
            self.mask_tensor.unwrap(),
            self.output_tensor.unwrap(),
        );
        self.output_tensor.unwrap()
    }

    fn compile(&mut self, input: TensorRef) -> TensorRef {
        let input_shape = self.tensor_context.borrow().get_tensor(input).shape.clone();
        self.layer_shape = Some(input_shape.iter().product());
        self.mask_tensor = Some(
            self.tensor_context
                .borrow_mut()
                .new_tensor(input_shape, vec![0.0; self.layer_shape.unwrap()]),
        );
        self.output_tensor = Some(
            self.tensor_context
                .borrow_mut()
                .mul(input, self.mask_tensor.unwrap()),
        );
        self.output_tensor.unwrap()
    }
}

impl Dropout {
    pub fn new(tensor_context: Rc<RefCell<TensorContext>>, rate: f64, training: bool) -> Dropout {
        Dropout {
            tensor_context,
            rate,
            training,
            output_tensor: None,
            mask_tensor: None,
            distribution: Bernoulli::new(rate).unwrap(),
            layer_shape: None,
        }
    }
}
