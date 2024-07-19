use std::{cell::RefCell, rc::Rc};

use crate::math::{tensor::Tensor, tensor_context::{TensorContext, TensorRef}};

pub struct SineWaveGenerator {
    pub amplitude: f64,
    pub frequency: f64,
    pub phase: f64,
    pub noise: f64,
}
impl SineWaveGenerator  {
    pub fn generate_data(&self, count: usize) -> (Tensor, Tensor) {
        let tensor_context: Rc<RefCell<TensorContext>> = create_tensor_context!(count*2 + 256);
        let mut labels = Vec::new();
        let mut data = Vec::new();
        for i in 0..count {
            // Let x be a random number between 0 and 2π
            let x = rand::random::<f64>() * 2.0 * std::f64::consts::PI;
            let y = self.amplitude * (self.frequency * x + self.phase).sin() + self.noise * rand::random::<f64>();
            data.push(x);
            labels.push(y);

        }
        let data_tensor = tensor_context.borrow_mut().new_tensor(vec![count], data);
        let labels_tensor = tensor_context.borrow_mut().new_tensor(vec![count], labels);

        let data_tensor = tensor_context.borrow_mut().get_tensor(data_tensor);
        let labels_tensor = tensor_context.borrow_mut().get_tensor(labels_tensor);
        (data_tensor, labels_tensor)
    }
}