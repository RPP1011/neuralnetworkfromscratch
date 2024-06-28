use std::rc::Rc;

use crate::layers::layers::layers::Layer;

use super::{loss_function::LossFunction, optimizer::Optimizer};


pub struct Sequential {
    pub layers: std::vec::Vec<Rc<dyn Layer>>
}

impl Sequential {
    pub fn compile(optimizer: Optimizer, loss: LossFunction) {
        todo!()
    }

    pub fn fit(data:Vec<f64>, labels:Vec<f64>, epochs: usize) {
        todo!()
    }
}

