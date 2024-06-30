use std::rc::Rc;

use crate::layers::layers::layers::Layer;

use super::{loss_function::LossFunction, network_metric::Metric, optimizer::Optimizer};


pub struct Sequential {
    pub layers: std::vec::Vec<Rc<dyn Layer>>
}

impl Sequential {
    pub fn compile(&self, optimizer: Optimizer, loss: LossFunction, metrics: Vec<Metric>) {
        todo!()
    }

    pub fn fit(&self, data:Vec<f64>, labels:Vec<f64>, epochs: usize) {
        todo!()
    }

    pub fn predict(&self, data:Vec<f64>) -> f64 {
        todo!()
    }

    pub fn evaluate(&self, data:Vec<f64>, labels:Vec<f64>) -> (f64, f64) {
        todo!()
    }

    pub fn save(&self) {
        todo!()
    }
}

