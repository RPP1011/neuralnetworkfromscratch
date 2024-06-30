use std::rc::Rc;

use crate::layers::layers::layers::Layer;

use super::{loss_function::LossFunction, network_metric::Metric, optimizer::Optimizer};


pub struct Sequential {
    pub layers: std::vec::Vec<Rc<dyn Layer>>
}

impl Model for Sequential {
    fn compile(&self, optimizer: Optimizer, loss: LossFunction, metrics: Vec<Metric>) {
        todo!()
    }

    fn fit(&self, data:Vec<f64>, labels:Vec<f64>, epochs: usize) {
        todo!()
    }

    fn predict(&self, data:Vec<f64>) -> f64 {
        todo!()
    }

    fn evaluate(&self, data:Vec<f64>, labels:Vec<f64>) -> (f64, f64) {
        todo!()
    }

    fn save(&self) {
        todo!()
    }
}

pub trait Model {
    fn compile(&self, optimizer: Optimizer, loss: LossFunction, metrics: Vec<Metric>);
    fn fit(&self, data:Vec<f64>, labels:Vec<f64>, epochs: usize);
    fn predict(&self, data:Vec<f64>) -> f64;
    fn evaluate(&self, data:Vec<f64>, labels:Vec<f64>) -> (f64, f64);
    fn save(&self);
}

