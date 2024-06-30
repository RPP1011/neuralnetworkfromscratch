use std::rc::Rc;

use graph::{graph::Sequential, loss_function::LossFunction, network_metric::Metric, optimizer::Optimizer};
use layers::{dense::Dense, dropout::Dropout, flatten::Flatten};
use nuerons::activation_function::ActivationFunction::ReLU;

pub mod layers;
pub mod nuerons;
pub mod math;
pub mod graph;
pub mod file;

fn main() {
    let network = Sequential {
        layers: vec![
            Rc::new(Flatten::new(vec![28,28])),
            Rc::new(Dense::new(128, ReLU)),
            Rc::new(Dropout::new(0.2, false)),
            Rc::new(Dense::new(10, ReLU)),
        ]
    };

    network.compile(Optimizer::SGD, LossFunction::MeanSquaredError, vec![Metric::Accuracy]);



}
