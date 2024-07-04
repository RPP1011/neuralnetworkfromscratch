use std::{borrow::{Borrow, BorrowMut}, rc::Rc};

// use file::idx_reader;
// use graph::{graph::{Model, Sequential}, loss_function::LossFunction, network_metric::Metric, optimizer::Optimizer};
// use layers::{dense::Dense, dropout::Dropout, flatten::Flatten};
use math::tensor::Tensor;
// use nuerons::activation_function::ActivationFunction::ReLU;

// pub mod layers;
pub mod math;
// pub mod graph;
// pub mod file;

fn main() {
    // let network = Sequential {
    //     layers: vec![
    //         Rc::new(Flatten::new(vec![28,28])),
    //         Rc::new(Dense::new(128, ReLU)),
    //         Rc::new(Dropout::new(0.2, false)),
    //         Rc::new(Dense::new(10, ReLU)),
    //     ]
    // };

    // let training_data = idx_reader::read_file("data/train-images-idx3-ubyte/train-images.idx3-ubyte").unwrap();
    // let training_labels = idx_reader::read_file("data/train-labels-idx1-ubyte/train-labels.idx1-ubyte").unwrap();

    // network.compile(Optimizer::SGD, LossFunction::MeanSquaredError, vec![Metric::Accuracy]);
    
    // let epochs = 10;
    // network.fit(training_data, training_labels, epochs)


    // let x = Tensor::new(vec![1], vec![1.0]);
    // let y = Tensor::new(vec![1], vec![1.0]);
    // let z = x + y;
    // let output = z.clone();
    // Tensor::backwards(output);
    // println!("{:?}", z);

    // visualize_tensor_graph(&z);
}
