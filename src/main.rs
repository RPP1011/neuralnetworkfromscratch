use std::{borrow::{Borrow, BorrowMut}, cell::RefCell, rc::Rc};

// use file::idx_reader;
use graph::{graph::{Model, Sequential}, loss_function::LossFunction, network_metric::Metric, optimizer::Optimizer};
use layers::{dense::Dense, dropout::Dropout, flatten::Flatten};
use math::{tensor::Tensor, tensor_context};
use nuerons::activation_function::ActivationFunction;

pub mod layers;
pub mod math;
pub mod graph;
pub mod file;
pub mod nuerons;

fn main() {
    let tensor_context = create_tensor_context!(1024);
    let network = Sequential {
        context: tensor_context.clone(),
        layers:vec![
            Rc::new(Flatten::new(vec![28,28])),
            Rc::new(Dense::new(tensor_context.clone(), 128, ActivationFunction::ReLU)),
            Rc::new(Dropout::new(0.2, false)),
            Rc::new(Dense::new(tensor_context.clone(), 10, ActivationFunction::ReLU)),
        ]
    };

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
