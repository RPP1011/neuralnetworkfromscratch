use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    rc::Rc,
};

use file::idx_reader;
// use file::idx_reader;
use graph::{
    graph::{Model, Sequential},
    loss_function::LossFunction,
    network_metric::Metric,
    optimizer::Optimizer,
};
use layers::{dense::Dense, dropout::Dropout, flatten::Flatten, layers::layers::Layer};
use math::{tensor::Tensor, tensor_context};
use nuerons::activation_function::ActivationFunction;

pub mod file;
pub mod graph;
pub mod layers;
pub mod math;
pub mod nuerons;

fn main() {
    let tensor_context = create_tensor_context!(1024);
    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(Flatten::new(tensor_context.clone(), vec![28, 28])),
        Box::new(Dense::new(tensor_context.clone(), 128, ActivationFunction::ReLU)),
        // Box::new(Dropout::new(tensor_context.clone(), 0.2, false)),
        // Box::new(Dense::new(tensor_context.clone(), 10, ActivationFunction::ReLU)),
    ];
    let mut network = Sequential::new(tensor_context.clone(), layers);

    let training_data = idx_reader::read_file("data/train-images-idx3-ubyte/train-images.idx3-ubyte").unwrap();
    // let training_labels = idx_reader::read_file("data/train-labels-idx1-ubyte/train-labels.idx1-ubyte").unwrap();

    network.compile(
        vec![28, 28],
        vec![10],
        Optimizer::SGD,
        LossFunction::MeanSquaredError,
        vec![Metric::Accuracy],
    );

    // let data = tensor_context.borrow_mut(). vec![28, 28], training_data[0].clone());
    
    println!("{:?}", network.predict(training_data.data[0..28*28].to_vec()))

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
