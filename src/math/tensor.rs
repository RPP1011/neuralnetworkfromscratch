
#[derive(Debug)]
#[derive(Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}