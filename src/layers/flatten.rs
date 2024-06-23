use super::layers::layers::Layer;

pub struct Flatten {
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

impl Layer for Flatten {
    fn forward(&self) {
        todo!()
    }

    fn backward(&self) {
        todo!()
    }
}