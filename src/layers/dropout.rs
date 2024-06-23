use super::layers::layers::Layer;


pub struct Dropout {
    pub rate: f32,
    pub training: bool,
}

impl Layer for Dropout {
    fn forward(&self)  {
        todo!("Implement dropout Forward pass")
    }

    fn backward(&self) {
        todo!("Implement dropout backward pass")
    }
}