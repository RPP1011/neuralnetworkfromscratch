pub mod layers {
    use crate::math::tensor::Tensor;

    
    pub trait Layer {
        fn forward(&self, input: Tensor) -> Tensor;
        fn backward(&self, input: Tensor) -> Tensor;  
    } 
}