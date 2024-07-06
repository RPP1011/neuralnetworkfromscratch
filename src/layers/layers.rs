pub mod layers {
    

    use crate::math::{tensor_context::{TensorRef}};

    
    pub trait Layer {
        fn forward(&self, input: TensorRef) -> TensorRef;
        fn compile(&mut self, input: TensorRef) -> TensorRef;
        fn get_parameters(&self) -> Vec<TensorRef>;
    } 
}