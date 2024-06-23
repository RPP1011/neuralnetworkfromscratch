pub mod layers {
    
    pub trait Layer {
        fn forward(&self);
        fn backward(&self);  
    } 
}