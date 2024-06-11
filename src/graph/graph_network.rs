mod graph {
    struct Node {
        pub activation_function: Box<dyn Fn(f64) -> f64>,
    }

    struct Layer {
        pub layer_size : u16,
        pub nodes : Vec<Node>,
        pub biases : Vec<f64>,
        pub weights : Vec<f64>
    }

    fn activate(node: &Node, input: Vec<f64>, weights:Vec<f64>, bias: f64) -> f64 {
        let mut sum : f64 = 0.0;
        for i in 0..input.len() {
            sum += input[i] * weights[i];
        }
        sum += bias;
        (*(*node).activation_function)(sum)
    }
}