mod micrograd {
    use std::borrow::BorrowMut;
    use std::rc::{Rc};
    
    #[derive(Clone)]
    pub enum Operation {
        ADD,
        MUL,
        POW,
    }

    impl Operation {
        pub fn forward(&self, a: Rc<Value>, b: Rc<Value>) -> f64 {
            match self {
                Operation::ADD => a.clone().data + b.clone().data ,
                Operation::MUL => a.clone().data * b.clone().data,
                Operation::POW => a.clone().data.powf(b.clone().data),
            }
        }

        pub fn backwards(&self, a: Rc<Value>, b: Rc<Value>, out: Rc<Value>) {
            match self {
                Operation::ADD => {
                    a.clone().borrow_mut().grad += out.clone().grad;
                    b.clone().borrow_mut().grad += out.clone().grad;
                },
                Operation::MUL => {
                    a.clone().borrow_mut().grad += b.clone().data * out.clone().grad;
                    b.clone().borrow_mut().grad += a.clone().data * out.clone().grad;
                },
                Operation::POW => {
                    a.clone().borrow_mut().grad += b.clone().data * a.clone().data.powf(b.clone().data - 1.0) * out.clone().grad;
                    b.clone().borrow_mut().grad += a.clone().data.powf(b.clone().data) * out.clone().grad * (a.clone().data.ln());
                },
            }
        }
    }

    pub enum CreateValueOption {
        EMPTY,
        DATA(Value),
    }
   
    pub struct Graph {
        pub values: Vec<Rc<Value>>,
        pub layers: Vec<Layer>
    }

    impl Graph {
        pub fn add_value(&mut self, value: CreateValueOption) -> Rc<Value>{
            let out = Rc::new(Value {
                data: 0.0,
                grad: 0.0,
                index: self.values.len(),
                previous: vec![],
                op: Operation::ADD,
            });

            match value {
                CreateValueOption::EMPTY => {
                    self.values.push(out.clone());
                },
                CreateValueOption::DATA(value) => {
                    self.values.push(Rc::new(value));
                }          
            };
            
            out.clone()
        }

        pub fn add_layer(&mut self, n:usize) {
            for _ in 0..n {
                self.add_value(CreateValueOption::EMPTY);
            }

            let layer = Layer {
                values: vec![],
                weights: vec![],
                biases: vec![],
            };

            self.layers.push(layer);
        }
    }

    pub struct Layer {
        pub values: Vec<usize>,
        pub weights: Vec<f64>,
        pub biases: Vec<f64>,
    }

    #[derive(Clone)]
    pub struct Value {
        pub data: f64,
        pub grad: f64,
        pub index: usize,
        pub previous: Vec<usize>,
        pub op: Operation,
    }

    pub fn add(graph: Rc<Graph>, a:Rc<Value>, b:Rc<Value>) -> Rc<Value> {
        let mut graph_reference = graph.clone();

        let out = Value {
            data: graph_reference.values[a.index].data + graph_reference.values[b.index].data,
            grad: 0.0,
            index: graph_reference.values.len(),
            previous: vec![a.index, b.index],
            op: Operation::ADD,
        };

        graph_reference.borrow_mut().add_value(CreateValueOption::DATA(out.clone()))
    }

    pub fn mul(graph: Rc<Graph>, a:Rc<Value>, b:Rc<Value>) -> Rc<Value> {
        let mut graph_reference = graph.clone();

        let out = Value {
            data: graph_reference.values[a.index].data + graph_reference.values[b.index].data,
            grad: 0.0,
            index: graph_reference.values.len(),
            previous: vec![a.index, b.index],
            op: Operation::MUL,
        };

        graph_reference.borrow_mut().add_value(CreateValueOption::DATA(out.clone()))
    }

    pub fn pow(graph: Rc<Graph>, a:Rc<Value>, b:Rc<Value>) -> Rc<Value> {
        let mut graph_reference = graph.clone();

        let out = Value {
            data: graph_reference.values[a.index].data + graph_reference.values[b.index].data,
            grad: 0.0,
            index: graph_reference.values.len(),
            previous: vec![a.index, b.index],
            op: Operation::POW,
        };

        graph_reference.borrow_mut().add_value(CreateValueOption::DATA(out.clone()))
    }
}
