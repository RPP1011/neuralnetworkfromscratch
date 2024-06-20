mod micrograd {
    use std::borrow::{Borrow, BorrowMut};
    use std::cell::RefCell;
    use std::rc::{Rc};
    
    #[derive(Clone)]
    pub enum Operation {
        ADD,
        MUL,
    }

    impl Operation {
        pub fn forward(&self, a: Rc<RefCell<Value>>, b: Rc<RefCell<Value>>) -> f64 {
            let a_reference : &RefCell<Value> = a.borrow();
            let a_data = a_reference.borrow().data;
            
            let b_reference : &RefCell<Value> = b.borrow();
            let b_data = b_reference.borrow().data;
            
            match self {
                Operation::ADD => a_data + b_data,
                Operation::MUL => a_data * b_data
            }
        }

        pub fn backwards(&self,mut a: Rc<RefCell<Value>>, mut b: Rc<RefCell<Value>>, out: Rc<RefCell<Value>>) {
            let a_binding = a.borrow_mut();
            let b_binding = b.borrow_mut();
            let mut a_mut_reference = a_binding.as_ref().borrow_mut();
            let mut b_mut_reference = b_binding.as_ref().borrow_mut();
            
            let out_ref : &RefCell<Value> = out.borrow();
            let out_grad : f64 = out_ref.borrow().grad;


            match self {
                Operation::ADD => {
                    a_mut_reference.grad += out_grad;
                    b_mut_reference.grad += out_grad;
                },
                Operation::MUL => {
                    a_mut_reference.grad += b_mut_reference.data * out_grad;
                    b_mut_reference.grad += a_mut_reference.data * out_grad;
                }
            }
        }
    }

    pub enum CreateValueOption {
        EMPTY,
        DATA(Value),
    }
   
    pub struct Graph {
        pub values: Vec<Rc<RefCell<Value>>>,
        pub layers: Vec<Layer>
    }

    impl Graph {
        pub fn add_value(&mut self, value: CreateValueOption) -> Rc<RefCell<Value>>{
            match value {
                CreateValueOption::EMPTY => {
                    let out = Rc::new(RefCell::new(Value {
                        data: 0.0,
                        grad: 0.0,
                        index: self.values.len(),
                        previous: vec![],
                        op: Operation::ADD,
                    }));

                    self.values.push(out.clone());
                    out
                },
                CreateValueOption::DATA(node) => {
                    let out = Rc::new(RefCell::new(node.clone()));
                    self.values.push(out.clone());
                    out
                }          
            }
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

    pub fn add(graph: Rc<RefCell<Graph>>, a:Rc<RefCell<Value>>, b:Rc<RefCell<Value>>) -> Rc<RefCell<Value>> {
        
        let values: usize = graph.as_ref().borrow().values.len();

        

        let a_index = a.as_ref().borrow().index;
        let b_index = b.as_ref().borrow().index;

        let a_data = a.as_ref().borrow().data;
        let b_data = b.as_ref().borrow().data;

        let out = Value {
            data: a_data + b_data,
            grad: 0.0,
            index: values,
            previous: vec![a_index, b_index],
            op: Operation::ADD,
        };

        let graph_reference: Rc<RefCell<Graph>> = graph.clone();
        let ref_graph = graph_reference.as_ref();
        let mut mut_graph_reference = ref_graph.borrow_mut();
        mut_graph_reference.add_value(CreateValueOption::DATA(out.clone()))
    }

    pub fn mul(graph: Rc<RefCell<Graph>>, a:Rc<RefCell<Value>>, b:Rc<RefCell<Value>>) -> Rc<RefCell<Value>> {
        
        let values: usize = graph.as_ref().borrow().values.len();
        let a_index = a.as_ref().borrow().index;
        let b_index = b.as_ref().borrow().index;

        let a_data = a.as_ref().borrow().data;
        let b_data = b.as_ref().borrow().data;

        let out = Value {
            data: a_data * b_data,
            grad: 0.0,
            index: values,
            previous: vec![a_index, b_index],
            op: Operation::MUL,
        };

        let graph_reference: Rc<RefCell<Graph>> = graph.clone();

        let mut mut_graph_reference = graph_reference.as_ref().borrow_mut();
        mut_graph_reference.add_value(CreateValueOption::DATA(out.clone()))
    }
}


// Add Tests
#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use crate::micrograd::micrograd::micrograd::{add, mul, CreateValueOption, Graph, Operation, Value};

    #[test]
    fn test_add() {
        let mut graph = Graph {
            values: vec![],
            layers: vec![],
        };

        let a = graph.add_value(CreateValueOption::DATA(Value {
            data: 2.0,
            grad: 0.0,
            index: 0,
            previous: vec![],
            op: Operation::ADD,
        }));

        let b = graph.add_value(CreateValueOption::DATA(Value {
            data: 3.0,
            grad: 0.0,
            index: 1,
            previous: vec![],
            op: Operation::ADD,
        }));

        let out = add(Rc::new(RefCell::new(graph)), a, b);

        let out_data = out.as_ref().borrow().data;
        assert_eq!(out_data, 5.0);
    }

    #[test]
    fn test_mul() {
        let mut graph = Graph {
            values: vec![],
            layers: vec![],
        };

        let a = graph.add_value(CreateValueOption::DATA(Value {
            data: 2.0,
            grad: 0.0,
            index: 0,
            previous: vec![],
            op: Operation::ADD,
        }));

        let b = graph.add_value(CreateValueOption::DATA(Value {
            data: 3.0,
            grad: 0.0,
            index: 1,
            previous: vec![],
            op: Operation::ADD,
        }));

        let out = mul(Rc::new(RefCell::new(graph)), a, b);

        let out_data = out.as_ref().borrow().data;
        assert_eq!(out_data, 6.0);
    }
}