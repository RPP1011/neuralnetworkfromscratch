mod micrograd {
    use std::cell::RefCell;
    use std::rc::{Rc, Weak};
    pub enum Operation {
        ADD,
        MUL,
        POW,
    }
    pub struct Value {
        pub data: f64,
        pub grad: f64,

        pub backwards: Option<Box<dyn Fn()>>,
        pub previous: Vec<Weak<RefCell<Value>>>,
        pub op: Operation,
    }

    impl Value {
        pub fn add(&self, other: &Rc<RefCell<Self>>) -> Rc<RefCell<Value>> {
            let previous = vec![
                Rc::downgrade(&Rc::new(RefCell::new(self.clone()))),
                Rc::downgrade(&Rc::new(other.borrow())),
            ];
            let out = Rc::new(RefCell::new(Value {
                data: self.data + other.borrow().data,
                grad: 0.0,
                backwards: None,
                previous,
                op: Operation::ADD,
            }));

            let out_clone = Rc::clone(&out);
            let self_clone = Rc::downgrade(&Rc::new(RefCell::new(self.clone())));
            let other_clone = Rc::downgrade(other);

            out.borrow_mut().backwards = Some(Box::new(move || {
                let out = out_clone.borrow();
                if let Some(self_strong) = self_clone.upgrade() {
                    self_strong.borrow_mut().grad += out.grad;
                }
                if let Some(other_strong) = other_clone.upgrade() {
                    other_strong.borrow_mut().grad += out.grad;
                }
            }));

            out
        }

        pub fn mul(&self, other: &Value) -> Value {
            unsafe {
                let previous = vec![Weak::from_raw(self), Weak::from_raw(other)];
                let mut out = Value {
                    data: self.data * other.data,
                    grad: 0.0,
                    backwards: None,
                    previous,
                    op: Operation::MUL,
                };

                out.backwards = Some(Box::new(move || -> () {
                    self.grad *= other.grad * out.grad;
                    other.grad *= self.grad * out.grad;
                }));

                return out;
            }
        }

        pub fn repr(&self) -> String {
            return format!("Value({})", self.data);
        }
    }
}
