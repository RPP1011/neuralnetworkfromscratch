mod micrograd {
    use std::rc::Weak;

    pub struct Value {
        pub data: f64,
        pub grad: f64,

        pub backwards : fn(Weak<Value>, Weak<Value>),
        pub previous : std::vec::Vec<Weak<Value>>
    }

    impl Value {
        pub fn add(&self, other: &Value) -> Value {
            unsafe {  
            let previous =  vec![Weak::from_raw(self), Weak::from_raw(other)];
            return Value {
                data: self.data + other.data,
                grad: 0.0,
                backwards: |self_ref: Weak<Value>, other_ref: Weak<Value>|  {
                    self_ref.upgrade().unwrap().grad += other_ref.upgrade().unwrap().grad;
                },
                previous
            }}
        }

        pub fn mul(&self, other: &Value) -> Value {
            return Value {
                data: self.data * other.data
            }
        }

        pub fn repr(&self) -> String {
            return format!("Value({})", self.data);
        }
    }

    impl Iterator for Value {
        type Item = Value;
    
        fn next(&mut self) -> Option<Self::Item> {
            todo!()
        }
    }
}