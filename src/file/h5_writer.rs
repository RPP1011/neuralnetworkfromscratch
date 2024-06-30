use std::{fmt::Result, fs::File, io::BufWriter};

use crate::graph::graph::Model;
use serde_json; // Add this line to import serde_json crate


pub fn save_model(model: &dyn Model, path: &str) -> Result{
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let mut encoder = serde_json::Serializer::new(&mut writer);
    // model.serialize(&mut encoder)?;
    Ok(())
}