use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::c_char;
use std::path::PathBuf;
use color_eyre::eyre::{WrapErr, eyre, OptionExt};
use libloading::{Library, Symbol};
use ouroboros::self_referencing;
use crate::OLCParser::{OLCParser, Subprocess};
use crate::Rambo::scalar;

/// Wrapper struct providing safe bindings to the `BLHA2` interface of a shared library providing
/// the subprocesses defined in `order_file` and `contract_file`. The enumeration of subprocesses
/// is assumed to not skip values.
#[self_referencing]
pub struct OneLoopProvider {
    lib: Library,
    subprocess_table: Option<HashMap<Subprocess, i32>>,
    max_subprocess_id: Option<i32>,
    #[borrows(lib)]
    #[covariant]
    OLP_Start:Symbol<'this, unsafe extern fn(*const c_char, *mut i32)>,
    #[borrows(lib)]
    #[covariant]
    OLP_SetParameter: Symbol<'this, unsafe extern fn(*const c_char, *const f64, *const f64, *mut i32)>,
    #[borrows(lib)]
    #[covariant]
    OLP_Evaluate: Symbol<'this, unsafe extern fn(*const i32, *const f64, *const f64, *mut f64, *mut f64)>
}

impl OneLoopProvider {
    /// Creates a new `OneLoopProvider` from `contract_file` and `library_path`.
    ///
    /// Returns an error if the shared library cannot be opened or the contract file cannot be parsed.
    pub fn create(order_file: &PathBuf, library_path: &PathBuf) -> eyre::Result<OneLoopProvider> {
        if !&order_file.try_exists().wrap_err("Failed to read the order file")? {
            return Err(eyre!("Failed to read the order file: file does not exist"));
        }
        let lib: Library;
        unsafe {
            lib = Library::new(&library_path).wrap_err("Failed to access the OLP library")?;
        }
        return Ok(OneLoopProviderBuilder {
            lib,
            subprocess_table: None,
            max_subprocess_id: None,
            OLP_Start_builder: |l| unsafe { l.get(b"OLP_Start")
                .unwrap_or_else(|e| panic!("Failed to read symbol 'OLP_Start' from {library_path:#?}: {e}")) },
            OLP_SetParameter_builder: |l| unsafe { l.get(b"OLP_SetParameter")
                .unwrap_or_else(|e| panic!("Failed to read symbol 'OLP_SetParameter' from {library_path:#?}: {e}")) },
            OLP_Evaluate_builder: |l| unsafe { l.get(b"OLP_EvalSubProcess2")
                .unwrap_or_else(|e| panic!("Failed to read symbol 'OLP_EvalSubProcess2' from {library_path:#?}: {e}")) },
        }.build());
    }

    /// Initialize the `OneLoopProvider` by running `OLP_Start` with `order_file` and setting all
    /// parameters supplied in `model_parameters`.
    ///
    /// Returns an error if the order file cannot be opened or if any of the calls to `OLP_Start`
    /// or `OLP_SetParameter` return an error.
    pub fn init(&mut self, order_file: &PathBuf, contract_file: &PathBuf, model_parameters: &HashMap<String, f64>)
        -> eyre::Result<()> {
        if !&order_file.try_exists().wrap_err("Failed to read the order file")? {
            return Err(eyre!("Failed to read the order file {:#?}: file does not exist", order_file));
        }
        self.start(order_file)?;
        let subprocess_table = OLCParser::subprocess_table_from_file(&contract_file)
            .wrap_err("Failed to parse the contract file")?;
        let max_subprocess_id: i32 = *subprocess_table.values().max().ok_or_eyre("No subprocesses found in contract file")?;
        self.with_mut( |fields| {
            *fields.subprocess_table = Some(subprocess_table);
            *fields.max_subprocess_id = Some(max_subprocess_id);
        });

        for (parameter, value) in model_parameters {
            self.set_parameter(parameter, *value)?;
        }
        return Ok(());
    }
    /// Safe wrapper to `OLP_Start`.
    pub fn start(&self, order_file: &PathBuf) -> eyre::Result<()> {
        let order_file_c = CString::new(order_file.to_str().unwrap())?;
        let mut ierr: i32 = 1;
        unsafe {
            self.borrow_OLP_Start()(order_file_c.as_ptr(), &mut ierr as *mut i32);
        }
        if ierr != 1 {return Err(eyre!("OLP_Start returned error code {ierr}"))}
        return Ok(());
    }

    /// Safe wrapper to `OLP_SetParameter`.
    pub fn set_parameter(&self, parameter: &str, value: f64) -> eyre::Result<()> {
        let parameter_c = CString::new(parameter)?;
        let mut ierr: i32 = 1;
        unsafe {
            self.borrow_OLP_SetParameter()(parameter_c.as_ptr(), &value as *const f64, &0.0 as *const f64, &mut ierr as *mut i32);
        }
        return match ierr {
            0 => Err(eyre!("Failed to set parameter {parameter} to value {value}")),
            1 => Ok(()),
            2 => Err(eyre!("Failed to set unknown parameter {parameter}")),
            _ => Err(eyre!("OLP_SetParameter returned undefined error code {ierr}"))
        };
    }

    /// Safe wrapper to `OLP_EvalSubProcess2`. This implementation explicitly assume the result
    /// array to have four entries, which makes it invalid for amplitude types like `ccTree` and
    /// `scTree`. Additionally, note that the `BLHA2` interface defines the quantities contained
    /// in the result array (the three Laurent coefficients `A2`, `A1`, `A0` and `|Born|^2`), but
    /// their order can depend on the OLP and the requested amplitude type.
    pub fn evaluate(&self, process_id: &i32, momenta: &Vec<[f64;4]>, scale: f64) -> eyre::Result<[f64; 4]> {
        if *process_id > self.borrow_max_subprocess_id().ok_or_eyre("Evaluate called before intialization")? {
            return Err(eyre!("Requested subprocess id {process_id} is larger than largest assigned id {}",
                                self.borrow_max_subprocess_id().unwrap()));
        }
        let mut momenta_flat = vec![0.0; 5 * momenta.len()];
        let mut result = [0.0; 4];
        let mut accuracy = 0.0;
        for (i, momentum) in momenta.iter().enumerate() {
            for j in 0..=4 {
                momenta_flat[5 * i + j] = if j == 4 {
                    scalar(momentum, momentum)
                } else {
                    momentum[j]
                }
            }
        }
        unsafe {
            self.borrow_OLP_Evaluate()(
                process_id as *const i32,
                momenta_flat.as_ptr(),
                &scale as *const f64,
                &mut result as *mut f64,
                &mut accuracy as *mut f64
            )
        }
        return Ok(result);
    }

    /// Evaluate the subprocess specified by `process_spec` at the phase space point `momenta` with
    /// scale `scale`.
    /// The expected process spec is `(initial, final)` where `initial` and `final` are vectors of
    /// PDG codes of the incoming and outgoing particles.
    pub fn evaluate_subprocess(&self, process_spec: &Subprocess, momenta: &Vec<[f64; 4]>, scale: f64)
        -> eyre::Result<[f64; 4]>{
        let process_id = self.borrow_subprocess_table().as_ref()
            .ok_or_eyre("Evaluate_subprocess called before initialization")?.get(&process_spec)
            .ok_or_eyre("Requested subprocess {process_spec:?} not found in process table")?;
        return self.evaluate(process_id, momenta, scale);
    }

    /// Get reference to the subprocess table if it exists, error otherwise.
    pub fn get_sp_table(&self) -> eyre::Result<&HashMap<Subprocess, i32>> {
        return self.borrow_subprocess_table().as_ref().ok_or_eyre("Get_sp_table called before initialization");
    }
}