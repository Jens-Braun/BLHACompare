#![allow(non_snake_case)]

use std::collections::HashMap;
use std::path::PathBuf;
use clap::Parser;
use color_eyre::eyre::{WrapErr, eyre};
use indicatif::ProgressBar;
use rand::prelude::*;
use rand::random;
use rand_pcg::Pcg64Mcg;
use serde::{Deserialize, Serialize};
use crate::OLP::OneLoopProvider;
use crate::Rambo::rambo;
use tempfile::tempdir;

pub mod Rambo;
pub mod OLCParser;
pub mod Report;
pub mod OLP;

/// A program to numerically compare two libraries supplying the same subprocesses through a BLHA2
/// interface.
#[derive(Parser, Serialize, Deserialize, Clone)]
pub struct CLIConfig {
    /// Path to the config file
    #[arg(required = true)]
    pub config_file: String,
    /// Number of jobs to start in parallel
    #[arg(short = 'j', long = "jobs")]
    pub n_jobs: Option<usize>,
    /// Name of the folder to put the report in
    #[arg(short = 'o', long = "report_name")]
    pub report_name: Option<String>,
    /// Seed for the internal pseudo random number generator
    #[arg(short = 's', long = "seed")]
    pub seed: Option<u64>
}

/// Configuration for the BLHACompare run. The possible configuration options are
/// - `n_points`: Number of points to sample from all subprocesses combined
/// - `outlier_threshold`: Threshold value of `(olp_1 - olp_2)/(olp_2 + olp_2)` to classify a phase space
///                        point as outlier
/// - `scale`: Energy scale of the process
/// - `masses`: List of masses to use for the RAMBO phase space generator. The total number of particles
///             is inferred from the length of `masses`
/// - `olp_1`: Configuration for the first one loop provider
/// - `olp_2`: Configuration for the second one loop provider
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct Config {
    pub n_points: usize,
    pub outlier_threshold: f64,
    pub scale: f64,
    pub masses: Vec<f64>,
    pub olp_1: OLPConfig,
    pub olp_2: OLPConfig,
}

/// Configuration for a one loop provider used in the global config. The possible configuration options are
/// - `olp_name`: Name of the one loop provider used in the generated reports
/// - `order_file`: Path to the BLHA order file
/// - `contract_file`: Path to the BLHA contract file generated from `order_file`. If the contract file
///                    is generated at runtime of BLHACompare, e.g. by calling `OLP_Start`, this field
///                    has to contain only the name of the generated file and no path
/// - `library_path`: Path to the shared library providing the BLHA interface
/// - `model_parameters`: Table of parameters to be set by `OLP_SetParameter` before sampling
/// - `permutation`: Optional permutation used to rearrange the result array of `OLP_EvalSubProcess2`
///                  in case of differing orders of the result values for different one loop providers
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct OLPConfig {
    pub olp_name: Option<String>,
    pub order_file: PathBuf,
    pub contract_file: PathBuf,
    pub library_path: PathBuf,
    pub model_parameters: HashMap<String, f64>,
    pub permutation: Option<[usize; 4]>
}

#[derive(Serialize, Deserialize)]
pub struct Outlier {
    pub momenta: Vec<[f64; 4]>,
    pub olp_1_result: [f64; 4],
    pub olp_2_result: [f64; 4],
    pub difference_result: [f64; 4]
}

/// Generate the actual sample of the subprocesses.
pub fn generate_sample(cli_config: &CLIConfig, config: &Config)
                   -> eyre::Result<(HashMap<(Vec<i32>, Vec<i32>), Vec<[f64; 4]>>,
                                    HashMap<(Vec<i32>, Vec<i32>), Vec<(Vec<[f64; 4]>, [f64; 4], [f64; 4], [f64; 4])>>)> {
    println!("[2/4] Generating sample of size {}....", &config.n_points);
    procspawn::init();
    let mut jobs: usize = std::thread::available_parallelism()?.get();

    if let Some(cli_jobs) = cli_config.n_jobs {
        if jobs > cli_jobs {
            jobs = cli_jobs;
        }
    }
    let batch_size = config.n_points / jobs ;
    { // Initialize both OLPs once to catch possible errors, since errors from the workers are not properly propagated to the main process
        let mut olp_1 = OneLoopProvider::create(&config.olp_1.order_file, &config.olp_1.library_path)
            .wrap_err("Failed to create OneLoopProvider struct for olp_1").unwrap();
        let mut olp_2 = OneLoopProvider::create(&config.olp_2.order_file, &config.olp_2.library_path)
            .wrap_err("Failed to create OneLoopProvider struct for olp_2").unwrap();
        olp_1.init(&config.olp_1.order_file, &config.olp_1.contract_file, &config.olp_1.model_parameters)
            .wrap_err("Failed to initialize olp_1").unwrap();
        olp_2.init(&config.olp_2.order_file, &config.olp_2.contract_file, &config.olp_2.model_parameters)
            .wrap_err("Failed to initialize olp_2").unwrap();

        let common_subprocesses: Vec<(Vec<i32>, Vec<i32>)> = olp_1.get_sp_table().unwrap().keys()
            .map(|key| key.clone())
            .filter(|key| olp_2.get_sp_table().unwrap().contains_key(key))
            .collect();

        if common_subprocesses.len() == 0 {
            return Err(eyre!("No common subprocesses between olp_1 and olp_2."));
        }
    }
    let mut worker_handles = vec![];

    for i in 0..jobs {
        worker_handles.push(procspawn::spawn((i, batch_size, (*cli_config).clone(), (*config).clone()),
         |(i, batch_size, cli_config, config)| {
             let mut worker_res: Vec<((Vec<i32>, Vec<i32>), [f64; 4])> = Vec::with_capacity(batch_size);
             let mut outliers: Vec<((Vec<i32>, Vec<i32>), (Vec<[f64; 4]>, [f64; 4], [f64; 4], [f64; 4]))> = Vec::new();
             let mut rng = Pcg64Mcg::seed_from_u64(cli_config.seed.unwrap_or_else(|| random()));
             rng.advance((i * batch_size) as u128);

             let tempdir = tempdir().unwrap();
             std::env::set_current_dir(&tempdir).unwrap();

             let mut olp_1 = OneLoopProvider::create(&config.olp_1.order_file, &config.olp_1.library_path).unwrap();
             let mut olp_2 = OneLoopProvider::create(&config.olp_2.order_file, &config.olp_2.library_path).unwrap();
             olp_1.init(&config.olp_1.order_file, &config.olp_1.contract_file, &config.olp_1.model_parameters).unwrap();
             olp_2.init(&config.olp_2.order_file, &config.olp_2.contract_file, &config.olp_2.model_parameters).unwrap();

             let common_subprocesses: Vec<(Vec<i32>, Vec<i32>)> = olp_1.get_sp_table().unwrap().keys()
                 .map(|key| key.clone())
                 .filter(|key| olp_2.get_sp_table().unwrap().contains_key(key))
                 .collect();

             let n_in: usize = common_subprocesses[0].0.len();

             let mut key: &(Vec<i32>, Vec<i32>);
             let mut k : Vec<[f64; 4]>;
             let mut olp_1_res: [f64; 4];
             let mut olp_2_res: [f64; 4];

             let jobs = (config.n_points / batch_size) as u64;
             let mut pb: ProgressBar = ProgressBar::new(0);
             if i == 0 {
                 pb = ProgressBar::new(config.n_points as u64);
             }

             for _ in 0..batch_size {
                 key = common_subprocesses.choose(&mut rng).unwrap();
                 k = rambo(config.scale.powi(2), &config.masses, n_in, &mut rng).wrap_err("Failed to generate phase space point").unwrap();
                 olp_1_res = olp_1.evaluate_subprocess(key, &k, config.scale)
                     .wrap_err_with(|| format!("Failed to evaluate olp_1 at phase space point {:?}", k)).unwrap();
                 olp_2_res = olp_2.evaluate_subprocess(key, &k, config.scale)
                     .wrap_err_with(|| format!("Failed to evaluate olp_2 at phase space point {:?}", k)).unwrap();
                 let mut res = [0.; 4];
                 for j in 0..4 {
                     if olp_1_res[config.olp_1.permutation.unwrap()[j]] == olp_2_res[config.olp_2.permutation.unwrap()[j]] {
                         res[j] = 0.;
                     } else {
                         res[j] = (olp_1_res[config.olp_1.permutation.unwrap()[j]] - olp_2_res[config.olp_2.permutation.unwrap()[j]])
                             / (olp_1_res[config.olp_1.permutation.unwrap()[j]] + olp_2_res[config.olp_2.permutation.unwrap()[j]])
                     }
                 }
                 if !res.iter().any(|x| x.is_nan()) {
                     worker_res.push((key.clone(), res));
                     if res.iter().any(|x| x > &config.outlier_threshold) {
                         outliers.push((key.clone(), (k, olp_1_res, olp_2_res, res)));
                     }
                 }
                 if i == 0 {pb.inc(jobs);}
             }
             if i == 0 {pb.finish_and_clear();}
             return (worker_res, outliers);
         }));
    }
    let mut result_list = vec![];
    for handle in worker_handles {
        result_list.push(handle.join().wrap_err("Worker panicked")?);
    }

    println!("[3/4] Processing sample...");
    let mut result: HashMap<(Vec<i32>, Vec<i32>), Vec<[f64; 4]>> = HashMap::new();
    let mut outliers: HashMap<(Vec<i32>, Vec<i32>), Vec<(Vec<[f64; 4]>, [f64; 4], [f64; 4], [f64; 4])>> = HashMap::new();
    for (worker_res, worker_outliers) in &result_list {
        for (key, value) in worker_res {
            if result.contains_key(key) {
                result.get_mut(key).unwrap().push(*value);
            } else {
                result.insert((*key).clone(), vec![*value]);
            }
        }
        for (key, value) in worker_outliers {
            if outliers.contains_key(key) {
                outliers.get_mut(key).unwrap().push(value.clone());
            } else {
                outliers.insert((*key).clone(), vec![value.clone()]);
            }
        }
    }
    return Ok((result, outliers));
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use super::{Config, OLPConfig};
    use std::path::PathBuf;
    #[test]
    fn config_parse_test() {
        let config_string = r#"n_points = 10_000
        outlier_threshold = 1e-3
        scale = 13e3
        masses = [0.0, 0.0, 125.0, 125.0, 0.0, 0.0]
        [olp_1]
        olp_name = "Foo"
        order_file = "foo.olp"
        contract_file = "bar.olc"
        library_path = "foobar.so"
        [olp_1.model_parameters]
        "alpha" = 0.0078186082877247849883
        "mass(23)" = 91.1876
        "width(23)" = 2.4952

        [olp_2]
        olp_name = "Bar"
        order_file = "foo2.olp"
        contract_file = "bar2.olc"
        library_path = "foobar2.so"
        permutation = [4, 2, 3, 1]
        [olp_2.model_parameters]
        "alpha" = 0.0078186082877247849883
        "mass(24)" = 79.82435974619783467
        "width(24)" = 2.085
        "#;
        let cfg: Config = toml::from_str(config_string).unwrap_or_else(|e| panic!{"{e}"});
        let cfg_ref = Config {
            n_points: 10_000,
            outlier_threshold: 1E-3,
            scale: 13_000.,
            masses: vec![0., 0., 125., 125., 0., 0.],
            olp_1: OLPConfig {
                olp_name: Some(String::from("Foo")),
                order_file: PathBuf::from("foo.olp"),
                contract_file: PathBuf::from("bar.olc"),
                library_path: PathBuf::from("foobar.so"),
                model_parameters: HashMap::from([
                    (String::from("alpha"), 0.0078186082877247849883),
                    (String::from("mass(23)"), 91.1876),
                    (String::from("width(23)"), 2.4952)
                ]),
                permutation: None
            },
            olp_2: OLPConfig {
                olp_name: Some(String::from("Bar")),
                order_file: PathBuf::from("foo2.olp"),
                contract_file: PathBuf::from("bar2.olc"),
                library_path: PathBuf::from("foobar2.so"),
                model_parameters: HashMap::from([
                    (String::from("alpha"), 0.0078186082877247849883),
                    (String::from("mass(24)"), 79.82435974619783467),
                    (String::from("width(24)"), 2.085)
                ]),
                permutation: Some([4, 2, 3, 1])
            }
        };
        assert_eq!(cfg, cfg_ref);
    }
}
