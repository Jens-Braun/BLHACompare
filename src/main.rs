#![allow(non_snake_case)]

use std::collections::HashMap;
use std::io::BufWriter;
use std::path::PathBuf;
use clap::Parser;
use color_eyre::eyre::{WrapErr, eyre};
use BLHACompare::*;
use BLHACompare::Report::{draw_boxplots, draw_histograms, write_text_report};


fn main() -> eyre::Result<()>{
    println!("[1/4] Parsing configuration...");
    let cli_config = CLIConfig::parse();
    let config_string = std::fs::read_to_string(&cli_config.config_file)
        .wrap_err_with(|| format!("Failed to read config file {}", cli_config.config_file))?;
    let mut config: Config = toml::from_str(&config_string).wrap_err("Failed to parse config file")?;
    if let Some(permutation) = config.olp_1.permutation {
        let mut perm_sorted = permutation.clone();
        perm_sorted.sort();
        if perm_sorted != [0, 1, 2, 3] {
            return Err(eyre!("{permutation:?} is not a valid permutation of [0, 1, 2, 3] for olp_1."));
        }
    } else {
        config.olp_1.permutation = Some([0, 1, 2, 3]);
    }
    if let Some(permutation) = config.olp_2.permutation {
        let mut perm_sorted = permutation.clone();
        perm_sorted.sort();
        if perm_sorted != [0, 1, 2, 3] {
            return Err(eyre!("{permutation:?} is not a valid permutation of [0, 1, 2, 3] for olp_2."));
        }
    } else {
        config.olp_2.permutation = Some([0, 1, 2, 3]);
    }
    let (sample_result, outliers) =
        generate_sample(&cli_config, &config).wrap_err("Failed to generate sample")?;
    println!("[4/4] Generating report...");
    let outdir = PathBuf::from(cli_config.report_name.unwrap_or_else(|| String::from("BLHACompare_report")));
    if !outdir.exists() {
        std::fs::create_dir(&outdir).wrap_err("Failed to create report directory")?;
    }

    write_text_report(&outdir.join("BLHACompare.report"), config.outlier_threshold, &sample_result, &outliers)
        .wrap_err("Failed to generate text report")?;
    draw_histograms(&outdir.join("Histograms.svg"), config.outlier_threshold, &sample_result, &outliers)
        .wrap_err("Failed to draw histograms")?;
    draw_boxplots(&outdir.join("Boxplots.svg"), config.outlier_threshold, &sample_result)
        .wrap_err("Failed to draw boxplots")?;

    if outliers.len() > 0 {
        let mut outlier_map = HashMap::new();
        for (key, outlier_values) in &outliers {
            let key_string = Report::process_string(&key);
            let outlier_structs: Vec<Outlier> = outlier_values.iter().map(|outlier|
                Outlier {momenta: outlier.0.clone(), olp_1_result: outlier.1.clone(),
                    olp_2_result: outlier.2.clone(), difference_result: outlier.3.clone()}
            ).collect();
            outlier_map.insert(key_string, outlier_structs);
        }
        let outlier_file = std::fs::File::create(outdir.join("outliers.json"))?;
        let outlier_writer = BufWriter::new(outlier_file);
        serde_json::to_writer_pretty(outlier_writer, &outlier_map).wrap_err("Failed to save outliers")?;
    }
    let n_outliers: usize = outliers.values().map(|val| val.len()).sum();
    println!("-{:-^80}-", " Summary ");
    println!("Sampled {} subprocesses from which {} passed and {} contained outliers.",
             sample_result.len(), sample_result.len() - outliers.len(), outliers.len());
    println!("Found {} outliers in total, averaging {} outliers per subprocess.",
             n_outliers, n_outliers as f64 / sample_result.len() as f64);
    println!("-{:-^80}-", "");
    return Ok(());
}