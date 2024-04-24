#![allow(dead_code)]
use std::path::PathBuf;
use std::collections::HashMap;
use color_eyre::eyre::{WrapErr};
use std::io::Write;
use std::time::Duration;
use plotters::prelude::*;
use crate::OLCParser::Subprocess;
use crate::{Outlier, PSPResult};

const GREY: RGBColor = RGBColor(158, 158, 158);

fn mean_var(values: Vec<f64>) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / (values.len() as f64);
    let var = values.iter().map(|x| (x - mean.clone()) * (x - mean.clone())).sum::<f64>() / (values.len() as f64 - 1f64);
    return (mean, var);
}

fn mean(values: Vec<f64>) -> f64 {
    let mean = values.iter().sum::<f64>() / (values.len() as f64);
    return mean;
}

fn var(values: Vec<f64>) -> f64 {
    let mean = values.iter().sum::<f64>() / (values.len() as f64);
    let var = values.iter().map(|x| (x - mean.clone()) * (x - mean.clone())).sum::<f64>() / (values.len() as f64 - 1f64);
    return var;
}

fn nested_mean(values: &Vec<Vec<f64>>) -> f64 {
    return values.iter().map(|vec| vec.iter().sum::<f64>()).sum::<f64>()
        / values.iter().map(|vec| vec.len() as f64).sum::<f64>();
}

fn nested_var(values: &Vec<Vec<f64>>) -> f64 {
    let mean = nested_mean(values);
    let var = values.iter().map(|vec|
        vec.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>()
    ).sum::<f64>() / (values.iter().map(|vec| vec.len() as f64).sum::<f64>() - 1f64);
    return var;
}

pub fn nested_std_dev(values: &Vec<Vec<f64>>) -> f64 {
    return nested_var(values).sqrt();
}

/// Write a text report to `outfile_path` that contains the result of the full sample and only the outliers
/// for each subprocess
pub fn write_text_report(outfile_path: &PathBuf,
                         outlier_threshold: f64,
                         result: &HashMap<Subprocess, Vec<PSPResult>>,
                         outliers: &HashMap<Subprocess, Vec<Outlier>>)
    -> eyre::Result<()> {
    let outfile = std::fs::File::create(outfile_path)
        .wrap_err("Failed to create or access output file")?;
    let mut outfile_writer = std::io::BufWriter::new(outfile);

    let total_mean: Vec<f64> = (0..4).map(|i| result.values().map(|val|
        val.iter().map(|x| x.value[i]).sum::<f64>()).sum::<f64>() /
        result.values().map(|val| val.len() as f64).sum::<f64>()).collect();
    let total_variance: Vec<f64> = (0..4).map(|i|
        result.values().map(|val| val.iter().map(|x| (x.value[i] - total_mean[i]).powi(2)).sum::<f64>()).sum::<f64>()
         / (result.values().map(|val| val.len() as f64).sum::<f64>() - 1.)
    ).collect();
    let total_std_deviation: Vec<f64> = total_variance.iter().map(|x| x.sqrt()).collect();
    let olp_1_time_mean = nested_mean(&result.values().map(|vec|
        vec.iter().map(|x| x.olp_1_time.as_secs_f64()).collect()).collect());
    let olp_2_time_mean = nested_mean(&result.values().map(|vec|
        vec.iter().map(|x| x.olp_2_time.as_secs_f64()).collect()).collect());
    let olp_1_time_std_dev = nested_std_dev(&result.values().map(|vec|
        vec.iter().map(|x| x.olp_1_time.as_secs_f64()).collect()).collect());
    let olp_2_time_std_dev = nested_std_dev(&result.values().map(|vec|
        vec.iter().map(|x| x.olp_2_time.as_secs_f64()).collect()).collect());

    writeln!(&mut outfile_writer, "={:=^120}=", " Summary of all subprocesses ")?;
    for i in 0..4 {
        writeln!(&mut outfile_writer, "Result component {i}: {} ± {}", total_mean[i], total_std_deviation[i])?;
    }
    writeln!(&mut outfile_writer, "Average time per phase space point for OLP1: {:?} ± {:?}",
             Duration::from_secs_f64(olp_1_time_mean), Duration::from_secs_f64(olp_1_time_std_dev))?;
    writeln!(&mut outfile_writer, "Average time per phase space point for OLP2: {:?} ± {:?}",
             Duration::from_secs_f64(olp_2_time_mean), Duration::from_secs_f64(olp_2_time_std_dev))?;
    if olp_1_time_mean > olp_2_time_mean {
        writeln!(&mut outfile_writer, "OLP2 was, on average, {:?} ± {:?} times faster than OLP1",
            olp_1_time_mean / olp_2_time_mean,
            f64::sqrt((olp_1_time_std_dev / olp_2_time_mean).powi(2)
                + olp_1_time_mean.powi(2) * (olp_2_time_std_dev / olp_2_time_mean))
        )?;
    } else {
        writeln!(&mut outfile_writer, "OLP1 was, on average, {:?} ± {:?} times faster than OLP2",
                 olp_2_time_mean / olp_1_time_mean,
                 f64::sqrt((olp_2_time_std_dev / olp_1_time_mean).powi(2)
                     + olp_2_time_mean.powi(2) * (olp_1_time_std_dev / olp_1_time_mean))
        )?;
    }

    for key in result.keys() {

        let data = result.get(key).unwrap();
        let outlier_data = outliers.get(key);
        let n_outliers = match outlier_data {
            Some(vec) => vec.len(),
            None => 0
        };

        let mean: Vec<f64> = (0..4).map(|i| data.iter().map(|x| x.value[i]).sum::<f64>() / (data.len() as f64)).collect();
        let variance: Vec<f64> = (0..4).map(|i|
            data.iter().map(|x| (x.value[i] - mean[i]).powi(2)).sum::<f64>() / (data.len() as f64 - 1.)
        ).collect();
        let std_deviation: Vec<f64> = variance.iter().map(|x| x.sqrt()).collect();
        let (olp_1_time_mean, olp_1_time_var) = mean_var(data.iter().map(|x| x.olp_1_time.as_secs_f64()).collect());
        let (olp_2_time_mean, olp_2_time_var) = mean_var(data.iter().map(|x| x.olp_2_time.as_secs_f64()).collect());



        writeln!(&mut outfile_writer, "={:=^120}=", format!(" Subprocess {}", key.to_string()))?;
        for i in 0..4 {
            writeln!(&mut outfile_writer, "Result component {i}: {} ± {}", mean[i], std_deviation[i])?;
        }
        if outliers.len() > 0 {
            writeln!(&mut outfile_writer, "Found {} outliers in {} points with absolute deviations", n_outliers, data.len())?;
        } else {
            writeln!(&mut outfile_writer, "Found {} outliers in {} points.", n_outliers, data.len())?;
        }
        writeln!(&mut outfile_writer, "Average time per phase space point for OLP1: {:?} ± {:?}",
                 Duration::from_secs_f64(olp_1_time_mean), Duration::from_secs_f64(olp_1_time_var.sqrt()))?;
        writeln!(&mut outfile_writer, "Average time per phase space point for OLP2: {:?} ± {:?}",
                 Duration::from_secs_f64(olp_2_time_mean), Duration::from_secs_f64(olp_2_time_var.sqrt()))?;
        if olp_1_time_mean > olp_2_time_mean {
            writeln!(&mut outfile_writer, "OLP2 was, on average, {:?} ± {:?} times faster than OLP1",
                     olp_1_time_mean / olp_2_time_mean,
                     f64::sqrt((olp_1_time_var.sqrt() / olp_2_time_mean).powi(2)
                         + olp_1_time_mean.powi(2) * (olp_2_time_var.sqrt() / olp_2_time_mean))
            )?;
        } else {
            writeln!(&mut outfile_writer, "OLP1 was, on average, {:?} ± {:?} times faster than OLP2",
                     olp_2_time_mean / olp_1_time_mean,
                     f64::sqrt((olp_2_time_var.sqrt() / olp_1_time_mean).powi(2)
                         + olp_2_time_mean.powi(2) * (olp_1_time_var.sqrt() / olp_1_time_mean))
            )?;
        }

        match outlier_data {
            Some(d) => {
                let mut outlier_abs_mean: [f64; 4] = [0.0; 4];
                let mut outlier_variance: [f64; 4] = [0.0; 4];
                let mut outlier_std_deviation:[f64; 4] = [0.0; 4];
                for i in 0..4 {
                    let component_outliers: Vec<f64> = d.iter()
                        .filter(|x| x.difference_result[i] > outlier_threshold)
                        .map(|x| x.difference_result[i])
                        .collect();
                    outlier_abs_mean[i] = component_outliers.iter().sum::<f64>() / (component_outliers.len() as f64);
                    outlier_variance[i] = component_outliers.iter()
                        .map(|x| (x - outlier_abs_mean[i]).powi(2)).sum::<f64>()/ (component_outliers.len() as f64 - 1.);
                    outlier_std_deviation[i] = outlier_variance[i].sqrt();
                }
                for i in 0..4 {
                    writeln!(&mut outfile_writer, "Outlier absolute deviation component {i}: {} ± {}",
                             outlier_abs_mean[i], outlier_std_deviation[i])?;
                }
            }
            None => ()
        }
    }
    return Ok(());
}

/// Draw histograms for all components for each subprocess that contains at least one component with outliers
pub fn draw_histograms(outfile_path: &PathBuf,
                       outlier_threshold: f64,
                       result: &HashMap<Subprocess, Vec<PSPResult>>,
                       outliers: &HashMap<Subprocess, Vec<Outlier>>)
                       -> eyre::Result<()> {
    let canvas = SVGBackend::new(outfile_path, (4 * 640, outliers.len() as u32 * 360)).into_drawing_area();
    let areas = canvas.split_evenly((outliers.len(), 4));
    let mut charts = Vec::with_capacity(4 * outliers.len());
    let mut hists = Vec::with_capacity(4 * outliers.len());
    for (i, key) in outliers.keys().enumerate() {
        for j in 0..4 {
            let key_string = key.to_string();
            let component_data: Vec<f64> = result.get(key).unwrap().iter().map(|x| x.value[j]).collect();
            let data_max: f64 = *component_data.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
            let data_min: f64 = *component_data.iter().min_by(|x, y| x.total_cmp(y)).unwrap();
            let x_spec;
            let x_range;
            let style;
            if data_max == data_min {
                x_spec = (-1f64..1f64).step(1./50.).use_round().into_segmented();
                x_range = -1f64..1f64;
            } else {
                x_spec = (data_min..data_max).step((data_max - data_min)/50.).use_round().into_segmented();
                x_range = data_min..data_max;
            }
            if component_data.iter().any(|x| x.abs() > outlier_threshold) {
                style = RED.mix(0.8).filled();
            } else {
                style = GREEN.mix(0.8).filled();
            }
            let mut chart = ChartBuilder::on(&areas[4 * i + j])
                .caption(format!("Component {j} of subprocess {key_string}"), ("sans-serif", 20).into_font())
                .x_label_area_size(40)
                .y_label_area_size(40)
                .build_cartesian_2d(x_spec.clone(),
                                    (0u64..component_data.len() as u64).log_scale())?
                .set_secondary_coord(x_spec.clone(),
                                     (0u64..component_data.len() as u64).log_scale());
            chart.configure_mesh().disable_x_mesh().y_labels(10).draw()?;
            let chart2 = ChartBuilder::on(&areas[4 * i + j])
                .caption(format!("Component {j} of subprocess {key_string}"), ("sans-serif", 20).into_font())
                .x_label_area_size(40)
                .y_label_area_size(40)
                .build_cartesian_2d(x_range,
                                    (0u64..component_data.len() as u64).log_scale())?;
            let mut hist = Histogram::vertical(chart.borrow_secondary())
                .style(style).margin(0)
                .data(component_data.iter().map(|x| (*x, 1)));
            chart2.plotting_area().draw(
                &Rectangle::new([(- outlier_threshold, 0u64), (outlier_threshold, component_data.len() as u64)],
                                GREY.mix(0.4).filled())
            )?;
            chart.draw_series(&mut hist)?;
            charts.push(chart);
            hists.push(hist);
        }
    }
    return Ok(());
}

/// Draw one boxplot for each component containing all subprocesses
pub fn draw_boxplots(outfile_path: &PathBuf,
                     outlier_threshold: f64,
                     result: &HashMap<Subprocess, Vec<PSPResult>>)
                     -> eyre::Result<()> {
    let canvas = SVGBackend::new(outfile_path, (4 * 680, 80 + result.len() as u32 * 25)).into_drawing_area();
    let areas = canvas.split_evenly((1, 4));
    let mut charts = Vec::with_capacity(4);
    let keys: Vec<String> = result.keys().map(|k| k.to_string()).collect();
    let max: Vec<f64> = (0..4).into_iter().map(|i|
        result.values().map(|val| val.iter().map(|x| x.value[i]).max_by(|x, y| x.total_cmp(&y)).unwrap())
            .max_by(|x, y| x.total_cmp(&y)).unwrap()
    ).collect();
    let min: Vec<f64> = (0..4).into_iter().map(|i|
        result.values().map(|val| val.iter().map(|x| x.value[i]).min_by(|x, y| x.total_cmp(&y)).unwrap())
            .min_by(|x, y| x.total_cmp(&y)).unwrap()
    ).collect();
    for i in 0..4 {
        let x_range;
        let mut x_scale_factor = 1;
        let mut x_shift_factor = 0.0;
        if min[i] == max[i] {
            x_range = -1.0..1.0;
        } else {
            x_range = (min[i] - 0.05 * (max[i] - min[i]))..(max[i] + 0.05 * (max[i] - min[i]));
            if max[i] - min[i] < 1E-4 {
                x_scale_factor = (-(max[i] - min[i]).log10()).floor() as u32;
                if (max[i] - min[i]) < 1E-2 * (max[i] + min[i]) {
                    x_shift_factor = (max[i] + min[i]) / 2.;
                }
            }
        }

        let mut chart = ChartBuilder::on(&areas[i])
            .x_label_area_size(40)
            .y_label_area_size(120)
            .caption(format!("Boxplot for component {i}"), ("sans-serif", 20).into_font())
            .build_cartesian_2d(x_range.clone(), keys[..].into_segmented())?;

        if x_scale_factor != 1 {
            chart.configure_mesh()
                .x_desc("(olp_1 - olp_2)/(olp_1 + olp_2)")
                .y_desc("")
                .y_labels(keys.len())
                .x_label_formatter(&|x| {
                    format!("{:.4}", (x  - x_shift_factor) * 10i32.pow(x_scale_factor) as f64)
                })
                .draw()?;
        } else {
            chart.configure_mesh()
                .x_desc("(olp_1 - olp_2)/(olp_1 + olp_2)")
                .y_desc("")
                .y_labels(keys.len())
                .draw()?
        }

        chart.draw_series(result.iter().map(|(key, value)| {
            let key_string: &String = &keys.iter().filter(|s| **s == key.to_string()).collect::<Vec<&String>>()[0];
            let component_data: Vec<f64> = value.iter().map(|x| x.value[i]).collect();
            let mean = component_data.iter().sum::<f64>() / (component_data.len() as f64);
            let std_dev = (component_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / (component_data.len() as f64 - 1.)).sqrt();
            let style;
            let data_max: f64 = *component_data.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
            let data_min: f64 = *component_data.iter().min_by(|x, y| x.total_cmp(y)).unwrap();
            if component_data.iter().any(|x| x.abs() > outlier_threshold) {
                style = RED;
            } else {style = GREEN;}
            let bp = CustomBoxplot::CustomBoxplot::new(SegmentValue::CenterOf(key_string),
                                                       [data_min, mean - std_dev, mean, mean + std_dev, data_max])
                .width(20)
                .whisker_width(1.)
                .style(style);
            return bp;
        }))?;
        let chart2 = ChartBuilder::on(&areas[i])
            .x_label_area_size(40)
            .y_label_area_size(120)
            .caption(format!("Boxplot for component {i}"), ("sans-serif", 20).into_font())
            .build_cartesian_2d(x_range.clone(), 0f64..1f64)?;
        chart2.plotting_area().draw(
            &Rectangle::new([(- outlier_threshold, 0f64), (outlier_threshold, 1f64)],
                            GREY.mix(0.4).filled())
        )?;
        if x_scale_factor != 1 {
            chart2.plotting_area().draw(&(
                EmptyElement::at((min[i] - 0.15 * (max[i] - min[i]), -0.5)) +
                MultiLineText::from_string(format!("x10^(-{})\n {:+.2}", x_scale_factor, x_shift_factor * 10i32.pow(x_scale_factor) as f64),
                          (-60, 10), ("", 13).into_font(), 100)))?;
        }
        charts.push(chart);
    }

    return Ok(());
}

mod CustomBoxplot {
    #![allow(dead_code)]
    //! Modified version of plotters::element::Boxplot that allows custom whisker and box sizes
    use plotters_backend::{BackendCoord, DrawingBackend, DrawingErrorKind};
    use plotters::element::{Drawable, PointCollection};
    use plotters::style::{Color, ShapeStyle, BLACK};
    pub(crate) struct CustomBoxplot<K> {
        style: ShapeStyle,
        width: u32,
        whisker_width: f64,
        offset: f64,
        key: K,
        values: [f64; 5]
    }

    impl<K> CustomBoxplot<K> {
        pub fn style<S: Into<ShapeStyle>>(mut self, style: S) -> Self {
            self.style = style.into();
            self
        }
        pub fn width(mut self, width: u32) -> Self {
            self.width = width;
            self
        }
        pub fn whisker_width(mut self, whisker_width: f64) -> Self {
            self.whisker_width = whisker_width;
            self
        }
        pub fn offset<T: Into<f64> + Copy>(mut self, offset: T) -> Self {
            self.offset = offset.into();
            self
        }
    }

    impl<K> CustomBoxplot<K> {
        pub fn new(key: K, values: [f64; 5]) -> Self {
            Self {
                style: Into::<ShapeStyle>::into(&BLACK),
                width: 10,
                whisker_width: 1.0,
                offset: 0.0,
                key,
                values,
            }
        }
    }

    impl<'a, K: Clone> PointCollection<'a, (f64, K)>
    for &'a CustomBoxplot<K>
    {
        type Point = (f64, K);
        type IntoIter = Vec<Self::Point>;
        fn point_iter(self) -> Self::IntoIter {
            self.values
                .iter()
                .map(|v| (*v, self.key.clone()))
                .collect()
        }
    }

    fn with_offset(coord: BackendCoord, offset: f64) -> BackendCoord {
        (coord.0, coord.1 + offset as i32)
    }

    impl<K, DB: DrawingBackend> Drawable<DB> for CustomBoxplot<K> {
        fn draw<I: Iterator<Item=BackendCoord>>(
            &self,
            points: I,
            backend: &mut DB,
            _: (u32, u32),
        ) -> Result<(), DrawingErrorKind<DB::ErrorType>> {
            let points: Vec<_> = points.take(5).collect();
            if points.len() == 5 {
                let width = f64::from(self.width);
                let moved = |coord| with_offset(coord, self.offset);
                let start_bar = |coord| with_offset(moved(coord), -width / 2.0);
                let end_bar = |coord| with_offset(moved(coord), width / 2.0);
                let start_whisker =
                    |coord| with_offset(moved(coord), -width * self.whisker_width / 2.0);
                let end_whisker =
                    |coord| with_offset(moved(coord), width * self.whisker_width / 2.0);

                // |---[   |  ]----|
                // ^________________
                backend.draw_line(
                    start_whisker(points[0]),
                    end_whisker(points[0]),
                    &self.style,
                )?;

                // |---[   |  ]----|
                // _^^^_____________

                backend.draw_line(
                    moved(points[0]),
                    moved(points[1]),
                    &self.style.color.to_backend_color(),
                )?;

                // |---[   |  ]----|
                // ____^______^_____
                let corner1 = start_bar(points[3]);
                let corner2 = end_bar(points[1]);
                let upper_left = (corner1.0.min(corner2.0), corner1.1.min(corner2.1));
                let bottom_right = (corner1.0.max(corner2.0), corner1.1.max(corner2.1));
                backend.draw_rect(upper_left, bottom_right, &self.style, false)?;

                // |---[   |  ]----|
                // ________^________
                backend.draw_line(start_bar(points[2]), end_bar(points[2]), &self.style)?;

                // |---[   |  ]----|
                // ____________^^^^_
                backend.draw_line(moved(points[3]), moved(points[4]), &self.style)?;

                // |---[   |  ]----|
                // ________________^
                backend.draw_line(
                    start_whisker(points[4]),
                    end_whisker(points[4]),
                    &self.style,
                )?;
            }
            Ok(())
        }
    }
}