#[derive(PartialEq, Debug, Clone)]
pub enum Value {
    String(String),
    Number(f64),
}

impl Eq for Value {}

impl ToString for Value {
    fn to_string(&self) -> String {
        match self {
            Value::Number(n) => n.to_string(),
            Value::String(s) => s.clone(),
        }
    }
}

impl std::cmp::PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Value::Number(a), Value::Number(b)) => a.partial_cmp(b),
            (Value::String(a), Value::String(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}
impl std::cmp::Ord for Value {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Value::Number(a), Value::Number(b)) => {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            }
            (Value::String(a), Value::String(b)) => a.cmp(b),
            _ => std::cmp::Ordering::Equal,
        }
    }
}

pub struct Record {
    pub values: Vec<Value>,
}

#[derive(Default)]
pub struct Filter {
    pub min: FilterText,
    pub max: FilterText,
    pub request: FilterText,
    pub min_found: f64,
    pub max_found: f64,
}

#[derive(Default)]
pub struct FilterText {
    pub text: String,
    pub value: Option<f64>,
}

#[derive(Debug)]
pub struct TabSorting {
    pub reverse: bool,
    pub column: usize,
}

use std::f64;

use egui::{Color32, Ui, Widget};

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());

    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    dot / (norm_a * norm_b)
}
// similarity = 1.0 / (1.0 + distance)
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn read_csv(d: &[u8]) -> Content {
    let mut headers = vec![];
    let mut data = vec![];

    let mut rdr = csv::Reader::from_reader(&d[..]);
    if let Ok(headers_input) = rdr.headers() {
        for text in headers_input.iter() {
            headers.push(text.to_string());
        }
    }
    for result in rdr.records() {
        if let Ok(r) = result {
            let mut values = vec![];
            for text in r.iter() {
                if let Ok(n) = text.parse::<f64>() {
                    values.push(Value::Number(n));
                } else {
                    values.push(Value::String(text.to_string()));
                }
            }
            data.push(Record { values });
        }
    }
    headers.insert(0, "Euclidean Similarity".to_string());
    headers.insert(0, "Cosine Similarity".to_string());
    for d in data.iter_mut() {
        d.values.insert(0, Value::Number(1.));
        d.values.insert(0, Value::Number(1.));
    }
    let filtered_data: Vec<_> = data.iter().enumerate().map(|(i, _)| i).collect();
    let filters: Vec<_> = headers
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let mut f = Filter::default();
            f.max_found = data
                .iter()
                .filter_map(|d| d.values.get(i))
                .filter_map(|x| match x {
                    Value::String(_) => None,
                    Value::Number(x) => Some(*x),
                })
                .fold(f64::NEG_INFINITY, f64::max);
            f.min_found = data
                .iter()
                .filter_map(|d| d.values.get(i))
                .filter_map(|x| match x {
                    Value::String(_) => None,
                    Value::Number(x) => Some(*x),
                })
                .fold(f64::INFINITY, f64::min);
            return f;
        })
        .collect();

    return Content {
        headers,
        data,
        filtered_data,
        filters,
        shown_data: vec![],
    };
}
#[derive(Default)]
struct Content {
    data: Vec<Record>,
    headers: Vec<String>,
    filters: Vec<Filter>,
    filtered_data: Vec<usize>,
    shown_data: Vec<usize>,
}

fn main() {
    run("product similarity search", |cc| {
        let d = include_bytes!("../output.csv");
        let content = std::sync::Arc::new(egui::mutex::Mutex::new(read_csv(d)));

        let mut sorting = TabSorting {
            reverse: false,
            column: 0,
        };
        return Box::new(move |ctx| {
            let ui = ctx.ui;

            ui.horizontal(|ui| {
                {
                    let content = content.lock();
                    ui.label(format!(
                        "{} of {} rows",
                        content.filtered_data.len(),
                        content.data.len()
                    ));
                }

                if ui.button("open csv").clicked() {
                    let task = rfd::AsyncFileDialog::new().pick_file();
                    let c = content.clone();
                    wasm_bindgen_futures::spawn_local(async move {
                        let file = task.await;
                        if let Some(file) = file {
                            let contents = file.read().await;
                            let contents = read_csv(&contents);
                            {
                                let mut c = c.lock();
                                *c = contents;
                            }
                        }
                    });
                }
            });

            {
                let mut content = content.lock();
                show_content(ui, &mut content, &mut sorting);
            }
        });
    });
}

fn show_content(ui: &mut Ui, content: &mut Content, sorting: &mut TabSorting) {
    egui_plot::Plot::new("plot")
        .allow_zoom(false)
        .allow_axis_zoom_drag(false)
        .allow_drag(false)
        .allow_scroll(false)
        .x_axis_formatter(|f, _| format!("{}", &content.headers[f.value as usize]))
        .height(ui.available_height() * 0.25)
        .show(ui, |plot_ui| {
            for &i in content.shown_data.iter() {
                let article = &content.data[content.filtered_data[i]];
                let mut d_vec = vec![];

                for (i, v) in article.values.iter().enumerate() {
                    if i <= 1 {
                        d_vec.push(f64::NAN);
                    } else {
                         match v {
                        Value::String(s) => {
                            d_vec.push(0.5);
                        }
                        Value::Number(v) => {
                            let f = &content.filters[i];
                            let min = f.min_found;
                            let len = f.max_found - min;
                            d_vec.push((*v - min) / len);
                        }
                    }
                    }
                   
                }

                plot_ui.line(
                    egui_plot::Line::new("", egui_plot::PlotPoints::from_ys_f64(&d_vec))
                        .color(Color32::GRAY),
                );
            }
            // min
            let mut d_vec = vec![];
            for (f) in content.filters.iter() {
                let v = f.min.value.unwrap_or(f.min_found);
                let min = f.min_found;
                let len = f.max_found - min;
                d_vec.push((v - min) / len);
            }
            plot_ui.line(
                egui_plot::Line::new("min", egui_plot::PlotPoints::from_ys_f64(&d_vec))
                    .color(Color32::GREEN),
            );
            // max
            let mut d_vec = vec![];
            for (f) in content.filters.iter() {
                let v = f.max.value.unwrap_or(f.max_found);
                let min = f.min_found;
                let len = f.max_found - min;
                d_vec.push((v - min) / len);
            }
            plot_ui.line(
                egui_plot::Line::new("max", egui_plot::PlotPoints::from_ys_f64(&d_vec))
                    .color(Color32::GREEN),
            );

            // request
            let mut d_vec = vec![];
            for (f) in content.filters.iter() {
                let min = f.min_found;
                let len = f.max_found - min;
                if let Some(v) = f.request.value {
                    d_vec.push((v - min) / len);
                } else {
                    d_vec.push(0.5);
                }
            }
            plot_ui.line(
                egui_plot::Line::new("request", egui_plot::PlotPoints::from_ys_f64(&d_vec))
                    .color(Color32::RED),
            );
        });

    let mut filters_changed = false;

    let mut builder = egui_extras::TableBuilder::new(ui);
    for _ in content.headers.iter() {
        builder = builder.column(egui_extras::Column::remainder());
    }

    let table = builder.header(20.0, |mut header| {
        for (i, column) in content.headers.iter().enumerate() {
            header.col(|ui| {
                let after = if sorting.column == i {
                    if sorting.reverse { " ⬇" } else { " ⬆" }
                } else {
                    ""
                };
                if egui::Label::new(egui::RichText::from(format!("{}{}", column, after)).strong())
                    .selectable(false)
                    .sense(egui::Sense::click())
                    .ui(ui)
                    .clicked()
                {
                    if sorting.column == i {
                        sorting.reverse = !sorting.reverse;
                    } else {
                        sorting.column = i;
                        sorting.reverse = false;
                    }
                    sort_articles(&content.data, &sorting, &mut content.filtered_data);
                }
                if i <= 1 {
                    ui.label("");
                } else {
                    ui.horizontal(|ui| {
                        ui.label("request");
                        let f = &mut content.filters[i].request;
                        let text = f.text.clone();
                        ui.text_edit_singleline(&mut f.text);
                        filters_changed |= text != f.text;
                        f.value = f.text.parse::<f64>().ok();
                    });
                }

                ui.horizontal(|ui| {
                    ui.label("min");
                    let f = &mut content.filters[i].min;
                    let text = f.text.clone();
                    ui.text_edit_singleline(&mut f.text);
                    filters_changed |= text != f.text;
                    f.value = f.text.parse::<f64>().ok();
                });
                ui.horizontal(|ui| {
                    ui.label("max");
                    let f = &mut content.filters[i].max;
                    let text = f.text.clone();
                    ui.text_edit_singleline(&mut f.text);
                    filters_changed |= text != f.text;
                    f.value = f.text.parse::<f64>().ok();
                });
            });
        }
    });

    if filters_changed {
        filter_articles(
            &mut content.data,
            &mut content.filtered_data,
            &content.filters,
        );
        sort_articles(&content.data, &sorting, &mut content.filtered_data);
    }

    content.shown_data.clear();
    table.body(|mut body| {
        body.rows(20., content.filtered_data.len(), |mut row| {
            content.shown_data.push(row.index());
            let article = &content.data[content.filtered_data[row.index()]];

            for column in &article.values {
                row.col(|ui| {
                    let v = match column {
                        Value::String(s) => s.to_string(),
                        Value::Number(s) => s.to_string(),
                    };
                    if ui.label(v).double_clicked() {}
                });
            }
        });
    });
}

fn filter_articles(data: &mut Vec<Record>, filtered_data: &mut Vec<usize>, filters: &Vec<Filter>) {
    for d in data.iter_mut() {
        let mut r_vec = vec![];
        let mut d_vec = vec![];

        for (i, v) in d.values.iter().enumerate() {
            match v {
                Value::String(s) => {}
                Value::Number(v) => {
                    let f = &filters[i];
                    if let Some(request) = f.request.value {
                        let min = f.min.value.unwrap_or(f.min_found);
                        let len = f.max.value.unwrap_or(f.max_found) - min;
                        r_vec.push((request - min) / len);
                        d_vec.push((*v - min) / len);
                    }
                }
            }
        }
        if r_vec.len() > 0 {
            d.values[0] = Value::Number(cosine_similarity(&r_vec, &d_vec).abs());
            d.values[1] = Value::Number(1.0 / (1.0 + euclidean_distance(&r_vec, &d_vec)))
        } else {
            d.values[0] = Value::Number(1.);
            d.values[1] = Value::Number(1.);
        }
    }
    filtered_data.clear();
    filtered_data.extend(0..data.len());
    filtered_data.retain(|&index| {
        let record = &data[index];
        for (i, v) in record.values.iter().enumerate() {
            match v {
                Value::String(s) => {
                    if !filters[i]
                        .request
                        .text
                        .split_whitespace()
                        .all(|x| s.contains(x))
                    {
                        return false;
                    }
                }
                Value::Number(v) => {
                    if let Some(min) = &filters[i].min.value {
                        if v < min {
                            return false;
                        }
                    }
                    if let Some(max) = &filters[i].max.value {
                        if v > max {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    });
}

fn sort_articles(data: &Vec<Record>, sorting: &TabSorting, filtered_data: &mut Vec<usize>) {
    filtered_data.sort_by(|&a, &b| {
        let a_value = &data[a].values[sorting.column];
        let b_value = &data[b].values[sorting.column];
        if sorting.reverse {
            b_value.cmp(&a_value)
        } else {
            a_value.cmp(&b_value)
        }
    });
}

// fn generate_random(
//     rng: &mut rand::prelude::ThreadRng,
//     r: impl rand::distr::uniform::SampleRange<f64>,
//     decimal_places: i32,
// ) -> f64 {
//     let p = 10_f64.powi(decimal_places);
//     (rng.random_range(r) * p).round() / p
// }

pub struct Context<'a> {
    pub ui: &'a mut egui::Ui,
    pub frame: &'a mut eframe::Frame,
    pub needs_save: &'a mut Option<String>,
}

impl<'a> Context<'a> {
    pub fn save<T: serde::Serialize>(&mut self, value: &T) {
        if let Ok(string) = ron::ser::to_string(value) {
            *self.needs_save = Some(string);
        }
    }

    pub fn get_ui(&mut self) -> &mut egui::Ui {
        self.ui
    }
}
struct App {
    update: Box<dyn FnMut(Context)>,
    needs_save: Option<String>,
}

impl App {
    fn new(
        cc: &eframe::CreationContext<'_>,
        f: impl Fn(&eframe::CreationContext<'_>) -> Box<dyn FnMut(Context)>,
    ) -> Self {
        let update = (f)(cc);
        Self {
            update,
            needs_save: None,
        }
    }
}

impl eframe::App for App {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if let Some(string) = &self.needs_save {
            storage.set_string(eframe::APP_KEY, string.clone());
            self.needs_save = None;
        }
    }

    fn auto_save_interval(&self) -> std::time::Duration {
        if self.needs_save.is_some() {
            std::time::Duration::from_secs(0)
        } else {
            std::time::Duration::from_secs(30)
        }
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let context = Context {
                ui,
                frame,
                needs_save: &mut self.needs_save,
            };

            (self.update)(context);
        });
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn run(app_name: &str, f: impl Fn(&eframe::CreationContext<'_>) -> Box<dyn FnMut(Context)>) {
    let mut native_options = eframe::NativeOptions::default();
    native_options.viewport =
        egui::ViewportBuilder::default().with_inner_size(egui::vec2(1200.0, 700.0));
    eframe::run_native(
        app_name,
        native_options,
        Box::new(|cc| Ok(Box::new(App::new(cc, f)))),
    )
    .unwrap();
}

#[cfg(target_arch = "wasm32")]
pub fn run(
    app_name: &str,
    f: impl Fn(&eframe::CreationContext<'_>) -> Box<dyn FnMut(Context)> + 'static,
) {
    use eframe::wasm_bindgen::JsCast as _;
    use std::io::Write;

    let web_options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window()
            .expect("No window")
            .document()
            .expect("No document");

        let canvas = document
            .get_element_by_id("the_canvas_id")
            .expect("Failed to find the_canvas_id")
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .expect("the_canvas_id was not a HtmlCanvasElement");

        let start_result = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| Ok(Box::new(App::new(cc, f)))),
            )
            .await;

        // Remove the loading text and spinner:
        if let Some(loading_text) = document.get_element_by_id("loading_text") {
            match start_result {
                Ok(_) => {
                    loading_text.remove();
                }
                Err(e) => {
                    loading_text.set_inner_html(
                        "<p> The app has crashed. See the developer console for details. </p>",
                    );
                    panic!("Failed to start eframe: {e:?}");
                }
            }
        }
    });
}
