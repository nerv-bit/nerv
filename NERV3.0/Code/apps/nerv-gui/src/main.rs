//! Desktop entry point: `eframe::run_native` with the shared `NervApp`.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([960.0, 640.0])
            .with_min_inner_size([480.0, 360.0])
            .with_title("NERV Wallet")
            .with_icon(load_icon()),
        ..Default::default()
    };

    eframe::run_native(
        "NERV Wallet",
        options,
        Box::new(|cc| Box::new(nerv_gui::NervApp::new(cc))),
    )
}

fn load_icon() -> egui::IconData {
    // A 32×32 dark-blue "N" monogram rendered procedurally.
    let size = 32;
    let mut rgba = Vec::with_capacity(size * size * 4);
    for y in 0..size {
        for x in 0..size {
            let cx = x as f32 / size as f32;
            let cy = y as f32 / size as f32;
            // Background: rounded dark square.
            let dist = ((cx - 0.5).powi(2) + (cy - 0.5).powi(2)).sqrt();
            let bg = if dist < 0.48 {
                (0x16u32, 0x1B, 0x22)
            } else {
                (0, 0, 0)
            };
            // The "N": two verticals + diagonal.
            let in_n = (0.25..=0.35).contains(&cx)
                || (0.65..=0.75).contains(&cx)
                || (cx - cy).abs() < 0.07 && (0.25..=0.75).contains(&cx);
            if in_n && dist < 0.48 {
                rgba.extend_from_slice(&[0x58, 0xA6, 0xFF, 0xFF]);
            } else {
                rgba.extend_from_slice(&[bg.0 as u8, bg.1, bg.2, 0xFF]);
            }
        }
    }
    egui::IconData { rgba, width: size, height: size }
}
