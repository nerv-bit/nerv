//! Theme bridging: nerv-wallet-core's design system → egui's Visuals.

use egui::{Color32, Visuals};

/// Apply the NERV dark theme to the egui context.
pub fn install(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    // GitHub-dark palette (erratum 201).
    let bg = Color32::from_rgb(0x0D, 0x11, 0x17);
    let surface = Color32::from_rgb(0x16, 0x1B, 0x22);
    let border = Color32::from_rgb(0x30, 0x36, 0x3D);
    let text = Color32::from_rgb(0xE6, 0xED, 0xF3);
    let muted = Color32::from_rgb(0x8B, 0x94, 0x9E);
    let accent = Color32::from_rgb(0x58, 0xA6, 0xFF);
    let success = Color32::from_rgb(0x3F, 0xB9, 0x50);
    let warning = Color32::from_rgb(0xD2, 0x99, 0x22);
    let error = Color32::from_rgb(0xF8, 0x51, 0x49);

    style.visuals = Visuals::dark();
    style.visuals.panel_fill = bg;
    style.visuals.window_fill = surface;
    style.visuals.faint_bg_color = surface;
    style.visuals.extreme_bg_color = bg;
    style.visuals.code_bg_color = Color32::from_rgb(0x0D, 0x11, 0x17);
    style.visuals.widgets.noninteractive.bg_fill = surface;
    style.visuals.widgets.noninteractive.fg_stroke.color = muted;
    style.visuals.widgets.inactive.bg_fill = surface;
    style.visuals.widgets.inactive.fg_stroke.color = text;
    style.visuals.widgets.hovered.bg_fill = border;
    style.visuals.widgets.hovered.fg_stroke.color = text;
    style.visuals.widgets.active.bg_fill = accent;
    style.visuals.widgets.active.fg_stroke.color = Color32::WHITE;
    style.visuals.widgets.open.bg_fill = surface;
    style.visuals.widgets.open.fg_stroke.color = text;
    style.visuals.selection.bg_fill = accent;
    style.visuals.selection.stroke.color = Color32::WHITE;
    style.visuals.hyperlink_color = accent;
    style.visuals.warn_fg_color = warning;
    style.visuals.error_fg_color = error;

    // Typography: system monospace for data.
    style.override_font_id = Some(egui::FontId::monospace(14.0));

    // Spacing: generous, airy.
    style.spacing.item_spacing = egui::vec2(10.0, 8.0);
    style.spacing.button_padding = egui::vec2(16.0, 8.0);
    style.spacing.window_margin = egui::Margin::same(12);

    ctx.set_style(style);

    // Store semantic colors for later use.
    ctx.data_mut(|d| {
        d.insert_temp("nerv_bg", bg);
        d.insert_temp("nerv_surface", surface);
        d.insert_temp("nerv_border", border);
        d.insert_temp("nerv_text", text);
        d.insert_temp("nerv_muted", muted);
        d.insert_temp("nerv_accent", accent);
        d.insert_temp("nerv_success", success);
        d.insert_temp("nerv_warning", warning);
        d.insert_temp("nerv_error", error);
    });
}

/// Retrieve a semantic color from the context.
pub fn color(ctx: &egui::Context, name: &str) -> Color32 {
    ctx.data(|d| d.get_temp(name).unwrap_or(Color32::WHITE))
}
