//! The TUI rendering (ratatui): every screen, elegantly.

use ratatui::prelude::*;
use ratatui::widgets::*;
use nerv_wallet_core::state::{Direction, Screen, SyncStatus};
use nerv_wallet_core::Theme;

use crate::app::App;

pub fn draw(f: &mut Frame, app: &App) {
    let theme = Theme::dark();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),   // Title bar
            Constraint::Length(3),   // Tab bar
            Constraint::Min(1),      // Content
            Constraint::Length(1),   // Status bar
        ])
        .split(f.area());

    draw_title(f, app, chunks[0], &theme);
    draw_tabs(f, app, chunks[1], &theme);
    draw_content(f, app, chunks[2], &theme);
    draw_status(f, app, chunks[3], &theme);
    draw_notifications(f, app, &theme);
}

fn draw_title(f: &mut Frame, app: &App, area: Rect, t: &Theme) {
    let title = if app.state.is_unlocked() {
        "NERV Wallet"
    } else {
        "NERV Wallet — [N]ew wallet / [I]mport seed"
    };
    let widget = Paragraph::new(title)
        .style(Style::default().fg(Color::Rgb(t.text[0], t.text[1], t.text[2])))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(t.border[0], t.border[1], t.border[2])))
                .bg(Color::Rgb(t.background[0], t.background[1], t.background[2])),
        );
    f.render_widget(widget, area);
}

fn draw_tabs(f: &mut Frame, app: &App, area: Rect, t: &Theme) {
    let titles: Vec<Line> = Screen::ALL
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let style = if s.index() == app.state.screen.index() {
                Style::default().fg(Color::Rgb(t.accent[0], t.accent[1], t.accent[2])).bold()
            } else {
                Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2]))
            };
            Line::from(format!(" {}:{} ", i + 1, s.title())).style(style)
        })
        .collect();
    let tabs = Tabs::new(titles)
        .style(Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])))
        .highlight_style(Style::default().fg(Color::Rgb(t.accent[0], t.accent[1], t.accent[2])))
        .divider(symbols::line::VERTICAL);
    f.render_widget(tabs, area);
}

fn draw_content(f: &mut Frame, app: &App, area: Rect, t: &Theme) {
    match app.state.screen {
        Screen::Dashboard => draw_dashboard(f, app, area, t),
        Screen::Send => draw_send(f, app, area, t),
        Screen::Receive => draw_receive(f, app, area, t),
        Screen::Claim => draw_claim(f, app, area, t),
        Screen::Producer => draw_producer(f, app, area, t),
        Screen::History => draw_history(f, app, area, t),
        Screen::Settings => draw_settings(f, app, area, t),
        Screen::Help => draw_help(f, app, area, t),
    }
}

fn draw_dashboard(f: &mut Frame, app: &App, area: Rect, t: &Theme) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(5), Constraint::Min(1)])
        .split(area);

    // Balance card.
    let balance_nerv = app.state.balance_nano() as f64 / 1_000_000_000.0;
    let balance_str = format!("{balance_nerv:.9}");
    let sync_str = match app.state.sync {
        SyncStatus::Locked => "Locked".to_string(),
        SyncStatus::Scanning { progress_permille } => format!("Syncing… {}‰", progress_permille),
        SyncStatus::Synced { height } => format!("Synced @ block {height}"),
        SyncStatus::Disconnected => "Disconnected".to_string(),
    };
    let balance = Paragraph::new(vec![
        Line::from(Span::styled("  Balance", Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])))),
        Line::from(Span::styled(
            format!("  {balance_str} NERV"),
            Style::default().fg(Color::Rgb(t.success[0], t.success[1], t.success[2])).bold(),
        )),
        Line::from(""),
        Line::from(Span::styled(
            format!("  {sync_str}  ·  {} peers  ·  {} addresses", app.state.peer_count, app.state.address_count()),
            Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])),
        )),
    ])
    .block(surrounding_block("Dashboard", t));
    f.render_widget(balance, chunks[0]);

    // Recent activity.
    let items: Vec<ListItem> = if app.state.history.is_empty() {
        vec![ListItem::new(Span::styled(
            "  No transactions yet",
            Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])),
        ))]
    } else {
        app.state.history.iter().take(10).map(|e| {
            let (dir_str, color) = match e.direction {
                Direction::Incoming => ("↓", Color::Rgb(t.success[0], t.success[1], t.success[2])),
                Direction::Outgoing => ("↑", Color::Rgb(t.error[0], t.error[1], t.error[2])),
            };
            let amount = e.amount_nano as f64 / 1_000_000_000.0;
            ListItem::new(Line::from(vec![
                Span::styled(format!("  {dir_str} "), Style::default().fg(color)),
                Span::styled(format!("{amount:.9} NERV"), Style::default().fg(color)),
                Span::raw("  "),
                Span::styled(
                    format!("block {}", e.height),
                    Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])),
                ),
            ]))
        }).collect()
    };
    let list = List::new(items).block(surrounding_block("Recent Activity", t));
    f.render_widget(list, chunks[1]);
}

fn draw_send(f: &mut Frame, app: &App, area: Rect, t: &Theme) {
    if let Some(draft) = &app.state.draft {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Min(1),
            ])
            .split(area);

        let label = |text: &str| {
            Paragraph::new(Span::styled(
                format!("  {text}"),
                Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])),
            ))
            .block(surrounding_block("", t))
        };

        let value = |text: &str, ok: bool| {
            let color = if ok {
                Color::Rgb(t.text[0], t.text[1], t.text[2])
            } else {
                Color::Rgb(t.error[0], t.error[1], t.error[2])
            };
            Paragraph::new(Span::styled(format!("  {text}"), Style::default().fg(color)))
                .block(surrounding_block("", t))
        };

        f.render_widget(label("Recipient address:"), chunks[0]);
        f.render_widget(
            value(&if draft.recipient_hex.is_empty() { "…".into() } else {
                format!("{}…{}", &draft.recipient_hex[..8.min(draft.recipient_hex.len())],
                         &draft.recipient_hex[draft.recipient_hex.len().saturating_sub(8)..])
            }, draft.validation.recipient_ok),
            chunks[1],
        );

        f.render_widget(label("Amount (nano-NERV):"), chunks[2]);
        f.render_widget(value(&draft.amount_nano.to_string(), draft.validation.amount_ok), chunks[3]);

        f.render_widget(label("Fee (nano-NERV):"), chunks[4]);

        if let Some(err) = &draft.validation.error {
            let error_p = Paragraph::new(Span::styled(
                format!("  ⚠ {}", err.message()),
                Style::default().fg(Color::Rgb(t.warning[0], t.warning[1], t.warning[2])),
            ));
            f.render_widget(error_p, chunks[5]);
        } else {
            let ready = Paragraph::new(Span::styled(
                "  ✓ Ready — [Enter] to sign and send, [x] to cancel",
                Style::default().fg(Color::Rgb(t.success[0], t.success[1], t.success[2])),
            ));
            f.render_widget(ready, chunks[5]);
        }
    } else {
        let p = Paragraph::new("  No active draft. Press [2] to start.")
            .style(Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])));
        f.render_widget(p, area);
    }
}

fn draw_receive(f: &mut Frame, app: &App, area: Rect, t: &Theme) {
    if let Some(addr_hex) = app.state.primary_address_hex() {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(5), Constraint::Min(1)])
            .split(area);

        let display = if addr_hex.len() > 64 {
            format!("{}…{}", &addr_hex[..32], &addr_hex[addr_hex.len() - 32..])
        } else {
            addr_hex.clone()
        };

        let p = Paragraph::new(vec![
            Line::from(Span::styled("  Your address:", Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])))),
            Line::from(""),
            Line::from(Span::styled(
                format!("  {display}"),
                Style::default().fg(Color::Rgb(t.accent[0], t.accent[1], t.accent[2])),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "  [c] copy   [y] new address",
                Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])),
            )),
        ])
        .block(surrounding_block("Receive", t));
        f.render_widget(p, chunks[0]);

        if let Some(clip) = &app.clipboard {
            let note = Paragraph::new(Span::styled(
                "  ✓ Copied to clipboard",
                Style::default().fg(Color::Rgb(t.success[0], t.success[1], t.success[2])),
            ));
            f.render_widget(note, chunks[1]);
        }
    } else {
        let p = Paragraph::new("  No wallet loaded.")
            .style(Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])));
        f.render_widget(p, area);
    }
}

fn draw_claim(f: &mut Frame, app: &App, area: Rect, t: &Theme) {
    use nerv_wallet_core::action::ClaimBucket;
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(1),
        ])
        .split(area);

    let label = |text: &str| {
        Paragraph::new(Span::styled(
            format!("  {text}"),
            Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])),
        ))
        .block(surrounding_block("", t))
    };

    // Bucket picker (filtered to USER_BUCKETS — excludes Founder).
    let user_buckets = ClaimBucket::USER_BUCKETS;
    let picked = app.state.claim_draft.as_ref().and_then(|d| d.bucket);
    let mut bucket_line = String::from("  Bucket: ");
    for (idx, b) in user_buckets.iter().enumerate() {
        let marker = if Some(*b) == picked { "●" } else { "○" };
        bucket_line.push_str(&format!("  {}. {} {}  ", idx + 1, marker, b.label()));
    }
    let bucket_p = Paragraph::new(Span::styled(
        bucket_line,
        Style::default().fg(Color::Rgb(t.text[0], t.text[1], t.text[2])),
    ))
    .block(surrounding_block("Pick a bucket [1-3]", t));
    f.render_widget(bucket_p, chunks[0]);

    // Amount row.
    f.render_widget(label("Amount (nano-NERV):"), chunks[1]);
    let amount_nano = app
        .state
        .claim_draft
        .as_ref()
        .map(|d| d.amount_nano)
        .unwrap_or(0);
    let amount_nerv = amount_nano as f64 / 1_000_000_000.0;
    let amount_p = Paragraph::new(Span::styled(
        format!("  {amount_nano}  ({:.9} NERV)", amount_nerv),
        Style::default().fg(Color::Rgb(t.accent[0], t.accent[1], t.accent[2])),
    ))
    .block(surrounding_block("", t));
    f.render_widget(amount_p, chunks[2]);

    // Validation / status row.
    if let Some(draft) = &app.state.claim_draft {
        if let Some(err) = &draft.validation.error {
            let p = Paragraph::new(Span::styled(
                format!("  ⚠ {}", err.message()),
                Style::default().fg(Color::Rgb(t.warning[0], t.warning[1], t.warning[2])),
            ))
            .block(surrounding_block("", t));
            f.render_widget(p, chunks[3]);
        } else if draft.is_ready() {
            let p = Paragraph::new(Span::styled(
                "  ✓ Ready — [Enter] to claim, [x] to cancel",
                Style::default().fg(Color::Rgb(t.success[0], t.success[1], t.success[2])),
            ))
            .block(surrounding_block("", t));
            f.render_widget(p, chunks[3]);
        } else {
            f.render_widget(label(""), chunks[3]);
        }
    } else {
        f.render_widget(label("  Press [4] to start a claim."), chunks[3]);
    }

    // Footer / instructions.
    let footer = Paragraph::new(Span::styled(
        "  Enter=submit  x=cancel  Founder bucket is reserved (not shown)",
        Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])),
    ));
    f.render_widget(footer, chunks[4]);
}

fn draw_producer(f: &mut Frame, app: &App, area: Rect, t: &Theme) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Seed input
            Constraint::Length(3), // Shard picker
            Constraint::Length(3), // Stake amount
            Constraint::Length(3), // Status / validation
            Constraint::Min(1),    // Live state (if registered)
        ])
        .split(area);

    let label = |text: &str| {
        Paragraph::new(Span::styled(
            format!("  {text}"),
            Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])),
        ))
        .block(surrounding_block("", t))
    };

    // 1. Seed input.
    let seed_line = match app.state.producer_draft.as_ref().and_then(|d| d.seed) {
        Some(bytes) => {
            let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
            format!("  Seed: {}…  (32 bytes parsed)", &hex[..16.min(hex.len())])
        }
        None => "  Seed: (paste 64 hex chars)".to_string(),
    };
    let seed_p = Paragraph::new(Span::styled(
        seed_line,
        Style::default().fg(Color::Rgb(t.text[0], t.text[1], t.text[2])),
    ))
    .block(surrounding_block("Producer seed [s] to type, [S] to paste", t));
    f.render_widget(seed_p, chunks[0]);

    // 2. Shard picker. Show the chosen shard + an inline range note.
    let shard = app
        .state
        .producer_draft
        .as_ref()
        .and_then(|d| d.shard)
        .unwrap_or(0);
    let shard_p = Paragraph::new(Span::styled(
        format!("  Shard: {shard}  (0..=63 — genesis set)"),
        Style::default().fg(Color::Rgb(t.text[0], t.text[1], t.text[2])),
    ))
    .block(surrounding_block("Pick shard [←/→] or [0..=9]", t));
    f.render_widget(shard_p, chunks[1]);

    // 3. Stake amount.
    let stake_nano = app
        .state
        .producer_draft
        .as_ref()
        .map(|d| d.stake_nano)
        .unwrap_or(0);
    let stake_nerv = stake_nano as f64 / 1_000_000_000.0;
    let stake_p = Paragraph::new(Span::styled(
        format!("  {stake_nano}  ({stake_nerv:.9} NERV)"),
        Style::default().fg(Color::Rgb(t.accent[0], t.accent[1], t.accent[2])),
    ))
    .block(surrounding_block("Stake bond [+] / [-] to bump 1 NERV", t));
    f.render_widget(stake_p, chunks[2]);

    // 4. Status / validation row.
    if let Some(draft) = &app.state.producer_draft {
        if let Some(err) = &draft.validation.error {
            let p = Paragraph::new(Span::styled(
                format!("  ⚠ {}", err.message()),
                Style::default().fg(Color::Rgb(t.warning[0], t.warning[1], t.warning[2])),
            ))
            .block(surrounding_block("", t));
            f.render_widget(p, chunks[3]);
        } else if draft.is_ready() {
            let p = Paragraph::new(Span::styled(
                "  ✓ Ready — [Enter] to register, [x] to cancel",
                Style::default().fg(Color::Rgb(t.success[0], t.success[1], t.success[2])),
            ))
            .block(surrounding_block("", t));
            f.render_widget(p, chunks[3]);
        } else {
            f.render_widget(label(""), chunks[3]);
        }
    } else {
        f.render_widget(label("  Press [5] to register a producer."), chunks[3]);
    }

    // 5. Live producer state, if registered.
    if app.state.producer_state.stake_registered {
        let s = &app.state.producer_state;
        let payout_short = if s.payout_address_hex.len() > 16 {
            format!(
                "{}…",
                &s.payout_address_hex[..16.min(s.payout_address_hex.len())]
            )
        } else {
            s.payout_address_hex.clone()
        };
        let line = format!(
            "  ● Registered — VK {}…  payout {}  shard {}  balance {} NERV",
            &s.verifying_key_hex[..16.min(s.verifying_key_hex.len())],
            payout_short,
            s.assigned_shard,
            s.stake_balance_nano as f64 / 1_000_000_000.0,
        );
        let p = Paragraph::new(Span::styled(
            line,
            Style::default().fg(Color::Rgb(t.success[0], t.success[1], t.success[2])),
        ))
        .block(surrounding_block("Live state", t));
        f.render_widget(p, chunks[4]);
    } else {
        let p = Paragraph::new(Span::styled(
            "  No producer registered yet — submit the draft to bond stake.",
            Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])),
        ))
        .block(surrounding_block("Live state", t));
        f.render_widget(p, chunks[4]);
    }
}

fn draw_history(f: &mut Frame, app: &App, area: Rect, t: &Theme) {
    let items: Vec<ListItem> = if app.state.history.is_empty() {
        vec![ListItem::new(Span::styled(
            "  No transactions yet",
            Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])),
        ))]
    } else {
        app.state.history.iter().map(|e| {
            let (arrow, color) = match e.direction {
                Direction::Incoming => ("↓", Color::Rgb(t.success[0], t.success[1], t.success[2])),
                Direction::Outgoing => ("↑", Color::Rgb(t.error[0], t.error[1], t.error[2])),
            };
            let amount = e.amount_nano as f64 / 1_000_000_000.0;
            let conf = if e.confirmations > 0 {
                format!(" ✓{}", e.confirmations)
            } else {
                " ⋯".to_string()
            };
            ListItem::new(Line::from(vec![
                Span::styled(format!("  {arrow} "), Style::default().fg(color)),
                Span::styled(format!("{amount:>15.9} NERV"), Style::default().fg(color)),
                Span::styled(format!("{conf}"), Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2]))),
                Span::raw("  "),
                Span::styled(format!("block {:>6}", e.height), Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2]))),
            ]))
        }).collect()
    };
    let list = List::new(items).block(surrounding_block("History", t));
    f.render_widget(list, area);
}

fn draw_settings(f: &mut Frame, app: &App, area: Rect, t: &Theme) {
    let p = Paragraph::new(vec![
        Line::from(Span::styled("  Settings", Style::default().fg(Color::Rgb(t.text[0], t.text[1], t.text[2])).bold())),
        Line::from(""),
        Line::from(Span::styled(format!("  Shard coverage: {}", app.state.address_count()), Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])))),
        Line::from(Span::styled(format!("  Chain height: {}", app.state.chain_height), Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])))),
        Line::from(Span::styled(format!("  Peers: {}", app.state.peer_count), Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])))),
        Line::from(""),
        Line::from(Span::styled("  [L] lock wallet", Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2])))),
    ])
    .block(surrounding_block("Settings", t));
    f.render_widget(p, area);
}

fn draw_help(f: &mut Frame, _app: &App, area: Rect, t: &Theme) {
    let p = Paragraph::new(vec![
        Line::from(Span::styled("  Keyboard Shortcuts", Style::default().fg(Color::Rgb(t.text[0], t.text[1], t.text[2])).bold())),
        Line::from(""),
        Line::from("  1-7       Switch tab (Dashboard/Send/Receive/Claim/History/Settings/Help)"),
        Line::from("  h / ←     Previous tab"),
        Line::from("  l / →     Next tab"),
        Line::from("  Enter     Confirm & send (Send) / submit (Claim)"),
        Line::from("  x         Cancel draft (Send / Claim screen)"),
        Line::from("  c         Copy address (Receive screen)"),
        Line::from("  y         New address (Receive screen)"),
        Line::from("  1/2/3     Pick claim bucket (Claim screen)"),
        Line::from("  n         Dismiss notification"),
        Line::from("  Q         Quit"),
    ])
    .block(surrounding_block("Help", t));
    f.render_widget(p, area);
}

fn draw_status(f: &mut Frame, app: &App, area: Rect, t: &Theme) {
    let status = match app.state.sync {
        SyncStatus::Locked => "LOCKED".to_string(),
        SyncStatus::Scanning { .. } => "SYNCING".to_string(),
        SyncStatus::Synced { .. } => "SYNCED".to_string(),
        SyncStatus::Disconnected => "OFFLINE".to_string(),
    };
    let color = match app.state.sync {
        SyncStatus::Synced { .. } => Color::Rgb(t.success[0], t.success[1], t.success[2]),
        SyncStatus::Disconnected => Color::Rgb(t.error[0], t.error[1], t.error[2]),
        _ => Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2]),
    };
    let bar = Paragraph::new(Line::from(vec![
        Span::styled(format!(" {status}"), Style::default().fg(color).bold()),
        Span::styled(format!(" · {} peers", app.state.peer_count), Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2]))),
        Span::styled(format!(" · block {}", app.state.chain_height), Style::default().fg(Color::Rgb(t.text_muted[0], t.text_muted[1], t.text_muted[2]))),
        Span::raw(" "),
    ]))
    .style(Style::default().bg(Color::Rgb(t.surface[0], t.surface[1], t.surface[2])));
    f.render_widget(bar, area);
}

fn draw_notifications(f: &mut Frame, app: &App, t: &Theme) {
    if let Some(note) = app.state.notifications.back() {
        let area = Rect {
            x: f.area().width.saturating_sub(50),
            y: f.area().height.saturating_sub(3),
            width: 50.min(f.area().width),
            height: 3,
        };
        let color = match note.level {
            nerv_wallet_core::state::NotificationLevel::Info => Color::Rgb(t.accent[0], t.accent[1], t.accent[2]),
            nerv_wallet_core::state::NotificationLevel::Success => Color::Rgb(t.success[0], t.success[1], t.success[2]),
            nerv_wallet_core::state::NotificationLevel::Warning => Color::Rgb(t.warning[0], t.warning[1], t.warning[2]),
            nerv_wallet_core::state::NotificationLevel::Error => Color::Rgb(t.error[0], t.error[1], t.error[2]),
        };
        let p = Paragraph::new(format!(" {} ", note.message))
            .style(Style::default().fg(color).bg(Color::Rgb(t.surface[0], t.surface[1], t.surface[2])))
            .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(color)));
        f.render_widget(p, area);
    }
}

fn surrounding_block(title: &str, t: &Theme) -> Block<'_> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Rgb(t.border[0], t.border[1], t.border[2])))
        .title(Span::styled(
            format!(" {title} "),
            Style::default().fg(Color::Rgb(t.text[0], t.text[1], t.text[2])).bold(),
        ))
        .bg(Color::Rgb(t.background[0], t.background[1], t.background[2]))
}
