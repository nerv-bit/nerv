//! Native Terminal/CLI UI for Desktop and Developer Use.
//!
//! Implements an exceedingly elegant, modern terminal user interface using
//! `ratatui` and `crossterm`. This provides a native, responsive UI for
//! desktop environments and a powerful interface for developers, without
//! requiring a web browser or GUI dependencies.

use crate::wallets::ui::{WalletPresenter, UiAction, Screen, WalletUiState};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Span, Spans, Text},
    widgets::{Block, Borders, List, ListItem, Paragraph, Tabs, Gauge},
    Terminal,
};
use std::io::{self, Stdout};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The main CLI application runner.
pub struct CliApp {
    presenter: Arc<WalletPresenter>,
}

impl CliApp {
    pub fn new(presenter: Arc<WalletPresenter>) -> Self {
        Self { presenter }
    }

    /// Runs the main event loop.
    pub fn run(&self) -> io::Result<()> {
        // Setup terminal
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        // Initial state fetch
        self.presenter.refresh_balance();

        let tick_rate = Duration::from_millis(250);
        let mut last_tick = Instant::now();

        loop {
            // Draw UI
            let state = self.presenter.state.lock().unwrap().clone();
            terminal.draw(|f| self.draw(f, &state))?;

            // Handle input
            let timeout = tick_rate
                .checked_sub(last_tick.elapsed())
                .unwrap_or_else(|| Duration::from_secs(0));
            
            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        match key.code {
                            KeyCode::Char('q') => break,
                            KeyCode::Char('1') | KeyCode::Char('2') | KeyCode::Char('3') | KeyCode::Char('4') => {
                                let screen = match key.code {
                                    KeyCode::Char('1') => Screen::Dashboard,
                                    KeyCode::Char('2') => Screen::Send,
                                    KeyCode::Char('3') => Screen::Receive,
                                    KeyCode::Char('4') => Screen::Settings,
                                    _ => unreachable!(),
                                };
                                let presenter = self.presenter.clone();
                                tokio::spawn(async move {
                                    presenter.handle_action(UiAction::Navigate(screen)).await;
                                });
                            }
                            KeyCode::Char('s') => {
                                let presenter = self.presenter.clone();
                                tokio::spawn(async move {
                                    presenter.handle_action(UiAction::StartSync).await;
                                });
                            }
                            KeyCode::Enter => {
                                if state.current_screen == Screen::Send {
                                    let presenter = self.presenter.clone();
                                    tokio::spawn(async move {
                                        presenter.handle_action(UiAction::SendTransaction).await;
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            if last_tick.elapsed() >= tick_rate {
                last_tick = Instant::now();
            }
        }

        // Restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;

        Ok(())
    }

    /// Draws the current UI state.
    fn draw(&self, f: &mut ratatui::Frame<CrosstermBackend<Stdout>>, state: &WalletUiState) {
        let size = f.size();

        // Main layout: Header (Tabs) | Body | Footer
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3), // Tabs
                Constraint::Min(10),   // Body
                Constraint::Length(3), // Footer
            ].as_ref())
            .split(size);

        // 1. Header / Tabs
        let titles = vec!["1: Dashboard", "2: Send", "3: Receive", "4: Settings"];
        let tabs = Tabs::new(titles.iter().map(|t| Spans::from(Span::styled(*t, Style::default().fg(Color::White)))).collect())
            .block(Block::default().borders(Borders::ALL).title(" NERV Wallet V2.0 "))
            .style(Style::default().fg(Color::Cyan))
            .highlight_style(Style::default().add_modifier(Modifier::BOLD).fg(Color::Yellow))
            .select(match state.current_screen {
                Screen::Dashboard => 0,
                Screen::Send => 1,
                Screen::Receive => 2,
                Screen::History => 3, // Map History to Settings for now
                Screen::Settings => 3,
            });
        f.render_widget(tabs, chunks[0]);

        // 2. Body
        match state.current_screen {
            Screen::Dashboard => self.draw_dashboard(f, chunks[1], state),
            Screen::Send => self.draw_send(f, chunks[1], state),
            Screen::Receive => self.draw_receive(f, chunks[1], state),
            Screen::History | Screen::Settings => self.draw_settings(f, chunks[1], state),
        }

        // 3. Footer / Status Bar
        let footer_text = if state.is_syncing {
            format!(" Syncing... [{}] ", "█".repeat((state.sync_progress * 20.0) as usize))
        } else if let Some(notif) = &state.notification {
            format!(" {} ", notif)
        } else {
            " Press [1-4] to navigate | [s] to sync | [q] to quit ".to_string()
        };

        let footer = Paragraph::new(footer_text)
            .style(Style::default().fg(Color::Black).bg(Color::Cyan))
            .alignment(Alignment::Center);
        f.render_widget(footer, chunks[2]);
    }

    fn draw_dashboard(&self, f: &mut ratatui::Frame<CrosstermBackend<Stdout>>, area: Rect, state: &WalletUiState) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(5), Constraint::Min(0)].as_ref())
            .split(area);

        // Balance Box
        let balance_text = Text::styled(
            format!(" {:.9} NERV ", state.balance_nerv),
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        );
        let balance_block = Paragraph::new(balance_text)
            .block(Block::default().borders(Borders::ALL).title(" Available Balance "))
            .alignment(Alignment::Center);
        f.render_widget(balance_block, chunks[0]);

        // Info Box
        let info = vec![
            Spans::from(vec![Span::raw("Status: "), Span::styled(if state.is_syncing { "Syncing" } else { "Online" }, Style::default().fg(Color::Green))]),
            Spans::from(vec![Span::raw("Network: "), Span::styled("NERV Mainnet V2.0", Style::default().fg(Color::Cyan))]),
            Spans::from(vec![Span::raw("Privacy: "), Span::styled("PQ-Sphinx (5-hop)", Style::default().fg(Color::Magenta))]),
        ];
        let info_block = Paragraph::new(info)
            .block(Block::default().borders(Borders::ALL).title(" Network Info "))
            .alignment(Alignment::Left);
        f.render_widget(info_block, chunks[1]);
    }

    fn draw_send(&self, f: &mut ratatui::Frame<CrosstermBackend<Stdout>>, area: Rect, state: &WalletUiState) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(2)
            .constraints([Constraint::Length(3), Constraint::Length(3), Constraint::Length(3), Constraint::Min(0)].as_ref())
            .split(area);

        let recipient = Paragraph::new(state.send_form.recipient_address.as_str())
            .block(Block::default().borders(Borders::ALL).title(" Recipient Address (nerv1...) "));
        f.render_widget(recipient, chunks[0]);

        let amount = Paragraph::new(state.send_form.amount_nerv.as_str())
            .block(Block::default().borders(Borders::ALL).title(" Amount (NERV) "));
        f.render_widget(amount, chunks[1]);

        let memo = Paragraph::new(state.send_form.memo.as_str())
            .block(Block::default().borders(Borders::ALL).title(" Memo (Optional) "));
        f.render_widget(memo, chunks[2]);

        let status = if state.send_form.is_processing {
            Gauge::default()
                .block(Block::default().borders(Borders::ALL).title(" Status "))
                .gauge_style(Style::default().fg(Color::Yellow))
                .percent(50)
                .label("Processing via Mixnet...")
        } else {
            Paragraph::new(state.send_form.status_message.as_str())
                .block(Block::default().borders(Borders::ALL).title(" Status "))
                .style(Style::default().fg(Color::Green))
        };
        f.render_widget(status, chunks[3]);
    }

    fn draw_receive(&self, f: &mut ratatui::Frame<CrosstermBackend<Stdout>>, area: Rect, state: &WalletUiState) {
        let block = Block::default().borders(Borders::ALL).title(" Your Receive Address ");
        let paragraph = Paragraph::new(state.address.as_str())
            .block(block)
            .alignment(Alignment::Center)
            .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
        f.render_widget(paragraph, area);
    }

    fn draw_settings(&self, f: &mut ratatui::Frame<CrosstermBackend<Stdout>>, area: Rect, _state: &WalletUiState) {
        let items = vec![
            ListItem::new("Backup Mnemonic (Requires Password)"),
            ListItem::new("Export VDW History"),
            ListItem::new("Network Settings"),
            ListItem::new("About NERV V2.0"),
        ];
        let list = List::new(items)
            .block(Block::default().borders(Borders::ALL).title(" Settings "))
            .style(Style::default().fg(Color::White))
            .highlight_style(Style::default().add_modifier(Modifier::ITALIC));
        f.render_widget(list, area);
    }
}
