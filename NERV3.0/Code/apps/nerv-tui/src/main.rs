//! The NERV wallet terminal UI (ratatui; erratum 200–201).

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{prelude::*, Terminal};

use nerv_wallet_core::state::Screen;
use nerv_wallet_core::WalletAction;

mod app;
mod ui;

fn main() -> Result<()> {
    enable_raw_mode()?;
    crossterm::execute!(std::io::stdout(), EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(std::io::stdout());
    let mut terminal = Terminal::new(backend)?;

    let mut app = app::App::new();
    let res = run(&mut terminal, &mut app);

    disable_raw_mode()?;
    crossterm::execute!(std::io::stdout(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    if let Err(e) = res {
        eprintln!("error: {e}");
    }
    Ok(())
}

fn run(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>, app: &mut app::App) -> Result<()> {
    loop {
        // Gap 6: drain the send pipeline before the draw so progress
        // notifications refresh in the same frame as the spawn.
        app.drain_pipeline();

        terminal.draw(|f| ui::draw(f, app))?;

        if event::poll(std::time::Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    // Producer-seed input mode collects hex keystrokes
                    // into a buffer; Enter dispatches, Esc cancels.
                    if app.producer_input_mode {
                        match key.code {
                            KeyCode::Esc => app.exit_producer_input_mode(),
                            KeyCode::Enter => app.commit_producer_seed(),
                            KeyCode::Backspace => app.pop_producer_input(),
                            KeyCode::Char(c) if c.is_ascii_hexdigit() && app.producer_input_buffer.len() < 64 => {
                                app.push_producer_input(c);
                            }
                            _ => {}
                        }
                        continue;
                    }
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc if !app.state.is_unlocked() => break,
                        KeyCode::Char('Q') => break,
                        KeyCode::Char('h') | KeyCode::Left => app.prev_tab(),
                        KeyCode::Char('l') | KeyCode::Right | KeyCode::Tab => app.next_tab(),
                        // === Claim screen (must come first — match
                        // arms are tried top-down, so the global
                        // `1/2/3` navigation bindings below would
                        // shadow these if they were reversed).
                        KeyCode::Enter if app.state.screen == Screen::Claim => {
                            app.submit_claim();
                        }
                        KeyCode::Char('x') if app.state.screen == Screen::Claim => {
                            app.cancel_claim();
                        }
                        KeyCode::Char('1') if app.state.screen == Screen::Claim => {
                            app.pick_claim_bucket(0)
                        }
                        KeyCode::Char('2') if app.state.screen == Screen::Claim => {
                            app.pick_claim_bucket(1)
                        }
                        KeyCode::Char('3') if app.state.screen == Screen::Claim => {
                            app.pick_claim_bucket(2)
                        }
                        KeyCode::F(5) => app.open_claim(),
                        // === Producer screen
                        KeyCode::Enter if app.state.screen == Screen::Producer => {
                            app.submit_producer();
                        }
                        KeyCode::Char('x') if app.state.screen == Screen::Producer => {
                            app.cancel_producer();
                        }
                        KeyCode::Char('s') if app.state.screen == Screen::Producer => {
                            app.enter_producer_input_mode();
                        }
                        KeyCode::Char('+') if app.state.screen == Screen::Producer => {
                            app.bump_producer_stake(1_000_000_000);
                        }
                        KeyCode::Char('-') if app.state.screen == Screen::Producer => {
                            app.bump_producer_stake(-1_000_000_000);
                        }
                        KeyCode::Char('0') if app.state.screen == Screen::Producer => {
                            app.set_producer_shard(0);
                        }
                        KeyCode::Char('1') if app.state.screen == Screen::Producer => {
                            app.set_producer_shard(1);
                        }
                        KeyCode::Char('2') if app.state.screen == Screen::Producer => {
                            app.set_producer_shard(2);
                        }
                        KeyCode::Char('3') if app.state.screen == Screen::Producer => {
                            app.set_producer_shard(3);
                        }
                        KeyCode::Char('4') if app.state.screen == Screen::Producer => {
                            app.set_producer_shard(4);
                        }
                        KeyCode::Char('5') if app.state.screen == Screen::Producer => {
                            app.set_producer_shard(5);
                        }
                        KeyCode::Char('6') if app.state.screen == Screen::Producer => {
                            app.set_producer_shard(6);
                        }
                        KeyCode::Char('7') if app.state.screen == Screen::Producer => {
                            app.set_producer_shard(7);
                        }
                        KeyCode::Char('8') if app.state.screen == Screen::Producer => {
                            app.set_producer_shard(8);
                        }
                        KeyCode::Char('9') if app.state.screen == Screen::Producer => {
                            app.set_producer_shard(9);
                        }
                        KeyCode::F(6) => app.open_producer(),
                        // === Send screen
                        KeyCode::Enter if app.state.screen == Screen::Send => {
                            app.action(WalletAction::SignAndSend);
                        }
                        KeyCode::Char('x') if app.state.screen == Screen::Send => {
                            app.action(WalletAction::CancelSend);
                        }
                        // === Receive screen
                        KeyCode::Char('c') if app.state.screen == Screen::Receive => {
                            app.action(WalletAction::CopyAddress);
                        }
                        KeyCode::Char('y') if app.state.screen == Screen::Receive => {
                            app.action(WalletAction::NewAddress);
                        }
                        // === Global tab navigation
                        KeyCode::Char('1') => app.navigate(0),
                        KeyCode::Char('2') => app.navigate(1),
                        KeyCode::Char('3') => app.navigate(2),
                        KeyCode::Char('4') => app.navigate(3),
                        KeyCode::Char('5') => app.navigate(4),
                        KeyCode::Char('6') => app.navigate(5),
                        KeyCode::Char('7') => app.navigate(6),
                        KeyCode::Char('n') => app.action(WalletAction::DismissNotification),
                        _ => {}
                    }
                }
            }
        }

        if app.state.quitting {
            break;
        }
    }
    Ok(())
}
