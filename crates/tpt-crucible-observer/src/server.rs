//! The WebSocket telemetry server.
//!
//! [`TelemetryServer::bind`] opens a listener and spawns the axum app on the
//! current tokio runtime. Emitters push [`TelemetryEvent`]s through
//! [`TelemetryServer::emit`]; each connected `/ws` client receives every
//! event as a JSON text frame, fanned out through a `tokio::sync::broadcast`
//! channel so a slow dashboard never back-pressures the swarm.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::routing::get;
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::broadcast;

use crate::TelemetryEvent;
use tpt_crucible_common::Error;

/// Broadcast buffer per connected client; overflow marks the client lagged
/// (skipped events) rather than blocking emitters.
const CHANNEL_CAPACITY: usize = 1024;

type EventTx = broadcast::Sender<Arc<TelemetryEvent>>;

#[derive(Clone)]
struct AppState {
    tx: EventTx,
}

/// A bound telemetry server broadcasting [`TelemetryEvent`]s over WebSockets.
///
/// Clone-free emitters: hand [`TelemetryServer::subscribe`] receivers to
/// in-process dashboards or tests; remote clients connect to `GET /ws`.
#[derive(Debug, Clone)]
pub struct TelemetryServer {
    addr: SocketAddr,
    tx: EventTx,
}

impl TelemetryServer {
    /// Bind `addr` (`ip:port`) and spawn the serving task on the current
    /// runtime.
    ///
    /// Use port `0` to let the OS pick one; read it back with
    /// [`Self::local_addr`].
    ///
    /// # Errors
    /// * [`Error::InvalidArgument`] when `addr` does not parse,
    /// * [`Error::Io`] when the listener cannot bind.
    pub async fn bind(addr: impl AsRef<str>) -> tpt_crucible_common::Result<Self> {
        let parsed: SocketAddr = addr.as_ref().parse().map_err(|_| {
            Error::InvalidArgument(format!(
                "invalid bind address `{}` (expected ip:port)",
                addr.as_ref()
            ))
        })?;
        let listener = tokio::net::TcpListener::bind(parsed).await?;
        let local = listener.local_addr()?;
        let (tx, _) = broadcast::channel(CHANNEL_CAPACITY);
        let state = AppState { tx: tx.clone() };

        let app = Router::new()
            .route("/", get(root))
            .route("/ws", get(ws_upgrade))
            .with_state(state);
        tokio::spawn(async move {
            if let Err(err) = axum::serve(listener, app).await {
                // Serving only ends on bind/runtime teardown; surfacing to a
                // log sink is enough here (no tracing dep by design).
                eprintln!("observer server stopped: {err}");
            }
        });

        Ok(Self { addr: local, tx })
    }

    /// The resolved address (useful after binding port `0`).
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Fan an event out to every connected WebSocket client.
    ///
    /// Returns `false` when no client is connected (the event is dropped —
    /// telemetry is best-effort by design).
    pub fn emit(&self, event: TelemetryEvent) -> bool {
        self.tx.send(Arc::new(event)).is_ok()
    }

    /// In-process tap on the same stream the sockets receive.
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<TelemetryEvent>> {
        self.tx.subscribe()
    }

    /// Number of currently connected WebSocket clients.
    pub fn clients(&self) -> usize {
        self.tx.receiver_count()
    }
}

async fn root() -> &'static str {
    "tpt-crucible-observer telemetry"
}

async fn ws_upgrade(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> axum::response::Response {
    ws.on_upgrade(move |socket| stream_events(socket, state.tx.subscribe()))
}

async fn stream_events(socket: WebSocket, mut rx: broadcast::Receiver<Arc<TelemetryEvent>>) {
    let (mut sink, _incoming) = socket.split();
    loop {
        match rx.recv().await {
            Ok(event) => {
                let Ok(json) = serde_json::to_string(event.as_ref()) else {
                    continue;
                };
                if sink.send(Message::Text(json.into())).await.is_err() {
                    break; // client went away
                }
            }
            Err(broadcast::error::RecvError::Lagged(skipped)) => {
                // Slow client: tell it how much it missed, keep going.
                let note = format!("{{\"lagged\":{skipped}}}");
                if sink.send(Message::Text(note.into())).await.is_err() {
                    break;
                }
            }
            Err(broadcast::error::RecvError::Closed) => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use std::time::Duration;
    use tokio_tungstenite::tungstenite::Message as WsMessage;

    fn sample(id: &str, tps: f64) -> TelemetryEvent {
        TelemetryEvent {
            node_id: id.into(),
            hw_type: "alloy".into(),
            tokens_per_sec: Some(tps),
            mem_bandwidth_gbps: None,
            thermal_drift_c: None,
            latency_ms: Some(0.7),
            timestamp_ms: 42,
        }
    }

    async fn connect(
        addr: SocketAddr,
    ) -> tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>
    {
        let url = format!("ws://{addr}/ws");
        let (ws, _resp) = tokio_tungstenite::connect_async(url)
            .await
            .expect("client connects");
        ws
    }

    async fn next_text(
        ws: &mut tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    ) -> String {
        let msg = tokio::time::timeout(Duration::from_secs(3), ws.next())
            .await
            .expect("frame within 3s")
            .expect("stream open")
            .expect("no ws error");
        match msg {
            WsMessage::Text(t) => t.to_string(),
            other => panic!("expected text frame, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn binds_ephemeral_port_and_reports_it() {
        let server = TelemetryServer::bind("127.0.0.1:0").await.unwrap();
        assert_ne!(server.local_addr().port(), 0);
        assert_eq!(server.clients(), 0);
    }

    #[tokio::test]
    async fn websocket_clients_receive_emitted_events_as_json() {
        let server = TelemetryServer::bind("127.0.0.1:0").await.unwrap();
        let addr = server.local_addr();

        let mut client = connect(addr).await;
        // Wait for the upgrade to land on the server side.
        while server.clients() == 0 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        let first = sample("swarm-03", 12.5);
        let second = sample("fusion-00", 3.25);
        assert!(server.emit(first.clone()));
        assert!(server.emit(second));

        for expected in [&first, &sample("fusion-00", 3.25)] {
            let text = next_text(&mut client).await;
            let back: TelemetryEvent = serde_json::from_str(&text).unwrap();
            assert_eq!(&back, expected);
        }
    }

    #[tokio::test]
    async fn in_process_subscribers_see_the_same_stream() {
        let server = TelemetryServer::bind("127.0.0.1:0").await.unwrap();
        let mut rx = server.subscribe();

        let ev = sample("element-01", 1.0);
        assert!(server.emit(ev.clone()));
        let got = rx.recv().await.unwrap();
        assert_eq!(got.as_ref(), &ev);
    }

    #[tokio::test]
    async fn emit_without_clients_reports_false() {
        let server = TelemetryServer::bind("127.0.0.1:0").await.unwrap();
        assert!(!server.emit(sample("swarm-00", 0.0)));
    }
}
