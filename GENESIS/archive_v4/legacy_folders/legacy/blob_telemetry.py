# blob_telemetry.py
import json
import asyncio
import threading
import time

try:
    import websockets
except ImportError:
    websockets = None

# Global telemetry state matching the dashboard requirements
BLOB_STATE = {
    "pos": [0.0, 0.0],
    "food": [2.5, 2.5],
    "hunger": 0.8,
    "speed": 0.0,
    "da": 0.0,
    "ne": 0.0,
    "steps_since_food": 0,
    "episode": 0,
    "steps_to_food_history": [],
    "neuron_states": [0.0] * 32,
    "W": [[0.0]*32 for _ in range(32)],
    "nav_signal": [],
    "reflex_fired": False,
    "reflex_ratio": 0.0,
    "novelty": 0.0,
    "uncertainty": 0.0,
    "is_sleeping": False,
    "fatigue": 0.0,
    "sigma": 0.0,
    "speed_drive": 0.0,
    "dreamer_loss": 0.0,
    "dreamer_surprise": 0.0,
    "hdc_familiarity": 0.0
}

_state_changed_event = threading.Event()
_server_thread = None

def update_state(**kwargs):
    """Update fields and set event flag for clients."""
    BLOB_STATE.update(kwargs)
    _state_changed_event.set()

async def _ws_handler(websocket):
    print("  [BLOB TELEMETRY] Web client connected.")
    try:
        while True:
            # Wait for state updates
            await asyncio.get_event_loop().run_in_executor(None, _state_changed_event.wait, 0.5)
            _state_changed_event.clear()
            
            # Send payload
            payload = json.dumps(BLOB_STATE)
            await websocket.send(payload)
            
            # Limit rate to ~30Hz
            await asyncio.sleep(1/30.0)
    except websockets.exceptions.ConnectionClosed:
        print("  [BLOB TELEMETRY] Web client disconnected.")
    except Exception as e:
        print(f"  [BLOB TELEMETRY] Error: {e}")

async def _main():
    print("  [BLOB TELEMETRY] Starting WebSocket server on ws://localhost:8766")
    async with websockets.serve(_ws_handler, "localhost", 8766):
        await asyncio.Future()

def _run_server():
    asyncio.run(_main())

def start():
    global _server_thread
    if websockets is None:
        print("  [BLOB TELEMETRY] websockets package is missing! Telemetry offline. Run: pip install websockets")
        return
    _server_thread = threading.Thread(target=_run_server, daemon=True)
    _server_thread.start()
