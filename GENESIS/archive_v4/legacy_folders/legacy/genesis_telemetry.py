# genesis_telemetry.py
import json
import asyncio
import threading
import time

try:
    import websockets
except ImportError:
    print("  [TELEMETRY] Please run: pip install websockets")
    websockets = None

# Global schema matching the dashboard expectations
TELEMETRY_STATE = {
    "A": None,
    "B": None,
    "world": {
        "food": [],
        "danger_grid": [],
        "day_factor": 1.0
    }
}

_state_changed_event = threading.Event()
_server_thread = None
_loop = None

def get_empty_bot_state():
    return {
        "pos": [0.0, 0.0],
        "yaw": 0.0,
        "lidar": [5.0]*16,
        "nm": {"DA": 0.5, "NE": 0.2, "ACh": 0.4, "SHT": 0.6},
        "state": {
            "confidence": 0.5, "fear": 0.0, "curiosity": 0.5, 
            "hunger": 0.0, "grief": 0.0, "fatigue": 0.0
        },
        "mood": 0.5,
        "energy": 100.0,
        "mode": "IDLE",
        "reflex_fired": False,
        "reflex_ratio": 0.0,
        "neck": 0.15,
        "head_pan": 0.0,
        "step": 0,
        "target": [0.0, 0.0],
        "sonar_vector": [0.0, 0.0]
    }

TELEMETRY_STATE["A"] = get_empty_bot_state()
TELEMETRY_STATE["B"] = get_empty_bot_state()

def update_bot_state(b_name, **kwargs):
    """Update specific fields for a bot and flag a change."""
    if b_name in TELEMETRY_STATE:
        TELEMETRY_STATE[b_name].update(kwargs)
        _state_changed_event.set()

def update_world_state(**kwargs):
    """Update world fields and flag a change."""
    TELEMETRY_STATE["world"].update(kwargs)
    _state_changed_event.set()

async def _ws_handler(websocket):
    """Handles an individual websocket client."""
    print("  [TELEMETRY] Client connected.")
    try:
        while True:
            # Wait until there is new data to send
            await asyncio.get_event_loop().run_in_executor(None, _state_changed_event.wait, 1.0)
            
            # Clear the event flag. If another update happens while sending, it will be set again.
            _state_changed_event.clear()
            
            # Send the payload
            payload = json.dumps(TELEMETRY_STATE)
            await websocket.send(payload)
            
            # Throttle to ~30Hz max
            await asyncio.sleep(1/30.0)
    except websockets.exceptions.ConnectionClosed:
        print("  [TELEMETRY] Client disconnected.")
    except Exception as e:
        print(f"  [TELEMETRY] Error: {e}")

async def _main():
    print("  [TELEMETRY] Starting WebSocket server on ws://localhost:8765")
    async with websockets.serve(_ws_handler, "localhost", 8765):
        await asyncio.Future()  # run forever

def _run_server():
    """Runs the asyncio event loop for the websocket server."""
    asyncio.run(_main())

def start():
    """Starts the telemetry server in a background thread."""
    global _server_thread
    if websockets is None:
        return
    _server_thread = threading.Thread(target=_run_server, daemon=True)
    _server_thread.start()
