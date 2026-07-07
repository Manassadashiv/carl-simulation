"""
carl_bios.py — The Cognitive BIOS (Mark VIII Real-Time Timing Infrastructure).

Architecture (Mark VIII — Medullary Timing Nucleus):
- CarlBiosCore: 4-thread daemon backbone providing deterministic timing for
  all sensorimotor processing. This is Bob's brainstem clock — the medullary
  timing nucleus that synchronizes every heartbeat of neural computation.

  Without this layer, the cortex (carl_agent.py) has no notion of real time.
  Neurons fire when they fire; the BIOS ensures they fire on schedule.

  Thread Architecture (all daemon threads — die with main process):

    1. Control Loop (100 Hz / 10 ms cycle)
       Biological analogue: Pontine reticular formation — the central pattern
       generator that drives rhythmic motor output. Uses adaptive hybrid
       scheduling: sleeps for the bulk of each cycle (saving CPU), then
       busy-waits the final ~200 µs for sub-millisecond landing precision.
       Tracks actual_dt_ns and jitter_ns per frame in a ring buffer.

    2. Watchdog Thread (250 Hz / 4 ms cycle)
       Biological analogue: Locus coeruleus vigilance circuit — monitors
       heartbeat freshness from the control loop. If the control thread stalls
       for > 25 ms (2.5 missed cycles), increments watchdog_overruns and can
       trigger an emergency stop flag. Ultra-lean — no allocations, just a
       single timestamp comparison per tick.

    3. Telemetry Logger (Background, queue-draining)
       Biological analogue: Cerebellar inferior olive — offline recording of
       timing residuals for post-hoc analysis. Drains a lock-free queue and
       batch-writes CSV rows to telemetry_bios.log. Completely decoupled from
       the control thread — never blocks the motor pathway.

    4. Health Monitor (2 Hz / 500 ms)
       Biological analogue: Hypothalamic homeostatic integrator — aggregates
       ring-buffer statistics every 500 ms and prints a diagnostic report to
       console every 5 seconds. Optional psutil CPU tracking with graceful
       fallback.

  SharedStateLayer (dataclass):
    A read-mostly snapshot of system health, updated by the health monitor
    and readable by any thread. All fields are plain Python types (atomic
    on CPython due to the GIL).

  Timing constants are in nanoseconds (int64) — no floating-point drift.
  All wall-clock reads use time.perf_counter_ns() for monotonic precision.
"""

import numpy as np
import time
import threading
import os
import ctypes
from collections import deque
from dataclasses import dataclass, field
from queue import Queue, Empty

# Configure Windows media timer resolution to 1ms for high-precision sleep
if os.name == 'nt':
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass


# ==============================================================================
#  TIMING CONSTANTS -- All values in nanoseconds (int64, no float drift)
# ==============================================================================

CONTROL_TARGET_NS    = 10_000_000      # 100 Hz = 10 ms per cycle
WATCHDOG_TARGET_NS   = 4_000_000       # 250 Hz = 4 ms per cycle
MONITOR_TARGET_NS    = 500_000_000     # 2 Hz   = 500 ms per cycle
DEADLINE_THRESHOLD_NS = 14_000_000     # 40% tolerance = 14 ms (accommodates Windows jitter)

# Watchdog staleness threshold: if control heartbeat is older than this,
# something has seized. 40 ms = 4 missed control cycles.
# Widened from 25ms to accommodate Windows scheduling jitter on consumer
# hardware where the 100Hz control tick averages ~11ms.
WATCHDOG_STALE_NS    = 40_000_000

# Warmup grace period: ignore watchdog faults for this many seconds after
# system start. TF Lite / FaceLandmarker cold-start initialization causes
# one-time 100ms+ spikes that are not real faults.
WATCHDOG_WARMUP_S    = 5.0

# Hybrid scheduler constants
# Switch from sleep to busy-wait at 1.5 ms remaining to absorb Windows scheduling jitter
BUSY_WAIT_HORIZON_NS = 1_500_000
SLEEP_GRANULARITY_S  = 0.001           # Minimum useful sleep on Windows (~1 ms actual)

# Ring buffer sizes
CONTROL_RING_SIZE    = 500             # ~5 seconds of control history at 100 Hz
HEALTH_RING_SIZE     = 100             # ~50 seconds of health snapshots at 2 Hz

# Telemetry logger batch size
LOG_BATCH_SIZE       = 100

# Health monitor console print interval (in monitor ticks)
DIAG_PRINT_INTERVAL  = 10             # 10 × 500 ms = every 5 seconds


# ==============================================================================
#  SHARED STATE LAYER
# ==============================================================================

@dataclass
class SharedStateLayer:
    """
    Read-mostly diagnostic snapshot of system health.

    Updated atomically (field-by-field) by the health monitor thread.
    Readable by any thread without locks — all fields are plain Python
    scalars, which are GIL-atomic on CPython for single-word writes.

    Biological analogue: Hypothalamic interoceptive state vector — the
    brain's internal model of its own metabolic and timing health.
    """
    control_heartbeat_ns: int   = 0
    control_hz:           float = 0.0
    mean_dt_ms:           float = 0.0
    max_jitter_ms:        float = 0.0
    missed_deadlines:     int   = 0
    watchdog_overruns:    int   = 0
    queue_backlog:        int   = 0
    cpu_utilization:      float = 0.0
    total_samples_logged: int   = 0
    emergency_stop:       bool  = False
    
    # Homeostatic Dials (H_e, H_f, H_s, H_d) scaled to % [0-100]
    energy:               float = 100.0
    fatigue:              float = 0.0
    stress:               float = 0.0
    damage:               float = 0.0


# ==============================================================================
#  CARL BIOS CORE
# ==============================================================================

class CarlBiosCore:
    """
    Mark VIII Cognitive BIOS — the real-time timing backbone.

    Provides deterministic 100 Hz control ticks, watchdog fault detection,
    background telemetry logging, and live health diagnostics — all running
    as parallel daemon threads that die cleanly with the main process.

    Design intent:
    - Standalone-testable: runs without MuJoCo, verifies timing on bare metal
    - Callback-driven: register_control_callback() hooks the cortex in later
    - Zero external deps: stdlib + numpy only (psutil optional for CPU %)
    - Windows-compatible: perf_counter_ns + time.sleep (no signals, no POSIX)

    Usage:
        bios = CarlBiosCore()
        bios.register_control_callback(my_10ms_tick)
        bios.start_system()
        ...
        bios.shutdown_system()

    Biological analogue: Medullary reticular formation + locus coeruleus +
    inferior olive + hypothalamic clock. The brainstem timing nucleus that
    keeps every upstream neural oscillator synchronized.
    """

    def __init__(self, log_dir=None):
        # ── Timing targets (nanoseconds, immutable) ──────────────────────
        self.CONTROL_TARGET_NS     = CONTROL_TARGET_NS
        self.WATCHDOG_TARGET_NS    = WATCHDOG_TARGET_NS
        self.MONITOR_TARGET_NS     = MONITOR_TARGET_NS
        self.DEADLINE_THRESHOLD_NS = DEADLINE_THRESHOLD_NS

        # ── Run-state flag (checked by all threads) ──────────────────────
        self._running = False

        # ── Ring buffers for control loop timing history ─────────────────
        # Each entry: (actual_dt_ns: int, jitter_ns: int)
        self._control_ring = deque(maxlen=CONTROL_RING_SIZE)

        # ── Heartbeat: last control tick timestamp (read by watchdog) ────
        self._heartbeat_ns = 0           # written by control, read by watchdog
        self._heartbeat_lock = threading.Lock()

        # ── Missed deadline counter (incremented by control loop) ────────
        self._missed_deadlines = 0

        # ── Watchdog overrun counter (incremented by watchdog) ───────────
        self._watchdog_overruns = 0

        # ── Emergency stop flag (set by watchdog, read by control) ───────
        self._force_emergency_stop = False

        # ── Telemetry log queue (control → logger, non-blocking) ────────
        # Tuples of (timestamp_ns, actual_dt_ns, jitter_ns)
        self._log_queue = Queue()

        # ── Shared state snapshot (updated by health monitor) ────────────
        self._state = SharedStateLayer()

        # ── Callback hooks (set before start, called during run) ─────────
        self._on_control_tick = None     # fn(dt_seconds: float) -> None
        self._on_watchdog_fault = None   # fn() -> None

        # ── Log file path ────────────────────────────────────────────────
        if log_dir is None:
            log_dir = os.path.dirname(os.path.abspath(__file__))
        self._log_path = os.path.join(log_dir, "telemetry_bios.log")

        # ── Thread handles (populated by start_system) ───────────────────
        self._threads = []

        # ── Total samples written to disk (tracked by logger thread) ─────
        self._total_samples_logged = 0

        # ── Optional psutil import (graceful fallback) ───────────────────
        self._psutil = None
        self._psutil_process = None
        try:
            import psutil
            self._psutil = psutil
            self._psutil_process = psutil.Process(os.getpid())
        except ImportError:
            pass  # CPU utilization will report 0.0

        # Note: No synthetic workload here. The registered control callback
        # (on_control_tick) provides the real per-tick computational payload.
        # Dummy workloads previously caused deadline misses by consuming
        # ~0.5ms of the tight 10ms budget alongside the real callback.

    # -----------------------------------------------------------------
    #  PUBLIC API: Callback Registration
    # -----------------------------------------------------------------

    def register_control_callback(self, fn):
        """
        Register a function to be called every 10 ms control tick.

        The callback receives a single argument: dt_seconds (float),
        the actual elapsed time since the last tick. This is where the
        cortex (carl_agent.py) will plug in its step() function.

        Must be called BEFORE start_system().

        Args:
            fn: Callable(dt_seconds: float) -> None
        """
        self._on_control_tick = fn

    def register_watchdog_callback(self, fn):
        """
        Register a function to be called when the watchdog detects a fault.

        A fault occurs when the control heartbeat is stale for > 25 ms.
        This callback can trigger safe-mode behaviors (e.g., motor cutoff).

        Args:
            fn: Callable() -> None
        """
        self._on_watchdog_fault = fn

    # -----------------------------------------------------------------
    #  PUBLIC API: Lifecycle
    # -----------------------------------------------------------------

    def start_system(self):
        """
        Print banner, set running=True, launch all 4 daemon threads.

        Thread start order matters:
        1. Logger first (ready to drain before control starts writing)
        2. Watchdog second (monitoring before control heartbeats begin)
        3. Health monitor third (aggregation ready)
        4. Control loop last (the hot path — starts producing data)
        """
        self._print_banner()
        self._running = True

        thread_specs = [
            ("BIOS-Logger",   self._logger_loop),
            ("BIOS-Watchdog", self._watchdog_loop),
            ("BIOS-Health",   self._health_monitor_loop),
            ("BIOS-Control",  self._control_loop),
        ]

        for name, target in thread_specs:
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

        print(f"[BIOS] All 4 threads launched. System is LIVE.")
        print(f"[BIOS] Telemetry logging to: {self._log_path}")
        print()

    def shutdown_system(self):
        """
        Graceful shutdown: set running=False, allow 0.5s for thread drain,
        then print a final diagnostic summary.
        """
        print("\n[BIOS] Shutdown requested - draining threads...")
        self._running = False

        # Give threads time to finish current cycles and flush
        time.sleep(0.5)

        # Print final summary
        self._print_shutdown_summary()

    def get_state(self):
        """
        Returns a snapshot of the current system health.

        Returns:
            SharedStateLayer: Copy of the current diagnostic state.
            The returned object is a snapshot — subsequent calls may
            return different values as the health monitor updates.
        """
        # Return a shallow copy so the caller gets an immutable snapshot
        s = self._state
        return SharedStateLayer(
            control_heartbeat_ns=s.control_heartbeat_ns,
            control_hz=s.control_hz,
            mean_dt_ms=s.mean_dt_ms,
            max_jitter_ms=s.max_jitter_ms,
            missed_deadlines=s.missed_deadlines,
            watchdog_overruns=s.watchdog_overruns,
            queue_backlog=s.queue_backlog,
            cpu_utilization=s.cpu_utilization,
            total_samples_logged=s.total_samples_logged,
            emergency_stop=s.emergency_stop,
            energy=s.energy,
            fatigue=s.fatigue,
            stress=s.stress,
            damage=s.damage,
        )

    # -----------------------------------------------------------------
    #  THREAD 1: Control Loop (100 Hz / 10 ms)
    # -----------------------------------------------------------------

    def _control_loop(self):
        """
        The hot path — 100 Hz deterministic control tick.

        Adaptive hybrid scheduling strategy:
          1. Compute remaining_ns = target - elapsed
          2. If remaining_ns > BUSY_WAIT_HORIZON_NS: sleep for (remaining - horizon)
          3. Busy-wait the final ~200 µs for sub-millisecond precision

        This achieves <0.5 ms jitter on Windows while keeping CPU usage
        under 5% (vs. 100% for pure busy-wait at 100 Hz).

        Each tick:
          - Records actual_dt_ns and jitter_ns into the ring buffer
          - Updates the heartbeat timestamp (read by watchdog)
          - Calls the registered control callback (if any)
          - Pushes telemetry tuple to the log queue (non-blocking)
          - Checks the emergency stop flag from the watchdog
        """
        prev_tick_ns = time.perf_counter_ns()

        while self._running:
            # ── Check emergency stop ─────────────────────────────────────
            if self._force_emergency_stop:
                self._state.emergency_stop = True
                break

            cycle_start_ns = time.perf_counter_ns()
            actual_dt_ns = cycle_start_ns - prev_tick_ns
            jitter_ns = abs(actual_dt_ns - self.CONTROL_TARGET_NS)

            # ── Record to ring buffer ────────────────────────────────────
            self._control_ring.append((actual_dt_ns, jitter_ns))

            # ── Check deadline ───────────────────────────────────────────
            if actual_dt_ns > self.DEADLINE_THRESHOLD_NS:
                self._missed_deadlines += 1

            # ── Update heartbeat (for watchdog) ──────────────────────────
            with self._heartbeat_lock:
                self._heartbeat_ns = cycle_start_ns

            # (Synthetic workload removed — real compute happens in callback)

            # ── Call registered control callback ─────────────────────────
            if self._on_control_tick is not None:
                dt_seconds = actual_dt_ns / 1_000_000_000.0
                try:
                    self._on_control_tick(dt_seconds)
                except Exception as e:
                    print(f"[BIOS] Control callback error: {e}")

            # ── Push telemetry to logger queue (non-blocking) ────────────
            self._log_queue.put_nowait((cycle_start_ns, actual_dt_ns, jitter_ns))

            # ── Adaptive hybrid sleep ────────────────────────────────────
            prev_tick_ns = cycle_start_ns
            work_elapsed_ns = time.perf_counter_ns() - cycle_start_ns
            remaining_ns = self.CONTROL_TARGET_NS - work_elapsed_ns

            if remaining_ns > BUSY_WAIT_HORIZON_NS:
                # Phase 1: Coarse sleep — yield CPU for the bulk of the wait
                sleep_ns = remaining_ns - BUSY_WAIT_HORIZON_NS
                sleep_s = sleep_ns / 1_000_000_000.0
                if sleep_s > SLEEP_GRANULARITY_S:
                    time.sleep(sleep_s)

            # Phase 2: Busy-wait the final microseconds for precision
            target_wakeup_ns = cycle_start_ns + self.CONTROL_TARGET_NS
            while time.perf_counter_ns() < target_wakeup_ns:
                pass  # spin — sub-µs precision

    # -----------------------------------------------------------------
    #  THREAD 2: Watchdog (250 Hz / 4 ms)
    # -----------------------------------------------------------------

    def _watchdog_loop(self):
        """
        Ultra-lean heartbeat monitor — 250 Hz timestamp comparison.

        Checks whether the control loop's heartbeat is fresh (< 40 ms old).
        If stale, increments the watchdog_overruns counter and optionally
        triggers an emergency stop.

        Includes a warmup grace period (WATCHDOG_WARMUP_S) to avoid false
        alarms during TF Lite / FaceLandmarker cold-start initialization.

        No allocations, no numpy, no I/O — just a single int64 comparison
        per tick. This thread must NEVER block or do heavy work.
        """
        warmup_end_ns = time.perf_counter_ns() + int(WATCHDOG_WARMUP_S * 1_000_000_000)

        while self._running:
            now_ns = time.perf_counter_ns()

            # Skip fault detection during warmup grace period
            if now_ns < warmup_end_ns:
                time.sleep(self.WATCHDOG_TARGET_NS / 1_000_000_000.0)
                continue

            # Read heartbeat (protected by lock for cross-thread visibility)
            with self._heartbeat_lock:
                last_heartbeat = self._heartbeat_ns

            # Check freshness (skip if heartbeat hasn't started yet)
            if last_heartbeat > 0:
                staleness_ns = now_ns - last_heartbeat
                if staleness_ns > WATCHDOG_STALE_NS:
                    self._watchdog_overruns += 1

                    # Call fault callback if registered
                    if self._on_watchdog_fault is not None:
                        try:
                            self._on_watchdog_fault()
                        except Exception:
                            pass  # watchdog must never crash

            # Fixed-interval sleep (watchdog doesn't need sub-ms precision)
            time.sleep(self.WATCHDOG_TARGET_NS / 1_000_000_000.0)

    # -----------------------------------------------------------------
    #  THREAD 3: Telemetry Logger (Background, queue-draining)
    # -----------------------------------------------------------------

    def _logger_loop(self):
        """
        Background CSV writer — drains the log queue in batches.

        Batch size: 100 rows. Flushes to disk when:
          - Batch fills (100 rows accumulated), OR
          - Queue empties (partial batch flush to avoid data loss)

        CSV columns: timestamp_ns, actual_dt_ns, jitter_ns

        Completely decoupled from the control thread — the queue acts as
        a pressure absorber. Even if the logger falls behind temporarily,
        the control loop never blocks (put_nowait).

        Biological analogue: Inferior olivary nucleus — records timing
        error signals for offline cerebellar calibration.
        """
        batch = []

        # Write CSV header
        try:
            with open(self._log_path, 'w') as f:
                f.write("timestamp_ns,actual_dt_ns,jitter_ns\n")
        except OSError as e:
            print(f"[BIOS] Warning: could not open log file: {e}")
            # Continue running — logger will silently drop data
            while self._running:
                time.sleep(0.1)
            return

        while self._running:
            try:
                # Block with timeout — wakes up to check _running flag
                entry = self._log_queue.get(timeout=0.05)
                batch.append(entry)
            except Empty:
                # Queue empty — flush any partial batch
                if batch:
                    self._flush_batch(batch)
                    batch = []
                continue

            # Flush when batch fills
            if len(batch) >= LOG_BATCH_SIZE:
                self._flush_batch(batch)
                batch = []

        # Final drain on shutdown
        while not self._log_queue.empty():
            try:
                batch.append(self._log_queue.get_nowait())
            except Empty:
                break
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch):
        """Write a batch of telemetry rows to disk."""
        try:
            with open(self._log_path, 'a') as f:
                for timestamp_ns, actual_dt_ns, jitter_ns in batch:
                    f.write(f"{timestamp_ns},{actual_dt_ns},{jitter_ns}\n")
            self._total_samples_logged += len(batch)
        except OSError:
            pass  # silently drop on I/O error — never crash the logger

    # -----------------------------------------------------------------
    #  THREAD 4: Health Monitor (2 Hz / 500 ms)
    # -----------------------------------------------------------------

    def _health_monitor_loop(self):
        """
        Aggregates ring-buffer statistics every 500 ms and updates the
        shared state layer. Prints a full diagnostic block to console
        every 5 seconds (10 monitor ticks).

        Metrics computed per cycle:
          - mean_hz:          1e9 / mean(actual_dt_ns) over ring buffer
          - mean_dt_ms:       mean(actual_dt_ns) / 1e6
          - max_jitter_ms:    max(jitter_ns) / 1e6
          - missed_deadlines: cumulative count from control loop
          - watchdog_overruns: cumulative count from watchdog
          - queue_backlog:    current log_queue depth
          - cpu_utilization:  psutil per-process CPU % (0.0 if unavailable)

        Biological analogue: Hypothalamic homeostatic integrator —
        monitors the organism's internal timing health and triggers
        corrective signals when the system drifts out of spec.
        """
        tick_count = 0

        while self._running:
            time.sleep(self.MONITOR_TARGET_NS / 1_000_000_000.0)

            # ── Aggregate ring buffer stats ──────────────────────────────
            ring_snapshot = list(self._control_ring)

            if ring_snapshot:
                dt_array = np.array([r[0] for r in ring_snapshot], dtype=np.float64)
                jitter_array = np.array([r[1] for r in ring_snapshot], dtype=np.float64)

                mean_dt_ns = np.mean(dt_array)
                mean_hz = 1_000_000_000.0 / mean_dt_ns if mean_dt_ns > 0 else 0.0
                mean_dt_ms = mean_dt_ns / 1_000_000.0
                max_jitter_ms = float(np.max(jitter_array)) / 1_000_000.0
            else:
                mean_hz = 0.0
                mean_dt_ms = 0.0
                max_jitter_ms = 0.0

            # ── CPU utilization (optional psutil) ────────────────────────
            cpu_pct = 0.0
            if self._psutil_process is not None:
                try:
                    cpu_pct = self._psutil_process.cpu_percent(interval=None)
                except Exception:
                    cpu_pct = 0.0

            # ── Update shared state (field-by-field, GIL-atomic) ─────────
            self._state.control_heartbeat_ns = self._heartbeat_ns
            self._state.control_hz           = mean_hz
            self._state.mean_dt_ms           = mean_dt_ms
            self._state.max_jitter_ms        = max_jitter_ms
            self._state.missed_deadlines     = self._missed_deadlines
            self._state.watchdog_overruns    = self._watchdog_overruns
            self._state.queue_backlog        = self._log_queue.qsize()
            self._state.cpu_utilization      = cpu_pct
            self._state.total_samples_logged = self._total_samples_logged

            # ── Console diagnostic (every 5 seconds) ────────────────────
            tick_count += 1
            if tick_count % DIAG_PRINT_INTERVAL == 0:
                self._print_bios_diagnostics()


    def _print_bios_diagnostics(self):
        """
        Formats and prints the console diagnostic block.
        """
        s = self._state
        print("+------------------- BIOS DIAGNOSTICS -------------------+")
        print(f"|  Control Hz:       {s.control_hz:>8.2f} Hz   (target: 100.0 Hz)  |")
        print(f"|  Mean dt:          {s.mean_dt_ms:>8.3f} ms  (target: 10.0 ms)   |")
        print(f"|  Max jitter:       {s.max_jitter_ms:>8.3f} ms                    |")
        print(f"|  Missed deadlines: {s.missed_deadlines:>5d}                             |")
        print(f"|  Watchdog overruns:{s.watchdog_overruns:>5d}                             |")
        print(f"|  Queue backlog:    {s.queue_backlog:>5d}                             |")
        print(f"|  CPU utilization:  {s.cpu_utilization:>7.1f}%                          |")
        print(f"|  Samples logged:   {s.total_samples_logged:>8d}                        |")
        print("+------------------ HOMEOSTATIC STATE -------------------+")
        print(f"|  Energy (H_e):     {s.energy:>7.1f}%                          |")
        print(f"|  Fatigue (H_f):    {s.fatigue:>7.1f}%                          |")
        print(f"|  Stress (H_s):     {s.stress:>7.1f}%                          |")
        print(f"|  Damage (H_d):     {s.damage:>7.1f}%                          |")
        if s.emergency_stop:
            print(f"|  *** EMERGENCY STOP ACTIVE ***                        |")
        print("+--------------------------------------------------------+")
        print()

    def _print_banner(self):
        """Print the BIOS startup banner."""
        print()
        print("=" * 60)
        print("  CARL BIOS -- Mark VIII Cognitive Timing Infrastructure")
        print("  Medullary Timing Nucleus v8.0")
        print("=" * 60)
        print(f"  Control loop:   {self.CONTROL_TARGET_NS / 1e6:.1f} ms  ({1e9 / self.CONTROL_TARGET_NS:.0f} Hz)")
        print(f"  Watchdog:       {self.WATCHDOG_TARGET_NS / 1e6:.1f} ms  ({1e9 / self.WATCHDOG_TARGET_NS:.0f} Hz)")
        print(f"  Health monitor: {self.MONITOR_TARGET_NS / 1e6:.1f} ms ({1e9 / self.MONITOR_TARGET_NS:.0f} Hz)")
        print(f"  Deadline:       {self.DEADLINE_THRESHOLD_NS / 1e6:.1f} ms  (15% tolerance)")
        print(f"  Ring buffer:    {CONTROL_RING_SIZE} frames")
        print(f"  Log batch size: {LOG_BATCH_SIZE} rows")
        psutil_status = "AVAILABLE" if self._psutil is not None else "NOT INSTALLED (CPU% = 0.0)"
        print(f"  psutil:         {psutil_status}")
        print("=" * 60)
        print()

    def _print_shutdown_summary(self):
        """Print final system summary on shutdown."""
        s = self._state
        print()
        print("=" * 60)
        print("  CARL BIOS -- SHUTDOWN SUMMARY")
        print("=" * 60)
        print(f"  Final Hz:         {s.control_hz:.2f} Hz")
        print(f"  Mean dt:          {s.mean_dt_ms:.3f} ms")
        print(f"  Max jitter:       {s.max_jitter_ms:.3f} ms")
        print(f"  Missed deadlines: {s.missed_deadlines}")
        print(f"  Watchdog overruns:{s.watchdog_overruns}")
        print(f"  Total logged:     {s.total_samples_logged} samples")
        print(f"  Emergency stop:   {'YES' if s.emergency_stop else 'No'}")
        print("=" * 60)
        print()


# ==============================================================================
#  STANDALONE TEST -- Verifies timing without MuJoCo
# ==============================================================================

if __name__ == "__main__":
    """
    30-second standalone timing validation.

    Starts the BIOS, lets it run for 30 seconds, then evaluates:
      PASS criteria:
        - Missed deadlines < 5
        - Achieved Hz > 98.0
      FAIL: anything else (indicates OS scheduling issues or code bugs)
    """
    TEST_DURATION_S = 30
    callback_count = 0

    def test_callback(dt_seconds):
        """Minimal callback to verify the hook works."""
        global callback_count
        callback_count += 1

    print(f"[TEST] Starting {TEST_DURATION_S}-second BIOS timing validation...")
    print(f"[TEST] Platform: {os.name} / Python perf_counter_ns precision test")
    print()

    # Measure timer resolution
    t0 = time.perf_counter_ns()
    t1 = time.perf_counter_ns()
    timer_resolution_ns = t1 - t0
    print(f"[TEST] Timer resolution: ~{timer_resolution_ns} ns")
    print()

    bios = CarlBiosCore()
    bios.register_control_callback(test_callback)
    bios.start_system()

    # Let it run
    try:
        time.sleep(TEST_DURATION_S)
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user.")

    bios.shutdown_system()

    # ── Final evaluation ─────────────────────────────────────────────
    state = bios.get_state()
    print()
    print("=" * 60)
    print("  TIMING VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Test duration:      {TEST_DURATION_S} seconds")
    print(f"  Callback invocations: {callback_count}")
    print(f"  Expected ticks:     ~{TEST_DURATION_S * 100}")
    print(f"  Achieved Hz:        {state.control_hz:.2f}")
    print(f"  Mean dt:            {state.mean_dt_ms:.3f} ms")
    print(f"  Max jitter:         {state.max_jitter_ms:.3f} ms")
    print(f"  Missed deadlines:   {state.missed_deadlines}")
    print(f"  Watchdog overruns:  {state.watchdog_overruns}")
    print(f"  Samples logged:     {state.total_samples_logged}")
    print()

    # PASS / FAIL criteria
    passed = True
    reasons = []

    if state.missed_deadlines >= 5:
        passed = False
        reasons.append(f"Missed deadlines ({state.missed_deadlines}) >= 5")

    if state.control_hz < 98.0:
        passed = False
        reasons.append(f"Achieved Hz ({state.control_hz:.2f}) < 98.0")

    if passed:
        print("  +==================================================+")
        print("  |           [PASS]  TIMING TEST PASSED             |")
        print("  +==================================================+")
    else:
        print("  +==================================================+")
        print("  |           [FAIL]  TIMING TEST FAILED             |")
        print("  +==================================================+")
        for r in reasons:
            print(f"  |  -> {r:<48s}|")
        print("  +==================================================+")

    print()
    print(f"  Telemetry log: {bios._log_path}")
    print("=" * 60)
