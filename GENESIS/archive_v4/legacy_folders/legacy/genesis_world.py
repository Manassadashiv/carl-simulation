# genesis_world.py
# The World of CARL Genesis.
# Sound. Day and night. Commands. The interface between CARL and its creator.
#
# This file handles:
#   1. Generative sounds — emerge from neurotransmitter levels, not events
#   2. Day/night cycle — changes behaviour without instruction
#   3. Command interface — influence, not control
#   4. Cognitive maps — spatial memory shared between siblings

import numpy as np
import math, time, threading
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# SOUND ENGINE — Wall-E style generative tones
# No sound files. Pure mathematics.
# Sounds emerge from chemistry, not from events.
# ─────────────────────────────────────────────────────────────────────────────

_pygame_available = False
try:
    import pygame
    import pygame.mixer
    _pygame_available = True
except ImportError:
    pass

_sound_initialized = False
_last_sound_step   = {'A': 0, 'B': 0}
_sound_cooldown    = 120   # steps between sounds


def init_sound():
    global _sound_initialized
    if not _pygame_available:
        print('  [SOUND] pygame not available — silent mode')
        return
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
        _sound_initialized = True
        print('  [SOUND] Initialized: 44100Hz, mono')
    except Exception as e:
        print(f'  [SOUND] Init failed: {e}')


def _make_tone(frequency, duration_s, shape='sine', attack=0.02, decay=0.05):
    """Generate a pure tone as numpy int16 samples."""
    sr    = 44100
    n     = int(sr * duration_s)
    t     = np.linspace(0, duration_s, n, endpoint=False)

    if shape == 'sine':
        wave = np.sin(2 * math.pi * frequency * t)
    elif shape == 'square':
        wave = np.sign(np.sin(2 * math.pi * frequency * t))
    elif shape == 'saw':
        wave = 2.0 * (t * frequency - np.floor(t * frequency + 0.5))
    else:
        wave = np.sin(2 * math.pi * frequency * t)

    # Envelope: attack, sustain, decay
    env = np.ones(n)
    na  = int(attack * sr)
    nd  = int(decay  * sr)
    if na > 0:
        env[:na]  = np.linspace(0, 1, na)
    if nd > 0 and nd < n:
        env[-nd:] = np.linspace(1, 0, nd)

    wave = wave * env * 0.5   # 50% volume
    return (wave * 32767).astype(np.int16)


def _play_tone_async(frequency, duration_s, shape='sine'):
    """Play a tone in a background thread — non-blocking."""
    if not _sound_initialized:
        return

    def _play():
        try:
            samples = _make_tone(frequency, duration_s, shape)
            sound   = pygame.sndarray.make_sound(samples)
            sound.play()
            time.sleep(duration_s + 0.05)
        except Exception:
            pass

    t = threading.Thread(target=_play, daemon=True)
    t.start()


def _play_sequence_async(notes):
    """Play a melody sequence — list of (frequency, duration) tuples."""
    if not _sound_initialized:
        return

    def _play():
        try:
            for freq, dur in notes:
                samples = _make_tone(freq, dur)
                sound   = pygame.sndarray.make_sound(samples)
                sound.play()
                time.sleep(dur * 0.9)
        except Exception:
            pass

    t = threading.Thread(target=_play, daemon=True)
    t.start()


# ── Emotional sound patterns ──────────────────────────────────────────────────
# These are not triggered by events.
# They emerge continuously from neurotransmitter levels.
# The chemistry speaks through sound.

def update_sound(body_name, nm, self_model, step):
    """
    Generate appropriate sound based on current internal state.
    Called every step, but only plays when cooldown has elapsed
    and the internal state is strong enough.
    """
    if step - _last_sound_step.get(body_name, 0) < _sound_cooldown:
        return

    s = self_model.state

    # Determine dominant emotional state
    if s['fear'] > 0.65:
        # Fear: rapid high-pitched staccato
        _play_sequence_async([(880, 0.06), (0, 0.04), (880, 0.06), (0, 0.04), (660, 0.08)])
        _last_sound_step[body_name] = step

    elif s['hunger'] > 0.75:
        # Hunger: rhythmic searching call, faster as hunger grows
        rate = float(np.clip(s['hunger'], 0.5, 1.0))
        _play_tone_async(392, 0.08)
        _last_sound_step[body_name] = step

    elif s['grief'] > 0.6:
        # Grief: slow, low, mournful descending tone
        _play_sequence_async([(294, 0.4), (262, 0.5), (220, 0.6)])
        _last_sound_step[body_name] = step + 480   # Extra cooldown for grief

    elif nm.DA > 0.75 and s['fear'] < 0.3:
        # Joy: rising two-tone (Wall-E "Waaaall-Eee")
        _play_sequence_async([(523, 0.15), (659, 0.25)])
        _last_sound_step[body_name] = step

    elif s['curiosity'] > 0.70 and s['fear'] < 0.4:
        # Curiosity: questioning upward tone
        _play_sequence_async([(392, 0.12), (523, 0.18)])
        _last_sound_step[body_name] = step

    elif nm.DA > 0.5 and nm.NE < 0.3:
        # Contentment: soft low hum
        _play_tone_async(196, 0.3, shape='sine')
        _last_sound_step[body_name] = step


def play_eat_sound():
    """Food found! Happy chirp sequence."""
    _play_sequence_async([(523, 0.08), (659, 0.08), (784, 0.15)])


def play_goal_sound():
    """Goal reached! Triumphant sequence."""
    _play_sequence_async([(523, 0.1), (659, 0.1), (784, 0.1), (1047, 0.3)])


def play_sibling_died_sound():
    """Sibling death: one long mournful tone."""
    _play_sequence_async([(294, 0.5), (220, 0.8)])


def play_praised_sound():
    """Praise received: happy rising melody."""
    _play_sequence_async([(392, 0.1), (523, 0.1), (659, 0.2)])


def play_scolded_sound():
    """Scolded: descending alarmed beeps."""
    _play_sequence_async([(880, 0.06), (660, 0.06), (440, 0.10)])


# ─────────────────────────────────────────────────────────────────────────────
# DAY / NIGHT CYCLE
# Every 5 minutes (72,000 steps at 240Hz), the world shifts.
# Day: high NE, exploratory, fast.
# Night: high ACh, consolidating, careful.
# CARL does not know it is obeying a cycle.
# ─────────────────────────────────────────────────────────────────────────────

DAY_NIGHT_PERIOD = 72000   # steps per full cycle
DAWN_STEPS       = 7200    # gradual transition duration

def get_day_factor(step):
    """
    Returns float [0, 1] — 1.0 = full day, 0.0 = full night.
    Follows a smooth sinusoidal cycle.
    """
    phase = (step % DAY_NIGHT_PERIOD) / DAY_NIGHT_PERIOD
    return float(0.5 + 0.5 * math.sin(2.0 * math.pi * phase - math.pi / 2.0))


def apply_day_night(nm, day_factor):
    """
    Modulate neurotransmitters with day/night cycle.
    
    Day:   NE rises (alertness), exploration encouraged.
    Night: ACh rises (consolidation, careful learning), NE falls.
    
    This happens without instruction. CARL simply is more alert during day.
    """
    nm.NE  = float(np.clip(nm.NE  + 0.002 * (day_factor - 0.5),       0.0, 1.0))
    nm.ACh = float(np.clip(nm.ACh + 0.002 * (0.5 - day_factor) * 0.5, 0.0, 1.0))
    return day_factor


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND INTERFACE — The "Obey Me" Channel
# Not a remote control. Influence.
# Commands create salient inputs that temporarily dominate dynamics.
# After the influence fades, CARL's own dynamics reassert.
# ─────────────────────────────────────────────────────────────────────────────

class CommandChannel:
    """
    The interface between creator and creature.
    
    Commands do not override CARL's will. They create a strong signal
    that temporarily dominates the reservoir, like a loud voice.
    The influence decays exponentially. CARL's own dynamics return.
    """

    VALID_COMMANDS = {
        'good', 'bad', 'go', 'stop', 'come', 'explore', 'play',
        'follow', 'home', 'slow', 'fast',
    }

    def __init__(self):
        self._queue   = deque()
        self._lock    = threading.Lock()
        self.active   = {}   # body_name → {command, strength, waypoint}

    def issue(self, body_name, command, arg=None):
        """
        Issue a command to a body.
        
        body_name : 'A' or 'B' or 'both'
        command   : one of VALID_COMMANDS
        arg       : optional argument (e.g., (x, y) for 'go')
        """
        if body_name == 'both':
            for bn in ['A', 'B']:
                self.issue(bn, command, arg)
            return

        cmd = command.strip().lower()
        with self._lock:
            self._queue.append((body_name, cmd, arg))

    def process(self, body_name, nm, self_model):
        """
        Process any pending commands for this body.
        Returns waypoint override if 'go' command, else None.
        """
        waypoint = None
        with self._lock:
            to_process = [q for q in list(self._queue)
                          if q[0] == body_name]
            for item in to_process:
                self._queue.remove(item)

        for (_, cmd, arg) in to_process:
            # Commands create neurotransmitter spikes — not direct control
            if cmd == 'good':
                nm.DA  = min(1.0, nm.DA  + 0.5)
                nm.SHT = min(1.0, nm.SHT + 0.2)
                play_praised_sound()
                print(f'  [CARL-{body_name}] ← Praised! DA→{nm.DA:.2f}')

            elif cmd == 'bad':
                nm.NE  = min(1.0, nm.NE  + 0.6)
                nm.DA  = max(0.0, nm.DA  - 0.3)
                play_scolded_sound()
                print(f'  [CARL-{body_name}] ← Scolded! NE→{nm.NE:.2f}')

            elif cmd == 'stop':
                # Brief NE spike → freeze reflex activates
                nm.NE = min(1.0, nm.NE + 0.4)
                self.active[body_name] = {'cmd': 'stop', 'strength': 1.0}

            elif cmd == 'go' and arg is not None:
                # Set a waypoint override
                waypoint = (float(arg[0]), float(arg[1]))
                self.active[body_name] = {'cmd': 'go', 'waypoint': waypoint, 'strength': 1.0}
                print(f'  [CARL-{body_name}] → Go to {waypoint}')

            elif cmd == 'explore':
                nm.ACh = min(1.0, nm.ACh + 0.5)
                self.active[body_name] = {'cmd': 'explore', 'strength': 1.0}

            elif cmd == 'play':
                nm.DA  = min(1.0, nm.DA  + 0.3)
                nm.ACh = min(1.0, nm.ACh + 0.5)
                self.active[body_name] = {'cmd': 'play', 'strength': 1.0}

            elif cmd == 'slow':
                self.active[body_name] = {'cmd': 'slow', 'strength': 1.0}

            elif cmd == 'fast':
                nm.DA  = min(1.0, nm.DA  + 0.2)
                self.active[body_name] = {'cmd': 'fast', 'strength': 1.0}

            elif cmd == 'home':
                waypoint = (0.6, 0.6)
                self.active[body_name] = {'cmd': 'go', 'waypoint': waypoint, 'strength': 1.0}

        # Decay active command influence
        if body_name in self.active:
            self.active[body_name]['strength'] *= 0.98   # Exponential decay
            if self.active[body_name]['strength'] < 0.05:
                del self.active[body_name]

        # Return active waypoint if still influential
        active = self.active.get(body_name)
        if active and active.get('cmd') == 'go':
            return active.get('waypoint')
        return waypoint

    def get_speed_modifier(self, body_name):
        """Return speed modifier from active command (1.0 = normal)."""
        active = self.active.get(body_name)
        if not active:
            return 1.0
        cmd = active.get('cmd', '')
        strength = active.get('strength', 0.0)
        if cmd == 'stop':
            return max(0.0, 1.0 - strength)
        if cmd == 'slow':
            return max(0.2, 1.0 - strength * 0.5)
        if cmd == 'fast':
            return min(1.8, 1.0 + strength * 0.8)
        return 1.0

    def get_mode(self, body_name):
        """Return current active command mode, or None."""
        active = self.active.get(body_name)
        return active.get('cmd') if active else None


# ─────────────────────────────────────────────────────────────────────────────
# GOAL SELECTION — Free Will
# When CARL has no command and no pressing survival need,
# it chooses its own next destination based on internal state.
# This is not scripted. This is the internal state selecting behavior.
# ─────────────────────────────────────────────────────────────────────────────

def select_goal(body_name, xw, yw, nm, self_model, sibling_pos,
                primary_goal, visit_map, MAP_XMIN, MAP_XMAX,
                MAP_YMIN, MAP_YMAX, MAP_RES, food_nearest,
                deliberator, cmd_waypoint=None):
    """
    Select the next goal position using Quantum Interference Deliberation.
    
    Strategies compete probabilistically based on internal state.
    COMMAND overrides are absolute.
    """
    s = self_model.state

    # 1. Command override (Absolute)
    if cmd_waypoint is not None:
        return cmd_waypoint, 'COMMANDED'

    strategies = []

    # 2. Survival
    if food_nearest is not None:
        _, fx, fy, fd = food_nearest
        # High hunger = very low cost
        cost_survival = max(0.01, 1.0 - s['hunger'])
        strategies.append(('HUNGRY', (fx, fy), cost_survival))

    # 3. Social
    if sibling_pos is not None:
        sx, sy = sibling_pos
        d_sib  = math.sqrt((xw - sx)**2 + (yw - sy)**2)
        if d_sib > 0.8:
            cost_social = max(0.01, 1.0 - (s['fear'] * 0.6 + s['grief'] * 0.4))
            strategies.append(('SEEKING_SIBLING', (sx, sy), cost_social))

    # 4. Mission
    cost_mission = max(0.01, 1.0 - nm.DA)
    strategies.append(('MISSION', primary_goal, cost_mission))

    # 5. Curiosity
    if visit_map is not None:
        unknown = (visit_map == -1)
        if unknown.any():
            candidates = np.argwhere(unknown)
        else:
            safe_mask = (visit_map >= 0)
            if safe_mask.any():
                min_visits = visit_map[safe_mask].min()
                candidates = np.argwhere(safe_mask & (visit_map == min_visits))
            else:
                candidates = []

        if len(candidates) > 0:
            idx = candidates[np.random.randint(len(candidates))]
            gx  = MAP_XMIN + (idx[0] / MAP_RES) * (MAP_XMAX - MAP_XMIN)
            gy  = MAP_YMIN + (idx[1] / MAP_RES) * (MAP_YMAX - MAP_YMIN)
            cost_explore = max(0.01, 1.0 - nm.ACh)
            strategies.append(('EXPLORING', (gx, gy), cost_explore))

    # ── Quantum Deliberation ──
    # Format for deliberator: (name, u, s, cost) -> we use 'target' in place of 'u'
    q_strats = [(name, target, None, cost) for name, target, cost in strategies]
    
    if len(q_strats) > 0:
        chosen_name, chosen_target, _, _ = deliberator.deliberate(q_strats)
        return chosen_target, chosen_name
        
    return primary_goal, 'MISSION'




# ─────────────────────────────────────────────────────────────────────────────
# COGNITIVE MAP — Shared spatial memory
# Inherited from Phase 18. Both bodies write to and read from the same map.
# Where one has been, both know. Where one died, both fear.
# ─────────────────────────────────────────────────────────────────────────────

MAP_XMIN, MAP_XMAX = 0.0, 7.2
MAP_YMIN, MAP_YMAX = 0.0, 6.0
MAP_RES  = 60
DRES     = 20

def make_cognitive_map():
    """Create a fresh cognitive map (danger=0, trust=0, scent=0, visits=0)."""
    return np.zeros((MAP_RES, MAP_RES, 4), dtype=np.float32)


def load_cognitive_map(path):
    try:
        cm = np.load(path).astype(np.float32)
        print(f'  [COGMAP] Loaded from {path}')
        return cm
    except FileNotFoundError:
        print(f'  [COGMAP] Starting fresh')
        return make_cognitive_map()


def make_danger_grid():
    return np.zeros((DRES, DRES), dtype=np.float32)


def load_danger_grid(path):
    try:
        dg = np.load(path).astype(np.float32)
        print(f'  [DANGER] Loaded from {path}')
        return dg
    except FileNotFoundError:
        return make_danger_grid()


def update_cognitive_map(CM, xw, yw, surprise, delta):
    gi = int(np.clip((xw - MAP_XMIN) / (MAP_XMAX - MAP_XMIN) * MAP_RES, 0, MAP_RES - 1))
    gj = int(np.clip((yw - MAP_YMIN) / (MAP_YMAX - MAP_YMIN) * MAP_RES, 0, MAP_RES - 1))
    CM[gi, gj, 1] = float(np.clip(CM[gi, gj, 1] + delta, 0.0, 1.0))  # Trust
    CM[gi, gj, 3] = float(CM[gi, gj, 3] + 1.0)                        # Visit count
    return CM


def update_danger_grid(D, x, y, intensity, rate=0.1):
    di = int(np.clip((x - MAP_XMIN) / (MAP_XMAX - MAP_XMIN) * DRES, 0, DRES - 1))
    dj = int(np.clip((y - MAP_YMIN) / (MAP_YMAX - MAP_YMIN) * DRES, 0, DRES - 1))
    D[di, dj] = float(np.clip(D[di, dj] * (1.0 - rate) + intensity * rate, 0.0, 100.0))
    return D


def danger_at(D, xw, yw):
    di = int(np.clip((xw - MAP_XMIN) / (MAP_XMAX - MAP_XMIN) * DRES, 0, DRES - 1))
    dj = int(np.clip((yw - MAP_YMIN) / (MAP_YMAX - MAP_YMIN) * DRES, 0, DRES - 1))
    return float(np.clip(D[di, dj] / 50.0, 0.0, 1.0))


def make_visit_map():
    return np.zeros((MAP_RES, MAP_RES), dtype=np.float32) - 1.0   # -1 = unknown

def load_visit_map(path):
    try:
        vm = np.load(path).astype(np.float32)
        print(f'  [VISIT] Loaded from {path}')
        return vm
    except FileNotFoundError:
        return make_visit_map()


def mark_visited(visit_map, xw, yw):
    gi = int(np.clip((xw - MAP_XMIN) / (MAP_XMAX - MAP_XMIN) * MAP_RES, 0, MAP_RES - 1))
    gj = int(np.clip((yw - MAP_YMIN) / (MAP_YMAX - MAP_YMIN) * MAP_RES, 0, MAP_RES - 1))
    if visit_map[gi, gj] < 0:
        visit_map[gi, gj] = 0
    visit_map[gi, gj] += 1.0
    return visit_map


def seed_walls_into_map(CM, wall_positions):
    """
    Pre-seed known wall positions into cognitive map Obstacle Channel (CM[:,:,2]).
    wall_positions contains (x, y, half_width_x, half_width_y).
    """
    for (wx, wy, sx, sy) in wall_positions:
        # Bounding box of the wall in world coords
        # Add 0.30m safety margin so the robot body doesn't clip the wall and trigger reflexes
        margin = 0.30
        min_x = wx - sx - margin
        max_x = wx + sx + margin
        min_y = wy - sy - margin
        max_y = wy + sy + margin
        
        # Convert to grid indices
        min_gi = int(np.clip((min_x - MAP_XMIN) / (MAP_XMAX - MAP_XMIN) * MAP_RES, 0, MAP_RES - 1))
        max_gi = int(np.clip((max_x - MAP_XMIN) / (MAP_XMAX - MAP_XMIN) * MAP_RES, 0, MAP_RES - 1))
        min_gj = int(np.clip((min_y - MAP_YMIN) / (MAP_YMAX - MAP_YMIN) * MAP_RES, 0, MAP_RES - 1))
        max_gj = int(np.clip((max_y - MAP_YMIN) / (MAP_YMAX - MAP_YMIN) * MAP_RES, 0, MAP_RES - 1))
        
        # Fill the grid cells with obstacle value 1.0
        for gi in range(min_gi, max_gi + 1):
            for gj in range(min_gj, max_gj + 1):
                CM[gi, gj, 2] = 1.0   # Obstacle Channel
    return CM
