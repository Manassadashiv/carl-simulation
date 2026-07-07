"""
carl_expression.py — CARL's Emotional Expression System

Maps internal neuromodulatory state → physical body posture.
This is NOT learned. This is innate. Just like how a human baby
cries before it knows what crying means — these are hardwired
body-emotion mappings that exist at the genetic level.

The learning system (Actor-Critic) controls locomotion.
This system controls expression. They run in parallel.

Input:  BiologicalDrives (da, ne, cort, sero, hunger)
Output: Expression targets for the 9 expressive actuators:
         neck_height, head_pitch, head_yaw,
         eyelid_L, eyelid_R,
         shoulder_L, elbow_L, shoulder_R, elbow_R

Wall-E Expression Reference:
  - Happy:     neck tall, eyes wide, arms relaxed outward
  - Curious:   neck stretched, head tilted forward, eyes wide
  - Scared:    neck compressed, eyes squinting, arms hugged in
  - Sad:       neck low, head drooped, eyes half-closed
  - Excited:   neck bouncing, head scanning, arms waving
  - Stressed:  neck trembling, rapid blinks, arms crossed
  - Sleepy:    neck sinking, eyelids closing, arms hanging
  - Surprised: neck SNAP up, eyes WIDE, head frozen
"""

import numpy as np
import math
import io
import wave

try:
    import winsound
    WINSOUND_AVAILABLE = True
except ImportError:
    WINSOUND_AVAILABLE = False



class EmotionalState:
    """
    Continuous emotional dimensions derived from neuromodulatory signals.
    Each dimension is [0, 1].
    """
    def __init__(self):
        self.joy        = 0.0   # DA > 0 → joy
        self.fear       = 0.0   # CORT high → fear
        self.curiosity  = 0.0   # NE > 0, DA neutral → curiosity
        self.sadness    = 0.0   # low DA, low sero → sadness
        self.excitement = 0.0   # high DA + high NE → excitement
        self.calm       = 0.5   # high sero → calm
        self.hunger_urgency = 0.0
        self.surprise   = 0.0   # NE spike → surprise

    def update_from_drives(self, drives):
        """Map raw neuromodulators to emotional dimensions."""
        da   = drives.da       # [-1, 1]
        ne   = drives.ne       # [0, 1]
        cort = drives.cort     # [0, 1]
        sero = drives.sero     # [0, 1]
        hung = drives.hunger   # [0, 1]

        # Joy: dopamine positive + serotonin present
        self.joy = np.clip(da * 0.7 + sero * 0.3, 0, 1)

        # Fear: cortisol high, low serotonin
        self.fear = np.clip(cort * 0.8 + (1 - sero) * 0.2, 0, 1)

        # Curiosity: norepinephrine active but not stressed
        self.curiosity = np.clip(ne * 1.5 - cort * 0.8, 0, 1)

        # Sadness: low everything — depleted, unfed, unengaged
        self.sadness = np.clip((1 - da) * 0.3 + (1 - sero) * 0.3 + hung * 0.2 - ne * 0.3, 0, 1)

        # Excitement: both DA and NE firing — found food, new situation
        self.excitement = np.clip(max(0, da) * 0.5 + ne * 0.5, 0, 1)

        # Calm: high serotonin, low cortisol, low NE
        self.calm = np.clip(sero * 0.6 + (1 - cort) * 0.2 + (1 - ne) * 0.2, 0, 1)

        # Hunger urgency
        self.hunger_urgency = hung

        # Surprise: NE spike (derivative would be ideal, but threshold on raw NE)
        self.surprise = np.clip((ne - 0.6) * 3.0, 0, 1)

    def dominant_emotion(self):
        """Return name of strongest emotion."""
        emotions = {
            'joy': self.joy,
            'fear': self.fear,
            'curiosity': self.curiosity,
            'sadness': self.sadness,
            'excitement': self.excitement,
            'calm': self.calm,
            'surprise': self.surprise
        }
        return max(emotions, key=emotions.get)

    def to_vec(self):
        return np.array([
            self.joy, self.fear, self.curiosity, self.sadness,
            self.excitement, self.calm, self.hunger_urgency, self.surprise
        ])


class ExpressionController:
    """
    Generates smooth, physically plausible expression targets
    from the current emotional state.

    All outputs are actuator position targets (not velocities).
    The MuJoCo position actuators handle the smooth motion.
    """
    def __init__(self):
        self.emotion = EmotionalState()
        self.t = 0.0           # internal time for oscillations
        self.blink_timer = 0.0
        self.blink_state = 0.0  # 0=open, 1=closed
        self.scan_phase = 0.0   # for head scanning
        self.prev_targets = None
        self.tantrum_timer = 0.0  # tracks frustration duration
        self.grip_slam_phase = 0.0  # for tantrum grip oscillation

        # Expression noise for organic feel (13 channels: 9 original + 4 new arm DOFs)
        self.noise_phase = np.random.uniform(0, 2*math.pi, 13)

    def step(self, drives, dt=0.005, locomotion_v=0.0):
        """
        Compute expression targets based on current neuroendocrine state and locomotion.
        """
        self.emotion.update_from_drives(drives)
        self.t += dt
        e = self.emotion

        # ── Organic noise (subtle tremor, breathing) — 13 channels ──────
        noise = np.array([
            0.003 * math.sin(self.t * 2.1 + self.noise_phase[i])
            for i in range(13)
        ])
        
        # High cortisol (fear/stress) causes trembling
        if e.fear > 0.8:
            tremor_amp = (e.fear - 0.8) * 0.25
            noise += np.random.normal(0, tremor_amp, 13)
        
        # Safely capture the Mark VI fatigue variable from incoming structural signals
        fatigue = getattr(drives, 'fatigue', 0.0)

        # ── NECK HEIGHT ─────────────────────────────────────────────────
        # Range: 0.0 (compressed) to 0.20 (fully extended)
        # Curious/excited → tall. Scared/sad → low. Neutral → middle.
        neck_base = 0.10   # neutral resting height
        neck = neck_base
        neck += e.curiosity * 0.08     # stretch up when curious
        neck += e.excitement * 0.06    # bounce up when excited
        neck += e.joy * 0.04           # lift when happy
        neck -= e.fear * 0.08          # compress when scared
        neck -= e.sadness * 0.06       # droop when sad

        # Excitement causes gentle bobbing
        if e.excitement > 0.3:
            neck += 0.015 * math.sin(self.t * 8.0) * e.excitement

        # Surprise causes a snap-up
        if e.surprise > 0.5:
            neck = min(0.20, neck + 0.05 * e.surprise)

        # ── MARK VI EXHAUSTION OVERRIDE ──
        neck -= fatigue * 0.07 # Structural neck droop slonch

        neck = np.clip(neck + noise[0], 0.0, 0.20)

        # ── HEAD PITCH (tilt) ───────────────────────────────────────────
        # Range: -0.524 (look down) to +0.349 (look up)
        # Curious → tilt forward/down slightly (attentive lean)
        # Sad → droop. Scared → pull back. Excited → slight up.
        pitch = 0.0
        pitch -= e.curiosity * 0.15    # lean forward
        pitch -= e.sadness * 0.25      # droop down
        pitch += e.fear * 0.10         # pull back/up
        pitch += e.surprise * 0.15     # snap back
        
        # ── MARK VI EXHAUSTION OVERRIDE ──
        pitch -= fatigue * 0.20 # Head box leans heavily toward the floor plane
        
        pitch = np.clip(pitch + noise[1], -0.524, 0.349)

        # ── HEAD PAN (yaw) ──────────────────────────────────────────────
        # Range: -1.047 to +1.047
        # Curious/excited → scanning side to side
        # Scared → frozen. Calm → slow drift.
        self.scan_phase += dt
        pan = 0.0
        if e.curiosity > 0.3:
            # Scanning: slow sweep left-right
            pan = 0.5 * math.sin(self.scan_phase * 1.5) * e.curiosity
        if e.excitement > 0.4:
            # Faster scanning when excited
            pan += 0.3 * math.sin(self.scan_phase * 4.0) * e.excitement
        if e.fear > 0.5:
            # Freeze and face forward
            pan *= (1.0 - e.fear)

        pan = np.clip(pan + noise[2], -1.047, 1.047)

        # ── EYELIDS ─────────────────────────────────────────────────────
        # Range: 0.0 (fully open) to 1.3 (fully closed)
        # Joy/curiosity/surprise → wide open
        # Fear → squint (half-closed). Sad → droopy. Sleepy → closing.

        # Autonomous blinking
        self.blink_timer += dt
        blink_interval = 3.0 - e.fear * 2.0   # blink faster when scared
        if self.blink_timer > blink_interval:
            self.blink_state = 1.0
            self.blink_timer = 0.0
        if self.blink_state > 0:
            self.blink_state -= dt * 8.0   # blink duration ~0.125s
            self.blink_state = max(0, self.blink_state)

        eye_base = 0.0   # default: fully open
        eye_base += e.sadness * 0.5       # droop when sad
        eye_base += e.fear * 0.4          # squint when scared
        eye_base -= e.curiosity * 0.1     # extra wide when curious
        eye_base -= e.surprise * 0.3      # WIDE open on surprise
        eye_base -= e.joy * 0.1           # slightly wider when happy

        # Sleepiness (low sero, low NE, low hunger)
        sleepy = max(0, e.calm * 0.3 - e.curiosity * 0.5)
        eye_base += sleepy * 0.6

        eye_base = np.clip(eye_base, -0.2, 1.0)

        # ── MARK VI EXHAUSTION OVERRIDE ──
        # Inject fatigue mapping smoothly directly underneath the dynamic blinking registers
        eyelid_L = np.clip(eye_base + self.blink_state * 1.3 + noise[3] * 2 + fatigue * 0.75, 0.0, 1.3)
        eyelid_R = np.clip(eye_base + self.blink_state * 1.3 + noise[4] * 2 + fatigue * 0.75, 0.0, 1.3)

        # Asymmetric eyelids for character (Wall-E often has one eye different)
        if e.curiosity > 0.4:
            eyelid_L *= 0.8   # left eye opens wider when curious

        # ── ARMS ────────────────────────────────────────────────────────
        # Shoulder range: -1.57 to +1.57
        # Elbow range: -2.0 to +0.5
        #
        # Relaxed: arms at sides (shoulder ~0.3, elbow ~-0.5)
        # Scared: arms pulled in tight (shoulder ~-0.3, elbow ~-1.5 = hugging)
        # Excited: arms up and waving (shoulder oscillating, elbow opening)
        # Sad: arms hanging limp (shoulder ~0.8, elbow ~-1.0)
        # Curious: one arm reaches forward slightly

        # Default: relaxed at sides, swinging with locomotion
        arm_swing = math.sin(self.t * 8.0) * locomotion_v * 0.6
        sh_L = 0.3 + arm_swing
        el_L = -0.5 - abs(arm_swing) * 0.2
        sh_R = 0.3 - arm_swing  # Opposite phase for natural walking
        el_R = -0.5 - abs(arm_swing) * 0.2

        # Fear → hug self
        if e.fear > 0.2:
            sh_L += e.fear * (-0.6)   # pull in
            sh_R += e.fear * (-0.6)
            el_L += e.fear * (-1.0)   # tighten
            el_R += e.fear * (-1.0)

        # Sadness → hang limp
        if e.sadness > 0.3:
            sh_L = 0.3 + e.sadness * 0.4
            sh_R = 0.3 + e.sadness * 0.4
            el_L = -0.5 - e.sadness * 0.3
            el_R = -0.5 - e.sadness * 0.3

        # Excitement → wave arms
        if e.excitement > 0.3:
            wave = math.sin(self.t * 6.0) * e.excitement
            sh_L = 0.3 + wave * 0.5
            sh_R = 0.3 - wave * 0.5  # opposite phase
            el_L = -0.3 + abs(wave) * 0.3
            el_R = -0.3 + abs(wave) * 0.3

        # Curiosity → one arm reaches
        if e.curiosity > 0.4 and e.excitement < 0.3:
            sh_L = 0.3 - e.curiosity * 0.3   # left arm reaches forward
            el_L = -0.3 + e.curiosity * 0.2

        # Joy → gentle arm lift
        if e.joy > 0.5:
            sh_L = 0.3 - e.joy * 0.2
            sh_R = 0.3 - e.joy * 0.2
            el_L = -0.3
            el_R = -0.3

        # Surprise → arms snap up
        if e.surprise > 0.5:
            sh_L = -0.5 * e.surprise
            sh_R = -0.5 * e.surprise
            el_L = 0.2 * e.surprise
            el_R = 0.2 * e.surprise

        # ── MARK VI EXHAUSTION OVERRIDE ──
        if fatigue > 0.5:
            # Overrides transient waves, dropping limbs completely flat down at sides
            sh_L, sh_R = 0.7, 0.7
            el_L, el_R = -0.8, -0.8

        sh_L = np.clip(sh_L + noise[5], -1.57, 1.57)
        el_L = np.clip(el_L + noise[6], -2.0, 0.5)
        sh_R = np.clip(sh_R + noise[7], -1.57, 1.57)
        el_R = np.clip(el_R + noise[8], -2.0, 0.5)

        # ── WRIST & GRIPPER (emotional hand language) ─────────────────
        # Ranges: wrist ±1.57 rad, grip 0 (open) → 0.52 (closed)
        wr_L = 0.0   # neutral level wrist
        wr_R = 0.0
        gr_L = 0.0   # open hand
        gr_R = 0.0

        # Fear → clenched fists, wrists curled inward (self-protective)
        if e.fear > 0.2:
            wr_L = e.fear * 0.6      # curl wrist inward
            wr_R = e.fear * 0.6
            gr_L = e.fear * 0.52     # clench fingers
            gr_R = e.fear * 0.52

        # Curiosity → left wrist extends forward, hand opens wide
        if e.curiosity > 0.4 and e.excitement < 0.3:
            wr_L = -e.curiosity * 0.4  # extend wrist out
            gr_L = 0.0                  # open hand to explore

        # Excitement → grabby hands — grippers open/close rhythmically
        if e.excitement > 0.3:
            self.grip_slam_phase += 0.05
            grab_pulse = (math.sin(self.grip_slam_phase * 4.0) + 1.0) * 0.5
            gr_L = grab_pulse * 0.35 * e.excitement
            gr_R = grab_pulse * 0.35 * e.excitement
            wr_L = math.sin(self.t * 3.0) * 0.3 * e.excitement
            wr_R = -math.sin(self.t * 3.0) * 0.3 * e.excitement

        # Sadness → limp wrists, fingers slightly curled
        if e.sadness > 0.3:
            wr_L = e.sadness * 0.5
            wr_R = e.sadness * 0.5
            gr_L = e.sadness * 0.15
            gr_R = e.sadness * 0.15

        # Surprise → startle — wrists snap outward, hands SPLAY open
        if e.surprise > 0.5:
            wr_L = -0.8 * e.surprise   # snap outward
            wr_R = -0.8 * e.surprise
            gr_L = 0.0                  # fully open
            gr_R = 0.0

        # Joy → relaxed open palms, slight upward wrist tilt
        if e.joy > 0.5:
            wr_L = -0.2
            wr_R = -0.2
            gr_L = 0.0
            gr_R = 0.0

        # ── TANTRUM OVERRIDE (frustration = high cortisol + low dopamine) ──
        # When CARL is deeply stressed and unrewarded, he melts down
        frustration = np.clip(e.fear * 0.5 + (1.0 - e.joy) * 0.3 + e.sadness * 0.2, 0, 1)
        if frustration > 0.75:
            self.tantrum_timer += dt
            self.grip_slam_phase += 0.15  # rapid slam
            slam = math.sin(self.grip_slam_phase * 8.0)
            # Arms flail rapidly and asymmetrically
            sh_L = math.sin(self.t * 9.0 + 1.2) * 1.4
            sh_R = math.sin(self.t * 11.0) * 1.4
            el_L = math.sin(self.t * 7.0) * 1.0 - 0.5
            el_R = math.sin(self.t * 13.0 + 0.8) * 1.0 - 0.5
            wr_L = math.sin(self.t * 15.0) * 1.2
            wr_R = math.sin(self.t * 17.0 + 0.5) * 1.2
            # Grip slams open and shut repeatedly
            gr_L = (slam + 1.0) * 0.26
            gr_R = (-slam + 1.0) * 0.26
        else:
            self.tantrum_timer = max(0.0, self.tantrum_timer - dt * 2.0)

        # ── MARK VI EXHAUSTION OVERRIDE (arms too) ──
        if fatigue > 0.5:
            wr_L, wr_R = 0.4, 0.4   # limp drooped wrists
            gr_L, gr_R = 0.1, 0.1   # barely open

        wr_L = np.clip(wr_L + noise[9]  * 0.02, -1.57, 1.57)
        wr_R = np.clip(wr_R + noise[10] * 0.02, -1.57, 1.57)
        gr_L = np.clip(gr_L + noise[11] * 0.01,  0.0,  0.52)
        gr_R = np.clip(gr_R + noise[12] * 0.01,  0.0,  0.52)

        # ── Build target dict ───────────────────────────────────────────
        targets = {
            'neck_height': float(neck),
            'head_pitch':  float(pitch),
            'head_yaw':    float(pan),
            'eyelid_L':    float(eyelid_L),
            'eyelid_R':    float(eyelid_R),
            'shoulder_L':  float(sh_L),
            'elbow_L':     float(el_L),
            'wrist_L':     float(wr_L),
            'grip_L':      float(gr_L),
            'shoulder_R':  float(sh_R),
            'elbow_R':     float(el_R),
            'wrist_R':     float(wr_R),
            'grip_R':      float(gr_R),
        }

        self.prev_targets = targets
        return targets

    def describe(self):
        """Human-readable description of current emotional state."""
        e = self.emotion
        dom = e.dominant_emotion()
        parts = []
        if e.joy > 0.3:        parts.append(f"happy({e.joy:.0%})")
        if e.fear > 0.3:       parts.append(f"scared({e.fear:.0%})")
        if e.curiosity > 0.3:  parts.append(f"curious({e.curiosity:.0%})")
        if e.sadness > 0.3:    parts.append(f"sad({e.sadness:.0%})")
        if e.excitement > 0.3: parts.append(f"excited({e.excitement:.0%})")
        if e.surprise > 0.3:   parts.append(f"surprised({e.surprise:.0%})")
        if e.calm > 0.5:       parts.append(f"calm({e.calm:.0%})")
        if not parts:          parts.append("neutral")
        return f"[{dom.upper()}] " + " + ".join(parts)


class ProceduralAudioSynthesizer:
    """
    Mark VIII Real-time Procedural Audio Synthesizer.
    
    Generates standard WAV container byte arrays directly in-memory from
    mathematical sin/saw/square waves and streams them asynchronously to the Windows
    multimedia subsystems (winsound) to bypass the Python GIL and eliminate blocking latency.
    """
    def __init__(self, sample_rate=11025):
        self.sample_rate = sample_rate

    def play_mood_sound(self, drives, is_stationary=False):
        """
        Synthesizes a 350ms mono audio frame from drives and plays it asynchronously.
        Preempts any currently playing sound with zero latency.
        """
        if not WINSOUND_AVAILABLE:
            return
            
        da = drives.da          # [-1, 1]
        ne = drives.ne          # [0, 1]
        cort = drives.cort      # [0, 1]
        sero = drives.sero      # [0, 1]
        hung = drives.hunger    # [0, 1]
        
        # Calculate emotional vectors consistent with EmotionalState
        joy = np.clip(da * 0.7 + sero * 0.3, 0.0, 1.0)
        fear = np.clip(cort * 0.8 + (1.0 - sero) * 0.2, 0.0, 1.0)
        curiosity = np.clip(ne * 1.5 - cort * 0.8, 0.0, 1.0)
        sadness = np.clip((1.0 - da) * 0.3 + (1.0 - sero) * 0.3 + hung * 0.2 - ne * 0.3, 0.0, 1.0)

        duration = 0.35  # seconds (matches CPG frame cycle rhythm)
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        waveform = np.zeros_like(t)

        # ── PURE MATHEMATICAL SOUND DESIGN GRID ──
        if fear > 0.5:
            # 1. Stressed Whimper: Rapid frequency tremor (FM) + volume warble (AM)
            freq = 900.0 + 180.0 * np.sin(2.0 * np.pi * 20.0 * t)  # 20Hz vibrato
            phase = 2.0 * np.pi * np.cumsum(freq) / self.sample_rate
            waveform = np.sin(phase) * 0.7
            # 15Hz volume tremolo
            waveform *= (0.4 + 0.6 * np.sin(2.0 * np.pi * 15.0 * t))
            
        elif is_stationary and joy > 0.4:
            # 2. Comfortable Purr: Rhythmic low frequency base rumbling + harmonic breath
            freq = 75.0 + 15.0 * np.sin(2.0 * np.pi * 1.5 * t)  # 1.5Hz pitch breathing
            phase = 2.0 * np.pi * np.cumsum(freq) / self.sample_rate
            # 1st + 2nd harmonic blend
            waveform = (np.sin(phase) + 0.35 * np.sin(2.0 * phase)) * 0.6
            # 3.0Hz rumble modulation
            waveform *= (0.55 + 0.45 * np.sin(2.0 * np.pi * 3.0 * t))
            
        elif joy > 0.5:
            # 3. Happy Ascending Chirp
            freq = 400.0 + 600.0 * (t / duration)  # 400 -> 1000Hz sweep
            phase = 2.0 * np.pi * np.cumsum(freq) / self.sample_rate
            waveform = np.sin(phase) * 0.5 * (1.0 - t / duration)  # fade out
            
        elif curiosity > 0.4:
            # 4. Curiosity Double Beep (ascending pitch pairs with silence gap)
            freq = np.zeros_like(t)
            half = len(t) // 2
            # Chirp A
            t1 = t[:half]
            freq[:half] = 600.0 + 500.0 * (t1 / (duration/2))
            # Chirp B
            t2 = t[half:]
            freq[half:] = 750.0 + 600.0 * ((t2 - duration/2) / (duration/2))
            
            phase = 2.0 * np.pi * np.cumsum(freq) / self.sample_rate
            waveform = np.sin(phase) * 0.5
            # Zero out the inter-beep gap to separate them cleanly
            waveform[int(half * 0.75):half] = 0.0
            
        elif sadness > 0.4:
            # 5. Sad Sigh: Ramped descending sweep fading into zero amplitude
            freq = 380.0 - 260.0 * (t / duration)  # 380 -> 120Hz slide
            phase = 2.0 * np.pi * np.cumsum(freq) / self.sample_rate
            waveform = np.sin(phase) * 0.4 * (1.0 - t / duration)  # fade out
            
        else:
            # 6. Neutral Hum: Low steady state sine
            freq = 110.0 * np.ones_like(t)
            phase = 2.0 * np.pi * np.cumsum(freq) / self.sample_rate
            waveform = np.sin(phase) * 0.3

        # Scale and convert to 16-bit PCM amplitude space (-32768 to 32767)
        audio_data = (waveform * 16384).astype(np.int16)

        # Build WAV file header + data block in a memory buffer
        byte_io = io.BytesIO()
        with wave.open(byte_io, 'wb') as wav_file:
            wav_file.setnchannels(1)       # Mono
            wav_file.setsampwidth(2)        # 16-bit (2 bytes per sample)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        # Fire and forget asynchronously
        try:
            winsound.PlaySound(byte_io.getvalue(), winsound.SND_MEMORY | winsound.SND_ASYNC)
        except Exception:
            pass  # Protect loop from audio driver crashes


# ==============================================================================
#  STANDALONE AUDITORY VALIDATION
# ==============================================================================
if __name__ == "__main__":
    import time
    
    class FakeDrives:
        def __init__(self, da=0.0, ne=0.0, cort=0.0, sero=0.5, hunger=0.0):
            self.da = da
            self.ne = ne
            self.cort = cort
            self.sero = sero
            self.hunger = hunger
            
    print("=" * 72)
    print("  carl_expression.py -- Procedural Audio Synthesis Validation")
    print("=" * 72)
    
    if not WINSOUND_AVAILABLE:
        print("  [FAIL] winsound not available on this platform. Standalone test skipped.")
        sys.exit(0)

    synth = ProceduralAudioSynthesizer()
    
    # Test cases mapping to all 6 expressions
    test_suite = [
        ("Neutral Hum",      FakeDrives(da=0.0, ne=0.0, cort=0.0, sero=0.5, hunger=0.0), False),
        ("Comfortable Purr", FakeDrives(da=0.8, ne=0.0, cort=0.0, sero=0.8, hunger=0.0), True),
        ("Happy Chirp",      FakeDrives(da=0.9, ne=0.5, cort=0.0, sero=0.6, hunger=0.0), False),
        ("Curious Beeps",    FakeDrives(da=0.2, ne=0.9, cort=0.0, sero=0.4, hunger=0.0), False),
        ("Sad Sigh",         FakeDrives(da=-0.5, ne=0.0, cort=0.0, sero=0.0, hunger=0.8), False),
        ("Stressed Whimper", FakeDrives(da=0.0, ne=0.4, cort=0.9, sero=0.1, hunger=0.0), False),
    ]

    print("\n  Looping through the 6 core emotional sound sweeps...")
    print("  (WAV buffers will stream asynchronously and overlap/preempt correctly)\n")
    
    for name, drives, stationary in test_suite:
        print(f"  -> Playing: {name:<20s} (da={drives.da:>4.1f}, ne={drives.ne:>3.1f}, cort={drives.cort:>3.1f}, sero={drives.sero:>3.1f}, hung={drives.hunger:>3.1f})")
        synth.play_mood_sound(drives, is_stationary=stationary)
        # Give it a short pause to hear the sound before the next sweep preempts it
        time.sleep(0.7)
        
    print("\n  [PASS] Auditory wave synthesis finished successfully.")
    print("=" * 72)

