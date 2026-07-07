html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CARL Phase 14 — FSD Autopilot</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700;900&family=JetBrains+Mono:wght@400;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root {
  --bg: #000000;
  --glass: rgba(15, 15, 20, 0.7);
  --border: rgba(255, 255, 255, 0.05);
  --cyan: #00f0ff;
  --blue: #1e78ff;
  --tesla-blue: #2e86ff;
  --red: #ff3366;
  --text: #e2e8f0;
  --dim: #64748b;
  --font-sans: 'Outfit', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: var(--font-sans);
  background-color: var(--bg);
  color: var(--text);
  min-height: 100vh;
  overflow-x: hidden;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

/* Premium FSD Background Gradients */
body::before {
  content: ''; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background: radial-gradient(circle at 50% 0%, rgba(46, 134, 255, 0.1), transparent 50%),
              radial-gradient(circle at 100% 100%, rgba(255, 51, 102, 0.05), transparent 50%);
  z-index: -1;
}

.glass-panel {
  background: var(--glass);
  backdrop-filter: blur(40px);
  -webkit-backdrop-filter: blur(40px);
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8), inset 0 1px 0 rgba(255, 255, 255, 0.02);
  padding: 1.5rem;
  position: relative;
  overflow: hidden;
}

/* Header */
header { display: flex; justify-content: space-between; align-items: center; }
.brand-title {
  font-size: 2.5rem; font-weight: 900; letter-spacing: -1px;
  background: linear-gradient(135deg, #fff, var(--tesla-blue));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.brand-subtitle {
  font-family: var(--font-mono); font-size: 0.7rem; color: var(--dim);
  letter-spacing: 3px; text-transform: uppercase;
}

/* Layout */
.main-grid {
  display: grid;
  grid-template-columns: 300px 1fr 350px;
  gap: 1.5rem;
  flex: 1;
}

/* Cognitive Map Container */
.map-container {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  position: relative;
  background: radial-gradient(circle at center, rgba(30, 120, 255, 0.05) 0%, transparent 70%);
}
#cmCanvas {
  width: 100%; max-width: 600px;
  aspect-ratio: 1;
  border-radius: 8px;
  /* Box shadow creates a glowing edge */
  box-shadow: 0 0 30px rgba(46, 134, 255, 0.1);
  background: #050505;
  image-rendering: pixelated;
}

/* UI Elements */
.metric-box { margin-bottom: 1rem; }
.metric-label { font-size: 0.75rem; color: var(--dim); text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-size: 1.8rem; font-weight: 700; font-family: var(--font-mono); }

/* Bars */
.bar-container { width: 8px; background: rgba(255,255,255,0.05); border-radius: 4px; height: 100px; margin-top: 10px; position: relative; overflow: hidden; }
.bar-fill { position: absolute; bottom: 0; width: 100%; border-radius: 4px; transition: height 0.2s ease; }

.status-badge {
  padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.8rem; font-weight: 700;
  display: inline-flex; align-items: center; gap: 8px;
}
.status-badge.online { background: rgba(0, 240, 255, 0.1); color: var(--cyan); border: 1px solid rgba(0, 240, 255, 0.3); }
.status-badge.offline { background: rgba(255, 51, 102, 0.1); color: var(--red); border: 1px solid rgba(255, 51, 102, 0.3); }
.dot { width: 8px; height: 8px; background: currentColor; border-radius: 50%; box-shadow: 0 0 10px currentColor; }

.body-card {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.75rem; background: rgba(255,255,255,0.02);
  border-radius: 8px; margin-bottom: 0.5rem;
  border-left: 3px solid var(--dim);
}
.body-card.alive { border-left-color: var(--cyan); }
.body-card.dead { border-left-color: var(--red); opacity: 0.5; }

/* Grid specific layout for flex items */
.flex-row { display: flex; justify-content: space-between; }
.flex-col { display: flex; flex-direction: column; }
</style>
</head>
<body>

<header>
  <div>
    <div class="brand-title">CARL FSD</div>
    <div class="brand-subtitle">Asynchronous Hippocampal A* Engine</div>
  </div>
  <div id="conn-badge" class="status-badge offline"><div class="dot"></div><span>CONNECTING</span></div>
</header>

<div class="main-grid">
  <!-- Left Panel: Internal State -->
  <div class="flex-col" style="gap: 1.5rem;">
    <div class="glass-panel">
      <h3 style="color:var(--dim); font-size:0.8rem; letter-spacing:2px; margin-bottom:1rem;">ENDOCRINE LEVELS</h3>
      <div class="flex-row" style="text-align:center;">
        <div>
          <div style="font-size:0.7rem; color:var(--tesla-blue);">DA</div>
          <div class="bar-container"><div id="bar-da" class="bar-fill" style="background:var(--tesla-blue); height:50%;"></div></div>
        </div>
        <div>
          <div style="font-size:0.7rem; color:var(--red);">NE</div>
          <div class="bar-container"><div id="bar-ne" class="bar-fill" style="background:var(--red); height:30%;"></div></div>
        </div>
      </div>
    </div>
    
    <div class="glass-panel">
      <h3 style="color:var(--dim); font-size:0.8rem; letter-spacing:2px; margin-bottom:1rem;">SYSTEM DIAGNOSTICS</h3>
      <div class="metric-box">
        <div class="metric-label">Allostatic Load</div>
        <div id="allo-load" class="metric-value">0.000</div>
      </div>
      <div class="metric-box">
        <div class="metric-label">Prefrontal Cortex State</div>
        <div id="pfc-state" class="metric-value" style="color:var(--tesla-blue); font-size:1.2rem;">IDLE</div>
      </div>
    </div>
  </div>

  <!-- Center Panel: Tesla FSD Cognitive Map -->
  <div class="glass-panel map-container">
    <div style="position:absolute; top:1.5rem; left:1.5rem; z-index:10;">
      <h3 style="color:var(--dim); font-size:0.8rem; letter-spacing:2px;">OCCUPANCY GRID & A* TRAJECTORY</h3>
    </div>
    <canvas id="cmCanvas" width="500" height="500"></canvas>
  </div>

  <!-- Right Panel: Swarm Telemetry -->
  <div class="glass-panel" style="display:flex; flex-direction:column; gap:1rem;">
    <h3 style="color:var(--dim); font-size:0.8rem; letter-spacing:2px;">SWARM TELEMETRY</h3>
    <div id="bodies-container">
      <!-- Injected via JS -->
    </div>
    <div style="margin-top:auto;">
      <canvas id="chartHistory" height="150"></canvas>
    </div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const f2 = n => Number(n||0).toFixed(2);
const pct = n => Math.min(100, Math.max(0, Number(n||0)*100)) + '%';

// A* Map Rendering
const cmCanvas = $('cmCanvas');
const cmCtx = cmCanvas.getContext('2d');

function renderTeslaMap(danger_flat, astar_path, gx, gy, rx, ry) {
  // Clear canvas
  cmCtx.fillStyle = '#050505';
  cmCtx.fillRect(0, 0, 500, 500);
  
  const cellW = 500 / 25;
  const cellH = 500 / 25;

  // Draw Occupancy Grid (Danger)
  if (danger_flat) {
    for(let i=0; i<625; i++) {
      const d = danger_flat[i];
      if (d > 0.05) {
        const x = (i % 25) * cellW;
        const y = Math.floor(i / 25) * cellH;
        cmCtx.fillStyle = `rgba(255, 51, 102, ${d * 0.8})`; // Red glowing walls
        cmCtx.fillRect(x, y, cellW, cellH);
        cmCtx.strokeStyle = `rgba(255, 51, 102, ${d})`;
        cmCtx.strokeRect(x, y, cellW, cellH);
      }
    }
  }

  // Draw A* Path (Tesla Blue Line)
  if (astar_path && astar_path.length > 0) {
    cmCtx.beginPath();
    cmCtx.strokeStyle = '#2e86ff';
    cmCtx.lineWidth = 4;
    cmCtx.lineJoin = 'round';
    cmCtx.lineCap = 'round';
    
    astar_path.forEach((pt, idx) => {
      // Map global coordinates to grid pixels
      // Assuming MAP_XMIN=-0.5, MAP_XMAX=7.7
      const px = ((pt[0] - (-0.5)) / 8.2) * 500;
      const py = ((pt[1] - (-0.5)) / 7.0) * 500; // MAP_YMAX=6.5
      
      if (idx === 0) cmCtx.moveTo(px, py);
      else cmCtx.lineTo(px, py);
    });
    
    cmCtx.shadowColor = '#2e86ff';
    cmCtx.shadowBlur = 10;
    cmCtx.stroke();
    cmCtx.shadowBlur = 0; // reset
  }

  // Draw Goal (Neon Green/Cyan)
  const gpx = ((gx - (-0.5)) / 8.2) * 500;
  const gpy = ((gy - (-0.5)) / 7.0) * 500;
  cmCtx.beginPath();
  cmCtx.arc(gpx, gpy, 8, 0, Math.PI*2);
  cmCtx.fillStyle = '#00f0ff';
  cmCtx.shadowColor = '#00f0ff';
  cmCtx.shadowBlur = 15;
  cmCtx.fill();
  cmCtx.shadowBlur = 0;
  
  // Draw Robots
  if (rx && ry) {
    rx.forEach((rx_pos, i) => {
      const rpx = ((rx_pos - (-0.5)) / 8.2) * 500;
      const rpy = ((ry[i] - (-0.5)) / 7.0) * 500;
      cmCtx.beginPath();
      cmCtx.arc(rpx, rpy, 6, 0, Math.PI*2);
      cmCtx.fillStyle = '#ffffff';
      cmCtx.fill();
      cmCtx.lineWidth = 2;
      cmCtx.strokeStyle = '#2e86ff';
      cmCtx.stroke();
    });
  }
}

// Websocket
let ws;
function connect() {
  ws = new WebSocket('ws://localhost:8765');
  ws.onopen = () => {
    $('conn-badge').className = 'status-badge online';
    $('conn-badge').innerHTML = '<div class="dot"></div><span>FSD ONLINE</span>';
  };
  ws.onclose = () => {
    $('conn-badge').className = 'status-badge offline';
    $('conn-badge').innerHTML = '<div class="dot"></div><span>LINK LOST</span>';
    setTimeout(connect, 2000);
  };
  ws.onmessage = (e) => updateDashboard(JSON.parse(e.data));
}

function updateDashboard(d) {
  $('bar-da').style.height = pct(d.DA);
  $('bar-ne').style.height = pct(d.NE);
  $('allo-load').innerText = f2(d.allostatic_load);
  $('pfc-state').innerText = d.mode || 'PLANNING';

  // Build Body Cards
  const bc = $('bodies-container');
  bc.innerHTML = '';
  if (d.alive_flags) {
    for (let i=0; i<d.alive_flags.length; i++) {
      const alive = d.alive_flags[i];
      bc.innerHTML += `
        <div class="body-card ${alive ? 'alive' : 'dead'}">
          <div>
            <div style="font-weight:700;">AGENT ${i+1}</div>
            <div style="font-size:0.7rem; color:var(--dim); font-family:var(--font-mono);">SURVIVAL: ${f2(d.survivals[i])}s</div>
          </div>
          <div style="font-size:0.75rem; font-weight:700; color:${alive ? 'var(--cyan)' : 'var(--red)'};">
            ${alive ? 'ACTIVE' : 'OFFLINE'}
          </div>
        </div>
      `;
    }
  }

  // Render FSD Map
  renderTeslaMap(d.cognitive_map, d.astar_path, d.goal_x, d.goal_y, d.robot_x, d.robot_y);
}

connect();
</script>
</body>
</html>"""

with open('dashboard/maze_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html_content)
print("Dashboard completely rewritten to Tesla FSD style.")
