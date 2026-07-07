html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CARL Phase 14 — Neural Autopilot</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700;900&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #090a0f;
  --glass: rgba(20, 22, 30, 0.6);
  --border: rgba(255, 255, 255, 0.08);
  --cyan: #00f0ff;
  --purple: #b53cff;
  --gold: #ffb800;
  --red: #ff2a55;
  --text: #ffffff;
  --dim: #8b9bb4;
  --font-sans: 'Inter', sans-serif;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: var(--font-sans); background-color: var(--bg); color: var(--text);
  min-height: 100vh; overflow-x: hidden; padding: 2rem; display: flex; flex-direction: column; gap: 2rem;
}

body::before {
  content: ''; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
  background: radial-gradient(circle at 15% 50%, rgba(0, 240, 255, 0.08), transparent 40%),
              radial-gradient(circle at 85% 30%, rgba(181, 60, 255, 0.08), transparent 40%);
  z-index: -1; filter: blur(60px);
}

.glass-panel {
  background: var(--glass); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--border); border-radius: 16px;
  box-shadow: 0 20px 50px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
  padding: 1.5rem; position: relative; overflow: hidden;
}

header { display: flex; justify-content: space-between; align-items: center; }
.brand-title { font-size: 2.5rem; font-weight: 900; letter-spacing: -1px; 
  background: linear-gradient(135deg, #fff, var(--cyan)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.brand-subtitle { font-size: 0.8rem; color: var(--dim); font-weight: 500; letter-spacing: 2px; text-transform: uppercase; margin-top: 4px; }

.main-grid { display: grid; grid-template-columns: 320px 1fr 320px; gap: 2rem; flex: 1; }

.map-container { display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(0,0,0,0.3); }
#cmCanvas {
  width: 100%; max-width: 650px; aspect-ratio: 1; border-radius: 12px;
  box-shadow: 0 0 50px rgba(0, 240, 255, 0.05), inset 0 0 20px rgba(0,0,0,1);
  background: radial-gradient(circle at center, #0a0c14 0%, #030408 100%);
}

.metric-label { font-size: 0.75rem; color: var(--dim); text-transform: uppercase; letter-spacing: 1px; font-weight: 700; margin-bottom: 4px; }
.metric-value { font-size: 2rem; font-weight: 300; letter-spacing: -1px; }

.status-badge { padding: 0.5rem 1rem; border-radius: 30px; font-size: 0.75rem; font-weight: 700; display: inline-flex; align-items: center; gap: 8px; letter-spacing: 1px; }
.status-badge.online { background: rgba(0, 240, 255, 0.1); color: var(--cyan); border: 1px solid rgba(0, 240, 255, 0.2); }
.status-badge.offline { background: rgba(255, 42, 85, 0.1); color: var(--red); border: 1px solid rgba(255, 42, 85, 0.2); }
.dot { width: 8px; height: 8px; background: currentColor; border-radius: 50%; box-shadow: 0 0 12px currentColor; }

.body-card { padding: 1rem; background: rgba(255,255,255,0.02); border-radius: 12px; margin-bottom: 0.8rem; border-left: 3px solid var(--dim); transition: all 0.3s ease; }
.body-card.alive { border-left-color: var(--cyan); background: linear-gradient(90deg, rgba(0,240,255,0.05) 0%, transparent 100%); }
.body-card.dead { border-left-color: var(--red); opacity: 0.5; }

.flex-row { display: flex; justify-content: space-between; align-items: center; }
.flex-col { display: flex; flex-direction: column; gap: 1.5rem; }
</style>
</head>
<body>

<header>
  <div>
    <div class="brand-title">CARL Neural Engine</div>
    <div class="brand-subtitle">Hyper-Realtime Biological Autonomy</div>
  </div>
  <div id="conn-badge" class="status-badge offline"><div class="dot"></div><span>CONNECTING</span></div>
</header>

<div class="main-grid">
  <div class="flex-col">
    <div class="glass-panel">
      <div class="metric-label">Neural Allostatic Load</div>
      <div id="allo-load" class="metric-value" style="color:var(--cyan);">0.000</div>
    </div>
    <div class="glass-panel">
      <div class="metric-label">Prefrontal Cortex Status</div>
      <div id="pfc-state" class="metric-value" style="color:var(--purple); font-size:1.5rem;">INITIALIZING</div>
    </div>
    <div class="glass-panel">
      <div class="metric-label">Endocrine Balance</div>
      <div class="flex-row" style="margin-top:1rem; gap:1rem;">
        <div style="flex:1;"><div style="font-size:0.7rem; color:var(--cyan); margin-bottom:4px;">DOPAMINE</div><div style="height:6px; background:rgba(255,255,255,0.1); border-radius:3px; overflow:hidden;"><div id="bar-da" style="height:100%; background:var(--cyan); width:0%; transition:width 0.3s;"></div></div></div>
        <div style="flex:1;"><div style="font-size:0.7rem; color:var(--red); margin-bottom:4px;">NORADRENALINE</div><div style="height:6px; background:rgba(255,255,255,0.1); border-radius:3px; overflow:hidden;"><div id="bar-ne" style="height:100%; background:var(--red); width:0%; transition:width 0.3s;"></div></div></div>
      </div>
    </div>
  </div>

  <div class="glass-panel map-container">
    <canvas id="cmCanvas" width="800" height="800"></canvas>
  </div>

  <div class="glass-panel flex-col" style="justify-content:flex-start;">
    <div class="metric-label">Swarm Telemetry</div>
    <div id="bodies-container" style="margin-top:1rem;"></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const f2 = n => Number(n||0).toFixed(2);
const pct = n => Math.min(100, Math.max(0, Number(n||0)*100)) + '%';

const cmCanvas = $('cmCanvas');
const ctx = cmCanvas.getContext('2d');
let timeOffset = 0;

function drawGlowRect(x, y, w, h, color, blur) {
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = blur;
  ctx.fillStyle = color;
  ctx.fillRect(x, y, w, h);
  ctx.restore();
}

function renderNeuralMap(danger_flat, astar_path, gx, gy, rx, ry) {
  timeOffset += 0.05;
  ctx.clearRect(0, 0, 800, 800);
  
  const cellW = 800 / 25;
  const cellH = 800 / 25;

  // 1. Render Occupancy Grid (Soft glowing walls instead of raw pixels)
  if (danger_flat) {
    for(let i=0; i<625; i++) {
      const d = danger_flat[i];
      if (d > 0.01) {
        const x = (i % 25) * cellW;
        const y = Math.floor(i / 25) * cellH;
        // Soft gradient block
        ctx.fillStyle = `rgba(255, 42, 85, ${d * 0.15})`;
        ctx.fillRect(x+2, y+2, cellW-4, cellH-4);
        // Highlight edge
        if (d > 0.4) {
          ctx.strokeStyle = `rgba(255, 42, 85, ${d * 0.5})`;
          ctx.lineWidth = 1;
          ctx.strokeRect(x+4, y+4, cellW-8, cellH-8);
        }
      }
    }
  }

  // 2. Render A* Path (Hyper-realistic pulsing energy beam)
  if (astar_path && astar_path.length > 1) {
    ctx.beginPath();
    astar_path.forEach((pt, idx) => {
      const px = ((pt[0] - (-0.5)) / 8.2) * 800;
      const py = ((pt[1] - (-0.5)) / 7.0) * 800;
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    
    // Outer glow
    ctx.strokeStyle = `rgba(181, 60, 255, 0.4)`;
    ctx.lineWidth = 12;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowColor = '#b53cff';
    ctx.shadowBlur = 30;
    ctx.stroke();
    
    // Inner core
    ctx.strokeStyle = `rgba(0, 240, 255, ${0.7 + 0.3*Math.sin(timeOffset)})`;
    ctx.lineWidth = 4;
    ctx.shadowColor = '#00f0ff';
    ctx.shadowBlur = 15;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // 3. Render Goal (Pulsing beacon)
  const gpx = ((gx - (-0.5)) / 8.2) * 800;
  const gpy = ((gy - (-0.5)) / 7.0) * 800;
  const gPulse = 8 + 4 * Math.sin(timeOffset * 2);
  
  ctx.beginPath();
  ctx.arc(gpx, gpy, gPulse, 0, Math.PI*2);
  ctx.fillStyle = '#00f0ff';
  ctx.shadowColor = '#00f0ff';
  ctx.shadowBlur = 30;
  ctx.fill();
  
  ctx.beginPath();
  ctx.arc(gpx, gpy, 4, 0, Math.PI*2);
  ctx.fillStyle = '#fff';
  ctx.fill();
  ctx.shadowBlur = 0;
  
  // 4. Render Robots
  if (rx && ry) {
    rx.forEach((rx_pos, i) => {
      const rpx = ((rx_pos - (-0.5)) / 8.2) * 800;
      const rpy = ((ry[i] - (-0.5)) / 7.0) * 800;
      
      // Drone Aura
      ctx.beginPath();
      ctx.arc(rpx, rpy, 15, 0, Math.PI*2);
      ctx.fillStyle = `rgba(0, 240, 255, 0.15)`;
      ctx.fill();
      
      // Drone Core
      ctx.beginPath();
      ctx.arc(rpx, rpy, 6, 0, Math.PI*2);
      ctx.fillStyle = '#ffffff';
      ctx.shadowColor = '#00f0ff';
      ctx.shadowBlur = 20;
      ctx.fill();
      ctx.shadowBlur = 0;
    });
  }
}

let ws;
function connect() {
  ws = new WebSocket('ws://localhost:8765');
  ws.onopen = () => {
    $('conn-badge').className = 'status-badge online';
    $('conn-badge').innerHTML = '<div class="dot"></div><span>NEURAL LINK ACTIVE</span>';
  };
  ws.onclose = () => {
    $('conn-badge').className = 'status-badge offline';
    $('conn-badge').innerHTML = '<div class="dot"></div><span>LINK SEVERED</span>';
    setTimeout(connect, 2000);
  };
  ws.onmessage = (e) => {
    const d = JSON.parse(e.data);
    $('bar-da').style.width = pct(d.DA);
    $('bar-ne').style.width = pct(d.NE);
    $('allo-load').innerText = f2(d.allostatic_load);
    $('pfc-state').innerText = d.mode || 'COMPUTING...';

    const bc = $('bodies-container');
    if (d.alive_flags && bc.children.length === 0) {
      bc.innerHTML = '';
      for (let i=0; i<d.alive_flags.length; i++) {
        bc.innerHTML += `
          <div class="body-card alive" id="bc-${i}">
            <div class="flex-row">
              <div>
                <div style="font-weight:700; font-size:1rem;">AGENT 0${i+1}</div>
                <div style="font-size:0.75rem; color:var(--dim); margin-top:2px;">LIFETIME: <span id="bsv-${i}">0.0s</span></div>
              </div>
              <div id="bst-${i}" style="font-size:0.8rem; font-weight:700; color:var(--cyan);">ACTIVE</div>
            </div>
          </div>
        `;
      }
    } else if (d.alive_flags) {
      for (let i=0; i<d.alive_flags.length; i++) {
        const alive = d.alive_flags[i];
        const card = $(`bc-${i}`);
        const stat = $(`bst-${i}`);
        const surv = $(`bsv-${i}`);
        if(card) {
          card.className = alive ? 'body-card alive' : 'body-card dead';
          stat.innerText = alive ? 'ACTIVE' : 'OFFLINE';
          stat.style.color = alive ? 'var(--cyan)' : 'var(--red)';
          surv.innerText = f2(d.survivals[i]) + 's';
        }
      }
    }

    renderNeuralMap(d.cognitive_map, d.astar_path, d.goal_x, d.goal_y, d.robot_x, d.robot_y);
  };
}

// Start render loop even when disconnected to keep animation smooth
function renderLoop() {
  requestAnimationFrame(renderLoop);
}
renderLoop();

connect();
</script>
</body>
</html>"""

with open('dashboard/maze_dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("Hyper-realistic UI applied.")
