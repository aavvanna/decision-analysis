/**
 * visualize.js — Canvas rendering, bipartite graph, hero animation
 */

const VIZ = (() => {

  // ── COLORS ──────────────────────────────────────────
  const C = {
    bg:      '#0e0e10',
    surface: '#1f1f23',
    border:  '#2e2e34',
    ink:     '#f0ede8',
    ink2:    '#a8a5a0',
    ink3:    '#6a6760',
    accentA: '#e8f54e',   // applicant yellow
    accentE: '#4ef5c8',   // employer mint
    accentD: '#f54e7a',   // diff pink
  };

  // ── HERO ANIMATION ──────────────────────────────────
  let heroAnim = null;
  let heroNodes = [];

  function initHeroCanvas(canvas) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;

    // Generate sample nodes
    const nA = 6, nE = 6;
    const applicants = Array.from({ length: nA }, (_, i) => ({
      x: 80, y: 40 + i * ((H - 80) / (nA - 1)),
      vy: (Math.random() - 0.5) * 0.3,
      type: 'a', idx: i,
    }));
    const employers = Array.from({ length: nE }, (_, i) => ({
      x: W - 80, y: 40 + i * ((H - 80) / (nE - 1)),
      vy: (Math.random() - 0.5) * 0.3,
      type: 'e', idx: i,
    }));

    // Random stable matching for decoration
    const matching = [1, 4, 0, 3, 2, 5];
    const t = { v: 0 };

    function draw() {
      ctx.clearRect(0, 0, W, H);

      // Draw faint grid
      ctx.strokeStyle = 'rgba(46,46,52,0.4)';
      ctx.lineWidth = 1;
      for (let gx = 40; gx < W; gx += 60) {
        ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
      }
      for (let gy = 40; gy < H; gy += 60) {
        ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
      }

      t.v += 0.008;

      // Animate node positions
      applicants.forEach((n, i) => {
        n.y += n.vy;
        if (n.y < 20 || n.y > H - 20) n.vy *= -1;
      });
      employers.forEach((n, i) => {
        n.y += n.vy;
        if (n.y < 20 || n.y > H - 20) n.vy *= -1;
      });

      // Draw faint all-possible edges
      applicants.forEach((a, ai) => {
        employers.forEach((e, ei) => {
          if (matching[ai] === ei) return; // draw matched separately
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);

          // Bezier curve
          const mx = (a.x + e.x) / 2;
          const cy = (a.y + e.y) / 2 + Math.sin((ai + ei) * 1.3 + t.v) * 20;
          ctx.quadraticCurveTo(mx, cy, e.x, e.y);
          ctx.strokeStyle = 'rgba(106,103,96,0.07)';
          ctx.lineWidth = 1;
          ctx.stroke();
        });
      });

      // Draw matched edges (animated)
      matching.forEach((ei, ai) => {
        const a = applicants[ai], e = employers[ei];
        const progress = (Math.sin(t.v * 1.5 + ai * 0.7) + 1) / 2;

        const grad = ctx.createLinearGradient(a.x, a.y, e.x, e.y);
        grad.addColorStop(0, `rgba(232,245,78,${0.3 + progress * 0.4})`);
        grad.addColorStop(1, `rgba(78,245,200,${0.3 + progress * 0.4})`);

        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        const mx = (a.x + e.x) / 2;
        ctx.quadraticCurveTo(mx, (a.y + e.y) / 2 - 20, e.x, e.y);
        ctx.strokeStyle = grad;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });

      // Draw applicant nodes
      applicants.forEach((n, i) => {
        const glow = Math.sin(t.v * 2 + i * 0.8) * 0.3 + 0.7;
        ctx.beginPath();
        ctx.arc(n.x, n.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(232,245,78,${0.15 * glow})`;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(n.x, n.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = C.accentA;
        ctx.fill();

        ctx.font = '11px DM Mono';
        ctx.fillStyle = C.ink3;
        ctx.textAlign = 'right';
        ctx.fillText(`A${i + 1}`, n.x - 14, n.y + 4);
      });

      // Draw employer nodes
      employers.forEach((n, i) => {
        const glow = Math.sin(t.v * 2 + i * 1.1 + Math.PI) * 0.3 + 0.7;
        ctx.beginPath();
        ctx.arc(n.x, n.y, 8, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(78,245,200,${0.15 * glow})`;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(n.x, n.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = C.accentE;
        ctx.fill();

        ctx.font = '11px DM Mono';
        ctx.fillStyle = C.ink3;
        ctx.textAlign = 'left';
        ctx.fillText(`E${i + 1}`, n.x + 14, n.y + 4);
      });

      // Labels
      ctx.font = '10px DM Mono';
      ctx.fillStyle = 'rgba(232,245,78,0.5)';
      ctx.textAlign = 'center';
      ctx.fillText('APPLICANTS', 80, H - 10);
      ctx.fillStyle = 'rgba(78,245,200,0.5)';
      ctx.fillText('EMPLOYERS', W - 80, H - 10);

      heroAnim = requestAnimationFrame(draw);
    }

    if (heroAnim) cancelAnimationFrame(heroAnim);
    draw();
  }


  // ── BIPARTITE GRAPH ─────────────────────────────────

  let graphView = 'applicant'; // 'applicant' | 'employer' | 'diff'
  let graphData = null;

  function drawBipartiteGraph(canvas, data, resultA, resultE, view) {
    graphData = { canvas, data, resultA, resultE };
    graphView = view;

    const DPR = window.devicePixelRatio || 1;
    const displayW = canvas.parentElement.clientWidth;
    const nMax = Math.max(data.applicants.length, data.employers.length);
    const displayH = Math.max(420, nMax * 36 + 80);

    canvas.style.width  = displayW + 'px';
    canvas.style.height = displayH + 'px';
    canvas.width  = displayW * DPR;
    canvas.height = displayH * DPR;

    const ctx = canvas.getContext('2d');
    ctx.scale(DPR, DPR);

    const W = displayW, H = displayH;
    ctx.clearRect(0, 0, W, H);

    const nA = data.applicants.length;
    const nE = data.employers.length;

    const pad = 80;
    const leftX  = pad + 40;
    const rightX = W - pad - 40;

    // Node y positions
    function nodeY(i, total) {
      if (total === 1) return H / 2;
      return pad + i * ((H - 2 * pad) / (total - 1));
    }

    const posA = data.applicants.map((_, i) => ({ x: leftX,  y: nodeY(i, nA) }));
    const posE = data.employers.map((_, i)  => ({ x: rightX, y: nodeY(i, nE) }));

    // Determine which matching to draw
    const matchA = resultA ? resultA.matching : null;
    const matchE = resultE ? resultE.matching : null;
    const diffSet = (matchA && matchE) ? new Set(GS.diffMatchings(matchA, matchE, nA)) : new Set();

    // Draw edges
    function drawEdge(ax, ay, ex, ey, color, width, alpha, dashed) {
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      if (dashed) ctx.setLineDash([6, 4]);
      else ctx.setLineDash([]);

      const mx = (ax + ex) / 2;
      const cy = (ay + ey) / 2;

      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.bezierCurveTo(mx, ay, mx, ey, ex, ey);
      ctx.stroke();
      ctx.restore();
    }

    // Choose what to draw
    if (view === 'applicant' && matchA) {
      // Draw all applicant-optimal edges
      for (const [ai, ei] of matchA) {
        const isDiff = diffSet.has(ai);
        drawEdge(
          posA[ai].x, posA[ai].y,
          posE[ei].x, posE[ei].y,
          isDiff ? C.accentD : C.accentA,
          isDiff ? 2.5 : 1.5,
          isDiff ? 0.9 : 0.6,
          false
        );
      }
    } else if (view === 'employer' && matchE) {
      for (const [ai, ei] of matchE) {
        const isDiff = diffSet.has(ai);
        drawEdge(
          posA[ai].x, posA[ai].y,
          posE[ei].x, posE[ei].y,
          isDiff ? C.accentD : C.accentE,
          isDiff ? 2.5 : 1.5,
          isDiff ? 0.9 : 0.6,
          false
        );
      }
    } else if (view === 'diff' && matchA && matchE) {
      // Draw A-optimal in yellow (faint)
      for (const [ai, ei] of matchA) {
        if (!diffSet.has(ai)) {
          drawEdge(posA[ai].x, posA[ai].y, posE[ei].x, posE[ei].y, C.accentA, 1.5, 0.3, false);
        }
      }
      // Draw pairs that differ: show both
      for (const ai of diffSet) {
        const eiA = matchA.get(ai);
        const eiE = matchE.get(ai);
        if (eiA !== undefined) {
          drawEdge(posA[ai].x, posA[ai].y, posE[eiA].x, posE[eiA].y, C.accentA, 2, 0.8, false);
        }
        if (eiE !== undefined) {
          drawEdge(posA[ai].x, posA[ai].y, posE[eiE].x, posE[eiE].y, C.accentE, 2, 0.8, true);
        }
      }
    }

    // Draw applicant nodes
    posA.forEach((pos, i) => {
      const a = data.applicants[i];
      const isChanged = diffSet.has(i);
      const r = 7;

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r + 4, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(232,245,78,0.08)`;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
      ctx.fillStyle = isChanged ? C.accentD : C.accentA;
      ctx.fill();

      ctx.font = '11px DM Mono';
      ctx.fillStyle = C.ink2;
      ctx.textAlign = 'right';
      ctx.fillText(a.name, pos.x - 16, pos.y + 4);

      // rank annotations
      if (view !== 'diff') {
        const ranks = view === 'applicant' ? resultA?.perApplicantRank : resultE?.perApplicantRank;
        if (ranks && ranks[i] !== null) {
          ctx.font = '10px DM Mono';
          ctx.fillStyle = C.ink3;
          ctx.textAlign = 'left';
          ctx.fillText(`#${ranks[i]}`, pos.x - 60, pos.y - 4);
        }
      }
    });

    // Draw employer nodes
    posE.forEach((pos, i) => {
      const e = data.employers[i];
      const r = 7;

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r + 4, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(78,245,200,0.08)`;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
      ctx.fillStyle = C.accentE;
      ctx.fill();

      ctx.font = '11px DM Mono';
      ctx.fillStyle = C.ink2;
      ctx.textAlign = 'left';
      ctx.fillText(e.name, pos.x + 16, pos.y + 4);

      if (view !== 'diff') {
        const ranks = view === 'applicant' ? resultA?.perEmployerRank : resultE?.perEmployerRank;
        if (ranks && ranks[i] !== null) {
          ctx.font = '10px DM Mono';
          ctx.fillStyle = C.ink3;
          ctx.textAlign = 'right';
          ctx.fillText(`#${ranks[i]}`, pos.x + 60, pos.y - 4);
        }
      }
    });

    // Column labels
    ctx.font = '600 11px DM Mono';
    ctx.textAlign = 'center';
    ctx.fillStyle = `rgba(232,245,78,0.6)`;
    ctx.fillText('APPLICANTS', leftX, 24);
    ctx.fillStyle = `rgba(78,245,200,0.6)`;
    ctx.fillText('EMPLOYERS', rightX, 24);
  }

  function redrawGraph(view) {
    if (!graphData) return;
    const { canvas, data, resultA, resultE } = graphData;
    drawBipartiteGraph(canvas, data, resultA, resultE, view);
  }


  // ── SIMILARITY MATRIX TABLE ─────────────────────────

  function buildSimMatrixTable(data) {
    const { applicants, employers, simMatrix } = data;
    let html = '<table class="pref-table"><thead><tr><th>A \\ E</th>';
    employers.forEach(e => { html += `<th>${e.name}</th>`; });
    html += '</tr></thead><tbody>';

    applicants.forEach((a, ai) => {
      html += `<tr><td><b>${a.name}</b></td>`;
      const row = simMatrix[ai];
      const sortedVals = [...row].sort((x, y) => y - x);

      row.forEach((val, ei) => {
        const rankIdx = sortedVals.indexOf(val);
        let cls = rankIdx === 0 ? 'rank-1' : rankIdx === 1 ? 'rank-2' : 'rank-3';
        html += `<td class="${cls}">${val.toFixed(3)}</td>`;
      });
      html += '</tr>';
    });

    html += '</tbody></table>';
    return html;
  }

  function buildPrefsTable(data, side) {
    const { applicants, employers, applicantPrefs, employerPrefs } = data;

    if (side === 'applicant') {
      let html = '<table class="pref-table"><thead><tr><th>Applicant</th><th>Top Skills</th>';
      for (let i = 1; i <= employers.length; i++) html += `<th>Rank ${i}</th>`;
      html += '</tr></thead><tbody>';

      applicants.forEach((a, ai) => {
        html += `<tr><td><b>${a.name}</b></td><td style="font-size:10px;color:var(--ink3)">${(a.topSkills || []).join(', ')}</td>`;
        applicantPrefs[ai].forEach((ei, rank) => {
          const cls = rank === 0 ? 'rank-1' : rank === 1 ? 'rank-2' : 'rank-3';
          html += `<td class="${cls}">${employers[ei].name}</td>`;
        });
        html += '</tr>';
      });
      html += '</tbody></table>';
      return html;
    } else {
      let html = '<table class="pref-table"><thead><tr><th>Employer</th><th>Industry</th>';
      for (let i = 1; i <= applicants.length; i++) html += `<th>Rank ${i}</th>`;
      html += '</tr></thead><tbody>';

      employers.forEach((e, ei) => {
        html += `<tr><td><b>${e.name}</b></td><td style="font-size:10px;color:var(--ink3)">${e.industry}</td>`;
        employerPrefs[ei].forEach((ai, rank) => {
          const cls = rank === 0 ? 'rank-1' : rank === 1 ? 'rank-2' : 'rank-3';
          html += `<td class="${cls}">${applicants[ai].name}</td>`;
        });
        html += '</tr>';
      });
      html += '</tbody></table>';
      return html;
    }
  }


  // ── RANK BARS ───────────────────────────────────────

  function buildRankBars(result, data, side, containerEl) {
    const entities = side === 'applicant' ? data.applicants : data.employers;
    const ranks    = side === 'applicant' ? result.perApplicantRank : result.perEmployerRank;
    const total    = side === 'applicant' ? data.employers.length : data.applicants.length;
    const cls      = side === 'applicant' ? 'applicant-fill' : 'employer-fill';

    if (!ranks) return;

    containerEl.innerHTML = '';
    entities.forEach((entity, i) => {
      const rank = ranks[i];
      if (rank === null) return;

      const pct = Math.max(4, 100 - ((rank - 1) / (total - 1 || 1)) * 100);

      containerEl.insertAdjacentHTML('beforeend', `
        <div class="rank-bar-row">
          <div class="rank-bar-label" title="${entity.label || entity.name}">${entity.name}</div>
          <div class="rank-bar-track">
            <div class="rank-bar-fill ${cls}" style="width:${pct}%"></div>
          </div>
          <div class="rank-bar-num">#${rank}</div>
        </div>
      `);
    });
  }

  return {
    initHeroCanvas,
    drawBipartiteGraph,
    redrawGraph,
    buildSimMatrixTable,
    buildPrefsTable,
    buildRankBars,
    C,
  };
})();
