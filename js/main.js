/**
 * main.js — UI Orchestration
 * Wires together config sliders, buttons, simulation, and rendering.
 */

(function () {

  // ── INIT HERO ───────────────────────────────────────
  const heroCanvas = document.getElementById('heroCanvas');
  if (heroCanvas) VIZ.initHeroCanvas(heroCanvas);

  // ── SLIDER BINDINGS ─────────────────────────────────
  function bindSlider(id, valId, transform) {
    const slider = document.getElementById(id);
    const valEl  = document.getElementById(valId);
    slider.addEventListener('input', () => {
      const raw = parseFloat(slider.value);
      const display = transform ? transform(raw) : raw;
      valEl.textContent = display;
    });
  }

  bindSlider('cfg-applicants', 'val-applicants');
  bindSlider('cfg-employers',  'val-employers');

  // ── MODE TABS ────────────────────────────────────────
  let currentMode = 'both';
  document.querySelectorAll('.mode-tab').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mode-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentMode = btn.dataset.mode;
    });
  });

  // ── GRAPH VIEW TABS ──────────────────────────────────
  let graphView = 'applicant';
  document.querySelectorAll('.gc-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.gc-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      graphView = btn.dataset.view;
      VIZ.redrawGraph(graphView);
    });
  });

  // ── TABLE TABS ───────────────────────────────────────
  let currentTableTab = 'applicant-prefs';
  let lastData = null;

  document.querySelectorAll('.tt-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tt-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentTableTab = btn.dataset.tab;
      renderTableTab();
    });
  });

  function renderTableTab() {
    if (!lastData) return;
    const wrap = document.getElementById('table-wrap');
    if (currentTableTab === 'applicant-prefs') {
      wrap.innerHTML = VIZ.buildPrefsTable(lastData, 'applicant');
    } else if (currentTableTab === 'employer-prefs') {
      wrap.innerHTML = VIZ.buildPrefsTable(lastData, 'employer');
    } else {
      wrap.innerHTML = VIZ.buildSimMatrixTable(lastData);
    }
  }

  // ── MAIN RUN ─────────────────────────────────────────
  const runBtn = document.getElementById('btn-run');

  runBtn.addEventListener('click', runSimulation);

  function getConfig() {
    return {
      nApplicants: parseInt(document.getElementById('cfg-applicants').value),
      nEmployers:  parseInt(document.getElementById('cfg-employers').value),
    };
  }

  function runSimulation() {
    runBtn.classList.add('loading');
    runBtn.textContent = 'Running…';

    // Small delay to let the UI update before heavy computation
    setTimeout(async () => {
      try {
        const cfg = getConfig();
        const data = await DATA.generate(cfg);
        lastData = data;

        let resultA = null, resultE = null;

        if (currentMode === 'applicant' || currentMode === 'both') {
          resultA = GS.applicantOptimal(data);
        }
        if (currentMode === 'employer' || currentMode === 'both') {
          resultE = GS.employerOptimal(data);
        }

        renderResults(data, resultA, resultE);

        // Scroll to results
        document.getElementById('analysis').scrollIntoView({ behavior: 'smooth', block: 'start' });
      } catch (e) {
        console.error(e);
        alert('Simulation error: ' + e.message);
      }

      runBtn.classList.remove('loading');
      runBtn.innerHTML = '<span class="run-icon">▶</span> Run Again';
    }, 30);
  }

  function renderResults(data, resultA, resultE) {
    const n = data.applicants.length;

    // ── STATS CARDS ────────────────────────────────────
    const statsGrid = document.getElementById('stats-grid');
    statsGrid.innerHTML = '';

    const cards = [];

    if (resultA) {
      cards.push({
        label: 'Avg Applicant Rank (A-Opt)',
        value: resultA.avgApplicantRank.toFixed(2),
        sub:   `out of ${data.employers.length} employers`,
        cls:   'ac',
      });
      cards.push({
        label: 'Avg Employer Rank (A-Opt)',
        value: resultA.avgEmployerRank.toFixed(2),
        sub:   `out of ${data.applicants.length} applicants`,
        cls:   'em',
      });
    }

    if (resultE) {
      cards.push({
        label: 'Avg Applicant Rank (E-Opt)',
        value: resultE.avgApplicantRank.toFixed(2),
        sub:   `out of ${data.employers.length} employers`,
        cls:   'em',
      });
      cards.push({
        label: 'Avg Employer Rank (E-Opt)',
        value: resultE.avgEmployerRank.toFixed(2),
        sub:   `out of ${data.applicants.length} applicants`,
        cls:   'ac',
      });
    }

    if (resultA && resultE) {
      const diffCount = GS.diffMatchings(resultA.matching, resultE.matching, n).length;
      cards.push({
        label: 'Pairs Changed Between Modes',
        value: diffCount,
        sub:   `${((diffCount / n) * 100).toFixed(0)}% of all matches differ`,
        cls:   'diff',
      });

      const advantage = (resultA.avgApplicantRank - resultE.avgApplicantRank).toFixed(2);
      cards.push({
        label: 'Applicant Rank Gain (A-Opt vs E-Opt)',
        value: (advantage > 0 ? '−' : '+') + Math.abs(advantage),
        sub:   advantage > 0 ? 'Applicants benefit from proposing' : 'No advantage in this instance',
        cls:   'neutral',
      });

      const rounds = `${resultA.rounds} / ${resultE.rounds}`;
      cards.push({
        label: 'Convergence Rounds (A / E)',
        value: rounds,
        sub:   'Rounds until stable matching found',
        cls:   'neutral',
      });
    }

    // Total matched
    const mr = resultA || resultE;
    if (mr) {
      cards.push({
        label: 'Total Pairs Matched',
        value: mr.matching.size,
        sub:   `of ${Math.min(n, data.employers.length)} possible`,
        cls:   'neutral',
      });
    }

    cards.forEach(c => {
      statsGrid.insertAdjacentHTML('beforeend', `
        <div class="stat-card ${c.cls}">
          <div class="stat-label">${c.label}</div>
          <div class="stat-value">${c.value}</div>
          <div class="stat-sub">${c.sub}</div>
        </div>
      `);
    });

    // ── RANK BARS ──────────────────────────────────────
    if (resultA) {
      VIZ.buildRankBars(resultA, data, 'applicant', document.getElementById('bars-applicant'));
    }
    if (resultE) {
      VIZ.buildRankBars(resultE, data, 'applicant', document.getElementById('bars-employer'));
    }

    // Show/hide panels
    const panelA = document.getElementById('panel-applicant');
    const panelE = document.getElementById('panel-employer');
    const vsDiv  = document.querySelector('.vs-divider');

    if (currentMode === 'applicant') {
      panelE.style.display = 'none';
      vsDiv.style.display  = 'none';
      document.querySelector('.comparison-wrap').style.gridTemplateColumns = '1fr';
    } else if (currentMode === 'employer') {
      panelA.style.display = 'none';
      vsDiv.style.display  = 'none';
      document.querySelector('.comparison-wrap').style.gridTemplateColumns = '1fr';
    } else {
      panelA.style.display = '';
      panelE.style.display = '';
      vsDiv.style.display  = '';
      document.querySelector('.comparison-wrap').style.gridTemplateColumns = '1fr 60px 1fr';
    }

    // ── BLOCKING PAIRS INFO ────────────────────────────
    const bi = document.getElementById('blocking-info');
    let bHTML = '';
    if (resultA) {
      bHTML += `<span class="${resultA.isStable ? 'blocking-ok' : 'blocking-fail'}">
        A-Optimal: ${resultA.isStable ? '✓ Stable (0 blocking pairs)' : `✗ Unstable (${resultA.blockingPairs.length} blocking pairs)`}
      </span>`;
    }
    if (resultE) {
      if (bHTML) bHTML += '  ·  ';
      bHTML += `<span class="${resultE.isStable ? 'blocking-ok' : 'blocking-fail'}">
        E-Optimal: ${resultE.isStable ? '✓ Stable (0 blocking pairs)' : `✗ Unstable (${resultE.blockingPairs.length} blocking pairs)`}
      </span>`;
    }
    if (resultA && resultE) {
      const diffCount = GS.diffMatchings(resultA.matching, resultE.matching, n).length;
      bHTML += `  ·  <span style="color:var(--ink3)">${diffCount} pair(s) differ between the two stable matchings</span>`;
    }
    bi.innerHTML = bHTML;

    // ── GRAPH ──────────────────────────────────────────
    const graphCanvas = document.getElementById('graphCanvas');
    // Reset to applicant view when new data arrives
    graphView = 'applicant';
    document.querySelectorAll('.gc-btn').forEach((b, i) => {
      b.classList.toggle('active', i === 0);
    });

    VIZ.drawBipartiteGraph(graphCanvas, data, resultA, resultE, graphView);

    // ── PREFERENCE TABLE ───────────────────────────────
    renderTableTab();

    // ── SHOW ALL SECTIONS ──────────────────────────────
    document.getElementById('analysis').classList.remove('hidden');
    document.getElementById('graph').classList.remove('hidden');
    document.getElementById('pref-table-section').classList.remove('hidden');
  }

  // Redraw graph on resize
  window.addEventListener('resize', () => VIZ.redrawGraph(graphView));

  // Run a default simulation on page load so the page isn't empty
  setTimeout(runSimulation, 600);

})();
