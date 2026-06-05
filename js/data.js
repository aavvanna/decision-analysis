/**
 * data.js — Real-data loader
 *
 * Loads a deterministic 20x20 sample produced by analysis/02_experiments.py
 * from js/sample_data.json. The sample contains real LinkedIn job-posting
 * skill profiles and real resume skill profiles (extracted from Kaggle
 * datasets) along with their cosine-similarity matrix and derived strict
 * preference lists.
 *
 * Exposes the same API as the original synthetic generator so the rest of
 * the app does not need to change, except that `generate` is now async.
 */

const DATA = (() => {

  let cached = null;  // { applicants, employers, simMatrix, applicantPrefs, employerPrefs }

  async function loadSample() {
    if (cached) return cached;
    const resp = await fetch('js/sample_data.json');
    if (!resp.ok) {
      throw new Error(`Failed to load sample_data.json: ${resp.status}`);
    }
    cached = await resp.json();
    return cached;
  }

  /**
   * Return a slice of the loaded sample with the requested number of
   * applicants and employers. Preference lists are filtered to the chosen
   * subsets (preserving relative order).
   *
   * @param {Object} cfg - { nApplicants, nEmployers }  (other fields ignored)
   */
  async function generate(cfg) {
    const sample = await loadSample();
    const nA = Math.min(cfg.nApplicants ?? sample.applicants.length, sample.applicants.length);
    const nE = Math.min(cfg.nEmployers ?? sample.employers.length, sample.employers.length);

    const eSet = new Set(Array.from({ length: nE }, (_, i) => i));
    const aSet = new Set(Array.from({ length: nA }, (_, i) => i));

    const applicants = sample.applicants.slice(0, nA);
    const employers = sample.employers.slice(0, nE);

    // Slice the similarity matrix
    const simMatrix = applicants.map((_, ai) =>
      employers.map((_, ei) => sample.simMatrix[ai][ei])
    );

    // Filter preference lists to the active subsets, preserving order.
    const applicantPrefs = sample.applicantPrefs.slice(0, nA).map(list =>
      list.filter(ei => eSet.has(ei))
    );
    const employerPrefs = sample.employerPrefs.slice(0, nE).map(list =>
      list.filter(ai => aSet.has(ai))
    );

    return {
      applicants,
      employers,
      skills: [],  // kept for API compatibility; vocabulary is implicit in topSkills
      simMatrix,
      applicantPrefs,
      employerPrefs,
    };
  }

  /** Cosine similarity between two arrays — kept for any external callers. */
  function cosineSim(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    const denom = Math.sqrt(na) * Math.sqrt(nb);
    return denom === 0 ? 0 : dot / denom;
  }

  return { generate, cosineSim };
})();
