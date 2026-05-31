/**
 * data.js — Synthetic Data Generation
 * Simulates structure of:
 *   - Kaggle vacanciesru dataset (job postings)
 *   - Kaggle Resume Screening Dataset 200k candidates
 *
 * Each applicant/employer is described by a skill vector.
 * Preference lists are derived from cosine similarity.
 */

const DATA = (() => {

  // All possible skills drawn from IT/Data domains typical of both datasets
  const SKILL_POOL = [
    'Python', 'SQL', 'Machine Learning', 'JavaScript', 'Java',
    'Data Analysis', 'Excel', 'PowerBI', 'Tableau', 'R',
    'Deep Learning', 'NLP', 'Docker', 'Kubernetes', 'AWS',
    'React', 'Node.js', 'PostgreSQL', 'MongoDB', 'Git',
    'Statistics', 'Pandas', 'TensorFlow', 'Spark', 'Hadoop'
  ];

  // Industry labels for employers (from vacanciesru)
  const INDUSTRIES = [
    'Fintech', 'E-commerce', 'Healthcare IT', 'EdTech',
    'Cybersecurity', 'Data Science', 'SaaS', 'Consulting',
    'Media Tech', 'Logistics Tech'
  ];

  // Education levels (from resume dataset)
  const EDUCATION = ['Bachelor', 'Master', 'PhD', 'Bootcamp', 'Self-taught'];

  // Job title templates
  const JOB_TITLES = [
    'Data Scientist', 'Backend Developer', 'ML Engineer',
    'Frontend Developer', 'Data Analyst', 'DevOps Engineer',
    'Full-Stack Dev', 'BI Developer', 'Software Engineer',
    'Research Engineer', 'Platform Engineer', 'Analytics Lead'
  ];

  /**
   * Generate random skill vector (sparse binary with noise)
   * @param {number} poolSize - total number of skills
   * @param {number} density - avg fraction of skills to include
   * @param {number} noise - σ for continuous noise added
   * @returns {Float32Array}
   */
  function randomSkillVector(poolSize, density = 0.45, noise = 0.1) {
    const v = new Float32Array(poolSize);
    for (let i = 0; i < poolSize; i++) {
      // Bernoulli draw for skill presence + noise
      v[i] = (Math.random() < density ? (0.6 + Math.random() * 0.4) : 0)
             + Math.abs(gaussianRandom(0, noise));
      if (v[i] < 0) v[i] = 0;
    }
    return v;
  }

  /** Box-Muller normal random */
  function gaussianRandom(mean = 0, std = 1) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return mean + std * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** Cosine similarity between two Float32Arrays */
  function cosineSim(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na  += a[i] * a[i];
      nb  += b[i] * b[i];
    }
    const denom = Math.sqrt(na) * Math.sqrt(nb);
    return denom === 0 ? 0 : dot / denom;
  }

  /**
   * Generate full simulation data
   * @param {Object} cfg - { nApplicants, nEmployers, nSkills, noise }
   */
  function generate(cfg) {
    const { nApplicants, nEmployers, nSkills, noise } = cfg;
    const skills = SKILL_POOL.slice(0, nSkills);

    // --- Applicants ---
    const applicants = Array.from({ length: nApplicants }, (_, i) => {
      const vec = randomSkillVector(nSkills, 0.4, noise);
      const topSkills = skills
        .map((s, idx) => ({ s, v: vec[idx] }))
        .filter(x => x.v > 0.3)
        .sort((a, b) => b.v - a.v)
        .slice(0, 4)
        .map(x => x.s);

      return {
        id: i,
        name: `A${String(i + 1).padStart(2, '0')}`,
        label: `Candidate ${i + 1}`,
        education: EDUCATION[Math.floor(Math.random() * EDUCATION.length)],
        topSkills,
        vec,
      };
    });

    // --- Employers ---
    const employers = Array.from({ length: nEmployers }, (_, i) => {
      const vec = randomSkillVector(nSkills, 0.4, noise);
      const topSkills = skills
        .map((s, idx) => ({ s, v: vec[idx] }))
        .filter(x => x.v > 0.3)
        .sort((a, b) => b.v - a.v)
        .slice(0, 4)
        .map(x => x.s);

      return {
        id: i,
        name: `E${String(i + 1).padStart(2, '0')}`,
        label: JOB_TITLES[i % JOB_TITLES.length],
        industry: INDUSTRIES[i % INDUSTRIES.length],
        topSkills,
        vec,
      };
    });

    // --- Similarity matrix ---
    // sim[a][e] = cosine similarity between applicant a and employer e
    const simMatrix = applicants.map(a =>
      employers.map(e => parseFloat(cosineSim(a.vec, e.vec).toFixed(4)))
    );

    // --- Preference lists (strict rank = sort by similarity desc) ---
    // applicantPrefs[a] = ordered list of employer indices (most preferred first)
    const applicantPrefs = applicants.map((a, ai) =>
      employers
        .map((e, ei) => ({ ei, sim: simMatrix[ai][ei] }))
        .sort((x, y) => y.sim - x.sim)
        .map(x => x.ei)
    );

    // employerPrefs[e] = ordered list of applicant indices (most preferred first)
    const employerPrefs = employers.map((e, ei) =>
      applicants
        .map((a, ai) => ({ ai, sim: simMatrix[ai][ei] }))
        .sort((x, y) => y.sim - x.sim)
        .map(x => x.ai)
    );

    return {
      applicants,
      employers,
      skills,
      simMatrix,
      applicantPrefs,
      employerPrefs,
    };
  }

  return { generate, cosineSim };
})();
