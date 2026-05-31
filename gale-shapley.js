/**
 * gale-shapley.js — Deferred Acceptance Algorithm
 *
 * Implements both variants:
 *   - applicantOptimal: applicants propose (best for applicants)
 *   - employerOptimal: employers propose (best for employers)
 *
 * Each returns:
 *   matching: Map<proposerIdx, receiverIdx>
 *   rounds: number of rounds until convergence
 *   history: round-by-round proposal log (for animation)
 *   stable: boolean (verification)
 *   avgProposerRank: average rank the proposer got (1=best)
 *   avgReceiverRank: average rank the receiver got (1=best)
 */

const GS = (() => {

  /**
   * Core deferred acceptance algorithm.
   * @param {number[][]} proposerPrefs  - proposerPrefs[p] = ranked list of receiver indices
   * @param {number[][]} receiverPrefs  - receiverPrefs[r] = ranked list of proposer indices
   * @param {number} nProposers
   * @param {number} nReceivers
   */
  function deferredAcceptance(proposerPrefs, receiverPrefs, nProposers, nReceivers) {
    // rankLookup[r][p] = rank of proposer p in receiver r's preference list (0=best)
    const rankLookup = Array.from({ length: nReceivers }, (_, r) => {
      const rank = new Map();
      receiverPrefs[r].forEach((p, idx) => rank.set(p, idx));
      return rank;
    });

    // nextProposal[p] = index into proposerPrefs[p] of next proposal to make
    const nextProposal = new Array(nProposers).fill(0);

    // current tentative hold: heldBy[r] = proposer index currently held by receiver r, or -1
    const heldBy = new Array(nReceivers).fill(-1);

    // matchedTo[p] = receiver index proposer p is currently tentatively matched to, or -1
    const matchedTo = new Array(nProposers).fill(-1);

    const history = []; // for visualization
    let rounds = 0;
    let freeProposers = Array.from({ length: nProposers }, (_, i) => i);

    while (freeProposers.length > 0) {
      rounds++;
      const roundLog = [];
      const nextFree = [];

      for (const p of freeProposers) {
        if (nextProposal[p] >= nReceivers) continue; // exhausted list

        const r = proposerPrefs[p][nextProposal[p]];
        nextProposal[p]++;

        roundLog.push({ proposer: p, receiver: r });

        const currentHolder = heldBy[r];
        if (currentHolder === -1) {
          // Receiver is free — tentatively accept
          heldBy[r] = p;
          matchedTo[p] = r;
        } else {
          // Receiver compares: prefer better rank (lower index = better)
          const rankCurrent = rankLookup[r].get(currentHolder) ?? Infinity;
          const rankNew     = rankLookup[r].get(p) ?? Infinity;

          if (rankNew < rankCurrent) {
            // New proposer is preferred — dump current holder
            heldBy[r] = p;
            matchedTo[p] = r;
            matchedTo[currentHolder] = -1;
            nextFree.push(currentHolder); // current holder is now free
          } else {
            // Receiver keeps current — new proposer stays free
            nextFree.push(p);
          }
        }
      }

      history.push({
        round: rounds,
        proposals: roundLog,
        matching: [...heldBy], // snapshot
      });

      freeProposers = nextFree;
    }

    // Build final matching as Map<proposerIdx, receiverIdx>
    const matching = new Map();
    for (let p = 0; p < nProposers; p++) {
      if (matchedTo[p] !== -1) matching.set(p, matchedTo[p]);
    }

    return { matching, rounds, history };
  }

  /** Compute average rank (1-indexed) the proposer got */
  function avgProposerRank(matching, proposerPrefs) {
    let sum = 0, count = 0;
    for (const [p, r] of matching) {
      const rank = proposerPrefs[p].indexOf(r) + 1; // 1-indexed
      sum += rank; count++;
    }
    return count > 0 ? sum / count : 0;
  }

  /** Compute average rank (1-indexed) the receiver got */
  function avgReceiverRank(matching, receiverPrefs) {
    let sum = 0, count = 0;
    for (const [p, r] of matching) {
      const rank = receiverPrefs[r].indexOf(p) + 1; // 1-indexed
      sum += rank; count++;
    }
    return count > 0 ? sum / count : 0;
  }

  /**
   * Verify stability: find any blocking pairs.
   * @returns {Array} list of blocking pairs [{proposer, receiver}]
   */
  function findBlockingPairs(matching, proposerPrefs, receiverPrefs) {
    const nProposers = proposerPrefs.length;
    const nReceivers = receiverPrefs.length;

    // Build reverse lookup: receiverMatch[r] = p  and  proposerMatch[p] = r
    const proposerMatch = new Map(matching);
    const receiverMatch = new Map();
    for (const [p, r] of matching) receiverMatch.set(r, p);

    const blocking = [];

    for (let p = 0; p < nProposers; p++) {
      for (let r = 0; r < nReceivers; r++) {
        const pMatchedTo = proposerMatch.get(p) ?? -1;
        const rMatchedTo = receiverMatch.get(r) ?? -1;

        if (pMatchedTo === r) continue; // already matched to each other

        // p prefers r over their current match?
        const pRankCurrent = pMatchedTo === -1 ? Infinity : proposerPrefs[p].indexOf(pMatchedTo);
        const pRankR       = proposerPrefs[p].indexOf(r);
        const pPrefersR    = pRankR < pRankCurrent;

        // r prefers p over their current match?
        const rRankCurrent = rMatchedTo === -1 ? Infinity : receiverPrefs[r].indexOf(rMatchedTo);
        const rRankP       = receiverPrefs[r].indexOf(p);
        const rPrefersP    = rRankP < rRankCurrent;

        if (pPrefersR && rPrefersP) {
          blocking.push({ proposer: p, receiver: r });
        }
      }
    }
    return blocking;
  }

  /**
   * Run APPLICANT-OPTIMAL matching (applicants propose)
   */
  function applicantOptimal(data) {
    const { applicants, employers, applicantPrefs, employerPrefs } = data;
    const n = applicants.length;
    const m = employers.length;

    const result = deferredAcceptance(applicantPrefs, employerPrefs, n, m);

    // In this variant: proposers = applicants, receivers = employers
    const aPropRank = avgProposerRank(result.matching, applicantPrefs);
    const eRecvRank = avgReceiverRank(result.matching, employerPrefs);

    const blocking = findBlockingPairs(result.matching, applicantPrefs, employerPrefs);

    // Per-applicant rank in A-optimal
    const perApplicantRank = applicants.map((a, i) => {
      const r = result.matching.get(i);
      if (r === undefined) return null;
      return applicantPrefs[i].indexOf(r) + 1;
    });

    // Per-employer rank in A-optimal
    const perEmployerRank = employers.map((e, i) => {
      // find which applicant was matched to this employer
      let matchedApplicant = -1;
      for (const [p, r] of result.matching) {
        if (r === i) { matchedApplicant = p; break; }
      }
      if (matchedApplicant === -1) return null;
      return employerPrefs[i].indexOf(matchedApplicant) + 1;
    });

    return {
      ...result,
      avgApplicantRank: aPropRank,
      avgEmployerRank:  eRecvRank,
      perApplicantRank,
      perEmployerRank,
      blockingPairs: blocking,
      isStable: blocking.length === 0,
      type: 'applicant-optimal',
    };
  }

  /**
   * Run EMPLOYER-OPTIMAL matching (employers propose)
   * NOTE: In this run, "proposers" are employers. We need to invert the matching.
   */
  function employerOptimal(data) {
    const { applicants, employers, applicantPrefs, employerPrefs } = data;
    const n = applicants.length;
    const m = employers.length;

    // Employers propose → proposerPrefs=employerPrefs, receiverPrefs=applicantPrefs
    const result = deferredAcceptance(employerPrefs, applicantPrefs, m, n);

    // result.matching is employer→applicant. Invert to applicant→employer for consistency.
    const matchingInverted = new Map(); // applicant → employer
    for (const [e, a] of result.matching) {
      matchingInverted.set(a, e);
    }

    // In this variant: proposers=employers, receivers=applicants
    // avgProposerRank of employer = rank they got among their preferences
    const ePropRank = avgProposerRank(result.matching, employerPrefs);
    // avgReceiverRank of applicant (as receiver)
    const aRecvRank = avgReceiverRank(result.matching, applicantPrefs);

    // Verify stability using inverted matching (consistent interface)
    const blocking = findBlockingPairs(matchingInverted, applicantPrefs, employerPrefs);

    const perApplicantRank = applicants.map((a, i) => {
      const r = matchingInverted.get(i);
      if (r === undefined) return null;
      return applicantPrefs[i].indexOf(r) + 1;
    });

    const perEmployerRank = employers.map((e, i) => {
      const a = result.matching.get(i);
      if (a === undefined) return null;
      return employerPrefs[i].indexOf(a) + 1;
    });

    return {
      ...result,
      matching: matchingInverted,  // now applicant → employer for consistency
      avgApplicantRank: aRecvRank,
      avgEmployerRank:  ePropRank,
      perApplicantRank,
      perEmployerRank,
      blockingPairs: blocking,
      isStable: blocking.length === 0,
      type: 'employer-optimal',
    };
  }

  /**
   * Compute which pairs differ between two matchings
   */
  function diffMatchings(mA, mE, nApplicants) {
    const changed = [];
    for (let i = 0; i < nApplicants; i++) {
      const rA = mA.get(i);
      const rE = mE.get(i);
      if (rA !== rE) changed.push(i);
    }
    return changed;
  }

  return { applicantOptimal, employerOptimal, diffMatchings };
})();
