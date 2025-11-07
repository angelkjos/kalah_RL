# Training Verification Report

This report documents the verification and improvements made to the Kalah RL training system.

---

## Issues Investigated

### 1. Agent Persistence Between Training Iterations ✅

**Question:** When training in interactive mode, does the agent load the model and continue training on the previously trained model?

**Answer:** **YES** ✅

**Details:**
- The agent is created once at the start of the interactive session (line 19 in `src/training/rl-demo.js`)
- The same agent instance is reused throughout all menu options
- Each training iteration continues training the same neural network
- The model's weights persist and improve cumulatively

**Code Reference:**
```javascript
// src/training/rl-demo.js:19
const agent = new QLearningAgent({
    learningRate: 0.001,
    discountFactor: 0.95,
    epsilon: 1.0,
    epsilonDecay: 0.995,
    epsilonMin: 0.1
});

// Agent is reused for all training runs
```

---

### 2. Empty Pit Actions Filtering ✅

**Question:** Are empty pit moves impossible during training?

**Answer:** **YES** ✅

**Details:**
- The game engine's `getValidMoves()` method only returns pits with seeds > 0
- The agent's `selectAction()` method only selects from `validMoves`
- This applies to both exploration (random selection) and exploitation (best Q-value)
- Verified with 20+ consecutive action selections - all were valid

**Code Reference:**
```javascript
// src/engine/kalah-engine.js:70-85
getValidMoves() {
    if (this.gameOver) return [];

    const startPit = this.currentPlayer * this.pitsPerPlayer;
    const endPit = startPit + this.pitsPerPlayer;
    const validMoves = [];

    for (let i = startPit; i < endPit; i++) {
        if (this.board[i] > 0) {  // ← Only non-empty pits
            validMoves.push(i);
        }
    }
    return validMoves;
}

// src/ai/rl-agent.js:94-97
selectAction(state, validMoves) {
    if (Math.random() < this.epsilon) {
        // Random selection FROM validMoves
        return validMoves[Math.floor(Math.random() * validMoves.length)];
    }
    // ... best Q-value FROM validMoves
}
```

**Test Results:**
```
Board state: [ 0, 0, 0, 4, 4, 4 ]
Valid moves for Player 0: [ 3, 4, 5 ]
✅ Empty pits correctly filtered: true
✅ Agent only selects valid moves: true
```

---

### 3. Win/Draw/Loss Rate Verification ✅

**Question:** Are win/draw/loss rates calculated correctly?

**Answer:** **YES** ✅

**Details:**
- Winner detection correctly identifies:
  - Player 0 wins (stores[0] > stores[1]) → winner = 0
  - Player 1 wins (stores[1] > stores[0]) → winner = 1
  - Draw (stores[0] === stores[1]) → winner = null
- Stats counting is accurate: `wins + losses + draws = gamesPlayed`
- Verified with 100-game training run

**Code Reference:**
```javascript
// src/engine/kalah-engine.js:286-298
getWinner() {
    if (!this.gameOver) return -1;

    if (this.stores[0] > this.stores[1]) return 0;
    else if (this.stores[1] > this.stores[0]) return 1;
    else return null; // Draw
}

// src/ai/trainer.js:242-257
updateStats(game, loss) {
    this.stats.gamesPlayed++;
    const winner = game.getWinner();

    if (winner === 0) this.stats.wins++;
    else if (winner === 1) this.stats.losses++;
    else this.stats.draws++;  // winner === null (draw)
}
```

**Test Results:**
```
Game 1 - Player 0: 30, Player 1: 18
Winner: 0 (expected: 0) ✅

Game 2 - Player 0: 18, Player 1: 30
Winner: 1 (expected: 1) ✅

Game 3 - Player 0: 24, Player 1: 24
Winner: null (expected: null for draw) ✅

Training 100 games:
  Wins: 49, Losses: 47, Draws: 4
  Total: 100 ✅
```

---

## Improvements Made

### 1. Stats Reset Between Training Runs

**Problem:** Stats accumulated across multiple training runs in interactive mode.

**Solution:** Added `resetStats()` method that's called at the start of each training run.

**Implementation:**
```javascript
// src/ai/trainer.js:35-44
resetStats() {
    this.stats = {
        gamesPlayed: 0,
        wins: 0,
        losses: 0,
        draws: 0,
        totalReward: 0,
        recentWinRate: []
    };
}

// Called at the start of each training method
async trainSelfPlay(numEpisodes) {
    this.resetStats();  // ← Reset before training
    console.log(`Training via self-play for ${numEpisodes} episodes...`);
    // ...
}
```

**Test Results:**
```
Training run #1: 50 games
  Stats: 28W 20L 2D (Total: 50)

Training run #2: 30 games
  Stats: 13W 17L 0D (Total: 30)  ← Not 80!

✅ PASS: Stats correctly reset between training runs
```

---

### 2. Curriculum Learning Stats

**Behavior:** Stats are **cumulative across all stages** by default.

**Rationale:** Curriculum learning is a single training session with progressive difficulty, so cumulative stats show overall performance.

**Implementation:**
```javascript
async trainCurriculum(numEpisodes, resetStatsPerStage = false) {
    if (!resetStatsPerStage) {
        this.resetStats(); // Reset once at the start
    }

    // Override resetStats() during stages to prevent resetting
    const originalResetStats = this.resetStats.bind(this);
    if (!resetStatsPerStage) {
        this.resetStats = () => {};  // Temporarily disable
    }

    // ... run all stages ...

    this.resetStats = originalResetStats;  // Restore
}
```

**Test Results:**
```
Curriculum training: 150 games (45 + 60 + 45)

Stage 1/3: Random opponent - 45 games
  Stats: 26W 16L 3D (Total: 45)

Stage 2/3: Self-play - 60 games
  Stats: 57W 43L 5D (Total: 105)  ← Cumulative

Stage 3/3: Self-play - 45 games
  Stats: 82W 58L 10D (Total: 150)  ← Cumulative

✅ PASS: Curriculum stats are cumulative
```

---

### 3. Self-Play Clarification

**Note Added:** In self-play mode, "wins" and "losses" represent which side won (Player 0 vs Player 1), not agent performance, since the agent plays both sides.

**Implementation:**
```javascript
async trainSelfPlay(numEpisodes) {
    this.resetStats();
    console.log(`Training via self-play for ${numEpisodes} episodes...`);
    console.log('(Note: Agent plays both sides, so wins/losses show game balance, not performance)');
    // ...
}
```

**Explanation:**
- In self-play, both players are the same agent
- A "win" means Player 0 side won, a "loss" means Player 1 side won
- Both outcomes contribute equally to learning
- Ideally, self-play win rate should be ~50% (balanced game)

---

## New Tests Added

### 1. `tests/verify-stats.js`

Verifies core functionality:
- ✅ Empty pit filtering
- ✅ Winner calculation (wins/losses/draws)
- ✅ Agent only selects valid moves
- ✅ Stats counting accuracy

**Run with:** `npm run test:verify`

---

### 2. `tests/test-stats-reset.js`

Tests stats reset behavior:
- ✅ Stats reset between independent training runs
- ✅ Stats cumulative across curriculum stages

**Run with:** `npm run test:stats`

---

### 3. New npm Scripts

```bash
npm run test          # Original unit tests (18 tests)
npm run test:verify   # Verification tests
npm run test:stats    # Stats reset tests
npm run test:all      # Run all tests
```

---

## Summary

| Issue | Status | Details |
|-------|--------|---------|
| **Agent persistence** | ✅ Working | Agent persists and improves cumulatively |
| **Empty pit filtering** | ✅ Working | Only non-empty pits are selected |
| **Win/loss/draw accuracy** | ✅ Working | All games accounted for correctly |
| **Stats reset** | ✅ Improved | Now resets between training runs |
| **Curriculum stats** | ✅ Improved | Cumulative across stages |
| **Self-play clarity** | ✅ Improved | Added explanatory note |

---

## Recommendations

### For Users:

1. **Multiple training runs:** Stats now reset automatically between runs, so you can train multiple times and see accurate stats for each run.

2. **Self-play interpretation:** When using self-play, focus on:
   - Learning progress (epsilon decay, buffer size)
   - Final evaluation against random opponent
   - Not the self-play win/loss ratio (should be ~50%)

3. **Evaluation:** Always evaluate with `evaluate()` method or option 4 in rl-demo to test against random opponent with ε=0 (no exploration).

### For Developers:

1. **Adding new training modes:** Call `this.resetStats()` at the start of your method.

2. **Custom stats:** Override `updateStats()` or `printFinalStats()` as needed.

3. **Testing:** Run `npm run test:all` to verify all functionality.

---

## Files Modified

- `src/ai/trainer.js` - Added resetStats() method and improved curriculum handling
- `package.json` - Added new test scripts
- `tests/verify-stats.js` - New verification test suite
- `tests/test-stats-reset.js` - New stats reset test suite

---

**All tests pass ✅**

**Verification complete: 2025-11-07**
