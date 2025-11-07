# Kalah RL Training & Testing Guide

Complete guide for training and testing the RL models in this project.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Training Commands](#training-commands)
4. [Testing & Evaluation](#testing--evaluation)
5. [Playing the Game](#playing-the-game)
6. [Model Management](#model-management)
7. [Advanced Training](#advanced-training)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. Install dependencies
npm install

# 2. Train a model (recommended - 10,000 episodes with curriculum learning)
npm run train

# 3. Test the trained model
npm test

# 4. Play against the AI
npm run play

# 5. View in browser
npm run serve
# Then open: http://localhost:8080
```

---

## Installation

### Prerequisites
- Node.js (v14 or higher)
- npm

### Install Dependencies

```bash
npm install
```

This installs:
- `@tensorflow/tfjs` - TensorFlow.js for neural networks
- `@tensorflow/tfjs-node` - Native bindings for faster training

---

## Training Commands

### 1. Quick Training (Recommended)

**Command:**
```bash
npm run train
```

**What it does:**
- Trains agent with curriculum learning (3 stages)
- 10,000 total episodes
- Saves model to `./models/kalah-agent/`
- Shows training progress every 100 episodes
- Evaluates final model performance

**Expected output:**
```
🎮 Training Kalah/Mancala RL Agent

📚 Starting curriculum learning...
Stage 1/3: Training against random opponent (3000 episodes)...
Stage 2/3: Self-play training (4000 episodes)...
Stage 3/3: Advanced self-play (3000 episodes)...

📊 Evaluating trained agent...
Win rate vs random: 85-90%

💾 Saving model to ./models/kalah-agent/
✅ Model saved successfully!
```

**Training time:** ~5-10 minutes on modern CPU

---

### 2. Interactive Training Menu

**Command:**
```bash
npm run rl-demo
```

**What it does:**
Opens an interactive menu with options:
1. Train via self-play (10,000 episodes)
2. Train against random opponent (1,000 episodes)
3. **Train with curriculum learning (10,000 episodes)** ⭐ RECOMMENDED
4. Evaluate agent vs random opponent
5. Play against the agent
6. Save model
7. Load model
8. Exit

**Use this when:**
- You want to experiment with different training modes
- You need to load/save models manually
- You want to play against the agent in CLI
- You want to evaluate performance mid-training

---

### 3. Custom Training Script

**Edit:** `src/training/train-agent.js`

**Customize:**
```javascript
// Option A: Curriculum learning (default)
await trainer.trainCurriculum(10000);  // 10,000 episodes

// Option B: Pure self-play
await trainer.trainSelfPlay(10000);

// Option C: Against random opponent
await trainer.trainAgainstOpponent(10000);
```

**Then run:**
```bash
npm run train
```

---

## Testing & Evaluation

### 1. Unit Tests (Game Engine)

**Command:**
```bash
npm test
```

**What it tests:**
- Game initialization
- Move validation
- Seed distribution
- Capture mechanics
- Extra turn rules
- Multi-lap moves
- Game end conditions
- Winner determination

**Expected output:**
```
✅ 18 tests passed, 0 failed
```

---

### 2. Evaluate Trained Model

**Method A - Interactive Menu:**
```bash
npm run rl-demo
# Select option 4: Evaluate agent vs random opponent
```

**Method B - Code:**
Add to `src/training/train-agent.js`:
```javascript
// After training
await trainer.evaluate(100);  // Play 100 evaluation games
```

**Metrics:**
- Win rate vs random opponent
- Average game length
- Average reward per episode

**Good performance:**
- 80-90% win rate vs random opponent
- 85-95% win rate after 10,000 episodes

---

### 3. Play Against the Agent (CLI)

**Command:**
```bash
npm run play
```

**What it does:**
- Loads trained model from `./models/kalah-agent/`
- If no model found, trains a quick agent (200 episodes)
- Lets you play as Player 0 against AI as Player 1

**How to play:**
```
You are Player 0 (bottom row, pits 0-5)
AI is Player 1 (top row, pits 6-11)

Player 1: [11:4] [10:4] [9:4] [8:4] [7:4] [6:4]
P1 Store: 0                        P0 Store: 0
Player 0: [0:4] [1:4] [2:4] [3:4] [4:4] [5:4]

Your turn! Enter pit number (0-5): 2
```

---

### 4. Play in Browser

**Command:**
```bash
npm run serve
```

**Then open:** http://localhost:8080

**Features:**
- Visual game board
- Play vs Human or AI
- AI difficulty levels:
  - Easy (Minimax depth 2)
  - Medium (Minimax depth 4)
  - Hard (RL Agent)
- Undo button
- Game history

**Note:** Server must be running to load RL model from browser

---

## Playing the Game

### CLI Gameplay

```bash
npm run play
```

### Browser Gameplay

```bash
npm run serve
# Open http://localhost:8080
```

### Demo (Watch AI vs Random)

```bash
npm run demo
```

Shows a complete game with detailed logging.

---

## Model Management

### Model Location

Trained models are saved in:
```
models/
├── kalah-agent/          # Primary trained model
│   └── model.json
└── test-agent/           # Backup/test model
    └── model.json
```

### Save Model

**Method A - Interactive Menu:**
```bash
npm run rl-demo
# Select option 6: Save model
```

**Method B - Code:**
```javascript
await agent.save('./models/my-model');
```

### Load Model

**Method A - Interactive Menu:**
```bash
npm run rl-demo
# Select option 7: Load model
```

**Method B - Code:**
```javascript
await agent.load('./models/kalah-agent');
```

### Compare Models

```javascript
// Evaluate model A
const agentA = new QLearningAgent();
await agentA.load('./models/kalah-agent');
await trainer.evaluate(100, agentA);

// Evaluate model B
const agentB = new QLearningAgent();
await agentB.load('./models/test-agent');
await trainer.evaluate(100, agentB);
```

---

## Advanced Training

### 1. Hyperparameter Tuning

**Edit:** `src/training/train-agent.js`

```javascript
const agent = new QLearningAgent({
    learningRate: 0.001,      // Learning rate (default: 0.001)
    discountFactor: 0.95,     // Gamma (default: 0.95)
    epsilon: 1.0,             // Initial exploration (default: 1.0)
    epsilonDecay: 0.995,      // Decay rate (default: 0.995)
    epsilonMin: 0.1,          // Min exploration (default: 0.1)
    replayBufferSize: 10000,  // Experience buffer (default: 10000)
    batchSize: 32             // Training batch size (default: 32)
});
```

**Recommendations:**
- Higher `learningRate` (0.01) → faster learning, less stable
- Lower `learningRate` (0.0001) → slower learning, more stable
- Higher `epsilon` → more exploration
- Lower `epsilonMin` → more exploitation
- Larger `batchSize` → more stable updates, slower training

---

### 2. Curriculum Learning Stages

**Edit:** `src/ai/trainer.js`

```javascript
async trainCurriculum(totalEpisodes) {
    const stage1 = Math.floor(totalEpisodes * 0.3);  // 30% vs random
    const stage2 = Math.floor(totalEpisodes * 0.4);  // 40% self-play
    const stage3 = totalEpisodes - stage1 - stage2;  // 30% advanced

    await this.trainAgainstOpponent(stage1);
    await this.trainSelfPlay(stage2);
    await this.trainSelfPlay(stage3);
}
```

**Adjust ratios for:**
- More basic training: Increase stage1 %
- More advanced training: Increase stage3 %

---

### 3. Training Monitoring

**View training statistics:**
```javascript
console.log('Agent stats:', agent.stats);
// {
//   episodeCount: 10000,
//   totalReward: 5234.5,
//   avgLoss: 0.123
// }
```

**Track win rate over time:**
```javascript
// In trainer.js
if (episode % 1000 === 0) {
    const winRate = this.stats.wins / this.stats.gamesPlayed;
    console.log(`Episode ${episode}: Win rate = ${winRate.toFixed(2)}`);
}
```

---

### 4. Neural Network Architecture

**Current architecture:**
```
Input: 21 features
  ↓
Dense(64) + ReLU
  ↓
Dense(32) + ReLU
  ↓
Dense(6)  [Q-values for each action]
```

**Modify in:** `src/ai/rl-agent.js`

```javascript
buildModel() {
    const model = tf.sequential();

    model.add(tf.layers.dense({
        units: 64,              // Change layer size
        activation: 'relu',
        inputShape: [21]
    }));

    model.add(tf.layers.dense({
        units: 32,              // Change layer size
        activation: 'relu'
    }));

    // Add dropout for regularization
    model.add(tf.layers.dropout({ rate: 0.2 }));

    model.add(tf.layers.dense({
        units: 6
    }));

    model.compile({
        optimizer: tf.train.adam(this.learningRate),
        loss: 'meanSquaredError'
    });

    return model;
}
```

---

## Troubleshooting

### Model Not Found

**Error:**
```
Model not found at ./models/kalah-agent/model.json
```

**Solution:**
Train a model first:
```bash
npm run train
```

---

### TensorFlow Installation Issues

**Error:**
```
Cannot find module '@tensorflow/tfjs-node'
```

**Solution:**
```bash
rm -rf node_modules package-lock.json
npm install
```

---

### Training Too Slow

**Solutions:**

1. **Use smaller network:**
   - Edit `src/ai/rl-agent.js`
   - Change layer sizes: 64→32, 32→16

2. **Reduce episodes:**
   - Edit `src/training/train-agent.js`
   - Change: `await trainer.trainCurriculum(5000);`

3. **Use smaller batch size:**
   ```javascript
   const agent = new QLearningAgent({
       batchSize: 16  // Instead of 32
   });
   ```

---

### Poor Performance

**Agent not learning well?**

1. **Train longer:**
   ```bash
   npm run train  # Increase episodes in train-agent.js
   ```

2. **Check hyperparameters:**
   - Increase `learningRate` (try 0.01)
   - Increase training episodes (try 20,000)

3. **Use curriculum learning:**
   ```bash
   npm run rl-demo
   # Select option 3
   ```

4. **Verify training:**
   - Win rate should reach 80-90% vs random
   - Check training logs for improvement

---

### Server Port Already in Use

**Error:**
```
Error: listen EADDRINUSE: address already in use :::8080
```

**Solution:**
See `docs/START-STOP-SERVER.md`

```bash
# Find process using port 8080
lsof -i :8080

# Kill the process
kill -9 <PID>

# Or use a different port in scripts/server.js
```

---

## Performance Benchmarks

### Expected Results

| Training Mode | Episodes | Time | Win Rate vs Random |
|--------------|----------|------|--------------------|
| Random opponent | 1,000 | ~1 min | 70-75% |
| Self-play | 5,000 | ~5 min | 75-80% |
| Curriculum | 10,000 | ~10 min | 85-90% |
| Extended training | 50,000 | ~50 min | 90-95% |

### Hardware

- **CPU:** 5,000-10,000 episodes per minute (M1/M2 Mac)
- **Memory:** ~200-500 MB during training
- **Disk:** ~100 KB per saved model

---

## Next Steps

1. ✅ **Train your first model:**
   ```bash
   npm run train
   ```

2. ✅ **Test it:**
   ```bash
   npm test
   npm run play
   ```

3. ✅ **Try it in browser:**
   ```bash
   npm run serve
   # Open http://localhost:8080
   ```

4. 🎯 **Experiment:**
   - Adjust hyperparameters
   - Try different training modes
   - Train for longer
   - Compare different models

5. 🚀 **Extend:**
   - Add new features to game engine
   - Implement tournament mode
   - Track training metrics
   - Add more AI opponents

---

## Additional Resources

- **Game Engine API:** See `README.md`
- **RL Agent Deep Dive:** See `docs/HOW-RL-AGENT-WORKS.md`
- **Project Status:** See `docs/PROJECT-STATUS.md`
- **Server Setup:** See `docs/START-STOP-SERVER.md`

---

**Happy Training! 🎮🤖**
