# RL Agent Upgrade: V1 → V2 (DQN Implementation)

Complete overhaul of the RL agent to implement modern DQN with improved architecture and hyperparameters.

---

## Summary of Changes

### ✅ **1. State Representation: 21 → 15 Features**

**Old (21 features):**
- 12 pits (absolute indexing 0-11)
- 2 stores
- 2 current player (one-hot)
- 2 seeds on each side
- 2 empty pits ratio
- 1 store difference

**New (15 features) - Player-relative:**
- 6 my pits (normalized by /20)
- 6 opponent's pits (normalized by /20)
- 1 my captured seeds (normalized by /48)
- 1 opponent's captured seeds (normalized by /48)
- 1 seeds remaining on board (normalized by /48)

**Benefits:**
- Consistent representation regardless of which player is current
- Simpler feature space (15 vs 21)
- Better generalization
- Matches specification requirements

---

### ✅ **2. Network Architecture: 64→32 → 128→128→64**

**Old:**
```
Input: 21 → Hidden: 64 → Hidden: 32 → Output: 6
```

**New:**
```
Input: 15 → Hidden: 128 → Hidden: 128 → Hidden: 64 → Output: 6
```

**Changes:**
- Reduced input size (21 → 15) due to better features
- Increased network capacity (64→32 → 128→128→64)
- Added third hidden layer for more complex patterns
- All hidden layers use ReLU activation
- Output layer uses linear activation (Q-values)

**Benefits:**
- More capacity to learn complex strategies
- Better approximation of Q-function
- Improved performance on harder opponents

---

### ✅ **3. Target Network (DQN)**

**Added:**
- Separate target network (copy of online network)
- Target network updated every 1000 training steps
- Online network trained every batch

**Why it matters:**
```
Old: Q(s,a) = r + γ * max Q(s',a')  ← same network for both!
New: Q(s,a) = r + γ * max Q_target(s',a')  ← separate target network
```

The old approach had a **moving target problem**: The network you're training is also the one giving you the targets, causing instability.

With target network:
- Targets remain stable for 1000 steps
- Online network can train without chasing a moving target
- Much more stable learning
- Critical for DQN convergence

---

### ✅ **4. Gradient Clipping**

**Added:** Gradient clipping with max norm = 1.0

**Why it matters:**
- Prevents exploding gradients during training
- Especially important with larger networks
- Stabilizes training on difficult transitions
- Integrated into Adam optimizer

---

### ✅ **5. Improved Hyperparameters**

| Parameter | Old | New | Why |
|-----------|-----|-----|-----|
| γ (discount) | 0.95 | **0.99** | Better long-term planning |
| Replay buffer | 10,000 | **100,000** | More diverse experience |
| Batch size | 32 | **64** | More stable updates |
| ε_min | 0.1 | **0.05** | More exploitation |
| ε decay | Exponential | **Linear over 50k** | Predictable convergence |
| LR decay | None | **1e-3 → 5e-4** | Fine-tuning in late training |

---

### ✅ **6. Linear Decay Schedules**

**Epsilon Decay:**
```python
# Old: Exponential decay
epsilon *= 0.995 after each episode

# New: Linear decay over 50k steps
progress = min(1.0, trainingStep / 50000)
epsilon = 1.0 - progress * (1.0 - 0.05)
```

**Learning Rate Decay:**
```python
# Old: No decay (constant 0.001)

# New: Linear decay
progress = min(1.0, trainingStep / 50000)
lr = 0.001 - progress * (0.001 - 0.0005)
```

**Benefits:**
- Predictable exploration schedule
- High exploration early (ε=1.0)
- Low exploration late (ε=0.05)
- Fine-tuning with lower LR in late training
- No guessing when convergence will happen

---

## API Changes

### Constructor

**Old:**
```javascript
const agent = new QLearningAgent({
    learningRate: 0.001,
    discountFactor: 0.95,
    epsilon: 1.0,
    epsilonDecay: 0.995,
    epsilonMin: 0.1,
    replayBufferSize: 10000,
    batchSize: 32
});
```

**New:**
```javascript
const agent = new QLearningAgent({
    learningRate: 0.001,           // Initial LR
    learningRateEnd: 0.0005,       // Final LR (NEW)
    discountFactor: 0.99,          // Increased
    epsilon: 1.0,                  // Same
    epsilonMin: 0.05,              // Decreased
    epsilonDecaySteps: 50000,      // Linear decay (NEW)
    replayBufferSize: 100000,      // Increased 10x
    batchSize: 64,                 // Doubled
    targetUpdateFreq: 1000,        // Target network (NEW)
    gradientClipValue: 1.0         // Clipping (NEW)
});
```

### New Methods

```javascript
// Sync target network with online network
agent.syncTargetNetwork();

// Get Q-values from target network
agent.getTargetQValues(state);

// Update learning rate (called automatically in replay())
agent.updateLearningRate();

// Update epsilon (called automatically in replay())
agent.updateEpsilon();
```

### Deprecated Methods

```javascript
// This method is now a no-op (kept for compatibility)
agent.decayEpsilon();  // ← Don't call this anymore
```

---

## Training Improvements

### Training Step Counter

The agent now tracks `trainingStep` which increments with each `replay()` call:
- Used for epsilon decay
- Used for learning rate decay
- Used for target network updates
- Saved/loaded with model

### Automatic Updates

During training, `replay()` automatically:
1. Increments training step
2. Updates learning rate
3. Updates epsilon
4. Samples batch from replay buffer
5. Computes targets using **target network**
6. Trains online network
7. Updates target network every 1000 steps

No manual decay calls needed!

---

## Backwards Compatibility

### ⚠️ Breaking Changes

1. **State representation changed** (21 → 15 features)
   - Old models will not work
   - Need to retrain from scratch

2. **Network architecture changed**
   - Old saved models incompatible
   - Load will fail with dimension mismatch

3. **Hyperparameters changed**
   - Old training scripts may need updates

### ✅ Compatibility Maintained

1. **API mostly the same**
   - Constructor signature extended (not changed)
   - Main methods (selectAction, replay, save/load) work the same

2. **Training scripts work**
   - Trainer class unchanged
   - npm scripts work the same

3. **`decayEpsilon()` kept**
   - Now a no-op
   - Won't break existing code

---

## Testing

Run the comprehensive test suite:

```bash
node tests/test-new-rl-agent.js
```

**Tests:**
1. ✅ Feature extraction (15 features)
2. ✅ Agent initialization with new hyperparameters
3. ✅ Network architecture (128→128→64)
4. ✅ Target network exists
5. ✅ Action selection
6. ✅ Training with new parameters
7. ✅ Target network updates every 1000 steps
8. ✅ Epsilon linear decay

---

## Training Recommendations

### Quick Training (Testing)
```bash
npm run train  # 10,000 episodes (~10 minutes)
```

### Proper Training (Better Results)
```javascript
// Edit src/training/train-agent.js
await trainer.trainCurriculum(50000);  // 50k episodes

// Expected performance:
// - After 10k: 70-80% win rate vs random
// - After 25k: 85-90% win rate vs random
// - After 50k: 90-95% win rate vs random
```

### Hyperparameter Tuning

**For faster training:**
```javascript
const agent = new QLearningAgent({
    learningRate: 0.003,          // Higher LR
    learningRateEnd: 0.001,
    epsilonDecaySteps: 25000,     // Faster decay
    batchSize: 128,               // Larger batches
});
```

**For better final performance:**
```javascript
const agent = new QLearningAgent({
    learningRate: 0.0005,         // Lower LR
    learningRateEnd: 0.0001,
    epsilonDecaySteps: 100000,    // Slower decay
    replayBufferSize: 200000,     // More experience
});
```

---

## Performance Comparison

### Expected Results

| Metric | Old Agent | New Agent | Improvement |
|--------|-----------|-----------|-------------|
| Win rate @ 10k | 60-70% | 75-85% | +15% |
| Win rate @ 50k | 80-85% | 90-95% | +10% |
| Training stability | Unstable | Stable | ✅ |
| Convergence | ~30k episodes | ~20k episodes | Faster |
| Memory usage | ~50 MB | ~100 MB | +50 MB |
| Training speed | ~1000 ep/min | ~800 ep/min | -20% |

**Trade-offs:**
- Slower per-episode (larger network)
- More memory (larger buffer)
- **But:** Better convergence, higher performance, more stable

---

## Migration Guide

### Step 1: Delete Old Models
```bash
rm -rf models/kalah-agent/*
rm -rf models/test-agent/*
```

### Step 2: Update Training Scripts (Optional)

If you have custom training scripts, update hyperparameters:

```javascript
// OLD
const agent = new QLearningAgent({
    learningRate: 0.001,
    discountFactor: 0.95,
    epsilon: 1.0,
    epsilonDecay: 0.995,
    epsilonMin: 0.1,
    replayBufferSize: 10000,
    batchSize: 32
});

// NEW (recommended)
const agent = new QLearningAgent({
    learningRate: 0.001,
    learningRateEnd: 0.0005,
    discountFactor: 0.99,
    epsilon: 1.0,
    epsilonMin: 0.05,
    epsilonDecaySteps: 50000,
    replayBufferSize: 100000,
    batchSize: 64,
    targetUpdateFreq: 1000,
    gradientClipValue: 1.0
});
```

### Step 3: Retrain
```bash
npm run train
```

### Step 4: Test
```bash
npm test
npm run play
npm run serve  # Test in browser
```

---

## Technical Details

### Target Network Update Logic

```javascript
async replay() {
    // ... sample batch and compute targets using TARGET network ...

    // Train ONLINE network
    await this.model.fit(states, targets);

    // Update TARGET network every 1000 steps
    if (this.trainingStep % 1000 === 0) {
        this.syncTargetNetwork();  // Copy online → target
    }
}
```

### Epsilon Schedule

```javascript
updateEpsilon() {
    const progress = Math.min(1.0, this.trainingStep / 50000);
    this.epsilon = 1.0 - progress * (1.0 - 0.05);
}

// Examples:
// Step 0: epsilon = 1.0
// Step 25000: epsilon = 0.525
// Step 50000: epsilon = 0.05
// Step 100000: epsilon = 0.05 (capped)
```

### Learning Rate Schedule

```javascript
updateLearningRate() {
    const progress = Math.min(1.0, this.trainingStep / 50000);
    this.learningRate = 0.001 - progress * (0.001 - 0.0005);
    this.model.optimizer.learningRate = this.learningRate;
}

// Examples:
// Step 0: lr = 0.001
// Step 25000: lr = 0.00075
// Step 50000: lr = 0.0005
// Step 100000: lr = 0.0005 (capped)
```

---

## Troubleshooting

### Issue: Training is slow

**Solution:** This is expected with the larger network. You can:
1. Use smaller network (64→64→32) for faster iteration
2. Reduce batch size to 32
3. Install `@tensorflow/tfjs-node` for native speed

### Issue: Win rate not improving

**Check:**
1. Training steps > 10k?
2. Epsilon decaying correctly? (check `agent.epsilon`)
3. Replay buffer filling up? (check `agent.replayBuffer.length`)
4. Target network updating? (check logs for "Target network synced")

### Issue: Model save/load fails

**Cause:** Format changed with new hyperparameters

**Solution:** Don't load old models. Retrain from scratch.

### Issue: Out of memory

**Solution:** Reduce replay buffer size:
```javascript
const agent = new QLearningAgent({
    replayBufferSize: 50000,  // Instead of 100000
});
```

---

## References

- **DQN Paper:** [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- **Target Networks:** Used in original DQN to stabilize training
- **Gradient Clipping:** Standard practice in deep RL
- **Linear Schedules:** Simpler and more predictable than exponential

---

## Changelog

### Version 2.0 (Current)
- ✅ 15-feature player-relative state
- ✅ 128→128→64 architecture
- ✅ Target network with DQN
- ✅ Gradient clipping
- ✅ Improved hyperparameters
- ✅ Linear decay schedules
- ✅ Comprehensive tests

### Version 1.0 (Previous)
- 21-feature absolute state
- 64→32 architecture
- Single network (Q-learning)
- No gradient clipping
- Exponential epsilon decay
- No learning rate decay

---

**Migration Date:** 2025-11-07

**Status:** ✅ All tests pass, ready for production training
