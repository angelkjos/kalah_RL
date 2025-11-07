# 🧠 How the RL Agent Works - Complete Explanation

## Overview

The RL (Reinforcement Learning) agent uses **Q-Learning** with a **neural network** to learn how to play Kalah/Mancala. It learns by playing thousands of games and getting better over time.

---

## Part 1: The Neural Network Architecture

### Input Layer (21 features)
The agent looks at the game board and extracts 21 numbers that describe the current state:

```javascript
// Board state (12 features)
[4, 4, 4, 4, 4, 4,    // Player 0's pits (normalized 0-1)
 4, 4, 4, 4, 4, 4]    // Player 1's pits (normalized 0-1)

// Store counts (2 features)
[0, 0]                // Player 0's store, Player 1's store (normalized)

// Current player (2 features - one-hot encoding)
[1, 0]                // [isPlayer0, isPlayer1]

// Derived features (5 features)
[0.5,                 // Seeds on player 0's side (normalized)
 0.5,                 // Seeds on player 1's side (normalized)
 0.0,                 // Empty pits on player 0's side
 0.0,                 // Empty pits on player 1's side
 0.0]                 // Store difference (normalized)

// Total = 12 + 2 + 2 + 5 = 21 features
```

### Hidden Layers
```
Input (21) → Dense(64, relu) → Dense(32, relu) → Output(6)
```

- **Layer 1**: 64 neurons with ReLU activation
- **Layer 2**: 32 neurons with ReLU activation
- **Output**: 6 neurons (one for each pit the agent can play)

### Output Layer (6 Q-values)
The network outputs 6 numbers called **Q-values**, one for each pit:
```javascript
[0.3, 0.8, -0.2, 0.5, 0.1, 0.4]
 ↑    ↑     ↑    ↑    ↑    ↑
pit0 pit1 pit2 pit3 pit4 pit5
```

**Higher Q-value = Better move!**

---

## Part 2: Training Process (Q-Learning)

### Step 1: Play Games and Collect Experiences

The agent plays games and remembers what happened:

```javascript
{
  state: [game board before move],
  action: 3,                      // Played pit 3
  reward: 0.05,                   // Got 5 seeds score advantage
  nextState: [game board after],
  done: false                     // Game not over yet
}
```

### Step 2: Experience Replay

Instead of learning from one move at a time, the agent stores 10,000 experiences in a "replay buffer" and randomly samples 32 at a time to learn from. This prevents it from forgetting old lessons.

### Step 3: Q-Learning Update

For each experience, the agent updates its Q-values using the **Bellman equation**:

```
Q(state, action) = reward + γ * max(Q(nextState, all actions))
                   ↑        ↑
              immediate   future
               reward     reward
```

Where:
- `reward` = What I got for this move (+1 for winning, -1 for losing, small bonuses for score)
- `γ (gamma)` = 0.95 = How much we care about future rewards
- `max Q(nextState)` = Best possible value from the next position

**Example:**
```
Current state: Q-values = [0.2, 0.5, 0.3, 0.1, 0.4, 0.6]
I played pit 1 (action=1)
Got reward = 0.05 (scored 5 seeds)
Next state best Q = 0.8

New Q-value for pit 1 = 0.05 + 0.95 * 0.8 = 0.81

Updated Q-values = [0.2, 0.81, 0.3, 0.1, 0.4, 0.6]
                          ↑ improved!
```

### Step 4: Neural Network Training

The neural network is trained to predict these Q-values:
- **Input**: Game state (21 features)
- **Target**: Updated Q-values (6 numbers)
- **Loss function**: Mean Squared Error
- **Optimizer**: Adam with learning rate 0.001

### Step 5: Exploration vs Exploitation (Epsilon-Greedy)

The agent balances:
- **Exploration** (trying new things): Pick random moves
- **Exploitation** (using knowledge): Pick best Q-value move

```javascript
if (random < epsilon) {
    // Explore: try random move
    return randomMove();
} else {
    // Exploit: pick best Q-value
    return bestMove();
}
```

**Epsilon decay:**
- Start: ε = 1.0 (100% random - explore everything)
- Middle: ε = 0.5 (50% random, 50% best)
- End: ε = 0.1 (10% random, 90% best)

This means early in training it explores a lot, but later it mostly uses what it learned.

### Step 6: Curriculum Learning (3-Stage Training)

Instead of just playing against itself, the agent trains in stages:

```
Stage 1 (30%): vs Random opponent
├─ Learn basic rules
├─ Learn that capturing is good
└─ Learn that stores are valuable

Stage 2 (40%): Self-play (intermediate)
├─ Develop strategies
├─ Learn counter-strategies
└─ Improve decision making

Stage 3 (30%): Self-play (advanced)
├─ Master the game
├─ Fine-tune strategies
└─ Reach peak performance
```

This prevents the "weak playing weak" problem where the agent just learns to beat weak opponents.

---

## Part 3: Making Decisions During Play

### When Playing in Browser:

1. **Load Model** (rl-agent-browser.js)
   ```javascript
   // Fetch the saved model.json
   const modelConfig = await fetch('./models/kalah-agent/model.json');

   // Reconstruct neural network
   model = tf.models.modelFromJSON(modelConfig.modelTopology);

   // Restore trained weights
   model.setWeights(weightValues);
   ```

2. **Extract Features** (same 21 features as training)
   ```javascript
   features = extractFeatures(gameState);
   // [board pits, stores, player, derived features]
   ```

3. **Run Neural Network**
   ```javascript
   qValues = model.predict(features);
   // Returns: [0.3, 0.8, -0.2, 0.5, 0.1, 0.4]
   ```

4. **Pick Best Valid Move**
   ```javascript
   validMoves = [0, 1, 2, 4, 5];  // Pit 3 is empty

   qValues = [0.3, 0.8, -0.2, 0.5, 0.1, 0.4];
              ✓    ✓    ✗     ✓    ✓    ✓

   bestMove = 1;  // Highest Q-value among valid moves!
   ```

5. **Make the Move**
   ```javascript
   game.makeMove(bestMove);
   ```

---

## Part 4: Complete Flow (Training → Browser)

### Training (Node.js)

```
1. Create Agent
   ├─ Build neural network (21 → 64 → 32 → 6)
   ├─ Set hyperparameters (ε=1.0, γ=0.95, lr=0.001)
   └─ Create replay buffer (size 10,000)

2. Train via Curriculum Learning
   ├─ Stage 1: 3,000 games vs random
   │   ├─ Play game
   │   ├─ Store experiences
   │   ├─ Sample batch (32 experiences)
   │   ├─ Calculate Q-targets
   │   ├─ Train network
   │   └─ Decay epsilon
   │
   ├─ Stage 2: 4,000 games self-play
   │   └─ [same process]
   │
   └─ Stage 3: 3,000 games self-play advanced
       └─ [same process]

3. Save Model
   ├─ Extract network architecture
   ├─ Extract all weights (thousands of numbers)
   ├─ Save hyperparameters
   └─ Write to models/kalah-agent/model.json
```

### Browser Play

```
1. User Opens http://localhost:8080
   └─ Loads HTML, CSS, JS files

2. User Clicks "🤖 Play vs AI" + "Hard"
   └─ Triggers model loading

3. Load Model
   ├─ Fetch models/kalah-agent/model.json (via HTTP)
   ├─ Parse JSON (architecture + weights)
   ├─ Reconstruct neural network in TensorFlow.js
   └─ Load trained weights

4. User Makes Move
   └─ Updates game state

5. AI's Turn
   ├─ Extract 21 features from current state
   ├─ Run neural network forward pass
   │   └─ Input → Layer1 → Layer2 → Output
   ├─ Get 6 Q-values
   ├─ Filter to valid moves
   ├─ Pick highest Q-value
   └─ Execute move

6. Repeat until game ends
```

---

## Part 5: What Makes It "Smart"?

### Pattern Recognition
After 10,000 games, the neural network learns patterns like:
- **Capturing is valuable** → High Q-value when opposite pit has many seeds
- **Extra turns are good** → High Q-value for moves ending in own store
- **Empty pits are bad** → Low Q-value for leaving pits vulnerable
- **Score matters** → Adjusts strategy based on current score difference

### Strategic Thinking
The network implicitly learns:
- **Early game**: Spread seeds evenly, create capturing opportunities
- **Mid game**: Balance offense (capturing) and defense (protecting seeds)
- **End game**: Maximize final score, prevent opponent captures

### Adaptation
It doesn't use hard-coded rules! Instead:
- The network has 64×21 + 32×64 + 6×32 = ~3,600 weights
- These weights encode all the knowledge
- During play, it's just matrix multiplication (very fast!)

---

## Part 6: Key Numbers

### Training
- **Episodes**: 10,000 games
- **Training time**: 5-10 minutes
- **Win rate**: ~85-90% vs random opponent
- **Epsilon decay**: 1.0 → 0.1 over training
- **Replay buffer**: 10,000 experiences
- **Batch size**: 32 experiences per update

### Neural Network
- **Total parameters**: ~3,600 trainable weights
- **Input size**: 21 features
- **Hidden layers**: 64 + 32 neurons
- **Output size**: 6 Q-values
- **Activation**: ReLU (hidden), Linear (output)

### Performance
- **Inference time**: <10ms per move
- **Model size**: ~112 KB (model.json)
- **Browser compatible**: Yes! (TensorFlow.js)
- **No backend needed**: Runs entirely in browser

---

## Summary

The RL agent works by:

1. **Learning**: Playing 10,000 games and adjusting neural network weights
2. **Remembering**: Storing experiences and learning from random batches
3. **Improving**: Using Q-learning to estimate move quality
4. **Deciding**: Running the network to get Q-values and picking best move
5. **Adapting**: Balancing exploration (trying new things) and exploitation (using knowledge)

**The magic:** After enough training, the neural network learns to "see" good moves without being explicitly programmed with rules. It discovers strategies on its own through trial and error!

🎮 **That's how your trained agent can beat you in the browser!** 🤖
