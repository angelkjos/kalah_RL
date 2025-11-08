/**
 * Q-Learning Agent for Kalah/Mancala
 *
 * Uses a neural network to approximate Q-values for state-action pairs.
 * Implements epsilon-greedy exploration and experience replay.
 */

const tf = require('@tensorflow/tfjs-node');
const { extractFeatures } = require('../utils/ml-examples.js');

class QLearningAgent {
    /**
     * Create a new Q-Learning agent
     * @param {Object} options - Configuration options
     */
    constructor(options = {}) {
        const {
            learningRate = 0.001,
            learningRateEnd = 0.0005,
            discountFactor = 0.99,
            epsilon = 1.0,
            epsilonMin = 0.05,
            epsilonDecaySteps = 50000,
            replayBufferSize = 100000,
            batchSize = 64,
            targetUpdateFreq = 1000,
            gradientClipValue = 1.0
        } = options;

        this.learningRate = learningRate;
        this.learningRateStart = learningRate;
        this.learningRateEnd = learningRateEnd;
        this.gamma = discountFactor;
        this.epsilon = epsilon;
        this.epsilonStart = epsilon;
        this.epsilonMin = epsilonMin;
        this.epsilonDecaySteps = epsilonDecaySteps;
        this.batchSize = batchSize;
        this.targetUpdateFreq = targetUpdateFreq;
        this.gradientClipValue = gradientClipValue;

        // Experience replay buffer
        this.replayBuffer = [];
        this.replayBufferSize = replayBufferSize;

        // Build neural networks (online and target)
        this.model = this.buildModel();
        this.targetModel = this.buildModel();
        this.syncTargetNetwork();

        // Training step counter for target network updates and epsilon decay
        this.trainingStep = 0;

        // Statistics
        this.stats = {
            episodeCount: 0,
            totalReward: 0,
            avgLoss: 0
        };
    }

    /**
     * Build the Q-network
     * Input: game state features (15 dimensions)
     * Output: Q-values for each action (6 pits)
     *
     * Architecture: 15 → 128 → 128 → 64 → 6
     */
    buildModel() {
        const model = tf.sequential();

        // Input layer (15 features)
        model.add(tf.layers.dense({
            inputShape: [15],
            units: 128,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));

        // Hidden layer 1
        model.add(tf.layers.dense({
            units: 128,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));

        // Hidden layer 2
        model.add(tf.layers.dense({
            units: 64,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));

        // Output layer - Q-values for 6 possible actions (pits 0-5)
        model.add(tf.layers.dense({
            units: 6,
            activation: 'linear',
            kernelInitializer: 'glorotUniform'
        }));

        // Compile model with gradient clipping
        model.compile({
            optimizer: tf.train.adam(this.learningRate, undefined, undefined, undefined, this.gradientClipValue),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return model;
    }

    /**
     * Sync target network weights with online network
     */
    syncTargetNetwork() {
        const weights = this.model.getWeights();
        const weightsCopy = weights.map(w => w.clone());
        this.targetModel.setWeights(weightsCopy);
    }

    /**
     * Select an action using epsilon-greedy policy
     * @param {Object} state - Game state from KalahEngine
     * @param {number[]} validMoves - Array of valid pit indices
     * @returns {number} Selected pit index
     */
    selectAction(state, validMoves) {
        // Exploration: random action
        if (Math.random() < this.epsilon) {
            return validMoves[Math.floor(Math.random() * validMoves.length)];
        }

        // Exploitation: best Q-value action
        return tf.tidy(() => {
            const features = extractFeatures(state);
            const featureTensor = tf.tensor2d([features]);
            const qValues = this.model.predict(featureTensor);
            const qArray = Array.from(qValues.dataSync());

            // Q-values are indexed 0-5 for the current player's pits
            // validMoves contains absolute pit indices
            let bestAction = validMoves[0];
            let bestQ = -Infinity;

            for (const move of validMoves) {
                // Convert absolute pit index to relative (0-5 for current player)
                const relativeMove = move - (state.currentPlayer * 6);
                if (relativeMove >= 0 && relativeMove < 6 && qArray[relativeMove] > bestQ) {
                    bestQ = qArray[relativeMove];
                    bestAction = move;
                }
            }

            return bestAction;
        });
    }

    /**
     * Get Q-values for a state
     * @param {Object} state - Game state
     * @returns {number[]} Q-values for each action
     */
    getQValues(state) {
        return tf.tidy(() => {
            const features = extractFeatures(state);
            const featureTensor = tf.tensor2d([features]);
            const qValues = this.model.predict(featureTensor);
            return Array.from(qValues.dataSync());
        });
    }

    /**
     * Store an experience in the replay buffer
     * @param {Object} experience - {state, action, reward, nextState, done}
     */
    remember(experience) {
        this.replayBuffer.push(experience);

        // Keep buffer size limited
        if (this.replayBuffer.length > this.replayBufferSize) {
            this.replayBuffer.shift();
        }
    }

    /**
     * Train the model on a batch of experiences using DQN with target network
     * @returns {number} Average loss for the batch
     */
    async replay() {
        // Need enough experiences
        if (this.replayBuffer.length < this.batchSize) {
            return 0;
        }

        // Increment training step
        this.trainingStep++;

        // Update learning rate (linear decay)
        this.updateLearningRate();

        // Update epsilon (linear decay over epsilonDecaySteps)
        this.updateEpsilon();

        // Sample random batch
        const batch = [];
        const bufferCopy = [...this.replayBuffer];
        for (let i = 0; i < this.batchSize; i++) {
            const index = Math.floor(Math.random() * bufferCopy.length);
            batch.push(bufferCopy[index]);
            bufferCopy.splice(index, 1);
        }

        // Prepare training data
        const states = [];
        const targets = [];

        for (const exp of batch) {
            const features = extractFeatures(exp.state);
            states.push(features);

            // Get current Q-values from online network
            const currentQ = this.getQValues(exp.state);

            // Calculate target Q-value using target network
            let targetQ;
            if (exp.done) {
                // Terminal state - just use the reward
                targetQ = exp.reward;
            } else {
                // DQN update: Q(s,a) = r + γ * V(s')
                const nextQ = this.getTargetQValues(exp.nextState);
                let nextValue = Math.max(...nextQ);

                // CRITICAL: In zero-sum alternating games, negate value when player changes
                // If next state is opponent's turn, their positive value is our negative value
                if (exp.state.currentPlayer !== exp.nextState.currentPlayer) {
                    nextValue = -nextValue;
                }

                targetQ = exp.reward + this.gamma * nextValue;
            }

            // Update only the Q-value for the action taken
            const relativeAction = exp.action - (exp.state.currentPlayer * 6);
            currentQ[relativeAction] = targetQ;

            targets.push(currentQ);
        }

        // Train the online network
        const xs = tf.tensor2d(states);
        const ys = tf.tensor2d(targets);

        const history = await this.model.fit(xs, ys, {
            epochs: 1,
            verbose: 0
        });

        const loss = history.history.loss[0];

        // Clean up tensors
        xs.dispose();
        ys.dispose();

        // Update target network periodically
        if (this.trainingStep % this.targetUpdateFreq === 0) {
            this.syncTargetNetwork();
        }

        return loss;
    }

    /**
     * Get Q-values from target network
     * @param {Object} state - Game state
     * @returns {number[]} Q-values for each action
     */
    getTargetQValues(state) {
        return tf.tidy(() => {
            const features = extractFeatures(state);
            const featureTensor = tf.tensor2d([features]);
            const qValues = this.targetModel.predict(featureTensor);
            return Array.from(qValues.dataSync());
        });
    }

    /**
     * Update learning rate with linear decay
     */
    updateLearningRate() {
        const progress = Math.min(1.0, this.trainingStep / this.epsilonDecaySteps);
        this.learningRate = this.learningRateStart - progress * (this.learningRateStart - this.learningRateEnd);

        // Update optimizer learning rate
        this.model.optimizer.learningRate = this.learningRate;
    }

    /**
     * Update epsilon with linear decay over epsilonDecaySteps
     */
    updateEpsilon() {
        const progress = Math.min(1.0, this.trainingStep / this.epsilonDecaySteps);
        this.epsilon = this.epsilonStart - progress * (this.epsilonStart - this.epsilonMin);
        this.epsilon = Math.max(this.epsilonMin, this.epsilon);
    }

    /**
     * Decay epsilon (deprecated - now handled in updateEpsilon)
     * Kept for backwards compatibility
     */
    decayEpsilon() {
        // Epsilon is now updated automatically in replay()
        // This method is kept for backwards compatibility but does nothing
    }

    /**
     * Save the model to disk
     * @param {string} path - Save path
     */
    async save(path) {
        const fs = require('fs');

        // Create directory if it doesn't exist
        if (!fs.existsSync(path)) {
            fs.mkdirSync(path, { recursive: true });
        }

        // Get model weights as JSON
        const weights = this.model.getWeights();
        const weightsData = await Promise.all(
            weights.map(async (w) => ({
                name: w.name,
                shape: w.shape,
                dtype: w.dtype,
                data: Array.from(await w.data())
            }))
        );

        // Save model architecture and weights
        const modelConfig = {
            modelTopology: this.model.toJSON(null, false),
            weightsData: weightsData,
            hyperparameters: {
                learningRate: this.learningRate,
                learningRateStart: this.learningRateStart,
                learningRateEnd: this.learningRateEnd,
                gamma: this.gamma,
                epsilon: this.epsilon,
                epsilonStart: this.epsilonStart,
                epsilonMin: this.epsilonMin,
                epsilonDecaySteps: this.epsilonDecaySteps,
                replayBufferSize: this.replayBufferSize,
                batchSize: this.batchSize,
                targetUpdateFreq: this.targetUpdateFreq,
                gradientClipValue: this.gradientClipValue,
                trainingStep: this.trainingStep
            },
            stats: this.stats
        };

        fs.writeFileSync(
            `${path}/model.json`,
            JSON.stringify(modelConfig, null, 2)
        );

        console.log(`✅ Model saved to ${path}`);
    }

    /**
     * Load a model from disk
     * @param {string} path - Load path
     */
    async load(path) {
        const fs = require('fs');

        const modelPath = `${path}/model.json`;
        if (!fs.existsSync(modelPath)) {
            throw new Error(`Model not found at ${modelPath}`);
        }

        // Load model config
        const modelConfig = JSON.parse(fs.readFileSync(modelPath, 'utf8'));

        // Restore hyperparameters first (needed for compilation)
        if (modelConfig.hyperparameters) {
            this.learningRate = modelConfig.hyperparameters.learningRate || this.learningRate;
            this.learningRateStart = modelConfig.hyperparameters.learningRateStart || this.learningRateStart;
            this.learningRateEnd = modelConfig.hyperparameters.learningRateEnd || this.learningRateEnd;
            this.gamma = modelConfig.hyperparameters.gamma || this.gamma;
            this.epsilon = modelConfig.hyperparameters.epsilon || this.epsilon;
            this.epsilonStart = modelConfig.hyperparameters.epsilonStart || this.epsilonStart;
            this.epsilonMin = modelConfig.hyperparameters.epsilonMin || this.epsilonMin;
            this.epsilonDecaySteps = modelConfig.hyperparameters.epsilonDecaySteps || this.epsilonDecaySteps;
            this.replayBufferSize = modelConfig.hyperparameters.replayBufferSize || this.replayBufferSize;
            this.batchSize = modelConfig.hyperparameters.batchSize || this.batchSize;
            this.targetUpdateFreq = modelConfig.hyperparameters.targetUpdateFreq || this.targetUpdateFreq;
            this.gradientClipValue = modelConfig.hyperparameters.gradientClipValue || this.gradientClipValue;
            this.trainingStep = modelConfig.hyperparameters.trainingStep || 0;
        }

        // Restore model from topology
        this.model = await tf.models.modelFromJSON(modelConfig.modelTopology);

        // Restore weights
        const weightValues = modelConfig.weightsData.map(w =>
            tf.tensor(w.data, w.shape, w.dtype)
        );
        this.model.setWeights(weightValues);

        // Recompile the model with gradient clipping
        this.model.compile({
            optimizer: tf.train.adam(this.learningRate, undefined, undefined, undefined, this.gradientClipValue),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        // Rebuild target network and sync weights
        this.targetModel = this.buildModel();
        this.syncTargetNetwork();

        // Restore stats
        if (modelConfig.stats) {
            this.stats = modelConfig.stats;
        }

        console.log(`✅ Model loaded from ${path}`);
    }

    /**
     * Get training statistics
     */
    getStats() {
        return {
            ...this.stats,
            epsilon: this.epsilon,
            bufferSize: this.replayBuffer.length
        };
    }
}

module.exports = QLearningAgent;
