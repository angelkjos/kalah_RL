/**
 * Q-Learning Agent for Kalah/Mancala
 *
 * Uses a neural network to approximate Q-values for state-action pairs.
 * Implements epsilon-greedy exploration and experience replay.
 */

const tf = require('@tensorflow/tfjs');
const { extractFeatures } = require('./ml-examples.js');

class QLearningAgent {
    /**
     * Create a new Q-Learning agent
     * @param {Object} options - Configuration options
     */
    constructor(options = {}) {
        const {
            learningRate = 0.001,
            discountFactor = 0.95,
            epsilon = 1.0,
            epsilonDecay = 0.995,
            epsilonMin = 0.1,
            replayBufferSize = 10000,
            batchSize = 32
        } = options;

        this.learningRate = learningRate;
        this.gamma = discountFactor;
        this.epsilon = epsilon;
        this.epsilonDecay = epsilonDecay;
        this.epsilonMin = epsilonMin;
        this.batchSize = batchSize;

        // Experience replay buffer
        this.replayBuffer = [];
        this.replayBufferSize = replayBufferSize;

        // Build neural network
        this.model = this.buildModel();

        // Statistics
        this.stats = {
            episodeCount: 0,
            totalReward: 0,
            avgLoss: 0
        };
    }

    /**
     * Build the Q-network
     * Input: game state features (21 dimensions)
     * Output: Q-values for each action (6 pits)
     */
    buildModel() {
        const model = tf.sequential();

        // Input layer
        model.add(tf.layers.dense({
            inputShape: [21],
            units: 64,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));

        // Hidden layer
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu',
            kernelInitializer: 'heNormal'
        }));

        // Output layer - Q-values for 6 possible actions (pits 0-5 for player 0)
        model.add(tf.layers.dense({
            units: 6,
            activation: 'linear'
        }));

        // Compile model
        model.compile({
            optimizer: tf.train.adam(this.learningRate),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return model;
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

            // Only consider valid moves
            let bestAction = validMoves[0];
            let bestQ = -Infinity;

            for (const move of validMoves) {
                // Convert absolute pit index to relative (0-5 for current player)
                const relativeMove = state.currentPlayer === 0 ? move : move - 6;
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
     * Train the model on a batch of experiences
     * @returns {number} Average loss for the batch
     */
    async replay() {
        // Need enough experiences
        if (this.replayBuffer.length < this.batchSize) {
            return 0;
        }

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

            // Get current Q-values
            const currentQ = this.getQValues(exp.state);

            // Calculate target Q-value for the action taken
            let targetQ;
            if (exp.done) {
                // Terminal state - just use the reward
                targetQ = exp.reward;
            } else {
                // Q-learning update: Q(s,a) = r + γ * max Q(s',a')
                const nextQ = this.getQValues(exp.nextState);
                targetQ = exp.reward + this.gamma * Math.max(...nextQ);
            }

            // Update only the Q-value for the action taken
            const relativeAction = exp.state.currentPlayer === 0 ? exp.action : exp.action - 6;
            currentQ[relativeAction] = targetQ;

            targets.push(currentQ);
        }

        // Train the model
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

        return loss;
    }

    /**
     * Decay epsilon (reduce exploration over time)
     */
    decayEpsilon() {
        this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
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
                gamma: this.gamma,
                epsilon: this.epsilon,
                epsilonDecay: this.epsilonDecay,
                epsilonMin: this.epsilonMin,
                replayBufferSize: this.replayBufferSize,
                batchSize: this.batchSize
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

        // Restore model from topology
        this.model = await tf.models.modelFromJSON(modelConfig.modelTopology);

        // Restore weights
        const weightValues = modelConfig.weightsData.map(w =>
            tf.tensor(w.data, w.shape, w.dtype)
        );
        this.model.setWeights(weightValues);

        // Recompile the model
        this.model.compile({
            optimizer: tf.train.adam(this.learningRate),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        // Restore hyperparameters
        if (modelConfig.hyperparameters) {
            this.learningRate = modelConfig.hyperparameters.learningRate;
            this.gamma = modelConfig.hyperparameters.gamma;
            this.epsilon = modelConfig.hyperparameters.epsilon;
            this.epsilonDecay = modelConfig.hyperparameters.epsilonDecay;
            this.epsilonMin = modelConfig.hyperparameters.epsilonMin;
            this.replayBufferSize = modelConfig.hyperparameters.replayBufferSize;
            this.batchSize = modelConfig.hyperparameters.batchSize;
        }

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
