/**
 * Browser-compatible RL Agent Loader
 *
 * Loads and runs the trained TensorFlow.js model in the browser
 */

class RLAgent {
    constructor() {
        this.model = null;
        this.loaded = false;
        this.loading = false;
    }

    /**
     * Load the trained model from JSON
     */
    async loadModel(modelPath = 'models/kalah-agent/model.json') {
        if (this.loaded) return true;
        if (this.loading) return false;

        this.loading = true;
        console.log('🤖 Loading RL agent model...');

        try {
            // Fetch the model JSON
            const response = await fetch(modelPath);
            if (!response.ok) {
                throw new Error(`Model not found at ${modelPath}`);
            }

            const modelConfig = await response.json();

            // Restore model from topology
            this.model = await tf.models.modelFromJSON(modelConfig.modelTopology);

            // Restore weights
            const weightValues = modelConfig.weightsData.map(w =>
                tf.tensor(w.data, w.shape, w.dtype)
            );
            this.model.setWeights(weightValues);

            this.loaded = true;
            this.loading = false;
            console.log('✅ RL agent loaded successfully!');
            return true;

        } catch (error) {
            console.error('❌ Failed to load RL agent:', error);
            console.log('💡 Make sure you have trained a model with: npm run train');
            this.loading = false;
            return false;
        }
    }

    /**
     * Select best move using the RL agent
     */
    selectMove(state) {
        if (!this.loaded) {
            console.error('Model not loaded!');
            return null;
        }

        const validMoves = this.getValidMoves(state);
        if (validMoves.length === 0) return null;

        return tf.tidy(() => {
            // Extract features (same as in training)
            const features = this.extractFeatures(state);
            const featureTensor = tf.tensor2d([features]);

            // Get Q-values
            const qValues = this.model.predict(featureTensor);
            const qArray = Array.from(qValues.dataSync());

            // Find best valid move
            let bestMove = validMoves[0];
            let bestQ = -Infinity;

            for (const move of validMoves) {
                // Convert to relative index (0-5)
                const relativeMove = state.currentPlayer === 0 ? move : move - 6;
                if (relativeMove >= 0 && relativeMove < 6 && qArray[relativeMove] > bestQ) {
                    bestQ = qArray[relativeMove];
                    bestMove = move;
                }
            }

            return bestMove;
        });
    }

    /**
     * Extract features from game state (must match training!)
     */
    extractFeatures(state) {
        const features = [];

        // Normalized board state (0-1 range, max ~20 seeds per pit)
        for (const seeds of state.board) {
            features.push(seeds / 20.0);
        }

        // Normalized store counts (0-1 range, max 48 total seeds)
        features.push(state.stores[0] / 48.0);
        features.push(state.stores[1] / 48.0);

        // Current player (one-hot encoding)
        features.push(state.currentPlayer === 0 ? 1 : 0);
        features.push(state.currentPlayer === 1 ? 1 : 0);

        // Derived features
        const player0Pits = state.board.slice(0, 6);
        const player1Pits = state.board.slice(6, 12);

        // Seeds on each side
        const p0Seeds = player0Pits.reduce((a, b) => a + b, 0);
        const p1Seeds = player1Pits.reduce((a, b) => a + b, 0);
        features.push(p0Seeds / 24.0);
        features.push(p1Seeds / 24.0);

        // Number of empty pits on each side
        const p0Empty = player0Pits.filter(x => x === 0).length / 6.0;
        const p1Empty = player1Pits.filter(x => x === 0).length / 6.0;
        features.push(p0Empty);
        features.push(p1Empty);

        // Store difference
        const storeDiff = (state.stores[0] - state.stores[1]) / 48.0;
        features.push(storeDiff);

        return features;
    }

    /**
     * Get valid moves for current player
     */
    getValidMoves(state) {
        if (state.gameOver) return [];

        const startPit = state.currentPlayer * 6;
        const validMoves = [];

        for (let i = 0; i < 6; i++) {
            const pit = startPit + i;
            if (state.board[pit] > 0) {
                validMoves.push(pit);
            }
        }

        return validMoves;
    }
}

// Make available globally
if (typeof window !== 'undefined') {
    window.RLAgent = RLAgent;
}
