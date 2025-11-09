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

            // Load model from topology (same approach as Node.js version)
            this.model = await tf.models.modelFromJSON(modelConfig.modelTopology);

            // Restore weights
            const weightTensors = modelConfig.weightsData.map(w =>
                tf.tensor(w.data, w.shape, w.dtype)
            );
            this.model.setWeights(weightTensors);

            // Compile the model (optional for inference, but good practice)
            this.model.compile({
                optimizer: 'adam',
                loss: 'meanSquaredError'
            });

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
     * Player-relative representation (15 features)
     *
     * Features:
     * - 6 floats: my pits (normalized to [0,1])
     * - 6 floats: opponent's pits (normalized to [0,1])
     * - 1 float: my captured seeds (normalized)
     * - 1 float: opponent's captured seeds (normalized)
     * - 1 float: total seeds remaining on board (normalized)
     */
    extractFeatures(state) {
        const features = [];
        const MAX_SEEDS_PER_PIT = 20.0;  // Reasonable max for normalization
        const MAX_STORE = 48.0;  // Total seeds in game
        const MAX_REMAINING = 48.0;

        const currentPlayer = state.currentPlayer;
        const opponent = 1 - currentPlayer;

        // Extract my pits and opponent's pits (player-relative)
        const myPits = state.board.slice(currentPlayer * 6, (currentPlayer + 1) * 6);
        const oppPits = state.board.slice(opponent * 6, (opponent + 1) * 6);

        // 1-6: My pits (normalized)
        for (const seeds of myPits) {
            features.push(seeds / MAX_SEEDS_PER_PIT);
        }

        // 7-12: Opponent's pits (normalized)
        for (const seeds of oppPits) {
            features.push(seeds / MAX_SEEDS_PER_PIT);
        }

        // 13: My captured seeds (normalized)
        features.push(state.stores[currentPlayer] / MAX_STORE);

        // 14: Opponent's captured seeds (normalized)
        features.push(state.stores[opponent] / MAX_STORE);

        // 15: Seeds remaining on board (normalized)
        const remainingSeeds = state.board.reduce((a, b) => a + b, 0);
        features.push(remainingSeeds / MAX_REMAINING);

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
