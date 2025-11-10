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
            // Extract features using the global function
            const features = extractFeatures(state);
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
