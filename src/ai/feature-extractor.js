/**
 * Feature Extractor for Kalah RL Agent
 *
 * This function is the single source of truth for converting game state
 * into a feature vector for the neural network.
 */

function extractFeatures(state) {
    /**
     * Extract features from game state for ML model input
     * Player-relative representation (15 features)
     *
     * Features:
     * - 6 floats: my pits (normalized to [0,1])
     * - 6 floats: opponent's pits (normalized to [0,1])
     * - 1 float: my captured seeds (normalized)
     * - 1 float: opponent's captured seeds (normalized)
     * - 1 float: total seeds remaining on board (normalized)
     */
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

// Export for use in Node.js and browser
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { extractFeatures };
} else if (typeof window !== 'undefined') {
    window.extractFeatures = extractFeatures;
}
