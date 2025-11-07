/**
 * Example: Using KalahEngine for Machine Learning
 *
 * This file demonstrates how to use the KalahEngine for ML applications,
 * including training, game simulation, and move evaluation.
 */

// Import the engine (for Node.js environment)
const KalahEngine = require('../engine/kalah-engine.js');

// ============================================================================
// Example 1: Basic Game Simulation
// ============================================================================

function exampleBasicSimulation() {
    console.log('\n=== Example 1: Basic Game Simulation ===\n');

    const game = new KalahEngine({ enableLogging: false });

    // Simulate a game with random moves
    while (!game.gameOver) {
        const validMoves = game.getValidMoves();
        const randomMove = validMoves[Math.floor(Math.random() * validMoves.length)];

        console.log(`Player ${game.currentPlayer} plays pit ${randomMove}`);
        const result = game.makeMove(randomMove);
        console.log(`  -> Captured: ${result.captured}, Extra turn: ${result.extraTurn}`);
    }

    console.log('\nFinal State:');
    console.log(game.toString());
    console.log(`Winner: Player ${game.getWinner()}`);
}

// ============================================================================
// Example 2: State Management for ML Training
// ============================================================================

function exampleStateManagement() {
    console.log('\n=== Example 2: State Management ===\n');

    const game = new KalahEngine();

    // Make some moves
    game.makeMove(0);
    game.makeMove(7);

    // Save state for later
    const savedState = game.getState();
    console.log('Saved state:', savedState);

    // Make more moves
    game.makeMove(2);

    // Restore previous state (useful for tree search algorithms)
    game.setState(savedState);
    console.log('\nRestored to previous state:');
    console.log(game.toString());
}

// ============================================================================
// Example 3: Game Tree Search (Minimax example)
// ============================================================================

function evaluatePosition(game, player) {
    /**
     * Simple evaluation function for a position
     * Returns a score from the perspective of the given player
     */
    if (game.gameOver) {
        const winner = game.getWinner();
        if (winner === player) return 1000;
        if (winner === null) return 0;
        return -1000;
    }

    // Score difference + positional advantage
    const scoreDiff = game.stores[player] - game.stores[1 - player];
    const seedsOnOurSide = game.board
        .slice(player * 6, (player + 1) * 6)
        .reduce((a, b) => a + b, 0);

    return scoreDiff + seedsOnOurSide * 0.1;
}

function minimax(game, depth, maximizingPlayer, player) {
    /**
     * Simple minimax implementation (no alpha-beta pruning)
     * Returns the best score for the current position
     */
    if (depth === 0 || game.gameOver) {
        return evaluatePosition(game, player);
    }

    const validMoves = game.getValidMoves();

    if (maximizingPlayer) {
        let maxScore = -Infinity;
        for (const move of validMoves) {
            const clonedGame = game.clone();
            clonedGame.makeMove(move);
            const score = minimax(clonedGame, depth - 1, false, player);
            maxScore = Math.max(maxScore, score);
        }
        return maxScore;
    } else {
        let minScore = Infinity;
        for (const move of validMoves) {
            const clonedGame = game.clone();
            clonedGame.makeMove(move);
            const score = minimax(clonedGame, depth - 1, true, player);
            minScore = Math.min(minScore, score);
        }
        return minScore;
    }
}

function getBestMove(game, depth = 4) {
    /**
     * Find the best move using minimax search
     */
    const validMoves = game.getValidMoves();
    const player = game.currentPlayer;
    let bestMove = validMoves[0];
    let bestScore = -Infinity;

    for (const move of validMoves) {
        const clonedGame = game.clone();
        clonedGame.makeMove(move);
        const score = minimax(clonedGame, depth - 1, false, player);

        if (score > bestScore) {
            bestScore = score;
            bestMove = move;
        }
    }

    return { move: bestMove, score: bestScore };
}

function exampleMinimaxAI() {
    console.log('\n=== Example 3: Minimax AI ===\n');

    const game = new KalahEngine({ enableLogging: false });

    // Play a game with AI vs random
    let moveCount = 0;
    while (!game.gameOver && moveCount < 100) {
        const currentPlayer = game.currentPlayer;

        let move;
        if (currentPlayer === 0) {
            // Player 0 uses minimax AI
            const aiMove = getBestMove(game, 3);
            move = aiMove.move;
            console.log(`Player 0 (AI) plays pit ${move} (score: ${aiMove.score.toFixed(2)})`);
        } else {
            // Player 1 plays randomly
            const validMoves = game.getValidMoves();
            move = validMoves[Math.floor(Math.random() * validMoves.length)];
            console.log(`Player 1 (Random) plays pit ${move}`);
        }

        game.makeMove(move);
        moveCount++;
    }

    console.log('\n' + game.toString());
}

// ============================================================================
// Example 4: Collecting Training Data for ML
// ============================================================================

function collectTrainingData(numGames = 100) {
    /**
     * Collect training data from self-play games
     * Returns array of (state, action, reward) tuples
     */
    const trainingData = [];

    for (let gameNum = 0; gameNum < numGames; gameNum++) {
        const game = new KalahEngine({ enableLogging: false });
        const gameHistory = [];

        // Play a game with random moves
        while (!game.gameOver) {
            const state = game.getState();
            const validMoves = game.getValidMoves();
            const move = validMoves[Math.floor(Math.random() * validMoves.length)];

            gameHistory.push({
                state: state,
                action: move,
                player: game.currentPlayer
            });

            game.makeMove(move);
        }

        // Assign rewards based on game outcome
        const winner = game.getWinner();
        for (const record of gameHistory) {
            let reward;
            if (winner === null) {
                reward = 0; // Draw
            } else if (winner === record.player) {
                reward = 1; // Win
            } else {
                reward = -1; // Loss
            }

            trainingData.push({
                board: record.state.board,
                stores: record.state.stores,
                currentPlayer: record.state.currentPlayer,
                action: record.action,
                reward: reward
            });
        }

        if ((gameNum + 1) % 10 === 0) {
            console.log(`Collected ${gameNum + 1} games...`);
        }
    }

    return trainingData;
}

function exampleCollectTrainingData() {
    console.log('\n=== Example 4: Collecting Training Data ===\n');

    const data = collectTrainingData(10);
    console.log(`Collected ${data.length} training samples`);
    console.log('\nSample data point:');
    console.log(data[0]);
}

// ============================================================================
// Example 5: Feature Extraction for ML Models
// ============================================================================

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

function exampleFeatureExtraction() {
    console.log('\n=== Example 5: Feature Extraction ===\n');

    const game = new KalahEngine();
    game.makeMove(0);
    game.makeMove(7);

    const state = game.getState();
    const features = extractFeatures(state);

    console.log('Game state:');
    console.log(game.toString());
    console.log('\nExtracted features (length=' + features.length + '):');
    console.log(features);
}

// ============================================================================
// Run Examples
// ============================================================================

if (require.main === module) {
    // Only run examples if this file is executed directly
    console.log('Kalah Engine ML Examples\n');

    exampleBasicSimulation();
    exampleStateManagement();
    exampleMinimaxAI();
    exampleCollectTrainingData();
    exampleFeatureExtraction();

    console.log('\n=== All examples completed ===\n');
}

// Export functions for use in other modules
module.exports = {
    evaluatePosition,
    minimax,
    getBestMove,
    collectTrainingData,
    extractFeatures
};
