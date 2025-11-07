/**
 * Browser-compatible AI Player for Kalah/Mancala
 *
 * This is a lightweight AI that runs in the browser.
 * Uses a simple heuristic evaluation since we can't load TensorFlow models in the browser easily.
 */

class KalahAI {
    constructor(difficulty = 'medium') {
        this.difficulty = difficulty;

        // Difficulty settings
        this.settings = {
            easy: { depth: 2, randomness: 0.3 },
            medium: { depth: 4, randomness: 0.1 },
            hard: { depth: 6, randomness: 0.0 }
        };

        this.config = this.settings[difficulty] || this.settings.medium;
    }

    /**
     * Select best move for current player
     * @param {Object} state - Game state from KalahEngine
     * @returns {number} Best pit index to play
     */
    selectMove(state) {
        const validMoves = this.getValidMoves(state);

        if (validMoves.length === 0) return null;

        // Random move with some probability (for variety)
        if (Math.random() < this.config.randomness) {
            return validMoves[Math.floor(Math.random() * validMoves.length)];
        }

        // Use minimax to find best move
        let bestMove = validMoves[0];
        let bestScore = -Infinity;

        for (const move of validMoves) {
            const score = this.minimax(state, move, this.config.depth, false);

            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
            }
        }

        return bestMove;
    }

    /**
     * Minimax algorithm with alpha-beta pruning
     */
    minimax(state, move, depth, isMaximizing, alpha = -Infinity, beta = Infinity) {
        // Simulate the move
        const newState = this.simulateMove(state, move);

        // Terminal state or max depth
        if (newState.gameOver || depth === 0) {
            return this.evaluatePosition(newState, state.currentPlayer);
        }

        const validMoves = this.getValidMoves(newState);

        if (validMoves.length === 0) {
            return this.evaluatePosition(newState, state.currentPlayer);
        }

        if (isMaximizing) {
            let maxScore = -Infinity;
            for (const nextMove of validMoves) {
                const score = this.minimax(newState, nextMove, depth - 1, false, alpha, beta);
                maxScore = Math.max(maxScore, score);
                alpha = Math.max(alpha, score);
                if (beta <= alpha) break; // Prune
            }
            return maxScore;
        } else {
            let minScore = Infinity;
            for (const nextMove of validMoves) {
                const score = this.minimax(newState, nextMove, depth - 1, true, alpha, beta);
                minScore = Math.min(minScore, score);
                beta = Math.min(beta, score);
                if (beta <= alpha) break; // Prune
            }
            return minScore;
        }
    }

    /**
     * Evaluate a board position
     */
    evaluatePosition(state, player) {
        if (state.gameOver) {
            const winner = this.getWinner(state);
            if (winner === player) return 1000;
            if (winner === null) return 0;
            return -1000;
        }

        // Heuristic evaluation
        const opponent = 1 - player;

        // Score difference (most important)
        const scoreDiff = state.stores[player] - state.stores[opponent];

        // Seeds on our side (positional advantage)
        const playerStart = player * 6;
        const opponentStart = opponent * 6;
        const ourSeeds = state.board.slice(playerStart, playerStart + 6).reduce((a, b) => a + b, 0);
        const theirSeeds = state.board.slice(opponentStart, opponentStart + 6).reduce((a, b) => a + b, 0);

        // Potential captures (pits with seeds opposite empty pits)
        let captureValue = 0;
        for (let i = playerStart; i < playerStart + 6; i++) {
            if (state.board[i] === 0) {
                const opposite = 11 - i;
                captureValue += state.board[opposite] * 0.5;
            }
        }

        return scoreDiff * 10 + (ourSeeds - theirSeeds) * 0.5 + captureValue;
    }

    /**
     * Simulate a move without modifying the original state
     */
    simulateMove(state, pitIndex) {
        // Deep copy state
        const newState = {
            board: [...state.board],
            stores: [...state.stores],
            currentPlayer: state.currentPlayer,
            gameOver: state.gameOver,
            moveNumber: state.moveNumber + 1
        };

        // Simulate the move (simplified version of game logic)
        let seeds = newState.board[pitIndex];
        newState.board[pitIndex] = 0;

        let currentPit = pitIndex;
        let landedInStore = false;
        const playerStart = newState.currentPlayer * 6;

        while (seeds > 0) {
            currentPit = (currentPit + 1) % 12;

            // Check if we should drop in store
            const shouldDropInStore =
                (newState.currentPlayer === 0 && currentPit === 6) ||
                (newState.currentPlayer === 1 && currentPit === 0);

            if (shouldDropInStore && seeds > 0) {
                newState.stores[newState.currentPlayer]++;
                seeds--;
                if (seeds === 0) {
                    landedInStore = true;
                    break;
                }
            }

            if (seeds > 0) {
                const wasEmpty = newState.board[currentPit] === 0;
                newState.board[currentPit]++;
                seeds--;

                // Capture logic
                if (seeds === 0 && wasEmpty &&
                    currentPit >= playerStart && currentPit < playerStart + 6) {
                    const opposite = 11 - currentPit;
                    if (newState.board[opposite] > 0) {
                        newState.stores[newState.currentPlayer] += newState.board[opposite] + newState.board[currentPit];
                        newState.board[opposite] = 0;
                        newState.board[currentPit] = 0;
                    }
                }
            }
        }

        // Check game over
        const p0Seeds = newState.board.slice(0, 6).reduce((a, b) => a + b, 0);
        const p1Seeds = newState.board.slice(6, 12).reduce((a, b) => a + b, 0);

        if (p0Seeds === 0 || p1Seeds === 0) {
            newState.gameOver = true;
            newState.stores[0] += p0Seeds;
            newState.stores[1] += p1Seeds;
            for (let i = 0; i < 12; i++) newState.board[i] = 0;
        }

        // Switch player unless landed in store
        if (!landedInStore) {
            newState.currentPlayer = 1 - newState.currentPlayer;
        }

        return newState;
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

    /**
     * Get winner from state
     */
    getWinner(state) {
        if (!state.gameOver) return -1;
        if (state.stores[0] > state.stores[1]) return 0;
        if (state.stores[1] > state.stores[0]) return 1;
        return null;
    }
}

// Make available globally
if (typeof window !== 'undefined') {
    window.KalahAI = KalahAI;
}
