/**
 * Kalah/Mancala Game Engine - Pure Business Logic
 *
 * This engine is UI-agnostic and can be used for:
 * - Web UI gameplay
 * - ML model training and inference
 * - Game simulations
 * - Unit testing
 */

class KalahEngine {
    /**
     * Initialize a new game
     * @param {Object} options - Configuration options
     * @param {number} options.pitsPerPlayer - Number of pits per player (default: 6)
     * @param {number} options.seedsPerPit - Initial seeds per pit (default: 4)
     * @param {boolean} options.enableLogging - Enable console logging (default: false)
     */
    constructor(options = {}) {
        const {
            pitsPerPlayer = 6,
            seedsPerPit = 4,
            enableLogging = false
        } = options;

        this.pitsPerPlayer = pitsPerPlayer;
        this.totalPits = pitsPerPlayer * 2;

        // Initialize board: pits 0-(n-1) belong to Player 0, pits n-(2n-1) belong to Player 1
        this.board = Array(this.totalPits).fill(seedsPerPit);
        this.stores = [0, 0]; // [Player 0 store, Player 1 store]
        this.currentPlayer = 0; // 0 for Player 0, 1 for Player 1
        this.gameOver = false;
        this.moveNumber = 0;
        this.enableLogging = enableLogging;
    }

    /**
     * Get current game state
     * @returns {Object} Complete game state
     */
    getState() {
        return {
            board: [...this.board],
            stores: [...this.stores],
            currentPlayer: this.currentPlayer,
            gameOver: this.gameOver,
            moveNumber: this.moveNumber,
            winner: this.getWinner(),
            validMoves: this.getValidMoves()
        };
    }

    /**
     * Set game state (useful for ML training from specific positions)
     * @param {Object} state - Game state to restore
     */
    setState(state) {
        this.board = [...state.board];
        this.stores = [...state.stores];
        this.currentPlayer = state.currentPlayer;
        this.gameOver = state.gameOver;
        this.moveNumber = state.moveNumber || 0;
    }

    /**
     * Get all valid moves for the current player
     * @returns {number[]} Array of valid pit indices
     */
    getValidMoves() {
        if (this.gameOver) {
            return [];
        }

        const startPit = this.currentPlayer * this.pitsPerPlayer;
        const endPit = startPit + this.pitsPerPlayer;
        const validMoves = [];

        for (let i = startPit; i < endPit; i++) {
            if (this.board[i] > 0) {
                validMoves.push(i);
            }
        }

        return validMoves;
    }

    /**
     * Check if a move is valid
     * @param {number} pitIndex - Index of the pit to play
     * @returns {boolean} True if move is valid
     */
    isValidMove(pitIndex) {
        // Check if game is over
        if (this.gameOver) {
            return false;
        }

        // Check if it's the current player's pit
        const playerStartPit = this.currentPlayer * this.pitsPerPlayer;
        const playerEndPit = playerStartPit + this.pitsPerPlayer;

        if (pitIndex < playerStartPit || pitIndex >= playerEndPit) {
            return false;
        }

        // Check if pit has seeds
        if (this.board[pitIndex] === 0) {
            return false;
        }

        return true;
    }

    /**
     * Make a move
     * @param {number} pitIndex - Index of the pit to play
     * @returns {Object|null} Move result or null if invalid
     */
    makeMove(pitIndex) {
        if (!this.isValidMove(pitIndex)) {
            return null;
        }

        // Track move details for return value
        const moveDetails = {
            pitIndex,
            player: this.currentPlayer,
            moveNumber: ++this.moveNumber,
            seedsSown: this.board[pitIndex],
            captured: 0,
            extraTurn: false,
            gameEnded: false,
            sowingPath: []
        };

        if (this.enableLogging) {
            console.log(`\n🎯 MOVE #${this.moveNumber}: Player ${this.currentPlayer} plays pit ${pitIndex}`);
            console.log(`   Seeds picked up: ${this.board[pitIndex]}`);
        }

        // Pick up seeds from the selected pit
        let seeds = this.board[pitIndex];
        this.board[pitIndex] = 0;

        // Determine player's side
        const playerStartPit = this.currentPlayer * this.pitsPerPlayer;
        const playerEndPit = playerStartPit + this.pitsPerPlayer;
        const playerPits = Array.from({ length: this.pitsPerPlayer }, (_, i) => playerStartPit + i);

        // Distribute seeds counter-clockwise
        let currentPit = pitIndex;
        let lastPitWasEmpty = false;
        let landedInOwnStore = false;
        let firstRound = true;

        while (seeds > 0) {
            currentPit = (currentPit + 1) % this.totalPits;

            // Skip the starting pit ONLY on the first round
            if (currentPit === pitIndex && firstRound) {
                firstRound = false;
                if (this.enableLogging) {
                    console.log(`   ⏭️  Skipping starting pit ${pitIndex} (first round)`);
                }
                continue;
            }

            // Check if we should drop into our store BEFORE placing in a pit
            // Player 0: when we reach the opponent's first pit, drop in store first
            // Player 1: when we reach Player 0's first pit, drop in store first
            const shouldDropInStore =
                (this.currentPlayer === 0 && currentPit === this.pitsPerPlayer) ||
                (this.currentPlayer === 1 && currentPit === 0);

            if (shouldDropInStore && seeds > 0) {
                this.stores[this.currentPlayer]++;
                seeds--;
                moveDetails.sowingPath.push(`Store-P${this.currentPlayer}`);

                if (this.enableLogging) {
                    console.log(`   🏪 Seed → Player ${this.currentPlayer}'s Store (now ${this.stores[this.currentPlayer]})`);
                }

                if (seeds === 0) {
                    landedInOwnStore = true;
                    moveDetails.extraTurn = true;
                    if (this.enableLogging) {
                        console.log(`   ⭐ Last seed in own store! Extra turn!`);
                    }
                    break;
                }
            }

            // If we still have seeds, place one in the current pit
            if (seeds > 0) {
                const wasEmpty = this.board[currentPit] === 0;
                this.board[currentPit]++;
                seeds--;
                moveDetails.sowingPath.push(`pit-${currentPit}`);

                if (this.enableLogging) {
                    console.log(`   🌱 Seed → pit ${currentPit} (now ${this.board[currentPit]}${wasEmpty ? ', was empty' : ''})`);
                }

                // Track if this is the last seed and it landed in an empty pit on our side
                if (seeds === 0 && playerPits.includes(currentPit) && wasEmpty) {
                    lastPitWasEmpty = true;
                }
            }
        }

        // Capture: if last seed landed in an empty pit on our side, capture opposite pit
        if (lastPitWasEmpty) {
            const oppositePit = this.totalPits - 1 - currentPit;
            if (this.board[oppositePit] > 0) {
                const capturedSeeds = this.board[oppositePit] + this.board[currentPit];
                moveDetails.captured = capturedSeeds;

                if (this.enableLogging) {
                    console.log(`   💰 CAPTURE! Pit ${currentPit} (${this.board[currentPit]}) + opposite pit ${oppositePit} (${this.board[oppositePit]}) = ${capturedSeeds} seeds`);
                }

                this.stores[this.currentPlayer] += capturedSeeds;
                this.board[oppositePit] = 0;
                this.board[currentPit] = 0;
            }
        }

        // Check for game over
        const gameEnded = this._checkAndHandleGameOver();
        moveDetails.gameEnded = gameEnded;

        // Switch players (unless last seed landed in own store - then same player goes again)
        if (!landedInOwnStore) {
            this.currentPlayer = 1 - this.currentPlayer;
            if (this.enableLogging) {
                console.log(`   ➡️  Turn switches to Player ${this.currentPlayer}`);
            }
        } else {
            if (this.enableLogging) {
                console.log(`   🔄 Player ${this.currentPlayer} plays again!`);
            }
        }

        if (this.enableLogging && gameEnded) {
            console.log('🏁 GAME OVER!');
        }

        return moveDetails;
    }

    /**
     * Check for game over and handle end-game logic
     * @returns {boolean} True if game ended
     * @private
     */
    _checkAndHandleGameOver() {
        // Check for immediate win condition (score >= 25)
        if (this.stores[0] >= 25 || this.stores[1] >= 25) {
            this.gameOver = true;
            // Collect any remaining seeds on the board
            const player0Seeds = this.board.slice(0, this.pitsPerPlayer).reduce((a, b) => a + b, 0);
            const player1Seeds = this.board.slice(this.pitsPerPlayer).reduce((a, b) => a + b, 0);

            this.stores[0] += player0Seeds;
            for (let i = 0; i < this.pitsPerPlayer; i++) {
                this.board[i] = 0;
            }

            this.stores[1] += player1Seeds;
            for (let i = this.pitsPerPlayer; i < this.totalPits; i++) {
                this.board[i] = 0;
            }
            return true;
        }

        const player0Seeds = this.board.slice(0, this.pitsPerPlayer).reduce((a, b) => a + b, 0);
        const player1Seeds = this.board.slice(this.pitsPerPlayer).reduce((a, b) => a + b, 0);

        if (player0Seeds === 0 || player1Seeds === 0) {
            this.gameOver = true;

            // Each player collects all remaining seeds on their side
            this.stores[0] += player0Seeds;
            for (let i = 0; i < this.pitsPerPlayer; i++) {
                this.board[i] = 0;
            }

            this.stores[1] += player1Seeds;
            for (let i = this.pitsPerPlayer; i < this.totalPits; i++) {
                this.board[i] = 0;
            }

            return true;
        }

        return false;
    }

    /**
     * Get winner (-1 for in progress, 0 for player 0, 1 for player 1, null for draw)
     * @returns {number|null} Winner
     */
    getWinner() {
        if (!this.gameOver) {
            return -1;
        }

        if (this.stores[0] > this.stores[1]) {
            return 0;
        } else if (this.stores[1] > this.stores[0]) {
            return 1;
        } else {
            return null; // Draw
        }
    }

    /**
     * Get score for a specific player (for ML reward calculation)
     * @param {number} player - Player index (0 or 1)
     * @returns {number} Player's score
     */
    getScore(player) {
        return this.stores[player];
    }

    /**
     * Get score difference (current player's score - opponent's score)
     * Useful for ML model evaluation
     * @returns {number} Score difference
     */
    getScoreDifference() {
        return this.stores[this.currentPlayer] - this.stores[1 - this.currentPlayer];
    }

    /**
     * Clone the current game state (useful for ML tree search)
     * @returns {KalahEngine} New engine instance with same state
     */
    clone() {
        const cloned = new KalahEngine({
            pitsPerPlayer: this.pitsPerPlayer,
            seedsPerPit: 0, // Will be overridden by setState
            enableLogging: false // Don't log in clones
        });
        cloned.setState(this.getState());
        return cloned;
    }

    /**
     * Reset game to initial state
     * @param {number} seedsPerPit - Seeds per pit for new game
     */
    reset(seedsPerPit = 4) {
        this.board = Array(this.totalPits).fill(seedsPerPit);
        this.stores = [0, 0];
        this.currentPlayer = 0;
        this.gameOver = false;
        this.moveNumber = 0;
    }

    /**
     * Get a string representation of the board (for debugging)
     * @returns {string} Board representation
     */
    toString() {
        const lines = [];
        lines.push('='.repeat(60));
        lines.push(`Move #${this.moveNumber} - Player ${this.currentPlayer}'s turn`);
        lines.push('='.repeat(60));

        // Player 1's pits (top row, reversed for visual layout)
        const p1Pits = this.board.slice(this.pitsPerPlayer).reverse()
            .map((seeds, i) => `[${this.totalPits - 1 - i}:${seeds}]`)
            .join(' ');
        lines.push(`Player 1: ${p1Pits}`);
        lines.push(`P1 Store: ${this.stores[1]}${' '.repeat(40)}P0 Store: ${this.stores[0]}`);

        // Player 0's pits (bottom row)
        const p0Pits = this.board.slice(0, this.pitsPerPlayer)
            .map((seeds, i) => `[${i}:${seeds}]`)
            .join(' ');
        lines.push(`Player 0: ${p0Pits}`);
        lines.push('='.repeat(60));

        if (this.gameOver) {
            const winner = this.getWinner();
            if (winner === null) {
                lines.push(`GAME OVER - Draw! (${this.stores[0]} each)`);
            } else {
                lines.push(`GAME OVER - Player ${winner} wins! (${this.stores[winner]} vs ${this.stores[1 - winner]})`);
            }
        }

        return lines.join('\n');
    }
}

// Export for use in Node.js and browser
if (typeof module !== 'undefined' && module.exports) {
    module.exports = KalahEngine;
}
