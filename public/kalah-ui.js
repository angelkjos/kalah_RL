/**
 * Kalah/Mancala UI Wrapper
 *
 * This class wraps the KalahEngine with UI-specific functionality
 * like DOM manipulation, event handlers, and user feedback.
 */

class KalahUI {
    constructor() {
        // Create game engine with logging enabled
        this.engine = new KalahEngine({
            pitsPerPlayer: 6,
            seedsPerPit: 4,
            enableLogging: true
        });

        // UI-specific state
        this.moveHistory = [];

        // AI opponent
        this.aiEnabled = false;
        this.aiDifficulty = 'medium';
        this.minimaxAI = null;
        this.rlAgent = null;

        // Initialize minimax AI
        if (typeof KalahAI !== 'undefined') {
            this.minimaxAI = new KalahAI('easy'); // Will change based on difficulty
        }

        // Initialize RL agent
        if (typeof RLAgent !== 'undefined') {
            this.rlAgent = new RLAgent();
        }

        // Initialize UI
        this.initializeUI();
    }

    initializeUI() {
        // Add click listeners to all pits
        document.querySelectorAll('.pit').forEach(pit => {
            pit.addEventListener('click', () => {
                const pitIndex = parseInt(pit.dataset.pit);
                this.handlePitClick(pitIndex);
            });
        });

        // Reset button
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.reset();
        });

        // Undo button
        document.getElementById('undo-btn').addEventListener('click', () => {
            this.undo();
        });

        // AI toggle button
        const aiToggleBtn = document.getElementById('ai-toggle-btn');
        if (aiToggleBtn) {
            aiToggleBtn.addEventListener('click', () => {
                this.toggleAI();
            });
        }

        // AI difficulty selector
        const aiDifficultySelect = document.getElementById('ai-difficulty');
        if (aiDifficultySelect) {
            aiDifficultySelect.addEventListener('change', (e) => {
                this.changeDifficulty(e.target.value);
            });
        }

        this.updateUI();
        this.logState('Game initialized');
    }

    async toggleAI() {
        this.aiEnabled = !this.aiEnabled;
        const btn = document.getElementById('ai-toggle-btn');
        const difficultySelect = document.getElementById('ai-difficulty');

        if (this.aiEnabled) {
            btn.textContent = '👤 Play vs Human';
            btn.style.background = '#27ae60';

            // Show difficulty selector
            if (difficultySelect) {
                difficultySelect.style.display = 'inline-block';
            }

            // Load RL agent if hard difficulty selected
            if (this.aiDifficulty === 'hard' && this.rlAgent && !this.rlAgent.loaded) {
                console.log('🤖 Loading RL agent...');
                const loaded = await this.rlAgent.loadModel();
                if (!loaded) {
                    alert('RL agent model not found! Train a model with "npm run train" first.\nUsing Medium difficulty instead.');
                    this.aiDifficulty = 'medium';
                    if (difficultySelect) difficultySelect.value = 'medium';
                }
            }

            console.log(`🤖 AI opponent enabled! Difficulty: ${this.aiDifficulty}`);
            console.log('   You are Player 1 (bottom row)');
            console.log('   AI is Player 2 (top row)');

            // If it's AI's turn, make move
            if (this.engine.currentPlayer === 1 && !this.engine.gameOver) {
                this.makeAIMove();
            }
        } else {
            btn.textContent = '🤖 Play vs AI';
            btn.style.background = '#3498db';

            // Hide difficulty selector
            if (difficultySelect) {
                difficultySelect.style.display = 'none';
            }

            console.log('👥 Playing vs human');
        }
    }

    async changeDifficulty(difficulty) {
        this.aiDifficulty = difficulty;

        // Update minimax AI difficulty
        if (this.minimaxAI && difficulty !== 'hard') {
            this.minimaxAI = new KalahAI(difficulty);
        }

        // Load RL agent if switching to hard
        if (difficulty === 'hard' && this.rlAgent && !this.rlAgent.loaded) {
            console.log('🤖 Loading RL agent...');
            const loaded = await this.rlAgent.loadModel();
            if (!loaded) {
                alert('RL agent model not found! Train a model with "npm run train" first.\nSwitching to Medium difficulty.');
                this.aiDifficulty = 'medium';
                document.getElementById('ai-difficulty').value = 'medium';
                return;
            }
        }

        console.log(`🎚️ AI difficulty changed to: ${difficulty}`);
    }

    handlePitClick(pitIndex) {
        if (this.engine.gameOver) {
            alert('Game is over! Click "New Game" to play again.');
            return;
        }

        // Check if AI is enabled and it's AI's turn
        if (this.aiEnabled && this.engine.currentPlayer === 1) {
            alert("It's the AI's turn! Please wait...");
            return;
        }

        if (!this.engine.isValidMove(pitIndex)) {
            if (this.engine.board[pitIndex] === 0) {
                alert('This pit is empty! Choose another pit.');
            } else {
                alert('Choose one of your pits!');
            }
            return;
        }

        // Save state before making move
        this.saveState();

        // Make the move using the engine
        const moveResult = this.engine.makeMove(pitIndex);

        // Update UI
        this.updateUI();
        this.logState(`After move #${this.engine.moveNumber}`);

        // Handle game over
        if (moveResult.gameEnded) {
            setTimeout(() => {
                this.showGameOverMessage();
            }, 300);
            return;
        }

        // If AI is enabled and it's now AI's turn, make AI move
        if (this.aiEnabled && this.engine.currentPlayer === 1 && !this.engine.gameOver) {
            setTimeout(() => {
                this.makeAIMove();
            }, 800);
        }
    }

    makeAIMove() {
        if (!this.aiEnabled || this.engine.gameOver) {
            return;
        }

        if (this.engine.currentPlayer !== 1) {
            return; // Not AI's turn
        }

        console.log(`🤖 AI (${this.aiDifficulty}) is thinking...`);

        // Get AI's move based on difficulty
        const state = this.engine.getState();
        let aiMove = null;

        if (this.aiDifficulty === 'hard' && this.rlAgent && this.rlAgent.loaded) {
            // Use trained RL agent
            aiMove = this.rlAgent.selectMove(state);
        } else {
            // Use minimax AI
            if (!this.minimaxAI) {
                console.log('❌ AI not available');
                return;
            }
            aiMove = this.minimaxAI.selectMove(state);
        }

        if (aiMove === null) {
            console.log('❌ AI has no valid moves');
            return;
        }

        console.log(`🤖 AI plays pit ${aiMove}`);

        // Save state
        this.saveState();

        // Make the move
        const moveResult = this.engine.makeMove(aiMove);

        // Update UI
        this.updateUI();
        this.logState(`After AI move #${this.engine.moveNumber}`);

        // Handle game over
        if (moveResult.gameEnded) {
            setTimeout(() => {
                this.showGameOverMessage();
            }, 300);
            return;
        }

        // If AI gets another turn, make another move
        if (this.aiEnabled && this.engine.currentPlayer === 1 && !this.engine.gameOver) {
            console.log('⭐ AI gets another turn!');
            setTimeout(() => {
                this.makeAIMove();
            }, 1000);
        }
    }

    showGameOverMessage() {
        const winner = this.engine.getWinner();
        if (winner === null) {
            alert(`Game Over! It's a draw with ${this.engine.stores[0]} seeds each!`);
        } else {
            alert(`Game Over! Player ${winner + 1} wins with ${this.engine.stores[winner]} seeds!`);
        }
    }

    saveState() {
        this.moveHistory.push(this.engine.getState());
    }

    undo() {
        if (this.moveHistory.length === 0) {
            console.log('❌ No moves to undo');
            alert('No moves to undo!');
            return;
        }

        const previousState = this.moveHistory.pop();
        this.engine.setState(previousState);

        this.updateUI();
        console.log('⏪ UNDO - Restored to move #' + this.engine.moveNumber);
        this.logState('After undo');
    }

    reset() {
        this.engine.reset();
        this.moveHistory = [];
        this.updateUI();
        console.clear();
        this.logState('Game reset');

        // If AI is enabled and it's AI's turn, make move
        if (this.aiEnabled && this.engine.currentPlayer === 1) {
            setTimeout(() => {
                this.makeAIMove();
            }, 500);
        }
    }

    logState(message) {
        console.log('\n' + this.engine.toString());
    }

    updateUI() {
        const state = this.engine.getState();

        // Update pit displays
        for (let i = 0; i < 12; i++) {
            document.getElementById(`pit-${i}`).textContent = state.board[i];
        }

        // Update store displays
        document.querySelector('#store-0 .seeds').textContent = state.stores[0];
        document.querySelector('#store-1 .seeds').textContent = state.stores[1];

        // Update scores in top info
        document.getElementById('player1-score').textContent = state.stores[0];
        document.getElementById('player2-score').textContent = state.stores[1];

        // Update turn display
        const turnDisplay = document.getElementById('turn-display');
        if (state.gameOver) {
            turnDisplay.textContent = 'Game Over!';
        } else {
            turnDisplay.textContent = `Player ${state.currentPlayer + 1}'s Turn`;
        }

        // Update pit states (enable/disable based on current player)
        document.querySelectorAll('.pit').forEach(pit => {
            const pitIndex = parseInt(pit.dataset.pit);

            if (state.gameOver) {
                pit.classList.add('disabled');
                pit.classList.remove('active');
            } else if (state.validMoves.includes(pitIndex)) {
                pit.classList.remove('disabled');
                pit.classList.add('active');
            } else {
                pit.classList.add('disabled');
                pit.classList.remove('active');
            }
        });
    }
}

// Initialize game when page loads
let game;
window.addEventListener('DOMContentLoaded', () => {
    game = new KalahUI();
});
