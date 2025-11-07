/**
 * Trainer for Q-Learning Agent
 *
 * Supports multiple training modes:
 * - Self-play
 * - Against random opponent
 * - Curriculum learning
 */

const KalahEngine = require('../engine/kalah-engine.js');
const QLearningAgent = require('./rl-agent.js');

class Trainer {
    constructor(agent, options = {}) {
        this.agent = agent;
        this.options = {
            verbose: true,
            logInterval: 100,
            ...options
        };

        this.stats = {
            gamesPlayed: 0,
            wins: 0,
            losses: 0,
            draws: 0,
            totalReward: 0,
            recentWinRate: []
        };
    }

    /**
     * Train agent through self-play
     * @param {number} numEpisodes - Number of games to play
     */
    async trainSelfPlay(numEpisodes) {
        console.log(`\n🤖 Training via self-play for ${numEpisodes} episodes...`);

        for (let episode = 0; episode < numEpisodes; episode++) {
            const game = new KalahEngine({ enableLogging: false });
            const experiences = [];

            // Play one game
            while (!game.gameOver) {
                const state = game.getState();
                const validMoves = game.getValidMoves();
                const action = this.agent.selectAction(state, validMoves);
                const currentPlayer = state.currentPlayer;

                game.makeMove(action);

                const nextState = game.getState();
                const done = game.gameOver;

                // Calculate reward from player 0's perspective
                let reward = 0;
                if (done) {
                    const winner = game.getWinner();
                    if (winner === 0) reward = 1;
                    else if (winner === 1) reward = -1;
                    // Draw = 0
                } else {
                    // Small intermediate reward based on score difference
                    const scoreDiff = game.stores[0] - game.stores[1];
                    reward = scoreDiff * 0.01; // Small reward for score advantage
                }

                experiences.push({
                    state,
                    action,
                    reward: currentPlayer === 0 ? reward : -reward, // Flip for player 1
                    nextState,
                    done
                });
            }

            // Store all experiences
            for (const exp of experiences) {
                this.agent.remember(exp);
            }

            // Train on batch
            const loss = await this.agent.replay();

            // Decay exploration
            this.agent.decayEpsilon();

            // Update stats
            this.updateStats(game, loss);

            // Log progress
            if (this.options.verbose && (episode + 1) % this.options.logInterval === 0) {
                this.logProgress(episode + 1);
            }
        }

        console.log('\n✅ Self-play training complete!');
        this.printFinalStats();
    }

    /**
     * Train agent against a fixed opponent
     * @param {number} numEpisodes - Number of games to play
     * @param {Function} opponentPolicy - Function(state, validMoves) => action
     */
    async trainAgainstOpponent(numEpisodes, opponentPolicy = null) {
        // Default to random opponent
        if (!opponentPolicy) {
            opponentPolicy = (state, validMoves) => {
                return validMoves[Math.floor(Math.random() * validMoves.length)];
            };
        }

        console.log(`\n🎯 Training against opponent for ${numEpisodes} episodes...`);

        for (let episode = 0; episode < numEpisodes; episode++) {
            const game = new KalahEngine({ enableLogging: false });
            const agentExperiences = [];

            // Agent is player 0
            while (!game.gameOver) {
                const state = game.getState();
                const validMoves = game.getValidMoves();
                let action;

                if (state.currentPlayer === 0) {
                    // Agent's turn
                    action = this.agent.selectAction(state, validMoves);

                    game.makeMove(action);

                    const nextState = game.getState();
                    const done = game.gameOver;

                    let reward = 0;
                    if (done) {
                        const winner = game.getWinner();
                        if (winner === 0) reward = 1;
                        else if (winner === 1) reward = -1;
                    }

                    agentExperiences.push({
                        state,
                        action,
                        reward,
                        nextState,
                        done
                    });
                } else {
                    // Opponent's turn
                    action = opponentPolicy(state, validMoves);
                    game.makeMove(action);
                }
            }

            // Store agent's experiences
            for (const exp of agentExperiences) {
                this.agent.remember(exp);
            }

            // Train on batch
            const loss = await this.agent.replay();

            // Decay exploration
            this.agent.decayEpsilon();

            // Update stats
            this.updateStats(game, loss);

            // Log progress
            if (this.options.verbose && (episode + 1) % this.options.logInterval === 0) {
                this.logProgress(episode + 1);
            }
        }

        console.log('\n✅ Opponent training complete!');
        this.printFinalStats();
    }

    /**
     * Curriculum learning: gradually increase difficulty
     * @param {number} numEpisodes - Total number of episodes
     */
    async trainCurriculum(numEpisodes) {
        console.log(`\n📚 Training with curriculum learning for ${numEpisodes} episodes...`);

        const stages = [
            { name: 'Random opponent (warm-up)', episodes: Math.floor(numEpisodes * 0.3), policy: null },
            { name: 'Self-play (intermediate)', episodes: Math.floor(numEpisodes * 0.4), policy: 'self' },
            { name: 'Self-play (advanced)', episodes: Math.floor(numEpisodes * 0.3), policy: 'self' }
        ];

        for (const stage of stages) {
            console.log(`\n--- Stage: ${stage.name} (${stage.episodes} episodes) ---`);

            if (stage.policy === 'self') {
                await this.trainSelfPlay(stage.episodes);
            } else {
                await this.trainAgainstOpponent(stage.episodes, stage.policy);
            }
        }

        console.log('\n✅ Curriculum training complete!');
    }

    /**
     * Update training statistics
     */
    updateStats(game, loss) {
        this.stats.gamesPlayed++;
        const winner = game.getWinner();

        if (winner === 0) {
            this.stats.wins++;
        } else if (winner === 1) {
            this.stats.losses++;
        } else {
            this.stats.draws++;
        }

        this.stats.totalReward += (winner === 0 ? 1 : winner === 1 ? -1 : 0);

        // Track recent win rate (last 100 games)
        this.stats.recentWinRate.push(winner === 0 ? 1 : 0);
        if (this.stats.recentWinRate.length > 100) {
            this.stats.recentWinRate.shift();
        }
    }

    /**
     * Log training progress
     */
    logProgress(episode) {
        const winRate = (this.stats.wins / this.stats.gamesPlayed * 100).toFixed(1);
        const recentWR = this.stats.recentWinRate.length > 0 ?
            (this.stats.recentWinRate.reduce((a, b) => a + b, 0) / this.stats.recentWinRate.length * 100).toFixed(1) :
            '0.0';

        console.log(
            `Episode ${episode} | ` +
            `Win rate: ${winRate}% | ` +
            `Recent: ${recentWR}% | ` +
            `ε: ${this.agent.epsilon.toFixed(3)} | ` +
            `Buffer: ${this.agent.replayBuffer.length}`
        );
    }

    /**
     * Print final training statistics
     */
    printFinalStats() {
        const { gamesPlayed, wins, losses, draws } = this.stats;
        const winRate = (wins / gamesPlayed * 100).toFixed(1);
        const drawRate = (draws / gamesPlayed * 100).toFixed(1);

        console.log('\n' + '='.repeat(50));
        console.log('Training Statistics');
        console.log('='.repeat(50));
        console.log(`Total games: ${gamesPlayed}`);
        console.log(`Wins: ${wins} (${winRate}%)`);
        console.log(`Losses: ${losses} (${(losses/gamesPlayed*100).toFixed(1)}%)`);
        console.log(`Draws: ${draws} (${drawRate}%)`);
        console.log(`Final epsilon: ${this.agent.epsilon.toFixed(3)}`);
        console.log(`Buffer size: ${this.agent.replayBuffer.length}`);
        console.log('='.repeat(50) + '\n');
    }

    /**
     * Evaluate agent's performance
     * @param {number} numGames - Number of evaluation games
     * @param {Function} opponentPolicy - Opponent policy
     * @returns {Object} Evaluation results
     */
    async evaluate(numGames = 100, opponentPolicy = null) {
        console.log(`\n📊 Evaluating agent over ${numGames} games...`);

        // Default to random opponent
        if (!opponentPolicy) {
            opponentPolicy = (state, validMoves) => {
                return validMoves[Math.floor(Math.random() * validMoves.length)];
            };
        }

        const savedEpsilon = this.agent.epsilon;
        this.agent.epsilon = 0; // No exploration during evaluation

        let wins = 0, losses = 0, draws = 0;
        let totalScore = 0, totalOpponentScore = 0;

        for (let i = 0; i < numGames; i++) {
            const game = new KalahEngine({ enableLogging: false });

            while (!game.gameOver) {
                const state = game.getState();
                const validMoves = game.getValidMoves();

                if (state.currentPlayer === 0) {
                    // Agent's turn
                    const action = this.agent.selectAction(state, validMoves);
                    game.makeMove(action);
                } else {
                    // Opponent's turn
                    const action = opponentPolicy(state, validMoves);
                    game.makeMove(action);
                }
            }

            const winner = game.getWinner();
            if (winner === 0) wins++;
            else if (winner === 1) losses++;
            else draws++;

            totalScore += game.getScore(0);
            totalOpponentScore += game.getScore(1);
        }

        this.agent.epsilon = savedEpsilon; // Restore epsilon

        const results = {
            wins,
            losses,
            draws,
            winRate: (wins / numGames * 100).toFixed(1),
            avgScore: (totalScore / numGames).toFixed(1),
            avgOpponentScore: (totalOpponentScore / numGames).toFixed(1)
        };

        console.log('\nEvaluation Results:');
        console.log(`Win rate: ${results.winRate}%`);
        console.log(`Wins: ${wins}, Losses: ${losses}, Draws: ${draws}`);
        console.log(`Avg score: ${results.avgScore} vs ${results.avgOpponentScore}`);

        return results;
    }
}

module.exports = Trainer;
