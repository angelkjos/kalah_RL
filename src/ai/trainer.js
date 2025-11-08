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
     * Reset training statistics
     */
    resetStats() {
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
     * Train agent through self-play with checkpointing
     * @param {number} numEpisodes - Number of games to play
     * @param {Object} checkpointOptions - Checkpoint configuration
     *   - evalInterval: Evaluate every N episodes (default: 5000)
     *   - evalGames: Number of games per evaluation (default: 200)
     *   - savePath: Path to save checkpoints (default: './checkpoints')
     *   - keepBest: Keep only best checkpoint (default: true)
     *
     * Note: In self-play, the agent plays both sides, so "wins" and "losses"
     * represent which side (Player 0 vs Player 1) won, not agent performance.
     * Both outcomes contribute to learning.
     */
    async trainSelfPlay(numEpisodes, checkpointOptions = {}) {
        this.resetStats();

        const {
            evalInterval = 5000,
            evalGames = 200,
            savePath = './checkpoints',
            keepBest = true
        } = checkpointOptions;

        console.log(`\n🤖 Training via self-play for ${numEpisodes} episodes...`);
        console.log('(Note: Agent plays both sides, so wins/losses show game balance, not performance)');
        console.log(`\n📊 Evaluation: Every ${evalInterval} episodes (${evalGames} games, both sides)`);
        console.log(`💾 Checkpoints: ${savePath}`);

        let bestWinRate = 0;
        let bestEpisode = 0;
        const evalHistory = [];

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
                }
                // No intermediate rewards - only terminal rewards
                // Intermediate rewards cause Q-value explosion

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

            // Checkpoint evaluation
            if ((episode + 1) % evalInterval === 0 || episode + 1 === numEpisodes) {
                console.log(`\n${'='.repeat(70)}`);
                console.log(`🎯 Checkpoint Evaluation at Episode ${episode + 1}`);
                console.log('='.repeat(70));

                const evalResults = await this.evaluate(evalGames);
                const winRate = parseFloat(evalResults.overall.winRate);

                evalHistory.push({
                    episode: episode + 1,
                    winRate,
                    epsilon: this.agent.epsilon,
                    asPlayer0: parseFloat(evalResults.asPlayer0.winRate),
                    asPlayer1: parseFloat(evalResults.asPlayer1.winRate)
                });

                // Calculate moving average (last 3 checkpoints)
                const recentEvals = evalHistory.slice(-3);
                const movingAvgWinRate = recentEvals.reduce((sum, e) => sum + e.winRate, 0) / recentEvals.length;
                console.log(`\n📈 Moving Average (last ${recentEvals.length} checkpoints): ${movingAvgWinRate.toFixed(1)}%`);

                // Save checkpoint if best so far
                if (winRate > bestWinRate) {
                    console.log(`\n🌟 New best model! ${winRate}% > ${bestWinRate}% (previous best)`);
                    bestWinRate = winRate;
                    bestEpisode = episode + 1;

                    const checkpointPath = `${savePath}/best-checkpoint`;
                    await this.agent.save(checkpointPath);
                    console.log(`💾 Best checkpoint saved to ${checkpointPath}`);

                    // Save metadata
                    const fs = require('fs');
                    const metadata = {
                        episode: episode + 1,
                        winRate,
                        epsilon: this.agent.epsilon,
                        timestamp: new Date().toISOString(),
                        evalResults
                    };
                    fs.writeFileSync(
                        `${checkpointPath}/metadata.json`,
                        JSON.stringify(metadata, null, 2)
                    );
                } else {
                    console.log(`\nCurrent: ${winRate}% (Best: ${bestWinRate}% at episode ${bestEpisode})`);
                }

                // Optionally save non-best checkpoints
                if (!keepBest) {
                    const checkpointPath = `${savePath}/checkpoint-${episode + 1}`;
                    await this.agent.save(checkpointPath);
                    console.log(`💾 Checkpoint saved to ${checkpointPath}`);
                }

                console.log('='.repeat(70));
            }
        }

        console.log('\n✅ Self-play training complete!');
        this.printFinalStats();

        console.log('\n📊 Evaluation History:');
        console.log('='.repeat(70));
        for (const eval of evalHistory) {
            const marker = eval.winRate === bestWinRate ? ' ⭐ BEST' : '';
            console.log(
                `Ep ${eval.episode.toString().padStart(5)}: ` +
                `${eval.winRate.toFixed(1)}% overall ` +
                `(P0: ${eval.asPlayer0.toFixed(1)}%, P1: ${eval.asPlayer1.toFixed(1)}%) ` +
                `ε=${eval.epsilon.toFixed(3)}${marker}`
            );
        }
        console.log('='.repeat(70));
        console.log(`\n🏆 Best checkpoint: Episode ${bestEpisode} with ${bestWinRate.toFixed(1)}% win rate`);
        console.log(`📁 Best model saved at: ${savePath}/best-checkpoint`);

        return {
            bestEpisode,
            bestWinRate,
            evalHistory
        };
    }

    /**
     * Train agent against a fixed opponent
     * @param {number} numEpisodes - Number of games to play
     * @param {Function} opponentPolicy - Function(state, validMoves) => action
     */
    async trainAgainstOpponent(numEpisodes, opponentPolicy = null) {
        this.resetStats();

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

                    // Store experience with placeholder reward (will be updated if game ends)
                    agentExperiences.push({
                        state,
                        action,
                        reward: 0, // Placeholder, updated after game ends
                        nextState,
                        done
                    });
                } else {
                    // Opponent's turn
                    action = opponentPolicy(state, validMoves);
                    game.makeMove(action);
                }
            }

            // CRITICAL FIX: Update terminal rewards for all agent experiences
            // The last experience might not be terminal if opponent made the final move
            const winner = game.getWinner();
            const terminalReward = winner === 0 ? 1 : winner === 1 ? -1 : 0;

            // Update the reward for the last experience (and mark as done)
            if (agentExperiences.length > 0) {
                const lastExp = agentExperiences[agentExperiences.length - 1];
                lastExp.reward = terminalReward;
                lastExp.done = true;
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
     * @param {boolean} resetStatsPerStage - Reset stats between stages (default: false, keeps cumulative)
     */
    async trainCurriculum(numEpisodes, resetStatsPerStage = false) {
        if (!resetStatsPerStage) {
            this.resetStats(); // Reset once at the start for cumulative stats
        }

        console.log(`\n📚 Training with curriculum learning for ${numEpisodes} episodes...`);

        const stages = [
            { name: 'Random opponent (warm-up)', episodes: Math.floor(numEpisodes * 0.3), policy: null },
            { name: 'Self-play (intermediate)', episodes: Math.floor(numEpisodes * 0.4), policy: 'self' },
            { name: 'Self-play (advanced)', episodes: Math.floor(numEpisodes * 0.3), policy: 'self' }
        ];

        for (let i = 0; i < stages.length; i++) {
            const stage = stages[i];
            console.log(`\n--- Stage ${i + 1}/${stages.length}: ${stage.name} (${stage.episodes} episodes) ---`);

            // Save resetStats calls to avoid resetting during curriculum unless requested
            const originalResetStats = this.resetStats.bind(this);
            if (!resetStatsPerStage) {
                // Temporarily override resetStats to do nothing
                this.resetStats = () => {};
            }

            if (stage.policy === 'self') {
                await this.trainSelfPlay(stage.episodes);
            } else {
                await this.trainAgainstOpponent(stage.episodes, stage.policy);
            }

            // Restore original method
            this.resetStats = originalResetStats;
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
     * @param {number} numGames - Number of evaluation games (split evenly between both sides)
     * @param {Function} opponentPolicy - Opponent policy
     * @returns {Object} Evaluation results
     */
    async evaluate(numGames = 100, opponentPolicy = null) {
        console.log(`\n📊 Evaluating agent over ${numGames} games (both sides)...`);

        // Default to random opponent
        if (!opponentPolicy) {
            opponentPolicy = (state, validMoves) => {
                return validMoves[Math.floor(Math.random() * validMoves.length)];
            };
        }

        const savedEpsilon = this.agent.epsilon;
        this.agent.epsilon = 0; // No exploration during evaluation - pure greedy

        // Split games evenly between both player positions
        const gamesPerSide = Math.floor(numGames / 2);

        // Test as Player 0 (first player)
        console.log(`\n  Testing as Player 0 (first) vs Random...`);
        const resultsAsP0 = await this.evaluateSide(gamesPerSide, 0, opponentPolicy);

        // Test as Player 1 (second player)
        console.log(`  Testing as Player 1 (second) vs Random...`);
        const resultsAsP1 = await this.evaluateSide(gamesPerSide, 1, opponentPolicy);

        this.agent.epsilon = savedEpsilon; // Restore epsilon

        // Combine results
        const totalWins = resultsAsP0.wins + resultsAsP1.wins;
        const totalLosses = resultsAsP0.losses + resultsAsP1.losses;
        const totalDraws = resultsAsP0.draws + resultsAsP1.draws;
        const totalGames = totalWins + totalLosses + totalDraws;
        const totalScore = resultsAsP0.totalScore + resultsAsP1.totalScore;
        const totalOpponentScore = resultsAsP0.totalOpponentScore + resultsAsP1.totalOpponentScore;

        const results = {
            asPlayer0: resultsAsP0,
            asPlayer1: resultsAsP1,
            overall: {
                wins: totalWins,
                losses: totalLosses,
                draws: totalDraws,
                winRate: (totalWins / totalGames * 100).toFixed(1),
                avgScore: (totalScore / totalGames).toFixed(1),
                avgOpponentScore: (totalOpponentScore / totalGames).toFixed(1)
            }
        };

        console.log('\n' + '='.repeat(60));
        console.log('Evaluation Results (ε = 0, pure greedy):');
        console.log('='.repeat(60));
        console.log(`\nAs Player 0 (first):  ${resultsAsP0.winRate}% win rate ` +
                    `(${resultsAsP0.wins}W-${resultsAsP0.losses}L-${resultsAsP0.draws}D)`);
        console.log(`As Player 1 (second): ${resultsAsP1.winRate}% win rate ` +
                    `(${resultsAsP1.wins}W-${resultsAsP1.losses}L-${resultsAsP1.draws}D)`);
        console.log(`\nOverall: ${results.overall.winRate}% win rate ` +
                    `(${totalWins}W-${totalLosses}L-${totalDraws}D)`);
        console.log(`Avg score: ${results.overall.avgScore} vs ${results.overall.avgOpponentScore}`);
        console.log('='.repeat(60));

        return results;
    }

    /**
     * Evaluate agent playing as a specific player
     * @param {number} numGames - Number of games to play
     * @param {number} agentPlayer - Player number for agent (0 or 1)
     * @param {Function} opponentPolicy - Opponent policy
     * @returns {Object} Results for this side
     */
    async evaluateSide(numGames, agentPlayer, opponentPolicy) {
        let wins = 0, losses = 0, draws = 0;
        let totalScore = 0, totalOpponentScore = 0;
        const opponentPlayer = 1 - agentPlayer;

        for (let i = 0; i < numGames; i++) {
            const game = new KalahEngine({ enableLogging: false });

            while (!game.gameOver) {
                const state = game.getState();
                const validMoves = game.getValidMoves();

                if (state.currentPlayer === agentPlayer) {
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
            if (winner === agentPlayer) wins++;
            else if (winner === opponentPlayer) losses++;
            else draws++;

            totalScore += game.getScore(agentPlayer);
            totalOpponentScore += game.getScore(opponentPlayer);
        }

        return {
            wins,
            losses,
            draws,
            winRate: (wins / numGames * 100).toFixed(1),
            avgScore: (totalScore / numGames).toFixed(1),
            avgOpponentScore: (totalOpponentScore / numGames).toFixed(1),
            totalScore,
            totalOpponentScore
        };
    }
}

module.exports = Trainer;
