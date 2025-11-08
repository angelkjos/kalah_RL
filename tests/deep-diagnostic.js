#!/usr/bin/env node

/**
 * Deep Diagnostic - Trace through actual Q-value updates
 */

const QLearningAgent = require('../src/ai/rl-agent.js');
const KalahEngine = require('../src/engine/kalah-engine.js');
const { extractFeatures } = require('../src/utils/ml-examples.js');

console.log('🔬 Deep Q-Learning Diagnostic\n');

async function analyzeQValues() {
    // Create a fresh agent
    const agent = new QLearningAgent({ epsilon: 0 });

    console.log('📊 Initial Q-values for starting position:');
    const game = new KalahEngine({ enableLogging: false });
    const state = game.getState();
    const qValues = agent.getQValues(state);

    console.log('State: Player 0, initial position');
    console.log('Q-values for pits 0-5:', qValues.map(q => q.toFixed(3)).join(', '));
    console.log('Best action (argmax):', qValues.indexOf(Math.max(...qValues)));

    // Check if all Q-values are similar (untrained network)
    const mean = qValues.reduce((a, b) => a + b) / qValues.length;
    const variance = qValues.reduce((sum, q) => sum + Math.pow(q - mean, 2), 0) / qValues.length;
    console.log(`Mean: ${mean.toFixed(3)}, Variance: ${variance.toFixed(6)}`);

    if (variance < 0.001) {
        console.log('⚠️  Q-values are nearly identical - network is untrained/random\n');
    }

    // Now try to load the trained model
    try {
        await agent.load('./models/kalah-agent');
        console.log('\n📊 Q-values from TRAINED model:');

        const trainedQValues = agent.getQValues(state);
        console.log('Q-values for pits 0-5:', trainedQValues.map(q => q.toFixed(3)).join(', '));
        console.log('Best action (argmax):', trainedQValues.indexOf(Math.max(...trainedQValues)));

        const trainedMean = trainedQValues.reduce((a, b) => a + b) / trainedQValues.length;
        const trainedVariance = trainedQValues.reduce((sum, q) => sum + Math.pow(q - trainedMean, 2), 0) / trainedQValues.length;
        console.log(`Mean: ${trainedMean.toFixed(3)}, Variance: ${trainedVariance.toFixed(6)}`);

        if (trainedVariance < 0.001) {
            console.log('❌ BUG: Trained network still has identical Q-values!');
            console.log('   This means the network is not learning.\n');
        } else {
            console.log('✅ Q-values have variance - network has learned something\n');
        }

        // Test Q-values throughout a game
        console.log('📊 Tracing Q-values through a game:\n');
        const testGame = new KalahEngine({ enableLogging: false });

        for (let turn = 0; turn < 10 && !testGame.gameOver; turn++) {
            const s = testGame.getState();
            const validMoves = testGame.getValidMoves();
            const q = agent.getQValues(s);

            console.log(`Turn ${turn + 1} - Player ${s.currentPlayer}`);
            console.log(`  Valid moves: ${validMoves.join(', ')}`);
            console.log(`  Q-values: ${q.map(v => v.toFixed(2)).join(', ')}`);

            // Get best valid action
            let bestMove = validMoves[0];
            let bestQ = -Infinity;
            for (const move of validMoves) {
                const relMove = move - (s.currentPlayer * 6);
                if (q[relMove] > bestQ) {
                    bestQ = q[relMove];
                    bestMove = move;
                }
            }
            console.log(`  Agent chooses: pit ${bestMove} (Q=${bestQ.toFixed(2)})`);

            testGame.makeMove(bestMove);
        }

        // Check for a critical bug: are we only training on Player 0's perspective?
        console.log('\n📊 Checking Player 0 vs Player 1 Q-value magnitudes:');

        const p0Game = new KalahEngine({ enableLogging: false });
        const p0State = p0Game.getState(); // Player 0's turn
        const p0Q = agent.getQValues(p0State);
        const p0Max = Math.max(...p0Q);
        const p0Min = Math.min(...p0Q);

        p0Game.makeMove(0); // Player 0 moves
        const p1State = p0Game.getState(); // Now Player 1's turn
        const p1Q = agent.getQValues(p1State);
        const p1Max = Math.max(...p1Q);
        const p1Min = Math.min(...p1Q);

        console.log(`Player 0: Q-values range [${p0Min.toFixed(3)}, ${p0Max.toFixed(3)}]`);
        console.log(`Player 1: Q-values range [${p1Min.toFixed(3)}, ${p1Max.toFixed(3)}]`);

        if (Math.abs(p0Max) > 10 * Math.abs(p1Max) || Math.abs(p1Max) > 10 * Math.abs(p0Max)) {
            console.log('❌ BUG: Huge difference in Q-value magnitudes between players!');
            console.log('   This suggests asymmetric training.\n');
        } else {
            console.log('✅ Q-values have similar magnitudes for both players\n');
        }

    } catch (e) {
        console.log('\n⚠️  No trained model found, skipping trained model tests\n');
    }
}

// Test the reward assignment fix
async function testRewardPropagation() {
    console.log('=' .repeat(60));
    console.log('📊 Testing Reward Propagation\n');

    const agent = new QLearningAgent({ epsilon: 0.1 });

    // Play 3 games and show what gets stored
    for (let gameNum = 0; gameNum < 3; gameNum++) {
        const game = new KalahEngine({ enableLogging: false });
        const experiences = [];

        while (!game.gameOver) {
            const state = game.getState();
            const validMoves = game.getValidMoves();

            if (state.currentPlayer === 0) {
                const action = agent.selectAction(state, validMoves);
                game.makeMove(action);

                const nextState = game.getState();

                experiences.push({
                    state,
                    action,
                    reward: 0,
                    nextState,
                    done: false,
                    player: 0
                });
            } else {
                // Random opponent
                const action = validMoves[Math.floor(Math.random() * validMoves.length)];
                game.makeMove(action);
            }
        }

        // Apply fix: update terminal reward
        const winner = game.getWinner();
        const terminalReward = winner === 0 ? 1 : winner === 1 ? -1 : 0;

        if (experiences.length > 0) {
            experiences[experiences.length - 1].reward = terminalReward;
            experiences[experiences.length - 1].done = true;
        }

        console.log(`Game ${gameNum + 1}:`);
        console.log(`  Final score: P0=${game.getScore(0)}, P1=${game.getScore(1)}`);
        console.log(`  Winner: ${winner === null ? 'Draw' : 'Player ' + winner}`);
        console.log(`  Agent experiences: ${experiences.length}`);
        console.log(`  Terminal reward: ${terminalReward}`);
        console.log(`  Non-zero rewards: ${experiences.filter(e => e.reward !== 0).length}`);

        // Check if any intermediate states have rewards
        const nonTerminal = experiences.slice(0, -1);
        const hasIntermediateRewards = nonTerminal.some(e => e.reward !== 0);
        console.log(`  Intermediate rewards: ${hasIntermediateRewards ? 'YES' : 'NO (all zeros)'}`);

        if (!hasIntermediateRewards) {
            console.log('  ⚠️  Only terminal reward is non-zero - agent relies on credit assignment');
        }
        console.log();
    }
}

async function main() {
    await analyzeQValues();
    await testRewardPropagation();

    console.log('=' .repeat(60));
    console.log('Diagnostic complete!');
}

main().catch(console.error);
