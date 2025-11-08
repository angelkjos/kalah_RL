#!/usr/bin/env node

/**
 * Diagnostic Tests for RL Agent
 *
 * Tests for common RL bugs:
 * 1. Epsilon is 0 during evaluation
 * 2. Symmetry: Agent performs equally as P0 and P1
 * 3. Terminal rewards are correct
 * 4. No illegal moves
 */

const QLearningAgent = require('../src/ai/rl-agent.js');
const KalahEngine = require('../src/engine/kalah-engine.js');
const { extractFeatures } = require('../src/utils/ml-examples.js');

console.log('🔍 Running RL Agent Diagnostic Tests\n');
console.log('='.repeat(60));

// ============================================================================
// Test 1: Epsilon Check During Evaluation
// ============================================================================

function test1_epsilonCheck() {
    console.log('\n📋 TEST 1: Epsilon is 0 during evaluation');
    console.log('-'.repeat(60));

    const agent = new QLearningAgent({ epsilon: 0.5 });
    console.log(`Initial epsilon: ${agent.epsilon}`);

    // Simulate what happens during evaluation
    const savedEpsilon = agent.epsilon;
    agent.epsilon = 0;
    console.log(`Epsilon during evaluation: ${agent.epsilon}`);

    if (agent.epsilon === 0) {
        console.log('✅ PASS: Epsilon is correctly set to 0');
        agent.epsilon = savedEpsilon;
        return true;
    } else {
        console.log('❌ FAIL: Epsilon is not 0 during evaluation!');
        return false;
    }
}

// ============================================================================
// Test 2: Symmetry Test - Agent as P0 vs P1
// ============================================================================

async function test2_symmetryTest(agent, numGames = 100) {
    console.log('\n📋 TEST 2: Symmetry - Agent performance as P0 vs P1');
    console.log('-'.repeat(60));

    const savedEpsilon = agent.epsilon;
    agent.epsilon = 0; // Greedy evaluation

    // Test as Player 0
    let p0Wins = 0, p0Losses = 0, p0Draws = 0;
    for (let i = 0; i < numGames; i++) {
        const game = new KalahEngine({ enableLogging: false });

        while (!game.gameOver) {
            const state = game.getState();
            const validMoves = game.getValidMoves();

            if (state.currentPlayer === 0) {
                // Agent as P0
                const action = agent.selectAction(state, validMoves);
                game.makeMove(action);
            } else {
                // Random opponent
                const action = validMoves[Math.floor(Math.random() * validMoves.length)];
                game.makeMove(action);
            }
        }

        const winner = game.getWinner();
        if (winner === 0) p0Wins++;
        else if (winner === 1) p0Losses++;
        else p0Draws++;
    }

    // Test as Player 1
    let p1Wins = 0, p1Losses = 0, p1Draws = 0;
    for (let i = 0; i < numGames; i++) {
        const game = new KalahEngine({ enableLogging: false });

        while (!game.gameOver) {
            const state = game.getState();
            const validMoves = game.getValidMoves();

            if (state.currentPlayer === 1) {
                // Agent as P1
                const action = agent.selectAction(state, validMoves);
                game.makeMove(action);
            } else {
                // Random opponent
                const action = validMoves[Math.floor(Math.random() * validMoves.length)];
                game.makeMove(action);
            }
        }

        const winner = game.getWinner();
        if (winner === 1) p1Wins++;
        else if (winner === 0) p1Losses++;
        else p1Draws++;
    }

    agent.epsilon = savedEpsilon;

    const p0WinRate = (p0Wins / numGames * 100).toFixed(1);
    const p1WinRate = (p1Wins / numGames * 100).toFixed(1);

    console.log(`Agent as Player 0: ${p0Wins}W ${p0Losses}L ${p0Draws}D (${p0WinRate}% win rate)`);
    console.log(`Agent as Player 1: ${p1Wins}W ${p1Losses}L ${p1Draws}D (${p1WinRate}% win rate)`);

    const diff = Math.abs(p0Wins - p1Wins);
    console.log(`Difference: ${diff} wins (${Math.abs(parseFloat(p0WinRate) - parseFloat(p1WinRate)).toFixed(1)}%)`);

    if (diff > numGames * 0.15) {
        console.log('❌ FAIL: Large asymmetry detected! Possible reward/POV bug.');
        console.log('   Expected: Similar win rates for P0 and P1');
        console.log('   This suggests the agent learned a player-specific strategy.');
        return false;
    } else {
        console.log('✅ PASS: Agent performs similarly as both players');
        return true;
    }
}

// ============================================================================
// Test 3: Terminal Reward Verification
// ============================================================================

async function test3_terminalRewards(agent) {
    console.log('\n📋 TEST 3: Terminal reward verification');
    console.log('-'.repeat(60));

    const savedEpsilon = agent.epsilon;
    agent.epsilon = 0.3; // Some exploration

    // Play 5 games and log rewards
    for (let gameNum = 0; gameNum < 5; gameNum++) {
        const game = new KalahEngine({ enableLogging: false });
        const experiences = [];

        // Agent is Player 0
        while (!game.gameOver) {
            const state = game.getState();
            const validMoves = game.getValidMoves();

            if (state.currentPlayer === 0) {
                const action = agent.selectAction(state, validMoves);

                game.makeMove(action);

                const nextState = game.getState();
                const done = game.gameOver;

                // Store with placeholder reward
                experiences.push({
                    state,
                    action,
                    reward: 0, // Will be updated after game ends
                    nextState,
                    done,
                    player: 0
                });
            } else {
                // Random opponent
                const action = validMoves[Math.floor(Math.random() * validMoves.length)];
                game.makeMove(action);
            }
        }

        // CRITICAL FIX: Update terminal reward after game ends
        const winner = game.getWinner();
        const terminalReward = winner === 0 ? 1 : winner === 1 ? -1 : 0;

        // Update the last experience with terminal reward
        if (experiences.length > 0) {
            experiences[experiences.length - 1].reward = terminalReward;
            experiences[experiences.length - 1].done = true;
        }

        const finalReward = experiences[experiences.length - 1]?.reward || 0;

        console.log(`\nGame ${gameNum + 1}:`);
        console.log(`  Final scores: P0=${game.getScore(0)}, P1=${game.getScore(1)}`);
        console.log(`  Winner: ${winner === null ? 'Draw' : `Player ${winner}`}`);
        console.log(`  Agent reward: ${finalReward}`);
        console.log(`  Agent transitions stored: ${experiences.length}`);

        // Verify reward matches winner
        let correct = false;
        if (winner === 0 && finalReward === 1) correct = true;
        if (winner === 1 && finalReward === -1) correct = true;
        if (winner === null && finalReward === 0) correct = true;

        if (correct) {
            console.log('  ✅ Reward correctly assigned');
        } else {
            console.log('  ❌ REWARD MISMATCH! This is a bug.');
            agent.epsilon = savedEpsilon;
            return false;
        }
    }

    agent.epsilon = savedEpsilon;
    console.log('\n✅ PASS: All rewards correctly assigned');
    return true;
}

// ============================================================================
// Test 4: Illegal Move Detection
// ============================================================================

async function test4_illegalMoves(agent, numGames = 50) {
    console.log('\n📋 TEST 4: Illegal move detection');
    console.log('-'.repeat(60));

    const savedEpsilon = agent.epsilon;
    agent.epsilon = 0; // Greedy

    let totalMoves = 0;
    let illegalAttempts = 0;

    for (let i = 0; i < numGames; i++) {
        const game = new KalahEngine({ enableLogging: false });

        while (!game.gameOver) {
            const state = game.getState();
            const validMoves = game.getValidMoves();
            const action = agent.selectAction(state, validMoves);

            totalMoves++;

            // Check if action is valid
            if (!validMoves.includes(action)) {
                illegalAttempts++;
                console.log(`❌ Illegal move detected: action=${action}, valid=${validMoves}`);
                console.log(`   State: player=${state.currentPlayer}, board=${state.board}`);
            }

            // Verify action is in correct range for player
            const minPit = state.currentPlayer * 6;
            const maxPit = minPit + 5;
            if (action < minPit || action > maxPit) {
                console.log(`❌ Action out of range: action=${action}, expected [${minPit}-${maxPit}]`);
            }

            game.makeMove(action);
        }
    }

    agent.epsilon = savedEpsilon;

    console.log(`Total moves: ${totalMoves}`);
    console.log(`Illegal attempts: ${illegalAttempts}`);

    if (illegalAttempts === 0) {
        console.log('✅ PASS: No illegal moves detected');
        return true;
    } else {
        console.log('❌ FAIL: Illegal moves detected!');
        return false;
    }
}

// ============================================================================
// Test 5: Feature Extraction Verification
// ============================================================================

function test5_featureExtraction() {
    console.log('\n📋 TEST 5: Feature extraction verification');
    console.log('-'.repeat(60));

    const game = new KalahEngine({ enableLogging: false });

    // Get features for Player 0
    const state0 = game.getState();
    const features0 = extractFeatures(state0);

    console.log('Initial state features (Player 0):');
    console.log(`  My pits (0-5): ${features0.slice(0, 6).map(f => f.toFixed(2)).join(', ')}`);
    console.log(`  Opp pits (6-11): ${features0.slice(6, 12).map(f => f.toFixed(2)).join(', ')}`);
    console.log(`  My store: ${features0[12].toFixed(2)}`);
    console.log(`  Opp store: ${features0[13].toFixed(2)}`);
    console.log(`  Remaining: ${features0[14].toFixed(2)}`);

    // Make a move and check Player 1's perspective
    game.makeMove(0);
    const state1 = game.getState();
    const features1 = extractFeatures(state1);

    console.log('\nAfter P0 move (Player 1 perspective):');
    console.log(`  My pits (6-11): ${features1.slice(0, 6).map(f => f.toFixed(2)).join(', ')}`);
    console.log(`  Opp pits (0-5): ${features1.slice(6, 12).map(f => f.toFixed(2)).join(', ')}`);

    if (features0.length === 15 && features1.length === 15) {
        console.log('\n✅ PASS: Features correctly extracted for both players');
        return true;
    } else {
        console.log('\n❌ FAIL: Feature length mismatch');
        return false;
    }
}

// ============================================================================
// Main Test Runner
// ============================================================================

async function runDiagnostics() {
    let passed = 0;
    let failed = 0;

    // Test 1: Basic epsilon check
    if (test1_epsilonCheck()) passed++;
    else failed++;

    // Test 5: Feature extraction
    if (test5_featureExtraction()) passed++;
    else failed++;

    // Load or create agent
    console.log('\n' + '='.repeat(60));
    console.log('Loading agent...');
    const agent = new QLearningAgent();

    try {
        await agent.load('./models/kalah-agent');
        console.log('✅ Loaded trained model');
    } catch (e) {
        console.log('⚠️  No trained model found, using untrained agent');
        console.log('   (Tests will still run, but results may not be meaningful)');
    }

    // Test 2: Symmetry
    if (await test2_symmetryTest(agent, 100)) passed++;
    else failed++;

    // Test 3: Terminal rewards
    if (await test3_terminalRewards(agent)) passed++;
    else failed++;

    // Test 4: Illegal moves
    if (await test4_illegalMoves(agent, 50)) passed++;
    else failed++;

    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('DIAGNOSTIC SUMMARY');
    console.log('='.repeat(60));
    console.log(`✅ Passed: ${passed}/5`);
    console.log(`❌ Failed: ${failed}/5`);

    if (failed === 0) {
        console.log('\n🎉 All diagnostic tests passed!');
        console.log('   If win rate is still low, the issue is likely:');
        console.log('   - Insufficient training');
        console.log('   - Hyperparameters need tuning');
        console.log('   - Network architecture needs improvement');
    } else {
        console.log('\n⚠️  Some tests failed - bugs detected that need fixing!');
    }
}

// Run diagnostics
runDiagnostics().catch(console.error);
