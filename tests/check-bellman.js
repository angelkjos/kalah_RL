#!/usr/bin/env node

/**
 * Check if Bellman update handles player perspective correctly
 */

const QLearningAgent = require('../src/ai/rl-agent.js');
const KalahEngine = require('../src/engine/kalah-engine.js');
const { extractFeatures } = require('../src/utils/ml-examples.js');

console.log('🔍 Checking Bellman Update for Player Perspective Bug\n');

// Simulate what happens in training
function simulateUpdate() {
    const game = new KalahEngine({ enableLogging: false });

    // Player 0 makes a move
    const state = game.getState();  // Player 0's turn
    console.log('State: Player', state.currentPlayer, 'turn');

    const action = 2;  // Player 0 plays pit 2
    game.makeMove(action);

    const nextState = game.getState();  // Might be Player 1's turn (or P0 if extra turn)
    console.log('Next State: Player', nextState.currentPlayer, 'turn');

    // This is what the agent does:
    const agent = new QLearningAgent({ epsilon: 0 });

    const stateQ = agent.getQValues(state);
    const nextStateQ = agent.getQValues(nextState);

    console.log('\nCurrent state Q-values (Player', state.currentPlayer, 'perspective):');
    console.log('  ', stateQ.map(q => q.toFixed(3)).join(', '));

    console.log('\nNext state Q-values (Player', nextState.currentPlayer, 'perspective):');
    console.log('  ', nextStateQ.map(q => q.toFixed(3)).join(', '));

    // In the replay function, we do:
    const reward = 0;  // Intermediate reward
    const gamma = 0.99;
    const targetQ = reward + gamma * Math.max(...nextStateQ);

    console.log('\nTarget Q calculated as:', targetQ.toFixed(3));
    console.log('  = reward + gamma * max(nextQ)');
    console.log('  = ', reward, '+ 0.99 *', Math.max(...nextStateQ).toFixed(3));

    if (state.currentPlayer !== nextState.currentPlayer) {
        console.log('\n⚠️  PLAYER CHANGED! This is a zero-sum game.');
        console.log('   If next player expects Q =', Math.max(...nextStateQ).toFixed(3));
        console.log('   Then current player should expect Q = -', Math.max(...nextStateQ).toFixed(3));
        console.log('\n❌ BUG DETECTED: We need to negate nextQ when player changes!');
        console.log('   Correct formula should be:');
        console.log('   if (nextState.currentPlayer !== state.currentPlayer) {');
        console.log('       targetQ = reward + gamma * (-Math.max(...nextQ))');
        console.log('   }');
    } else {
        console.log('\n✅ Same player still playing (extra turn), no negation needed');
    }
}

simulateUpdate();
