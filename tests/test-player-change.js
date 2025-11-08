const KalahEngine = require('../src/engine/kalah-engine.js');
const QLearningAgent = require('../src/ai/rl-agent.js');

const game = new KalahEngine({ enableLogging: false });
const state = game.getState();
console.log('Player', state.currentPlayer, 'plays pit 0');

game.makeMove(0);  // Should NOT give extra turn

const nextState = game.getState();
console.log('Next state: Player', nextState.currentPlayer);

if (state.currentPlayer !== nextState.currentPlayer) {
    console.log('\n✅ Player changed - this is the case we need to check!');

    const agent = new QLearningAgent({epsilon: 0});
    const stateQ = agent.getQValues(state);
    const nextQ = agent.getQValues(nextState);

    console.log('\nPlayer', state.currentPlayer, 'Q-values:', stateQ.slice(0,3).map(q => q.toFixed(3)));
    console.log('Player', nextState.currentPlayer, 'Q-values:', nextQ.slice(0,3).map(q => q.toFixed(3)));

    console.log('\n❌ CRITICAL BUG FOUND!');
    console.log('   In Bellman update, we use max(nextQ) directly');
    console.log('   But nextQ is from Player ' + nextState.currentPlayer + ' perspective');
    console.log('   We are updating Player ' + state.currentPlayer + ' Q-value');
    console.log('   In a zero-sum game, we MUST NEGATE!');
    console.log('\n   Current code: targetQ = r + γ * max(nextQ)');
    console.log('   Should be:    targetQ = r + γ * (-max(nextQ))  [when player changes]');
}
