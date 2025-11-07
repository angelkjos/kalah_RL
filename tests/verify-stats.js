#!/usr/bin/env node

/**
 * Verify training statistics are calculated correctly
 */

const KalahEngine = require('../src/engine/kalah-engine.js');
const QLearningAgent = require('../src/ai/rl-agent.js');
const Trainer = require('../src/ai/trainer.js');

async function testEmptyPitFiltering() {
    console.log('\n🔍 Test 1: Empty pit filtering...');

    const game = new KalahEngine({ enableLogging: false });

    // Empty some pits manually
    game.board[0] = 0;
    game.board[1] = 0;
    game.board[2] = 0;

    const validMoves = game.getValidMoves();
    console.log('Board state:', game.board.slice(0, 6));
    console.log('Valid moves for Player 0:', validMoves);
    console.log('✅ Empty pits correctly filtered:', !validMoves.includes(0) && !validMoves.includes(1) && !validMoves.includes(2));
}

async function testWinnerCalculation() {
    console.log('\n🔍 Test 2: Winner calculation...');

    // Test win
    const game1 = new KalahEngine({ enableLogging: false });
    game1.stores = [30, 18];
    game1.board.fill(0);
    game1.gameOver = true;
    console.log('Game 1 - Player 0: 30, Player 1: 18');
    console.log('Winner:', game1.getWinner(), '(expected: 0)');
    console.log('✅ Player 0 wins:', game1.getWinner() === 0);

    // Test loss
    const game2 = new KalahEngine({ enableLogging: false });
    game2.stores = [18, 30];
    game2.board.fill(0);
    game2.gameOver = true;
    console.log('\nGame 2 - Player 0: 18, Player 1: 30');
    console.log('Winner:', game2.getWinner(), '(expected: 1)');
    console.log('✅ Player 1 wins:', game2.getWinner() === 1);

    // Test draw
    const game3 = new KalahEngine({ enableLogging: false });
    game3.stores = [24, 24];
    game3.board.fill(0);
    game3.gameOver = true;
    console.log('\nGame 3 - Player 0: 24, Player 1: 24');
    console.log('Winner:', game3.getWinner(), '(expected: null for draw)');
    console.log('✅ Draw detected:', game3.getWinner() === null);
}

async function testStatsCounting() {
    console.log('\n🔍 Test 3: Stats counting during training...');

    const agent = new QLearningAgent({
        epsilon: 0.1, // Low exploration for faster testing
        learningRate: 0.001
    });

    const trainer = new Trainer(agent, {
        verbose: false,
        logInterval: 1000
    });

    console.log('Training agent vs random for 100 games...');
    await trainer.trainAgainstOpponent(100);

    const { wins, losses, draws, gamesPlayed } = trainer.stats;
    const total = wins + losses + draws;

    console.log('\nStats after 100 games:');
    console.log(`Games played: ${gamesPlayed}`);
    console.log(`Wins: ${wins}`);
    console.log(`Losses: ${losses}`);
    console.log(`Draws: ${draws}`);
    console.log(`Total: ${total}`);
    console.log(`\n✅ All games accounted for: ${total === gamesPlayed && gamesPlayed === 100}`);

    if (total !== gamesPlayed) {
        console.log(`❌ ERROR: Total (${total}) doesn't match gamesPlayed (${gamesPlayed})`);
    }
}

async function testAgentSelectsValidMoves() {
    console.log('\n🔍 Test 4: Agent only selects valid moves...');

    const agent = new QLearningAgent();
    const game = new KalahEngine({ enableLogging: false });

    // Empty some pits
    game.board[0] = 0;
    game.board[2] = 0;
    game.board[4] = 0;

    const validMoves = game.getValidMoves();
    console.log('Valid moves:', validMoves);

    // Test 20 selections
    let allValid = true;
    for (let i = 0; i < 20; i++) {
        const state = game.getState();
        const action = agent.selectAction(state, validMoves);
        if (!validMoves.includes(action)) {
            console.log(`❌ Invalid action selected: ${action}`);
            allValid = false;
        }
    }

    console.log(`✅ Agent only selects valid moves: ${allValid}`);
}

async function main() {
    console.log('=' .repeat(60));
    console.log('KALAH RL STATISTICS VERIFICATION');
    console.log('='.repeat(60));

    await testEmptyPitFiltering();
    await testWinnerCalculation();
    await testAgentSelectsValidMoves();
    await testStatsCounting();

    console.log('\n' + '='.repeat(60));
    console.log('✅ VERIFICATION COMPLETE');
    console.log('='.repeat(60) + '\n');
}

main().catch(console.error);
