#!/usr/bin/env node

/**
 * Simple CLI Demo of KalahEngine
 *
 * Run: node simple-demo.js
 */

const KalahEngine = require('../engine/kalah-engine.js');

console.log('🎮 Kalah/Mancala Engine Demo\n');

// Create a new game with logging enabled
const game = new KalahEngine({
    pitsPerPlayer: 6,
    seedsPerPit: 4,
    enableLogging: true
});

console.log('Initial state:');
console.log(game.toString());
console.log('\n');

// Player 0 makes some moves
console.log('Player 0 plays pit 2:');
let result = game.makeMove(2);
console.log('Result:', {
    captured: result.captured,
    extraTurn: result.extraTurn,
    gameEnded: result.gameEnded
});
console.log('\n' + game.toString());
console.log('\n');

// Check who's turn it is and make a valid move
console.log(`Current player is: ${game.currentPlayer}`);
const validMoves = game.getValidMoves();
console.log(`Valid moves: ${validMoves}`);
const nextMove = validMoves[0];
console.log(`Player ${game.currentPlayer} plays pit ${nextMove}:`);
result = game.makeMove(nextMove);
console.log('Result:', {
    captured: result.captured,
    extraTurn: result.extraTurn,
    gameEnded: result.gameEnded
});
console.log('\n' + game.toString());
console.log('\n');

// Show valid moves
console.log(`Valid moves for Player ${game.currentPlayer}:`, game.getValidMoves());
console.log('\n');

// Show state object
console.log('Current state object:');
console.log(JSON.stringify(game.getState(), null, 2));
console.log('\n');

// Clone and make a speculative move
console.log('Testing clone functionality...');
const cloned = game.clone();
console.log('Making speculative move on clone (pit 3)...');
cloned.makeMove(3);
console.log('\nCloned game state:');
console.log(cloned.toString());
console.log('\nOriginal game (unchanged):');
console.log(game.toString());
console.log('\n');

// Play random moves until game ends
console.log('Playing random moves until game ends...\n');
game.enableLogging = false;  // Disable verbose logging for auto-play

let moveCount = 0;
const maxMoves = 200;

while (!game.gameOver && moveCount < maxMoves) {
    const moves = game.getValidMoves();
    const randomMove = moves[Math.floor(Math.random() * moves.length)];
    const result = game.makeMove(randomMove);

    console.log(`Move ${game.moveNumber}: Player ${result.player} plays pit ${randomMove} - ` +
                `Captured: ${result.captured}, Extra turn: ${result.extraTurn}`);

    moveCount++;
}

console.log('\n' + game.toString());
console.log('\n');

// Show final results
const winner = game.getWinner();
if (winner === null) {
    console.log('🤝 Game ended in a draw!');
} else {
    console.log(`🏆 Player ${winner} wins!`);
}

console.log(`Final scores: Player 0: ${game.getScore(0)}, Player 1: ${game.getScore(1)}`);
console.log(`Total moves: ${game.moveNumber}`);
console.log('\n✅ Demo complete!');
