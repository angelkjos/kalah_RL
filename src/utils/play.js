#!/usr/bin/env node

/**
 * Quick Play Script
 *
 * Play against a trained agent (or train a quick one if none exists)
 */

const QLearningAgent = require('../ai/rl-agent.js');
const Trainer = require('../ai/trainer.js');
const KalahEngine = require('../engine/kalah-engine.js');
const readline = require('readline');
const fs = require('fs');

async function main() {
    console.log('🎮 Kalah/Mancala - Play Against AI\n');

    const agent = new QLearningAgent();
    const modelPath = './models/kalah-agent';

    // Try to load existing model
    if (fs.existsSync(modelPath + '/model.json')) {
        console.log('📦 Loading trained model...');
        await agent.load(modelPath);
        console.log('✅ Model loaded!\n');
    } else {
        console.log('No trained model found. Training a quick agent...');
        console.log('(For a stronger agent, run: npm run train)\n');

        const trainer = new Trainer(agent, { verbose: true, logInterval: 50 });
        await trainer.trainAgainstOpponent(200);

        console.log('\n💾 Saving model for next time...');
        if (!fs.existsSync('./models')) {
            fs.mkdirSync('./models');
        }
        await agent.save(modelPath);
    }

    await playGame(agent);
}

async function playGame(agent) {
    console.log('\n' + '='.repeat(60));
    console.log('🎲 Starting Game!');
    console.log('='.repeat(60));
    console.log('You are Player 0 (bottom row, pits 0-5)');
    console.log('AI is Player 1 (top row, pits 6-11)');
    console.log('Enter pit numbers when prompted.');
    console.log('='.repeat(60) + '\n');

    const game = new KalahEngine({ enableLogging: false });
    agent.epsilon = 0; // No exploration during play

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    const playTurn = () => {
        // Show board
        console.log('\n' + game.toString());

        if (game.gameOver) {
            const winner = game.getWinner();
            console.log('\n' + '='.repeat(60));
            if (winner === null) {
                console.log('🤝 Game ended in a draw!');
            } else if (winner === 0) {
                console.log('🎉 YOU WIN! Congratulations!');
            } else {
                console.log('🤖 AI WINS! Better luck next time!');
            }
            console.log('='.repeat(60));
            console.log(`Final scores: You: ${game.getScore(0)}, AI: ${game.getScore(1)}`);
            console.log('\nWant to play again? Just run: npm run play');
            rl.close();
            return;
        }

        const state = game.getState();
        const validMoves = game.getValidMoves();

        if (state.currentPlayer === 0) {
            // Human's turn
            console.log(`\n🎯 Your turn! Valid moves: ${validMoves.join(', ')}`);
            rl.question('Enter pit number: ', (input) => {
                const pit = parseInt(input);

                if (isNaN(pit)) {
                    console.log('❌ Please enter a number.');
                    playTurn();
                    return;
                }

                if (!validMoves.includes(pit)) {
                    console.log(`❌ Invalid move. Choose from: ${validMoves.join(', ')}`);
                    playTurn();
                    return;
                }

                const result = game.makeMove(pit);

                if (result.extraTurn) {
                    console.log('⭐ Extra turn! You get to play again.');
                }
                if (result.captured > 0) {
                    console.log(`💰 Captured ${result.captured} seeds!`);
                }

                playTurn();
            });
        } else {
            // AI's turn
            console.log('\n🤖 AI is thinking...');

            setTimeout(() => {
                const action = agent.selectAction(state, validMoves);
                console.log(`🎯 AI plays pit ${action}`);

                const result = game.makeMove(action);

                if (result.extraTurn) {
                    console.log('⭐ AI gets an extra turn!');
                }
                if (result.captured > 0) {
                    console.log(`💰 AI captured ${result.captured} seeds!`);
                }

                playTurn();
            }, 800);
        }
    };

    playTurn();
}

main().catch(console.error);
