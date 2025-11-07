#!/usr/bin/env node

/**
 * RL Agent Demo
 *
 * Demonstrates training and playing with the Q-Learning agent
 */

const QLearningAgent = require('../ai/rl-agent.js');
const Trainer = require('../ai/trainer.js');
const KalahEngine = require('../engine/kalah-engine.js');
const readline = require('readline');

async function main() {
    console.log('🎮 Kalah/Mancala Q-Learning Agent Demo\n');

    // Create agent with improved DQN hyperparameters
    console.log('Creating DQN agent...');
    const agent = new QLearningAgent({
        learningRate: 0.001,
        learningRateEnd: 0.0005,
        discountFactor: 0.99,
        epsilon: 1.0,
        epsilonMin: 0.05,
        epsilonDecaySteps: 50000,
        replayBufferSize: 100000,
        batchSize: 64,
        targetUpdateFreq: 1000,
        gradientClipValue: 1.0
    });

    // Create trainer
    const trainer = new Trainer(agent, {
        verbose: true,
        logInterval: 500
    });

    // Show menu
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    const showMenu = () => {
        console.log('\n' + '='.repeat(50));
        console.log('What would you like to do?');
        console.log('='.repeat(50));
        console.log('1. Train via self-play (10k episodes) ⭐ RECOMMENDED');
        console.log('2. Train against random opponent (10k episodes)');
        console.log('3. Evaluate agent vs random opponent');
        console.log('4. Play against the agent');
        console.log('5. Save model');
        console.log('6. Load model');
        console.log('7. Exit');
        console.log('='.repeat(50));
    };

    const handleChoice = async (choice) => {
        switch (choice) {
            case '1':
                console.log('\n🤖 Training via self-play...');
                await trainer.trainSelfPlay(10000);
                showMenu();
                askChoice();
                break;

            case '2':
                console.log('\n🎯 Training against random opponent...');
                await trainer.trainAgainstOpponent(10000);
                showMenu();
                askChoice();
                break;

            case '3':
                await trainer.evaluate(100);
                showMenu();
                askChoice();
                break;

            case '4':
                await playAgainstAgent(agent);
                showMenu();
                askChoice();
                break;

            case '5':
                console.log('💾 Saving model...');
                await agent.save('./models/kalah-agent');
                showMenu();
                askChoice();
                break;

            case '6':
                try {
                    console.log('📦 Loading model...');
                    await agent.load('./models/kalah-agent');
                } catch (e) {
                    console.log('❌ Could not load model. Train a model first!');
                }
                showMenu();
                askChoice();
                break;

            case '7':
                console.log('\n👋 Goodbye!');
                rl.close();
                process.exit(0);
                break;

            default:
                console.log('Invalid choice. Please try again.');
                askChoice();
        }
    };

    const askChoice = () => {
        rl.question('\nEnter your choice (1-7): ', handleChoice);
    };

    // Quick training option for demo
    if (process.argv.includes('--quick-train')) {
        console.log('Running quick training demo...\n');
        await trainer.trainAgainstOpponent(200);
        await trainer.evaluate(50);
        console.log('\n✅ Quick training complete! Model is ready to use.');
        rl.close();
        process.exit(0);
    }

    showMenu();
    askChoice();
}

/**
 * Play an interactive game against the agent
 */
async function playAgainstAgent(agent) {
    console.log('\n🎲 Playing against the AI agent...');
    console.log('You are Player 0 (bottom row, pits 0-5)');
    console.log('AI is Player 1 (top row, pits 6-11)\n');

    const game = new KalahEngine({ enableLogging: false });
    const savedEpsilon = agent.epsilon;
    agent.epsilon = 0; // No exploration during play

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    const playTurn = () => {
        console.log('\n' + game.toString());

        if (game.gameOver) {
            const winner = game.getWinner();
            if (winner === null) {
                console.log('\n🤝 Game ended in a draw!');
            } else if (winner === 0) {
                console.log('\n🎉 You win!');
            } else {
                console.log('\n🤖 AI wins!');
            }
            rl.close();
            agent.epsilon = savedEpsilon;
            return;
        }

        const state = game.getState();
        const validMoves = game.getValidMoves();

        if (state.currentPlayer === 0) {
            // Human's turn
            console.log(`\nYour turn! Valid moves: ${validMoves.join(', ')}`);
            rl.question('Enter pit number: ', (input) => {
                const pit = parseInt(input);
                if (validMoves.includes(pit)) {
                    const result = game.makeMove(pit);
                    if (result.extraTurn) {
                        console.log('⭐ Extra turn! You get to play again.');
                    }
                    if (result.captured > 0) {
                        console.log(`💰 Captured ${result.captured} seeds!`);
                    }
                    playTurn();
                } else {
                    console.log('❌ Invalid move. Try again.');
                    playTurn();
                }
            });
        } else {
            // AI's turn
            console.log('\n🤖 AI is thinking...');
            setTimeout(() => {
                const action = agent.selectAction(state, validMoves);
                console.log(`AI plays pit ${action}`);
                const result = game.makeMove(action);
                if (result.extraTurn) {
                    console.log('⭐ AI gets an extra turn!');
                }
                if (result.captured > 0) {
                    console.log(`💰 AI captured ${result.captured} seeds!`);
                }
                playTurn();
            }, 500);
        }
    };

    playTurn();
}

// Run the demo
main().catch(console.error);
