#!/usr/bin/env node

/**
 * Custom Training Script
 *
 * Train an agent with custom number of episodes and save the model
 */

const QLearningAgent = require('../ai/rl-agent.js');
const Trainer = require('../ai/trainer.js');

async function trainAgent() {
    console.log('🎮 Training Kalah/Mancala RL Agent\n');

    // Create agent with improved DQN hyperparameters
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
        logInterval: 500  // Log every 500 episodes
    });

    // ===== CUSTOMIZE YOUR TRAINING HERE =====

    // Self-play training (recommended for DQN)
    // Agent plays against itself, learning from both perspectives
    console.log('\n🤖 Starting self-play training...');
    await trainer.trainSelfPlay(50000);  // 50,000 episodes for full ε decay

    // Alternative: Train against random opponent (faster, but less strategic)
    // console.log('\n🎯 Training against random opponent...');
    // await trainer.trainAgainstOpponent(50000);

    // =========================================

    // Evaluate the trained agent
    console.log('\n📊 Evaluating trained agent...');
    await trainer.evaluate(100);

    // Save the model
    const modelPath = './models/kalah-agent';
    console.log(`\n💾 Saving model to ${modelPath}...`);

    const fs = require('fs');
    if (!fs.existsSync('./models')) {
        fs.mkdirSync('./models');
    }

    await agent.save(modelPath);

    console.log('\n✅ Training complete! Model saved.');
    console.log('\nTo play against this agent, run:');
    console.log('  npm run rl-demo');
    console.log('  Then select option 7 to load the model');
    console.log('  Then select option 5 to play!');
}

trainAgent().catch(console.error);
