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

    // Create agent
    const agent = new QLearningAgent({
        learningRate: 0.001,
        discountFactor: 0.95,
        epsilon: 1.0,
        epsilonDecay: 0.995,
        epsilonMin: 0.1
    });

    // Create trainer
    const trainer = new Trainer(agent, {
        verbose: true,
        logInterval: 100  // Log every 100 episodes
    });

    // ===== CUSTOMIZE YOUR TRAINING HERE =====

    // Option A: Train with curriculum learning (recommended)
    console.log('\n📚 Starting curriculum learning...');
    await trainer.trainCurriculum(10000);  // 10,000 episodes total

    // Option B: Pure self-play (uncomment to use instead)
    // console.log('\n🤖 Starting self-play training...');
    // await trainer.trainSelfPlay(10000);  // 10,000 episodes

    // Option C: Against random opponent (uncomment to use instead)
    // console.log('\n🎯 Training against random opponent...');
    // await trainer.trainAgainstOpponent(10000);  // 10,000 episodes

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
