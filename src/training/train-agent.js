#!/usr/bin/env node

/**
 * Configurable Training Script with Checkpointing
 *
 * Usage:
 *   node src/training/train-agent.js [episodes] [evalInterval]
 *
 * Examples:
 *   node src/training/train-agent.js            # Default: 50k episodes, eval every 5k
 *   node src/training/train-agent.js 30000      # 30k episodes, eval every 5k
 *   node src/training/train-agent.js 30000 3000 # 30k episodes, eval every 3k
 */

const QLearningAgent = require('../ai/rl-agent.js');
const Trainer = require('../ai/trainer.js');

async function trainAgent() {
    // Parse command line arguments
    const args = process.argv.slice(2);
    const numEpisodes = parseInt(args[0]) || 50000;
    const evalInterval = parseInt(args[1]) || 5000;

    console.log('🎮 Training Kalah/Mancala RL Agent\n');
    console.log(`Episodes: ${numEpisodes}`);
    console.log(`Eval interval: ${evalInterval}`);
    console.log(`Architecture: 15 → 64 → 64 → 32 → 6`);

    // Create agent with DQN hyperparameters
    const agent = new QLearningAgent({
        learningRate: 0.001,
        learningRateEnd: 0.0005,
        discountFactor: 0.99,
        epsilon: 1.0,
        epsilonMin: 0.05,
        epsilonDecaySteps: numEpisodes,  // Decay over full training
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

    // Train with checkpointing
    console.log('\n🤖 Starting self-play training with checkpointing...');
    const results = await trainer.trainSelfPlay(numEpisodes, {
        evalInterval,
        evalGames: 200,
        savePath: './checkpoints',
        keepBest: true
    });

    // Copy best checkpoint to final model location
    const fs = require('fs');
    const path = require('path');

    console.log('\n📦 Copying best checkpoint to ./models/kalah-agent...');

    const sourcePath = './checkpoints/best-checkpoint';
    const targetPath = './models/kalah-agent';

    // Create models directory
    if (!fs.existsSync('./models')) {
        fs.mkdirSync('./models', { recursive: true });
    }

    // Copy model files
    if (fs.existsSync(sourcePath)) {
        // Copy model.json
        fs.copyFileSync(
            path.join(sourcePath, 'model.json'),
            path.join(targetPath, 'model.json')
        );

        // Copy metadata
        if (fs.existsSync(path.join(sourcePath, 'metadata.json'))) {
            fs.copyFileSync(
                path.join(sourcePath, 'metadata.json'),
                path.join(targetPath, 'metadata.json')
            );
        }

        console.log('✅ Best model copied to ./models/kalah-agent');
    } else {
        console.log('⚠️  No checkpoint found, saving current model...');
        await agent.save(targetPath);
    }

    console.log('\n' + '='.repeat(70));
    console.log('🎉 Training Complete!');
    console.log('='.repeat(70));
    console.log(`🏆 Best model: Episode ${results.bestEpisode} (${results.bestWinRate.toFixed(1)}% win rate)`);
    console.log(`📁 Model location: ./models/kalah-agent`);
    console.log(`📊 Checkpoints: ./checkpoints`);
    console.log('\n💡 To play against this agent:');
    console.log('   Open http://localhost:8080 and select "Hard (RL Agent)"');
    console.log('   Or run: npm run rl-demo');
    console.log('='.repeat(70));
}

trainAgent().catch(console.error);
