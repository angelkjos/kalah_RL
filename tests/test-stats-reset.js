#!/usr/bin/env node

/**
 * Test that stats reset between training runs
 */

const QLearningAgent = require('../src/ai/rl-agent.js');
const Trainer = require('../src/ai/trainer.js');

async function testStatsReset() {
    console.log('🔍 Testing stats reset between training runs...\n');

    const agent = new QLearningAgent({
        epsilon: 0.1,
        learningRate: 0.001
    });

    const trainer = new Trainer(agent, {
        verbose: false,
        logInterval: 1000
    });

    // First training run
    console.log('Training run #1: 50 games...');
    await trainer.trainAgainstOpponent(50);
    const stats1 = { ...trainer.stats };
    console.log(`Stats after run #1: ${stats1.wins}W ${stats1.losses}L ${stats1.draws}D (Total: ${stats1.gamesPlayed})`);

    // Second training run
    console.log('\nTraining run #2: 30 games...');
    await trainer.trainAgainstOpponent(30);
    const stats2 = { ...trainer.stats };
    console.log(`Stats after run #2: ${stats2.wins}W ${stats2.losses}L ${stats2.draws}D (Total: ${stats2.gamesPlayed})`);

    // Check that stats were reset
    console.log('\n' + '='.repeat(60));
    if (stats2.gamesPlayed === 30) {
        console.log('✅ PASS: Stats correctly reset between training runs');
        console.log(`   Run #1 had ${stats1.gamesPlayed} games, Run #2 has ${stats2.gamesPlayed} games (not ${stats1.gamesPlayed + 30})`);
    } else {
        console.log('❌ FAIL: Stats accumulated instead of resetting');
        console.log(`   Expected 30 games, got ${stats2.gamesPlayed}`);
    }
    console.log('='.repeat(60));
}

async function testCurriculumStats() {
    console.log('\n\n🔍 Testing curriculum learning stats...\n');

    const agent = new QLearningAgent({
        epsilon: 0.5,
        learningRate: 0.001
    });

    const trainer = new Trainer(agent, {
        verbose: false,
        logInterval: 1000
    });

    // Curriculum training
    console.log('Curriculum training: 150 games (30% + 40% + 30%)...');
    await trainer.trainCurriculum(150);
    const stats = { ...trainer.stats };
    console.log(`Stats after curriculum: ${stats.wins}W ${stats.losses}L ${stats.draws}D (Total: ${stats.gamesPlayed})`);

    console.log('\n' + '='.repeat(60));
    if (stats.gamesPlayed === 150) {
        console.log('✅ PASS: Curriculum stats are cumulative across all stages');
        console.log(`   Total games: ${stats.gamesPlayed} (expected 150)`);
    } else {
        console.log('❌ FAIL: Curriculum stats are not cumulative');
        console.log(`   Expected 150 games, got ${stats.gamesPlayed}`);
    }
    console.log('='.repeat(60));
}

async function main() {
    console.log('=' .repeat(60));
    console.log('STATS RESET TEST');
    console.log('='.repeat(60) + '\n');

    await testStatsReset();
    await testCurriculumStats();

    console.log('\n✅ All tests complete!\n');
}

main().catch(console.error);
