// Simple test framework
const tests = [];
let testResults = { passed: 0, failed: 0 };

function test(description, fn) {
    tests.push({ description, fn });
}

function expect(actual) {
    return {
        toBe(expected) {
            if (actual !== expected) {
                throw new Error(`Expected ${expected} but got ${actual}`);
            }
        },
        toEqual(expected) {
            if (JSON.stringify(actual) !== JSON.stringify(expected)) {
                throw new Error(`Expected ${JSON.stringify(expected)} but got ${JSON.stringify(actual)}`);
            }
        },
        toBeTruthy() {
            if (!actual) {
                throw new Error(`Expected truthy but got ${actual}`);
            }
        },
        toBeFalsy() {
            if (actual) {
                throw new Error(`Expected falsy but got ${actual}`);
            }
        }
    };
}

function runTests() {
    console.log('\n🎮 Running Kalah/Mancala Game Tests...\n');

    tests.forEach(({ description, fn }) => {
        try {
            fn();
            console.log(`✅ ${description}`);
            testResults.passed++;
        } catch (error) {
            console.log(`❌ ${description}`);
            console.log(`   ${error.message}\n`);
            testResults.failed++;
        }
    });

    console.log(`\n📊 Results: ${testResults.passed} passed, ${testResults.failed} failed\n`);
}

// Mock DOM for testing
class MockElement {
    constructor() {
        this.textContent = '';
        this.classList = {
            items: [],
            add(className) {
                if (!this.items.includes(className)) this.items.push(className);
            },
            remove(className) {
                this.items = this.items.filter(c => c !== className);
            }
        };
        this.dataset = {};
        this.eventListeners = {};
    }

    addEventListener(event, handler) {
        this.eventListeners[event] = handler;
    }
}

global.document = {
    elements: {},
    getElementById(id) {
        if (!this.elements[id]) {
            this.elements[id] = new MockElement();
        }
        return this.elements[id];
    },
    querySelector(selector) {
        if (!this.elements[selector]) {
            this.elements[selector] = new MockElement();
        }
        return this.elements[selector];
    },
    querySelectorAll(selector) {
        // Return array of mock elements for pits
        if (selector === '.pit') {
            const pits = [];
            for (let i = 0; i < 12; i++) {
                const pit = new MockElement();
                pit.dataset.pit = i.toString();
                pits.push(pit);
            }
            return pits;
        }
        return [];
    }
};

global.window = {
    addEventListener() {}
};

global.alert = () => {}; // Mock alert

// Load the game engine
const KalahEngine = require('../src/engine/kalah-engine.js');

// ============ TESTS ============

test('Game initializes with correct starting state', () => {
    const game = new KalahEngine();
    expect(game.board).toEqual([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]);
    expect(game.stores).toEqual([0, 0]);
    expect(game.currentPlayer).toBe(0);
    expect(game.gameOver).toBe(false);
});

test('Basic move distributes seeds correctly', () => {
    const game = new KalahEngine();
    // Player 0 plays pit 0 (has 4 seeds)
    game.makeMove(0);

    // Seeds go to pits 1, 2, 3, 4
    expect(game.board[0]).toBe(0); // Empty
    expect(game.board[1]).toBe(5); // 4 + 1
    expect(game.board[2]).toBe(5);
    expect(game.board[3]).toBe(5);
    expect(game.board[4]).toBe(5);
    expect(game.stores[0]).toBe(0); // No store reached
});

test('Seeds drop into own store during sowing', () => {
    const game = new KalahEngine();
    // Player 1 plays pit 5 (2 seeds will reach store)
    game.board[5] = 2;
    game.makeMove(5);

    // 1 seed in store, 1 seed in pit 6
    expect(game.stores[0]).toBe(1);
    expect(game.board[6]).toBe(5); // Originally 4 + 1
});

test('Player gets extra turn when landing in own store', () => {
    const game = new KalahEngine();
    // Set up so Player 1's pit 5 has exactly 1 seed (will land in store)
    game.board[5] = 1;
    const currentPlayer = game.currentPlayer;
    game.makeMove(5);

    // Player should still be the same (extra turn)
    expect(game.currentPlayer).toBe(currentPlayer);
    expect(game.stores[0]).toBe(1);
});

test('Seeds skip opponent store', () => {
    const game = new KalahEngine();
    game.currentPlayer = 1; // Player 2's turn
    // Player 2 plays pit 11 with 2 seeds - should go to store, then pit 0
    game.board[11] = 2;
    game.makeMove(11);

    expect(game.stores[1]).toBe(1); // Player 2's store gets 1
    expect(game.board[0]).toBe(5); // Pit 0 gets 1 (originally 4)
    expect(game.stores[0]).toBe(0); // Player 1's store is skipped
});

test('Capture works when landing in empty pit on own side', () => {
    const game = new KalahEngine();
    // Set up: Player 1's pit 2 is empty, pit 1 has 1 seed
    // Opponent's pit 9 (opposite of pit 2: 11-2=9) has 5 seeds
    game.board[1] = 1;
    game.board[2] = 0;
    game.board[9] = 5;

    game.makeMove(1); // 1 seed from pit 1 goes to pit 2 (empty)

    // Should capture pit 2 + pit 9
    expect(game.board[2]).toBe(0); // Captured
    expect(game.board[9]).toBe(0); // Captured
    expect(game.stores[0]).toBe(6); // 1 from pit 2 + 5 from pit 9
});

test('No capture when opposite pit is empty', () => {
    const game = new KalahEngine();
    // Set up: Player 1's pit 2 is empty, pit 1 has 1 seed
    // Opponent's pit 9 is also empty
    game.board[1] = 1;
    game.board[2] = 0;
    game.board[9] = 0;

    game.makeMove(1); // 1 seed from pit 1 goes to pit 2 (empty)

    // Should NOT capture because opposite pit is empty
    expect(game.board[2]).toBe(1); // Seed stays
    expect(game.stores[0]).toBe(0); // No capture
});

test('No capture when landing on opponent side', () => {
    const game = new KalahEngine();
    // Player 1 plays from pit 2 with 5 seeds
    // Path: 3,4,5,store,6 - lands in pit 6 (opponent)
    for (let i = 0; i < 12; i++) game.board[i] = 0;
    game.board[2] = 5;
    game.board[6] = 0; // opponent pit is empty

    game.makeMove(2);

    // No capture because pit 6 is opponent's side
    expect(game.board[6]).toBe(1);
    expect(game.stores[0]).toBe(1); // Just the store seed
});

test('Multi-lap: starting pit is skipped first round, gets seed on subsequent rounds', () => {
    const game = new KalahEngine();
    // With 13 seeds from pit 0: 1-5(5),store(1),6-11(6),skip 0,1(1) = 13 total
    // Verify pit 0 was skipped (stays 0) and pit 1 gets 2 seeds (4+1+1)
    game.board[0] = 13;
    game.makeMove(0);

    // Pit 0 should be 0 (was skipped on first round, not reached on second)
    expect(game.board[0]).toBe(0);
    // Pit 1 should have 6 seeds (4 original + 1 first round + 1 second round)
    expect(game.board[1]).toBe(6);
});

test('Multi-lap: starting pit is skipped on first round only', () => {
    const game = new KalahEngine();
    // 13 seeds from pit 0: pits 1-11 (11 seeds), store (1 seed), skip pit 0, pit 1 (1 seed)
    game.board[0] = 13;
    game.makeMove(0);

    expect(game.board[1]).toBe(6); // Originally 4 + 1 + 1 (second round)
    expect(game.stores[0]).toBe(1); // Store gets 1 seed
});

test('Game ends when one side is empty', () => {
    const game = new KalahEngine();
    // Clear Player 0's side
    game.board = [0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4];
    game._checkAndHandleGameOver();

    expect(game.gameOver).toBe(true);
});

test('Remaining seeds collected when game ends', () => {
    const game = new KalahEngine();
    // Clear Player 0's side, give them some store seeds
    game.board = [0, 0, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4];
    game.stores = [10, 5];
    game._checkAndHandleGameOver();

    // Player 1 should collect their remaining 24 seeds
    expect(game.stores[1]).toBe(29); // 5 + 24
    expect(game.stores[0]).toBe(10); // Unchanged
});

test('Invalid move on opponent pit is rejected', () => {
    const game = new KalahEngine();
    game.currentPlayer = 0; // Player 0
    const result = game.isValidMove(6); // Try to play opponent's pit

    expect(result).toBe(false);
});

test('Invalid move on empty pit is rejected', () => {
    const game = new KalahEngine();
    game.board[0] = 0;
    const result = game.isValidMove(0);

    expect(result).toBe(false);
});

test('Reset resets game to initial state', () => {
    const game = new KalahEngine();
    game.board = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    game.stores = [10, 15];
    game.currentPlayer = 1;
    game.gameOver = true;

    game.reset();

    expect(game.board).toEqual([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]);
    expect(game.stores).toEqual([0, 0]);
    expect(game.currentPlayer).toBe(0);
    expect(game.gameOver).toBe(false);
});

test('Player 2 store works correctly', () => {
    const game = new KalahEngine();
    game.currentPlayer = 1; // Player 1
    game.board[11] = 1; // 1 seed in last pit

    game.makeMove(11);

    // Should land in Player 1's store
    expect(game.stores[1]).toBe(1);
    expect(game.currentPlayer).toBe(1); // Extra turn
});

test('Capture calculation uses correct opposite pit formula', () => {
    const game = new KalahEngine();
    // Simple test: pit 2 empty, pit 1 has 1 seed, pit 9 (opposite of 2) has 3 seeds
    // Seed from pit 1 lands in pit 2, should capture 1+3=4
    for (let i = 0; i < 12; i++) game.board[i] = 0;
    game.board[1] = 1;
    game.board[9] = 3;
    game.stores = [0, 0];

    game.makeMove(1);

    expect(game.stores[0]).toBe(4); // 1 (from pit 2) + 3 (from pit 9)
    expect(game.board[2]).toBe(0);
    expect(game.board[9]).toBe(0);
});

test('Winner is determined correctly', () => {
    const game = new KalahEngine();
    game.board = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0];
    game.stores = [20, 15];

    game._checkAndHandleGameOver();

    expect(game.gameOver).toBe(true);
    expect(game.stores[0]).toBe(20); // Player 0 wins
    expect(game.stores[1]).toBe(16); // Player 1 gets remaining seed
});

test('Game ends when a player reaches 25 or more seeds in their store', () => {
    const game = new KalahEngine();
    // Set Player 0's store to 24, Player 1's to 10
    game.stores = [24, 10];
    // Set up board so Player 0 can make a move that adds 1 seed to their store
    // Player 0 plays pit 0 with 1 seed, landing in their store
    game.board = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    game.currentPlayer = 0;

    game.makeMove(0); // Player 0 moves 1 seed from pit 0 to their store

    expect(game.gameOver).toBe(true);
    expect(game.stores[0]).toBe(25);
    expect(game.getWinner()).toBe(0); // Player 0 should be the winner
});

// Run all tests
runTests();
