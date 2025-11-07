#!/usr/bin/env node

/**
 * Simple local web server for Kalah game
 * Serves files so the browser can load the RL agent model
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 8080;

// MIME types for different file extensions
const MIME_TYPES = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.svg': 'image/svg+xml'
};

const server = http.createServer((req, res) => {
    // Parse URL and remove query strings
    let filePath = '';

    // Determine which directory to serve from
    if (req.url === '/') {
        filePath = './public/index.html';
    } else if (req.url.startsWith('/models/')) {
        // Allow access to models directory
        filePath = '.' + req.url;
    } else {
        // Serve from public directory
        filePath = './public' + req.url;
    }

    // Get file extension for MIME type
    const extname = String(path.extname(filePath)).toLowerCase();
    const mimeType = MIME_TYPES[extname] || 'application/octet-stream';

    // Read and serve the file
    fs.readFile(filePath, (error, content) => {
        if (error) {
            if (error.code === 'ENOENT') {
                // File not found
                res.writeHead(404, { 'Content-Type': 'text/html' });
                res.end('<h1>404 - File Not Found</h1>', 'utf-8');
            } else {
                // Server error
                res.writeHead(500);
                res.end(`Server Error: ${error.code}`, 'utf-8');
            }
        } else {
            // Success - serve the file
            res.writeHead(200, { 'Content-Type': mimeType });
            res.end(content, 'utf-8');
        }
    });
});

server.listen(PORT, () => {
    console.log('🎮 Kalah Game Server Running!');
    console.log(`📡 Server: http://localhost:${PORT}`);
    console.log(`🤖 RL Agent Model: Ready to load`);
    console.log('');
    console.log('🌐 Open in browser: http://localhost:8080');
    console.log('');
    console.log('Press Ctrl+C to stop the server');
});
