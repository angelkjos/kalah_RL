# 🚀 Starting and Stopping the Server

## Starting the Server

### Option 1: Using npm (Recommended)
```bash
npm run serve
```

### Option 2: Direct command
```bash
node server.js
```

You'll see:
```
🎮 Kalah Game Server Running!
📡 Server: http://localhost:8080
🤖 RL Agent Model: Ready to load

🌐 Open in browser: http://localhost:8080

Press Ctrl+C to stop the server
```

## Stopping the Server

### Option 1: If server is running in your terminal
Press **Ctrl+C** (or **Cmd+C** on Mac)

### Option 2: If server is running in background
Find and kill the process:
```bash
# Find the process ID
lsof -ti:8080

# Kill it
lsof -ti:8080 | xargs kill
```

### Option 3: Kill all node servers (nuclear option)
```bash
killall node
```

## Checking if Server is Running

```bash
# Check if port 8080 is in use
lsof -i:8080

# Or try to access it
curl -I http://localhost:8080
```

If you see "Connection refused" - server is NOT running  
If you see "HTTP/1.1 200 OK" - server IS running ✅

## Common Issues

### "Port already in use" / "EADDRINUSE"
**Problem:** Another server is already using port 8080

**Solution:**
```bash
# Kill the existing server
lsof -ti:8080 | xargs kill

# Then start again
npm run serve
```

### Server won't stop with Ctrl+C
**Problem:** Process is stuck

**Solution:**
```bash
# Force kill
lsof -ti:8080 | xargs kill -9
```

## Quick Reference

| Action | Command |
|--------|---------|
| **Start** | `npm run serve` |
| **Stop** | `Ctrl+C` (in terminal) |
| **Force stop** | `lsof -ti:8080 \| xargs kill` |
| **Check status** | `lsof -i:8080` |
| **Access** | `http://localhost:8080` |

