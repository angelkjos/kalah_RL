#!/bin/bash

# Deploy Kalah RL to GitHub Pages
# Usage: ./deploy-to-pages.sh [commit-message]

set -e  # Exit on error

# Configuration
PAGES_REPO="$HOME/projects/angelkjos.github.io"
DEPLOY_DIR="kalah-rl"
SOURCE_DIR="./public"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}📦 Deploying Kalah RL to GitHub Pages...${NC}\n"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}❌ Error: Source directory '$SOURCE_DIR' not found${NC}"
    exit 1
fi

# Check if pages repo exists
if [ ! -d "$PAGES_REPO" ]; then
    echo -e "${RED}❌ Error: GitHub Pages repo not found at '$PAGES_REPO'${NC}"
    echo "Please clone your repository first:"
    echo "  git clone https://github.com/angelkjos/angelkjos.github.io.git $PAGES_REPO"
    exit 1
fi

# Create deploy directory if it doesn't exist
echo -e "${BLUE}📁 Creating deployment directory...${NC}"
mkdir -p "$PAGES_REPO/$DEPLOY_DIR"

# Update public folder with latest from src
echo -e "${BLUE}🔄 Syncing files from src/ to public/...${NC}"
cp src/ai/kalah-ai-browser.js public/js/
cp src/ai/rl-agent-browser.js public/js/
cp src/engine/kalah-engine.js public/js/
rsync -av models/kalah-agent/ public/models/kalah-agent/
echo -e "${GREEN}✅ Sync complete!${NC}\n"

# Copy files
echo -e "${BLUE}📋 Copying files from $SOURCE_DIR to $PAGES_REPO/$DEPLOY_DIR...${NC}"
rsync -av --delete \
    --exclude='.git' \
    --exclude='.DS_Store' \
    "$SOURCE_DIR/" "$PAGES_REPO/$DEPLOY_DIR/"

echo -e "${GREEN}✅ Files copied successfully!${NC}\n"

# Navigate to pages repo
cd "$PAGES_REPO"

# Check git status
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${BLUE}📝 Git changes detected:${NC}"
    git status --short
    echo ""

    # Get commit message
    if [ -n "$1" ]; then
        COMMIT_MSG="$1"
    else
        COMMIT_MSG="Update Kalah RL game - $(date '+%Y-%m-%d %H:%M')"
    fi

    # Ask for confirmation
    read -p "Commit and push these changes? (y/n) " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${BLUE}💾 Committing changes...${NC}"
        git add "$DEPLOY_DIR"
        git commit -m "$COMMIT_MSG"

        echo -e "${BLUE}🚀 Pushing to GitHub...${NC}"
        git push origin main || git push origin master

        echo -e "${GREEN}✅ Deployment complete!${NC}"
        echo -e "${GREEN}🌐 Your game will be available at: https://angelkjos.github.io/$DEPLOY_DIR/${NC}"
    else
        echo -e "${BLUE}⏸️  Changes staged but not committed${NC}"
        echo "You can commit manually with:"
        echo "  cd $PAGES_REPO"
        echo "  git commit -m 'Update Kalah RL'"
        echo "  git push"
    fi
else
    echo -e "${GREEN}✅ No changes to deploy (files are already up to date)${NC}"
fi

echo ""
echo -e "${GREEN}🎮 Done!${NC}"
