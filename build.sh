#!/usr/bin/env bash
set -o errexit

echo "🚀 Building EDA Python static site..."

# Create public directory for static files
mkdir -p public

# Copy all static files to public directory
echo "📋 Copying files to public directory..."
cp index.html public/
cp eda_report.html public/
cp eda_report.txt public/
cp README.md public/ 2>/dev/null || echo "README not copied"

# Copy any PNG files if they exist
cp *.png public/ 2>/dev/null || echo "No PNG files to copy"

echo "✅ Build complete!"
echo "📁 Files in public directory:"
ls -la public/
