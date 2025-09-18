#!/usr/bin/env bash
# Build script for Render static site deployment
set -o errexit

echo "🚀 Installing Python dependencies..."
pip install -r requirements.txt

echo "📊 Running EDA analysis and generating reports..."
python main.py

echo "📁 Copying generated files to public directory..."
mkdir -p public
cp index.html public/
cp eda_report.html public/
cp eda_report.txt public/
cp *.png public/ 2>/dev/null || echo "No PNG files to copy"

echo "✅ Build complete!"
ls -la public/
