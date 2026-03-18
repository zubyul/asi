#!/usr/bin/env bash
# Run this script on your LOCAL machine (not in Claude Code cloud)
# It will:
# 1. Download signal-cli
# 2. Link it as a secondary device to your Signal account
# 3. Package the config for upload to Claude Code
set -euo pipefail

SIGNAL_CLI_VERSION="0.14.1"
INSTALL_DIR="$HOME/.local/bin"
CONFIG_DIR="$HOME/.local/share/signal-cli"

echo "=== Signal-CLI Local Setup for Claude Code ==="
echo ""

# Detect OS
OS="$(uname -s)"
case "$OS" in
  Linux)  ASSET="signal-cli-${SIGNAL_CLI_VERSION}-Linux-native.tar.gz" ;;
  Darwin) echo "macOS: use the Java version (requires Java 25) or build from source"; exit 1 ;;
  *)      echo "Unsupported OS: $OS"; exit 1 ;;
esac

# Download
mkdir -p "$INSTALL_DIR"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "[1/4] Downloading signal-cli v${SIGNAL_CLI_VERSION}..."
curl -L -o "$TMPDIR/signal-cli.tar.gz" \
  "https://github.com/AsamK/signal-cli/releases/download/v${SIGNAL_CLI_VERSION}/${ASSET}"
tar xzf "$TMPDIR/signal-cli.tar.gz" -C "$TMPDIR"
cp "$TMPDIR/signal-cli" "$INSTALL_DIR/signal-cli"
chmod +x "$INSTALL_DIR/signal-cli"
echo "   Installed: $INSTALL_DIR/signal-cli"

# Link
echo ""
echo "[2/4] Linking as secondary device..."
echo "   A tsdevice:// URI will appear below."
echo "   On your phone: Signal > Settings > Linked Devices > Link New Device"
echo "   Then scan the QR code (or paste the URI into a QR code generator)."
echo ""
echo "   Waiting for link URI..."
"$INSTALL_DIR/signal-cli" link -n "ClaudeCode"

echo ""
echo "[3/4] Link successful!"

# Package config
echo ""
echo "[4/4] Packaging config for Claude Code upload..."
OUTPUT="$HOME/signal-cli-config.tar.gz"
tar czf "$OUTPUT" -C "$HOME/.local/share" signal-cli/
echo "   Config saved to: $OUTPUT"
echo ""
echo "=== Done! ==="
echo ""
echo "Next steps:"
echo "  1. Upload $OUTPUT to your Claude Code session"
echo "  2. In Claude Code, tell me: 'I uploaded the signal config'"
echo "  3. I'll extract it and enable the MCP server"
