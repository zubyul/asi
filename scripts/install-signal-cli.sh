#!/usr/bin/env bash
# Install signal-cli (native Linux build) v0.14.1
# This script downloads and installs the native GraalVM-compiled signal-cli binary,
# which does not require a Java runtime.

set -euo pipefail

SIGNAL_CLI_VERSION="0.14.1"
DOWNLOAD_URL="https://github.com/AsamK/signal-cli/releases/download/v${SIGNAL_CLI_VERSION}/signal-cli-${SIGNAL_CLI_VERSION}-Linux-native.tar.gz"
INSTALL_DIR="/usr/local/bin"

echo "Installing signal-cli v${SIGNAL_CLI_VERSION} (native build)..."

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

curl -L -o "$TMPDIR/signal-cli-native.tar.gz" "$DOWNLOAD_URL"
tar xzf "$TMPDIR/signal-cli-native.tar.gz" -C "$TMPDIR"

cp "$TMPDIR/signal-cli" "$INSTALL_DIR/signal-cli"
chmod +x "$INSTALL_DIR/signal-cli"

echo "Installed: $(signal-cli --version)"
