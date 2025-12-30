# VIBECODED & ONLY TESTED ON LINUX MACHINES.
#!/bin/bash

set -e  # Exit on error

# Color codes for prettier output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEFAULT_LANGUAGE="en"
DEFAULT_VARIANT="nopic"
KIWIX_PORT=8080
# More threads = Faster serving.
# Ultimately bandwidth-bound though :P
KIWIX_THREADS=16
DOWNLOAD_URL="https://download.kiwix.org/zim/wikipedia"
WIKIPEDIA_PATH="$HOME/bcy/wikipedia"
# If the server crashes more than this, we give up.
MAX_RESTARTS=100

# Function to print colored messages
print_info() {
    printf "${BLUE} ${NC}$1\n" >&2
}

print_success() {
    printf "${GREEN} ${NC} $1\n" >&2
}

print_warning() {
    printf "${YELLOW} ${NC} $1\n" >&2
}

print_error() {
    printf "${RED} ${NC} $1\n" >&2
}

# Function to keep kiwix alive with proper restart logic
keep_kiwix_alive() {
    local zim_file="$1"
    local restart_count=0
    
    while [ $restart_count -lt $MAX_RESTARTS ]; do
        restart_count=$((restart_count + 1))
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting kiwix-serve (attempt $restart_count/$MAX_RESTARTS)..." >> "$WIKIPEDIA_PATH/watchdog.log"
        
        # Start kiwix-serve and capture its PID
        kiwix-serve --port=$KIWIX_PORT "$zim_file" \
            -t $KIWIX_THREADS --address=127.0.0.1 \
            >> "$WIKIPEDIA_PATH/kiwix.log" 2>&1 &
        
        local kiwix_pid=$!
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server started with PID: $kiwix_pid" >> "$WIKIPEDIA_PATH/watchdog.log"
        
        # Wait for the process to exit
        wait $kiwix_pid
        local exit_code=$?
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server crashed with exit code: $exit_code" >> "$WIKIPEDIA_PATH/watchdog.log"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting in 1 second..." >> "$WIKIPEDIA_PATH/watchdog.log"
        sleep 1
    done
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Maximum restart limit ($MAX_RESTARTS) reached. Giving up." >> "$WIKIPEDIA_PATH/watchdog.log"
    exit 1
}

main() {
    local zim_file="$1"
    local daemon_mode="$2"
    
    if [ -z "$zim_file" ]; then
        print_error "No ZIM file specified"
        echo "Usage: $0 <path-to-zim-file>"
        exit 1
    fi
    
    if [ ! -f "$zim_file" ]; then
        print_error "ZIM file not found: $zim_file"
        exit 1
    fi
    
    mkdir -p "$WIKIPEDIA_PATH"
    
    # If not in daemon mode, start in background with nohup
    if [ "$daemon_mode" != "--daemon" ]; then
        echo ""
        print_success "=== Starting Kiwix Watchdog in Background ==="
        print_info "ZIM file: $zim_file"
        print_info "Port: $KIWIX_PORT"
        print_info "Threads: $KIWIX_THREADS"
        print_info "Watchdog log: $WIKIPEDIA_PATH/watchdog.log"
        print_info "Kiwix log: $WIKIPEDIA_PATH/kiwix.log"
        print_info "Access at: http://localhost:$KIWIX_PORT"
        echo ""
        
        # Re-execute this script in background with nohup
        nohup "$0" "$zim_file" --daemon > "$WIKIPEDIA_PATH/nohup.log" 2>&1 &
        local bg_pid=$!
        
        print_success "Watchdog started in background with PID: $bg_pid"
        print_info "Monitor logs with:"
        print_info "  tail -f $WIKIPEDIA_PATH/watchdog.log"
        print_info "  tail -f $WIKIPEDIA_PATH/kiwix.log"
        print_info ""
        print_info "Stop the server with:"
        print_info "  kill $bg_pid"
        echo ""
        
        exit 0
    fi
    
    # Daemon mode - run the keepalive loop
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Kiwix Watchdog Started in Daemon Mode ===" >> "$WIKIPEDIA_PATH/watchdog.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ZIM file: $zim_file" >> "$WIKIPEDIA_PATH/watchdog.log"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Port: $KIWIX_PORT" >> "$WIKIPEDIA_PATH/watchdog.log"
    
    keep_kiwix_alive "$zim_file"
}

main "$@"