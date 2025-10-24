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

# Function to print colored messages
print_info() {
    printf "${BLUE}ℹ ${NC}$1\n" >&2
}

print_success() {
    printf "${GREEN}✓${NC} $1\n" >&2
}

print_warning() {
    printf "${YELLOW}⚠${NC} $1\n" >&2
}

print_error() {
    printf "${RED}✗${NC} $1\n" >&2
}

# Function to check if kiwix-tools is installed
check_kiwix() {
    if command -v kiwix-serve &> /dev/null; then
        print_success "kiwix-serve is already installed"
        return 0
    else
        print_warning "kiwix-serve is not installed"
        return 1
    fi
}

# Function to install kiwix-tools on Linux
install_kiwix() {
    print_info "Attempting to install kiwix-tools..."
    
    # Detect the package manager
    if command -v apt-get &> /dev/null; then
        print_info "Using apt package manager..."
        sudo apt-get update
        sudo apt-get install -y kiwix-tools
    elif command -v dnf &> /dev/null; then
        print_info "Using dnf package manager..."
        sudo dnf install -y kiwix-tools
    elif command -v yum &> /dev/null; then
        print_info "Using yum package manager..."
        sudo yum install -y kiwix-tools
    elif command -v pacman &> /dev/null; then
        print_info "Using pacman package manager..."
        sudo pacman -S --noconfirm kiwix-tools
    else
        print_error "Could not detect package manager"
        print_info "Please install kiwix-tools manually:"
        print_info "  Visit: https://www.kiwix.org/en/downloads/kiwix-serve/"
        exit 1
    fi
    
    if [ $? -eq 0 ]; then
        print_success "kiwix-tools installed successfully!"
    else
        print_error "Installation failed"
        exit 1
    fi
}

# Function to display language and variant information
show_options_info() {
    echo ""
    print_info "=== Wikipedia Language Options ==="
    echo "  Common languages:"
    echo "    en         - English (full Wikipedia, ~50GB nopic, ~100GB maxi)"
    echo "    en_simple     - Simple English (easier vocabulary, ~1GB nopic)"
    echo "    fr         - French"
    echo "    de         - German"
    echo "    es         - Spanish"
    echo "    ja         - Japanese"
    echo "    zh         - Chinese"
    echo ""
    print_info "  Find more language codes at: https://en.wikipedia.org/wiki/List_of_Wikipedias"
    echo ""
    print_info "=== Wikipedia Variants ==="
    echo "  maxi       - Full version with all images (~100GB+ for English)"
    echo "  nopic      - Complete articles without images (~50GB for English)"
    echo "  mini       - Only article introductions (~5GB for English)"
    echo ""
}

# Function to get the latest Wikipedia ZIM file
get_latest_zim_filename() {
    local language="$1"
    local variant="$2"
    
    print_info "Finding the latest Wikipedia ${language} (${variant}) ZIM file..."
    
    local latest_file=$(curl -s "$DOWNLOAD_URL/" | \
        grep -oE "wikipedia_${language}_all_${variant}_[0-9]{4}-[0-9]{2}\.zim" | \
        sort -u | \
        tail -1)
    
    if [ -z "$latest_file" ]; then
        print_error "Could not automatically detect the latest file for language '${language}' variant '${variant}'"
        print_info "Browse available files at: $DOWNLOAD_URL/"
        read -p "Enter the exact ZIM filename to download (or 'q' to quit): " manual_filename
        if [[ "$manual_filename" == "q" ]] || [[ -z "$manual_filename" ]]; then
            print_error "Cannot proceed without a valid filename. Exiting."
            exit 1
        fi
        echo "$manual_filename"
    else
        print_success "Found latest file: $latest_file"
        
        local filesize=$(curl -s "$DOWNLOAD_URL/" | grep "$latest_file" | grep -oE '[0-9]+[KMG]' | tail -1)
        if [ -n "$filesize" ]; then
            print_info "Approximate size: $filesize"
        fi
        
        echo "$latest_file"
    fi
}

# Function to download the ZIM file
download_zim() {
    local zim_filename="$1"
    local zim_path="$WIKIPEDIA_PATH/$zim_filename"
    
    # Create directory if it doesn't exist
    mkdir -p "$WIKIPEDIA_PATH"
    
    # Check if file already exists
    if [ -f "$zim_path" ]; then
        print_warning "ZIM file already exists at: $zim_path"
        read -p "Do you want to re-download it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Skipping download, using existing file"
            echo "$zim_path"
            return 0
        fi
    fi
    
    print_info "Downloading Wikipedia ZIM file..."
    print_warning "This may be a large file and will take a while!"
    print_info "Download URL: $DOWNLOAD_URL/$zim_filename"
    echo ""
    
    # Use wget if available, otherwise curl
    if command -v wget &> /dev/null; then
        wget -c --show-progress "$DOWNLOAD_URL/$zim_filename" -O "$zim_path"
    elif command -v curl &> /dev/null; then
        curl -C - -L -# -o "$zim_path" "$DOWNLOAD_URL/$zim_filename"
    else
        print_error "Neither wget nor curl is available. Please install one of them."
        exit 1
    fi
    
    if [ $? -eq 0 ]; then
        echo ""
        print_success "Download completed!"
        echo "$zim_path"
    else
        print_error "Download failed. Please check your internet connection and try again."
        exit 1
    fi
}

# Main script execution
download_setup() {
    echo ""
    print_info "=== Kiwix Wikipedia Offline Setup (Native) ==="
    echo ""
    
    # Step 1: Check for kiwix-serve
    if ! check_kiwix; then
        read -p "Would you like to install kiwix-tools now? (Y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            print_error "kiwix-serve is required. Exiting."
            exit 1
        fi
        install_kiwix
    fi
    
    # Step 2: Choose language and variant
    echo ""
    show_options_info
    
    read -p "Enter Wikipedia language code (default: $DEFAULT_LANGUAGE): " language
    language="${language:-$DEFAULT_LANGUAGE}"
    language=$(echo "$language" | tr '[:upper:]' '[:lower:]')
    
    read -p "Enter variant (maxi/nopic/mini, default: $DEFAULT_VARIANT): " variant
    variant="${variant:-$DEFAULT_VARIANT}"
    variant=$(echo "$variant" | tr '[:upper:]' '[:lower:]')
    
    # Validate variant
    if [[ ! "$variant" =~ ^(maxi|nopic|mini)$ ]]; then
        print_warning "Invalid variant '$variant'. Using default: $DEFAULT_VARIANT"
        variant="$DEFAULT_VARIANT"
    fi
    
    print_success "Selected: Wikipedia ${language} (${variant} variant)"
    
    # Step 3: Get latest ZIM filename
    echo ""
    zim_filename=$(get_latest_zim_filename "$language" "$variant")
    
    # Step 4: Download ZIM file
    echo ""
    read -p "Download Wikipedia now? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        zim_path=$(download_zim "$zim_filename")
    else
        zim_path="$WIKIPEDIA_PATH/$zim_filename"
        print_warning "Skipping download. Make sure you have: $zim_path"
    fi
    echo $zim_path
}

main() {
    echo ""
    print_info "Starting Kiwix server... with Wikipedia ZIM file: $1"
    print_success "=== Setup Complete! ==="
    print_info "Access Wikipedia at: http://localhost:$KIWIX_PORT"
    print_info "Press Ctrl+C to stop the server"
    echo ""
    
    # Run kiwix-serve in the background
    nohup kiwix-serve --port=$KIWIX_PORT "$1" -t $KIWIX_THREADS --address=127.0.0.1 > $WIKIPEDIA_PATH/kiwix.log 2>&1 &
    print_info "Server started in background. PID: $!"
    print_info "View logs: tail -f $WIKIPEDIA_PATH/kiwix.log"
}

echo "Do you want to download and set up a Kiwix Wikipedia ZIM file? (y/N): "
read -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    zim_path=$(download_setup)
else 
    print_info "Skipping download setup."
    read -p "Enter the full path to your existing ZIM file: " zim_path
    if [ ! -f "$zim_path" ]; then
        print_error "The specified ZIM file does not exist. Exiting."
        exit 1
    fi
fi

main "$zim_path"