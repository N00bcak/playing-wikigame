# playing-wikigame
Using Language Models to play the Wikipedia Game

## Getting Started
0. Use Linux.
1. Download dependencies:
   ```bash
   curl -fsSL https://uv.dev/install.sh | bash
   source ~/.uv/uv.sh
   uv install python@3.10
   uv use python@3.10
   sudo apt-get install libjemalloc2 kiwix-tools
   ```
2. Clone the repository:
   ```bash
   git clone https://github.com/N00bcak/playing-wikigame.git
   cd playing-wikigame
   ```
3. Install the required packages from uv:
   ```bash
   ./handle-dependencies.sh
   ```
4. Set up the Kiwix server:
   ```bash
   ./kiwix-server.sh
   ```

## Training an agent
1. Start the Kiwix server if not already running:
   ```bash
   ./kiwix-deploy.sh <YOUR_ZIMFILE_PATH_HERE>
   ```

2. Edit training script as necessary before running:
   ```bash
   ./train.sh
   ```

## License

This project uses [GEM](https://github.com/axon-rl/gem) and is licensed under the Apache 2.0 License:
```
Copyright 2025 AxonRL Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```