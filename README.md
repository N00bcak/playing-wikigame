# playing-wikigame
Using Language Models to play the Wikipedia Game

## Getting Started
Unlike most other python projects, this project does **NOT** use `requirements.txt` for dependency management.

This is because the libraries used by this project don't quite play nicely together,
and manually specifying the order of operations within a script makes for a much smoother installation experience.

Steps to set up the environment:
1. Install Python 3.10 (for now, only 3.10 is supported)
2. Create a new virtual environment using `python -m venv <env_name>`
3. Activate the virtual environment using `source <env_name>/bin/activate`
4. Run `./handle-dependencies.sh` to install all dependencies

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