# Structure from motion

A simple structure from motion pipeline 

## Usage

### Windows:

    py -3.10 -m venv .venv
    .\.venv\Scripts\activate
    pip install -r requirements.txt
    python .\main.py

If you get the error message 

    \.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system. For more information, see about_Execution_Policies at
    https:/go.microsoft.com/fwlink/?LinkID=135170.
    At line:1 char:1
    + .\.venv\Scripts\activate

use 

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

to allow the current PowerShell session to run scripts.

### Linux/macOS:


    python3.10 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python main.py
