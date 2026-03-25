import subprocess
import argparse
def ui():
    subprocess.run(["streamlit", "run", "ui.py"])
def eval():
    subprocess.run(["streamlit", "run", "eval.py"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="name of file to run")

    args = parser.parse_args()
    if args.name == 'eval':
        eval()
    else:
        ui()
