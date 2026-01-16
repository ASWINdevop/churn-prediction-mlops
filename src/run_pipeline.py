import subprocess
import sys

def run_step(command, description):
    print(f"🚀 Starting: {description}...")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Error during: {description}")
        sys.exit(1)
    print(f"✅ Finished: {description}\n")

if __name__ == "__main__":

    run_step("python -m venv venv", "Virtual Environment Setup")

    run_step("venv\\Scripts\\activate", "Activating Virtual Environment")

    run_step("pip install -r requirements.txt", "Installing Dependencies")

    run_step("python src/ingest.py", "Data  Ingestion")
   
    run_step("python src/clean.py", "Data Cleaning ")
    
    run_step("python src/split.py", "Feature Splitting")
   
    run_step("python src/train_xgb.py", "Model Training and Serialization")
    
    print("🎉 Pipeline complete! You can now run the API with:")
    print("uvicorn app.main:app --reload")